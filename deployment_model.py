import logging
import re

import numpy as np
import torch
from djl_python.encode_decode import decode, encode
from djl_python.inputs import Input
from djl_python.outputs import Output
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, Pipeline, PreTrainedTokenizer)

device = "cuda"
model = None
tokenizer = None

"""
##########################################################################
        Instruct Pipeline
##########################################################################
The pipeline defines the inference prompt, the special token ID, and it  overrides the `_sanitize_parameters`, `preprocess`, `_forward`, and `postprocess` methods inherited from the base Pipeline class. 

The pipeline's preprocessing step tokenizes the instruction text, adds prompt text, and returns the processed inputs. The postprocessing step decodes the generated sequence to extract the response text. It uses the response key and end key to locate and extract the response portion from the generated tokens. If the keys are not found, regular expressions are used to extract the response. 
"""

# Create Instruct Pipeline

logger = logging.getLogger(__name__)

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
INFERENCE_PROMPT = """{intro}
                    {instruction_key}
                    {instruction}
                    {response_key}
                    """.format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.
    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.
    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token
    Raises:
        RuntimeError: if more than one ID was generated
    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(
            f"Expected only a single token for '{key}' but found {token_ids}"
        )
    return token_ids[0]


class InstructionTextGenerationPipeline(Pipeline):
    def __init__(
        self,
        *args,
        do_sample: bool = True,
        max_new_tokens: int = 256,
        top_p: float = 0.92,
        top_k: int = 0,
        **kwargs,
    ):
        super().__init__(
            *args,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )

    def _sanitize_parameters(self, return_instruction_text=False, **generate_kwargs):
        preprocess_params = {}

        # newer versions of the tokenizer configure the response key as a special token.  newer versions still may
        # append a newline to yield a single token.  find whatever token is configured for the response key.
        tokenizer_response_key = next(
            (
                token
                for token in self.tokenizer.additional_special_tokens
                if token.startswith(RESPONSE_KEY)
            ),
            None,
        )

        response_key_token_id = None
        end_key_token_id = None
        if tokenizer_response_key:
            try:
                response_key_token_id = get_special_token_id(
                    self.tokenizer, tokenizer_response_key
                )
                end_key_token_id = get_special_token_id(self.tokenizer, END_KEY)

                # Ensure generation stops once it generates "### End"
                generate_kwargs["eos_token_id"] = end_key_token_id
            except ValueError:
                pass

        forward_params = generate_kwargs
        postprocess_params = {
            "response_key_token_id": response_key_token_id,
            "end_key_token_id": end_key_token_id,
            "return_instruction_text": return_instruction_text,
        }

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, instruction_text, **generate_kwargs):
        prompt_text = INFERENCE_PROMPT.format(instruction=instruction_text)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        inputs["prompt_text"] = prompt_text
        inputs["instruction_text"] = instruction_text
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        generated_sequence = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )[0].cpu()
        instruction_text = model_inputs.pop("instruction_text")
        return {
            "generated_sequence": generated_sequence,
            "input_ids": input_ids,
            "instruction_text": instruction_text,
        }

    def postprocess(
        self,
        model_outputs,
        response_key_token_id,
        end_key_token_id,
        return_instruction_text,
    ):
        sequence = model_outputs["generated_sequence"]
        instruction_text = model_outputs["instruction_text"]

        # The response will be set to this variable if we can identify it.
        decoded = None

        # If we have token IDs for the response and end, then we can find the tokens and only decode between them.
        if response_key_token_id and end_key_token_id:
            # Find where "### Response:" is first found in the generated tokens.  Considering this is part of the
            # prompt, we should definitely find it.  We will return the tokens found after this token.
            response_pos = None
            response_positions = np.where(sequence == response_key_token_id)[0]
            if len(response_positions) == 0:
                logger.warn(
                    f"Could not find response key {response_key_token_id} in: {sequence}"
                )
            else:
                response_pos = response_positions[0]

            if response_pos:
                # Next find where "### End" is located.  The model has been trained to end its responses with this
                # sequence (or actually, the token ID it maps to, since it is a special token).  We may not find
                # this token, as the response could be truncated.  If we don't find it then just return everything
                # to the end.  Note that even though we set eos_token_id, we still see the this token at the end.
                end_pos = None
                end_positions = np.where(sequence == end_key_token_id)[0]
                if len(end_positions) > 0:
                    end_pos = end_positions[0]

                decoded = self.tokenizer.decode(
                    sequence[response_pos + 1 : end_pos]
                ).strip()
        else:
            # Otherwise we'll decode everything and use a regex to find the response and end.

            fully_decoded = self.tokenizer.decode(sequence)

            # The response appears after "### Response:".  The model has been trained to append "### End" at the
            # end.
            m = re.search(
                r"#+\s*Response:\s*(.+?)#+\s*End", fully_decoded, flags=re.DOTALL
            )

            if m:
                decoded = m.group(1).strip()
            else:
                # The model might not generate the "### End" sequence before reaching the max tokens.  In this case,
                # return everything after "### Response:".
                m = re.search(r"#+\s*Response:\s*(.+)", fully_decoded, flags=re.DOTALL)
                if m:
                    decoded = m.group(1).strip()
                else:
                    logger.warn(f"Failed to find response in:\n{fully_decoded}")

        if return_instruction_text:
            return {"instruction_text": instruction_text, "generated_text": decoded}

        return decoded


def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=256,
    **kwargs,
):
    dolly_gen = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
    response = dolly_gen(instruction)
    return response


def load_base_model(adapter_checkpoint, adapter_name):
    model_name = "databricks/dolly-v2-3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="balanced",
        load_in_8bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter_checkpoint, adapter_name)
    return model, tokenizer


def inference(inputs: Input):
    json_input = decode(inputs, "application/json")
    sequence = json_input.get("inputs")
    generation_kwargs = json_input.get("parameters", {})
    output = Output()
    outs = evaluate(sequence)
    encode(output, outs, "application/json")
    return output


def handle(inputs: Input):
    """
    Default handler function
    """
    global model, tokenizer
    if not model:
        # stateful model
        props = inputs.get_properties()
        model, tokenizer = load_base_model(
            props.get("adapter_checkpoint"), props.get("adapter_name")
        )

    if inputs.is_empty():
        # initialization request
        return None

    return inference(inputs)
