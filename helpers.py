from typing import Any, Dict, List, Tuple, Union

import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Pipeline,
                          PreTrainedTokenizer, TextDataset)

"""
##########################################################################
        Helper Variables
##########################################################################
"""
# The following variables are used to decorate the instruction and responses for training and inference
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
DEFAULT_SEED = 42

PROMPT = """{intro}
            {instruction_key}
            {instruction}
            {response_key}
            {response}
            {end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

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


"""
##########################################################################
        Helper Functions
##########################################################################
"""


# Function to generate token embeddings
def mlu_preprocess_batch(
    batch: Dict[str, List], tokenizer: AutoTokenizer, max_length: int
) -> dict:
    model_inputs = tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )
    return model_inputs


def mlu_get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
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


"""
##########################################################################
        Helper Classes
##########################################################################
"""

"""
Let's define a custom data collator class named `DataCollatorForCompletionOnlyLM` for a language modeling task. It extends the functionality of the base `DataCollatorForLanguageModeling` class from transformers library. This custom collator is designed to handle examples where a prompt is followed by a response in the input text and modifies the labels accordingly.
"""


class MLUDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:

        # The torch_call method overrides the same method in the base class and
        # takes a list of examples as input.
        batch = super().torch_call(examples)

        # The code then encodes a special token, RESPONSE_KEY_NL,
        # representing the end of the prompt followed by a newline.
        # It searches for this token in the sequence of tokens (labels)
        # and finds its index.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                # If the response token is not found in the sequence, it raises a RuntimeError.
                # Otherwise, it determines the end index of the response token.
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs \
                    {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # To train the model to predict only the response and ignore the prompt tokens,
            # it sets the label values before the response token to -100.
            # This ensures that those tokens are ignored by the PyTorch loss function during training.
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


"""
Let's define a custom pipeline `MLUInstructionTextGenerationPipeline` for instruction text generation using the Huggingface transformers library. The pipeline defines the inference prompt, the special token ID, and it  overrides the `_sanitize_parameters`, `preprocess`, `_forward`, and `postprocess` methods inherited from the base Pipeline class. 

The pipeline's preprocessing step tokenizes the instruction text, adds prompt text, and returns the processed inputs. The postprocessing step decodes the generated sequence to extract the response text. It uses the response key and end key to locate and extract the response portion from the generated tokens. If the keys are not found, regular expressions are used to extract the response. 
"""


class MLUInstructionTextGenerationPipeline(Pipeline):
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
                response_key_token_id = mlu_get_special_token_id(
                    self.tokenizer, tokenizer_response_key
                )
                end_key_token_id = mlu_get_special_token_id(self.tokenizer, END_KEY)

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
