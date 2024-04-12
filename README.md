Part of the AWS ML Certification

Fine-tuned(Transfer Learning) a large language model (LLM) on SageMaker. Utilized LoRa for efficient parameter reduction and transfer learning for fast inference with CloudFront. Leveraged a pre-trained model (databricks/dolly-v2-3b) from Hugging Face for optimal performance and resource conservation.


![Infrastructure](https://github.com/marlhex/AmazonSageMaker_LLM_JumpStart_FoundationModel_HuggingFace_FasterInference_LoRa_DatabricksDolly-V2-3B/assets/4165637/b90cb421-3ee7-4fdf-916e-252cf19d9758)


accelerate>=0.20.3,<1
bitsandbytes==0.39.0
click>=8.0.4,<9
datasets>=2.10.0,<3
deepspeed>=0.8.3,<0.9
faiss-cpu==1.7.4
ipykernel==6.22.0
langchain==0.0.161
torch>=1.13.1,<2
transformers==4.28.1
peft==0.3.0
pytest==7.3.2
numpy>=1.25.2
