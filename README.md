# Customer Support Chatbot (LLaMA Fine-Tuning Project)

## Objective
To build and fine-tune a LLaMA-based chatbot capable of assisting employees of a tech company with technical queries such as software troubleshooting, access issues, and system guidance.

## Dataset
**Primary:** [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)  


The dataset consists of human-generated instruction–response pairs across technical and workplace topics, making it suitable for training a corporate support chatbot.

## Model
- **Base Model:** LLaMA (Meta AI)
- **Fine-tuning method:** LoRA (Low-Rank Adaptation) via Hugging face
- **Frameworks:** PyTorch, Hugging Face Transformers, bitsandbytes, peft
- **Deployment target:** Gradio (local UI) or Hugging Face TGI (cloud)

## Business Use Case
This chatbot automates internal tech support for employees by providing quick, context-aware answers to common queries about IT systems, tools, and troubleshooting steps. It can reduce support workload and improve employee experience.

Multimodal Pipline
1. Vision Encoder

Model: openai/clip-vit-base-patch32

Converts input image → visual embeddings.

Defined in: src/multimodal/vision_encoder.py

2. Projection Head

Linear layer that maps visual embeddings → text embedding space.

Learns alignment between image and text.

Defined in: src/multimodal/projection_head.py

3. Multimodal Dataset Preparation

Uses mm_cleaned.jsonl (contains image path + caption text).

Generates:

train_mm.pt

validation_mm.pt

test_mm.pt

Defined in: src/multimodal/multimodal_dataset.py

4. Data Loading

Converts .pt data → Hugging Face Dataset objects.

Defined in: src/multimodal/data_loader.py

5. Fine-Tuning

Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

Adds LoRA adapters to fine-tune efficiently.

Uses image-text embeddings for multimodal learning.

Defined in: src/multimodal/finetune_multimodal.py

6. Evaluation

Computes BLEU and ROUGE on validation/test samples.

Uses small subset for quick evaluation.
Defined in: src/multimodal/evaluate_multimodal.py

7. Multimodal 

Fine tunned the model on image specific dataset using vision encoder

8. Inference

Used vllm and TGI for faster inference

9. Docker

Created a docker image containing my docker file, requirement.txt, Multimodal, shell file.
Pushed the docker image to docker hub and finally runing the fine tunned Multimodal on local host.

