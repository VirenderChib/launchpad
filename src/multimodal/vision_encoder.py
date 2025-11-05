from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image

class VisionEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
