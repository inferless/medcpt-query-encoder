import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer, AutoModel


class InferlessPythonModel:
    def initialize(self):
        model_id = "ncbi/MedCPT-Query-Encoder"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.model = AutoModel.from_pretrained(model_id,device_map="cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

    def infer(self, inputs):
        queries = inputs["queries"]
        embeds = None
        with torch.no_grad():
            encoded = self.tokenizer(
                queries,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=64,
            ).to("cuda")

            embeds = self.model(**encoded).last_hidden_state[:, 0, :]

        return {"embeds": embeds.tolist()}

    def finalize(self):
        self.model = None
