import torch
from transformers import AutoTokenizer, AutoModel


class InferlessPythonModel:
    def initialize(self):
        self.model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
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
            )

            embeds = self.model(**encoded).last_hidden_state[:, 0, :]

        return {"embeds": embeds.tolist()}

    def finalize(self):
        self.tokenizer = None
        self.model = None
