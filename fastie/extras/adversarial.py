import torch
from transformers import PreTrainedModel


class FGM:
    def __init__(self, model: PreTrainedModel, emb_name: str = "word_embeddings"):
        self.model = model
        self.backup = {}
        self.emb_name = emb_name

    def attack(self, epsilon: float = 1.0):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
