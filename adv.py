import torch


class FGM:
    def __init__(self, model, noise_budget=0.1):
        self.model = model
        self.backup = {}
        self.noise_budget = noise_budget
    
    def open(self):
        return self.noise_budget > 1e-4
    
    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name.startswith('embeddings'):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = (self.noise_budget / norm) * param.grad
                    param.data.add_(r_at)
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name.startswith('embeddings'):
                assert name in self.backup
                param.data = self.backup[name]
        self.backup.clear()
