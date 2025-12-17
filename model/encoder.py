import torch
import torch.nn as nn

class DrugEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DrugPairEncoder(nn.Module):
    def __init__(self, drug_encoder):
        super().__init__()
        self.drug_encoder = drug_encoder

    def forward(self, drug_a, drug_b):
        z_a = self.drug_encoder(drug_a)
        z_b = self.drug_encoder(drug_b)
        return torch.cat([z_a, z_b], dim=-1)
