import torch
from torch.utils.data import Dataset

class DrugPairDataset(Dataset):
    def __init__(self, num_pairs=1000, feature_dim=128):
        self.num_pairs = num_pairs
        self.feature_dim = feature_dim

        # Simulated drug feature vectors
        self.drug_features = torch.randn(num_pairs * 2, feature_dim)

        # Pair indices
        self.pairs = [(2 * i, 2 * i + 1) for i in range(num_pairs)]

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        idx_a, idx_b = self.pairs[idx]
        drug_a = self.drug_features[idx_a]
        drug_b = self.drug_features[idx_b]
        return drug_a, drug_b
