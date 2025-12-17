import torch
import numpy as np
from torch.utils.data import DataLoader

from data.load_data import DrugPairDataset
from model.encoder import DrugEncoder, DrugPairEncoder

def extract_embeddings(pair_encoder, loader):
    pair_encoder.eval()
    embeddings = []

    with torch.no_grad():
        for drug_a, drug_b in loader:
            z = pair_encoder(drug_a, drug_b)
            embeddings.append(z)

    return torch.cat(embeddings, dim=0).cpu().numpy()

def main():
    dataset = DrugPairDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Load trained encoder (reuse architecture)
    drug_encoder = DrugEncoder()
    pair_encoder = DrugPairEncoder(drug_encoder)

    # Load weights from Day 2 training
    pair_encoder.load_state_dict(torch.load("pair_encoder.pt"))

    learned_embeddings = extract_embeddings(pair_encoder, loader)

    # Random baseline
    random_embeddings = np.random.randn(
        learned_embeddings.shape[0],
        learned_embeddings.shape[1]
    )

    np.save("learned_embeddings.npy", learned_embeddings)
    np.save("random_embeddings.npy", random_embeddings)

    print("Embeddings saved:")
    print("learned_embeddings.npy")
    print("random_embeddings.npy")

if __name__ == "__main__":
    main()
