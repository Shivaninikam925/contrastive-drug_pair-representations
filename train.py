import torch
from torch.utils.data import DataLoader

from data.load_data import DrugPairDataset
from model.augmentations import make_views
from model.encoder import DrugEncoder, DrugPairEncoder
from model.loss import info_nce_loss

def main():
    dataset = DrugPairDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    drug_encoder = DrugEncoder()
    pair_encoder = DrugPairEncoder(drug_encoder)

    optimizer = torch.optim.Adam(pair_encoder.parameters(), lr=1e-3)

    for epoch in range(3):  # very small training
        total_loss = 0.0
        for drug_a, drug_b in loader:
            (v1_a, v1_b), (v2_a, v2_b) = make_views(drug_a, drug_b)

            z1 = pair_encoder(v1_a, v1_b)
            z2 = pair_encoder(v2_a, v2_b)

            loss = info_nce_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

if __name__ == "__main__":
    main()
