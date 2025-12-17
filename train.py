import torch
from torch.utils.data import DataLoader

from data.load_data import DrugPairDataset
from model.augmentations import make_views
from model.encoder import DrugEncoder, DrugPairEncoder
from model.loss import info_nce_loss


def main():
    # Dataset and loader
    dataset = DrugPairDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Model
    drug_encoder = DrugEncoder(input_dim=128, hidden_dim=256, output_dim=128)
    pair_encoder = DrugPairEncoder(drug_encoder)

    # Optimizer
    optimizer = torch.optim.Adam(pair_encoder.parameters(), lr=1e-3)

    # Training loop
    pair_encoder.train()
    num_epochs = 3

    for epoch in range(num_epochs):
        total_loss = 0.0

        for drug_a, drug_b in loader:
            # Create two contrastive views
            (v1_a, v1_b), (v2_a, v2_b) = make_views(drug_a, drug_b)

            # Encode drug pairs
            z1 = pair_encoder(v1_a, v1_b)
            z2 = pair_encoder(v2_a, v2_b)

            # Contrastive loss
            loss = info_nce_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

    # ✅ SAVE TRAINED MODEL (IMPORTANT FOR DAY 3)
    torch.save(pair_encoder.state_dict(), "pair_encoder.pt")
    print("Saved trained encoder to pair_encoder.pt")


if __name__ == "__main__":
    main()
