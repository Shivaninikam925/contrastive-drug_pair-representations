import os
import sys

# Always add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.load_data import DrugPairDataset
from model.augmentations import make_views

def main():
    dataset = DrugPairDataset()
    drug_a, drug_b = dataset[0]

    (v1_a, v1_b), (v2_a, v2_b) = make_views(drug_a, drug_b)

    print("View 1 A shape:", v1_a.shape)
    print("View 1 B shape:", v1_b.shape)
    print("View 2 A shape:", v2_a.shape)
    print("View 2 B shape:", v2_b.shape)

if __name__ == "__main__":
    main()
