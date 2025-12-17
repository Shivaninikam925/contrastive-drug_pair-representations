import torch

def feature_masking(x, mask_ratio=0.2):
    mask = torch.rand_like(x) > mask_ratio
    return x * mask

def add_noise(x, noise_std=0.1):
    noise = torch.randn_like(x) * noise_std
    return x + noise

def make_views(drug_a, drug_b):
    # First augmented view
    view1_a = feature_masking(drug_a)
    view1_b = add_noise(drug_b)

    # Second augmented view
    view2_a = add_noise(drug_a)
    view2_b = feature_masking(drug_b)

    return (view1_a, view1_b), (view2_a, view2_b)
