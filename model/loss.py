import torch
import torch.nn.functional as F

def info_nce_loss(z1, z2, temperature=0.5):
    """
    z1, z2: (batch_size, embedding_dim)
    """
    batch_size = z1.size(0)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)  # (2B, D)

    similarity_matrix = torch.matmul(representations, representations.T)

    # mask self-similarity
    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

    similarity_matrix /= temperature

    # positive pairs: i <-> i+B
    targets = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(z1.device)

    loss = F.cross_entropy(similarity_matrix, targets)
    return loss
