import numpy as np
import matplotlib.pyplot as plt
import umap

def plot(embeddings, title, filename):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(5, 5))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=5, alpha=0.6)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    learned = np.load("learned_embeddings.npy")
    random = np.load("random_embeddings.npy")

    plot(learned, "Contrastive Drug Pair Embeddings", "contrastive_embeddings.png")
    plot(random, "Random Embeddings (Baseline)", "random_embeddings.png")

    print("Plots saved:")
    print("contrastive_embeddings.png")
    print("random_embeddings.png")

if __name__ == "__main__":
    main()
