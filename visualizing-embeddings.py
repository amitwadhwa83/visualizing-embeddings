import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


def get_embeddings(sentences, model_name='all-MiniLM-L6-v2'):
    """Generates embeddings for given sentences using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    return np.array(model.encode(sentences))


def plot_heatmap(embeddings):
    """Plots a heatmap of pairwise cosine similarities between sentence embeddings."""
    similarity_matrix = np.dot(embeddings, embeddings.T)

    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Sentence Embeddings Similarity Heatmap")
    plt.show()


def plot_embeddings(embeddings, sentences, method='pca', random_state=42):
    """
    Plots 2D visualization of embeddings using PCA with labels for each point.

    :param embeddings: ndarray of shape (n_samples, n_features), input embeddings.
    :param sentences: List of original sentences corresponding to the embeddings.
    :param method: 'pca' for PCA.
    :param random_state: Random state for reproducibility.
    """
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        raise ValueError("Only 'pca' method is supported")

    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for i, sentence in enumerate(sentences):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], label=f"S{i + 1}")
        plt.text(reduced_embeddings[i, 0] + 0.02, reduced_embeddings[i, 1] + 0.02, sentence, fontsize=9)

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Embedding Visualization using PCA")
    plt.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    sentences = ["Missing flamingo discovered at swimming pool",
                 "Sea otter spotted on surfboard by beach",
                 "Baby panda enjoys boat ride",
                 "Breakfast themed food truck beloved by all!",
                 "New curry restaurant aims to please!",
                 "Python developers are wonderful people",
                 "TypeScript, C++ or Java? All are great!"
    ]

    embeddings = get_embeddings(sentences)
    #plot_heatmap(embeddings)
    plot_embeddings(embeddings, sentences)
