import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from FeatureExtractor import FeatureExtractor
from HashNetwork import Embedder
from build_faiss_db import FaissDatabase, load_model


def get_mnist_image(index):
    """
    Get a specific MNIST image by index
    
    Args:
        index: Index of the image
        is_test: Whether the image is from the test set
        
    Returns:
        image, label
    """
    test_dataset = MNIST(root="/data", train=False, download=True)
    
    return test_dataset.data[623], test_dataset.targets[623]

def get_images(similar_index):

    train_dataset = MNIST(root = "/data" , train = True , download = True)

    images = [train_dataset.data[idx] for idx in similar_index]
    labels = [train_dataset.targets[idx] for idx in similar_index]


    return images , labels


def query_similar_images(query_index, db, extractor, embedder, device, k=10):
    """
    Query the database for similar images
    
    Args:
        query_index: Index of the query image
        db: FAISS database
        extractor: Feature extractor model
        embedder: Hash embedding model
        device: Device to run inference on
        k: Number of similar images to retrieve
        
    Returns:
        query_image, query_label, similar_images, similar_labels, distances
    """

    query_image, query_label = get_mnist_image(query_index)
    
    with torch.no_grad():
        image_tensor = query_image.unsqueeze(0).unsqueeze(0).float().to(device) / 255.0  # Normalize
        
        features = extractor(image_tensor)
        hash_vector = embedder.embedder(features)
        hash_vector_np = hash_vector.cpu().numpy().astype('float32')
    
    distances, similar_indices = db.search(hash_vector_np, k)
    
    similar_images = []
    similar_labels = []

    images , labels = get_images(similar_indices)
    
    for id , idx in enumerate(similar_indices):  
        img, label = get_mnist_image(idx)
        similar_images.append(images[id])
        similar_labels.append(labels[id])
    
    return query_image, query_label, similar_images, similar_labels, distances

def plot_similar_images(query_image, query_label, similar_images, similar_labels, distances):
    """
    Plot query image and similar images
    
    Args:
        query_image: Query image
        query_label: Query label
        similar_images: List of similar images
        similar_labels: List of similar labels
        distances: List of distances
    """
    n_similar = len(similar_images)
    fig, axes = plt.subplots(1, n_similar + 1, figsize=(15, 3))
    
    axes[0].imshow(query_image, cmap='gray')
    axes[0].set_title(f"Query: {query_label}")
    axes[0].axis('off')
    
    for i, (img, label, dist) in enumerate(zip(similar_images, similar_labels, distances)):
        axes[i+1].imshow(img, cmap='gray')
        axes[i+1].set_title(f"#{i+1}: {label}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"similar_to_{query_label}.png")
    plt.show()

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    extractor, embedder = load_model("./model/best_model1.pt", device)
    
    db = FaissDatabase.load('./faiss_database/mnist_hash', use_gpu=torch.cuda.is_available())
    
    query_image, query_label, similar_images, similar_labels, distances = query_similar_images(
        0, db, extractor, embedder, device, 5
    )
    
    plot_similar_images(query_image, query_label, similar_images, similar_labels, distances)
    
    print(f"Found {len(similar_images)} similar images to query image with label {query_label}")
    print(f"Results saved to similar_to_{query_label}.png")


if __name__ == "__main__":
    main()