import torch
import faiss
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os
import pickle
from tqdm import tqdm
from FeatureExtractor import FeatureExtractor
from HashNetwork import Embedder
from torchvision.transforms import transforms

class FaissDatabase:
    def __init__(self, hash_dim, use_gpu=False):
        """
        Initialize FAISS database
        
        Args:
            hash_dim: Dimension of hash vectors
            use_gpu: Whether to use GPU for FAISS
        """
        self.hash_dim = hash_dim
        self.use_gpu = use_gpu
        self.index = faiss.IndexFlatL2(hash_dim)  
        
        if use_gpu and faiss.get_num_gpus() > 0:
            print(f"Using GPU for FAISS")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.image_ids = []  
    
    def add_vectors(self, vectors, image_ids):
        """
        Add vectors to the index
        
        Args:
            vectors: numpy array of vectors to add (n x hash_dim)
            image_ids: list of image IDs corresponding to vectors
        """
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.cpu().numpy()
        
        self.index.add(vectors)
        self.image_ids.extend(image_ids)
    
    def search(self, query_vector, k=10):
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector
            k: Number of similar vectors to retrieve
            
        Returns:
            distances, indices
        """
        if isinstance(query_vector, torch.Tensor):
            query_vector = query_vector.cpu().numpy()
        
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        distances, indices = self.index.search(query_vector, k)
        
        image_indices = [self.image_ids[idx] for idx in indices[0] if idx >= 0 and idx < len(self.image_ids)]
        
        return distances[0], image_indices
    
    def save(self, filepath):
        """
        Save the FAISS index and image IDs to disk
        
        Args:
            filepath: Path to save the database
        """
        if self.use_gpu and hasattr(faiss, "index_gpu_to_cpu"):
            index_cpu = faiss.index_gpu_to_cpu(self.index)
        else:
            index_cpu = self.index
            
        faiss.write_index(index_cpu, f"{filepath}.index")
        
        with open(f"{filepath}.pkl", "wb") as f:
            pickle.dump(self.image_ids, f)
        
        print(f"Database saved to {filepath}.index and {filepath}.pkl")
    
    @classmethod
    def load(cls, filepath, use_gpu=False):
        """
        Load a FAISS index and image IDs from disk
        
        Args:
            filepath: Path to load the database from
            use_gpu: Whether to use GPU for FAISS
            
        Returns:
            FaissDatabase instance
        """
        index = faiss.read_index(f"{filepath}.index")
        
        with open(f"{filepath}.pkl", "rb") as f:
            image_ids = pickle.load(f)
        
        db = cls(index.d, use_gpu)
        
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            db.index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            db.index = index
        
        db.image_ids = image_ids
        print(f"Database loaded from {filepath}.index and {filepath}.pkl")
        print(f"Database contains {len(image_ids)} vectors of dimension {index.d}")
        
        return db


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load the model on
        
    Returns:
        extractor, embedder
    """
    extractor = FeatureExtractor().to(device)
    embedder = Embedder(device=device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    extractor.load_state_dict(checkpoint['extractor_state_dict'])
    embedder.embedder.load_state_dict(checkpoint['embedder_state_dict'])
    
    extractor.eval()
    embedder.embedder.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    return extractor, embedder


def generate_hash_vectors(extractor, embedder, dataset, device, batch_size=256):
    """
    Generate hash vectors for all images in the dataset
    
    Args:
        extractor: Feature extractor model
        embedder: Hash embedding model
        dataset: MNIST dataset
        device: Device to run inference on
        batch_size: Batch size for inference
        
    Returns:
        hash_vectors: Tensor of hash vectors
        image_ids: List of image indices
    """
 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    hash_vectors = []
    image_ids = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Generating hash vectors")):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch_indices = list(range(start_idx, end_idx))
            
            data = data.to(device)
            
            features = extractor(data)
            hashes = embedder.embedder(features)
            
            hash_vectors.append(hashes.cpu())
            image_ids.extend(batch_indices)
    
    hash_vectors = torch.cat(hash_vectors, dim=0)
    
    return hash_vectors, image_ids


def main():
    CHECKPOINT_PATH = "./model/best_model.pt"
    DB_PATH = "./faiss_database/mnist_hash"
    BATCH_SIZE = 256
    USE_GPU = torch.cuda.is_available()
    
    transform = transforms.Compose(
       [ transforms.ToTensor(),]
    )
    
    
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    device = torch.device("cuda" if USE_GPU else "cpu")
    print(f"Using device: {device}")
    
    extractor, embedder = load_model(CHECKPOINT_PATH, device)
    
    with torch.no_grad():
        dummy_input = torch.zeros(1, 1, 28, 28).to(device)
        features = extractor(dummy_input)
        hash_vector = embedder.embedder(features)
        hash_dim = hash_vector.shape[1]
    
    print(f"Hash dimension: {hash_dim}")
    
    train_dataset = MNIST(root="/data", train=True, download=True , transform=transform)
    
    db = FaissDatabase(hash_dim, use_gpu=USE_GPU)
    
    print("Processing training set...")
    train_vectors, train_ids = generate_hash_vectors(extractor, embedder, train_dataset, device, BATCH_SIZE)
    db.add_vectors(train_vectors, train_ids)
    
    db.save(DB_PATH)
    
    print(f"FAISS database created with {len(db.image_ids)} vectors")
    print(f"Database saved to {DB_PATH}")


if __name__ == "__main__":
    main()