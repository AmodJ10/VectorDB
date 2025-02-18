import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from typing import Dict, List, Generator
from collections import defaultdict
import torch

def batch_generator(data: np.ndarray, labels: np.ndarray, batch_size: int) -> Generator:
    num_samples = len(data)
    indices = np.arange(num_samples)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield data[batch_indices], labels[batch_indices], batch_indices


def batch_simhash(feature_vectors: np.ndarray, hash_bits: int = 16, seed: int = 42) -> np.ndarray:
    batch_size, feature_dim = feature_vectors.shape

    norms = np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    feature_vectors = feature_vectors / (norms + 1e-10)  

    np.random.seed(seed)

    random_matrix = np.random.randn(feature_dim, hash_bits)
    random_matrix, _ = np.linalg.qr(random_matrix)  

    projections = np.dot(feature_vectors, random_matrix)

    binary_hashes = (projections > np.median(projections, axis=0)).astype(np.uint64)

    simhashes = np.zeros(batch_size, dtype=np.uint64)
    for i in range(hash_bits):
        simhashes |= binary_hashes[:, i] << i  

    return simhashes


def cluster_similar_images_batch(
    feature_vectors: np.ndarray, 
    labels: np.ndarray, 
    batch_size: int = 1000,
    hash_bits: int = 16,
) -> Dict[int, List[tuple]]:
 
    clusters = defaultdict(list)
    total_processed = 0
    
    for batch_data, batch_labels, batch_indices in batch_generator(feature_vectors, labels, batch_size):
        batch_hashes = batch_simhash(batch_data , hash_bits)
        
        for i, hash_value in enumerate(batch_hashes):
            clusters[int(hash_value)].append(batch_data[i])
        
        total_processed += len(batch_data)
    
    return clusters


def print_top_k_results(hash_list : list , k : int):
        print(f"Hash size: {hash_list[i][0]} , Similarity: {hash_list[i][1]}")

def create_dic(binary_hash_buckets):

    dic = {}

    for i , hash in enumerate(binary_hash_buckets):
        if hash in dic.keys():
            dic[hash].append(train_features[i].numpy())
            continue
        dic[hash] = [train_features[i].numpy()]
    
    return dic

def average_similarity_all_keys(dictionary):
    avg_similarity_scores = 0

    for hash_key, vectors in dictionary.items():
        vectors = np.array(vectors)  
        if len(vectors) < 2:
            continue

        similarity_matrix = cosine_similarity(vectors)

        triu_indices = np.triu_indices(len(vectors), k=1)
        similarity_scores = similarity_matrix[triu_indices]

        avg_similarity_scores += np.mean(similarity_scores) if similarity_scores.size > 0 else None

    return avg_similarity_scores/len(dictionary)

n_samples = 10000
feature_dim = 128
np.random.seed(42)

train_features = torch.load("database/train_features.pt")
train_labels = torch.load("database/train_labels.pt")

dic = {
    0 : (-1,-1,-1,-1),
    1 : (-1,-1,-1,1),
    2 : (-1,-1,1,-1),
    3 : (-1,-1,1,1),
    4 : (-1,1,-1,-1),
    5 : (-1,1,-1,1),
    6 : (-1,1,1,-1),
    7 : (-1,1,1,1),
    8 : (1,-1,-1,-1),
    9 : (1,-1,-1,1),
}

train_buckets = [dic[i.item()] for i in train_labels]

train_dic = create_dic(train_buckets)

feature_vectors = train_features.cpu()
labels = train_labels.cpu()

hash_sizes = [i for i in range(4,33)]

hash_dic =  {}

for hash_size in tqdm(hash_sizes):
    clusters = cluster_similar_images_batch(
        feature_vectors=feature_vectors,
        labels=labels,
        batch_size=512,
        hash_bits = hash_size
    )

    hash_dic[hash_size] = average_similarity_all_keys(clusters)

hash_dic = sorted(hash_dic.items() , key = lambda kv : (kv[1] , kv[0]) , reverse=True )

k = 5

print_top_k_results(hash_dic , k)

