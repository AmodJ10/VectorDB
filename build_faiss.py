import faiss
import numpy as np
import pickle
import torch

train_features = torch.load("database/train_features.pt")
train_labels = torch.load("database/train_labels.pt")

with open("data/train_dic.pkl", "rb") as f:
    train_dic = pickle.load(f)

faiss_indices = {index : faiss.IndexFlatL2(train_features.shape[1]) for index in train_dic.keys()}

for index in faiss_indices.keys():
    faiss_indices[index].add(np.array(train_dic[index]))


index_data = {key: faiss.serialize_index(index) for key, index in faiss_indices.items()}

with open("faiss_indices.pkl", "wb") as f:
    pickle.dump(index_data, f)
