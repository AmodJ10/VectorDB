import torch
from torch.utils.data import Dataset
import numpy as np

class TripletDataset(Dataset):
    def __init__(self, images, labels, device):
        self.device = device
        self.images = [image.unsqueeze(0) for image in images]
        self.labels = np.array(labels)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        anchor = self.images[idx]
        anchor_label = self.labels[idx]

        pos_idx = np.where(self.labels == anchor_label)[0]
        pos_idx = pos_idx[pos_idx != idx]  
        if len(pos_idx) == 0:
            pos_idx = [idx]  
        pos_choice = np.random.choice(pos_idx)
        positive = self.images[pos_choice]

        neg_idx = np.where(self.labels != anchor_label)[0]
        neg_choice = np.random.choice(neg_idx)
        negative = self.images[neg_choice]
        negative_label = self.labels[neg_choice]

        anchor = torch.tensor(anchor, dtype=torch.float32)
        positive = torch.tensor(positive, dtype=torch.float32)
        negative = torch.tensor(negative, dtype=torch.float32)

        return (anchor, positive, negative), (anchor_label, anchor_label, negative_label)
