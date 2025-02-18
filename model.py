import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

class HashingMLP(nn.Module):
    def __init__(self, input_dim=128, output_dim=4):
        super(HashingMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_dim), 
            nn.Tanh()  
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_features = torch.load("database/train_features.pt")
train_labels = torch.load("database/train_labels.pt")

hashing_mlp = HashingMLP().to(device)
optimizer = optim.Adam(hashing_mlp.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.TripletMarginLoss(margin=1.0, p=2)

train_features_tensor = torch.tensor(train_features, dtype=torch.float32).to(device)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)

batch_size = 32
num_samples = len(train_features_tensor)

num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0.0
    indices = torch.randperm(num_samples)  
    
    for i in tqdm(range(0, num_samples - batch_size, batch_size)):
        batch_indices = indices[i : i + batch_size]
        
        anchor = train_features_tensor[batch_indices]
        anchor_hash = hashing_mlp(anchor)

        positive_indices = [
            random.choice(torch.where(train_labels_tensor == train_labels_tensor[idx])[0]).item()
            for idx in batch_indices
        ]
        positive = train_features_tensor[positive_indices]
        positive_hash = hashing_mlp(positive)

        negative_indices = [
            random.choice(torch.where(train_labels_tensor != train_labels_tensor[idx])[0]).item()
            for idx in batch_indices
        ]
        negative = train_features_tensor[negative_indices]
        negative_hash = hashing_mlp(negative)

        optimizer.zero_grad()
        loss = criterion(anchor_hash, positive_hash, negative_hash)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()  
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / (num_samples // batch_size):.4f}")


torch.save(hashing_mlp , "models/improved_hash_temp.pt")
