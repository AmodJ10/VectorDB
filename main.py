import torch
import torch.nn as nn
from FeatureExtractor import FeatureExtractor
from PreFeatureExtractor import FeatureExtractorNet
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, SubsetRandomSampler
from TripletDataset import TripletDataset
from HashNetwork import Embedder
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import numpy as np
from tqdm import tqdm
import os
import multiprocessing
import time

BATCH_SIZE = 512  
NUM_WORKERS = 8
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
CHECKPOINT_DIR = "./model"
SAMPLE_FRACTION = 0.2 

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"checkpoint directory created: {CHECKPOINT_DIR}")
print(f"saving model version 0.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  

extractor = FeatureExtractor().to(device)
embedder = Embedder(device=device)

loss_fn = nn.TripletMarginLoss()
optimizer = Adam(
    params=embedder.embedder.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=1  
)

train_dataset = MNIST(root="/data", train=True, download=True)

indices = list(range(len(train_dataset)))
np.random.shuffle(indices)
split = int(np.floor(0.1 * len(train_dataset)))
train_indices, val_indices = indices[split:], indices[:split]

if SAMPLE_FRACTION < 1.0:
    train_subset_size = int(len(train_indices) * SAMPLE_FRACTION)
    train_indices = train_indices[:train_subset_size]
    print(f"Using {train_subset_size} samples ({SAMPLE_FRACTION*100}% of training data) for faster training")

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

triplet_train = TripletDataset(train_dataset.data, train_dataset.targets, device="cpu")  # Keep data on CPU until batch loading

triplet_train_loader = DataLoader(
    triplet_train, 
    batch_size=BATCH_SIZE, 
    sampler=train_sampler,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    prefetch_factor=2
)

triplet_val_loader = DataLoader(
    triplet_train, 
    batch_size=BATCH_SIZE, 
    sampler=val_sampler,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    prefetch_factor=2
)

scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

def train_epoch(epoch):
    extractor.train()
    embedder.embedder.train()
    
    epoch_loss = 0.0
    batch_times = []
    start_time = time.time()
    
    with tqdm(enumerate(triplet_train_loader), total=len(triplet_train_loader), 
              desc=f"Epoch {epoch+1}/{EPOCHS}") as batch_bar:
        for batch_idx, ((a, p, n), _) in batch_bar:
            batch_start = time.time()
            
            a, p, n = a.to(device, non_blocking=True), p.to(device, non_blocking=True), n.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True) 
            
            if scaler:
                with torch.cuda.amp.autocast():
                    ae = extractor(a)
                    pe = extractor(p)
                    ne = extractor(n)

                    ah = embedder.embedder(ae)
                    ph = embedder.embedder(pe)
                    nh = embedder.embedder(ne)
                    
                    loss = loss_fn(ah, ph, nh)
                
                scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(extractor.parameters()) + list(embedder.embedder.parameters()), 
                    max_norm=1.0
                )
                
                scaler.step(optimizer)
                scaler.update()
            else:
                ae = extractor(a)
                pe = extractor(p)
                ne = extractor(n)
                
                ah = embedder.embedder(ae)
                ph = embedder.embedder(pe)
                nh = embedder.embedder(ne)
                
                loss = loss_fn(ah, ph, nh)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    list(extractor.parameters()) + list(embedder.embedder.parameters()), 
                    max_norm=1.0
                )
                
                optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            avg_batch_time = sum(batch_times) / len(batch_times)
            elapsed = batch_end - start_time
            eta = avg_batch_time * (len(triplet_train_loader) - batch_idx - 1)
            
            batch_bar.set_postfix({
                "Loss": f"{batch_loss:.4f}",
                "Avg": f"{epoch_loss/(batch_idx+1):.4f}",
                "Batch time": f"{batch_time:.2f}s",
                "ETA": f"{eta/60:.1f}m"
            })
    
    epoch_time = time.time() - start_time
    print(f"Epoch time: {epoch_time/60:.2f} minutes")
    return epoch_loss / len(triplet_train_loader)

def validate():
    extractor.eval()
    embedder.embedder.eval()
    embedder.training = False
    
    val_loss = 0.0
    start_time = time.time()
    
    with torch.no_grad():
        with tqdm(enumerate(triplet_val_loader), total=len(triplet_val_loader), 
                  desc="Validation") as batch_bar:
            for batch_idx, ((a, p, n), _) in batch_bar:
                a, p, n = a.to(device, non_blocking=True), p.to(device, non_blocking=True), n.to(device, non_blocking=True)
                
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        ae = extractor(a)
                        pe = extractor(p)
                        ne = extractor(n)
                        
                        ah = embedder.embedder(ae)
                        ph = embedder.embedder(pe)
                        nh = embedder.embedder(ne)
                        
                        loss = loss_fn(ah, ph, nh)
                else:
                    ae = extractor(a)
                    pe = extractor(p)
                    ne = extractor(n)
                    
                    ah = embedder.embedder(ae)
                    ph = embedder.embedder(pe)
                    nh = embedder.embedder(ne)
                    
                    loss = loss_fn(ah, ph, nh)
                
                val_loss += loss.item()
                
                batch_bar.set_postfix({
                    "Val Loss": f"{loss.item():.4f}",
                    "Avg": f"{val_loss/(batch_idx+1):.4f}"
                })
    
    val_time = time.time() - start_time
    print(f"Validation time: {val_time:.2f} seconds")
    return val_loss / len(triplet_val_loader)

def main():
    print(f"Starting training with batch size: {BATCH_SIZE}, workers: {NUM_WORKERS}")
    print(f"Data fraction: {SAMPLE_FRACTION*100}%, DataLoader batches: {len(triplet_train_loader)}")
    
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        train_loss = train_epoch(epoch)
        
        if epoch % 2 == 0 or epoch == EPOCHS - 1:
            val_loss = validate()
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'extractor_state_dict': extractor.state_dict(),
                    'embedder_state_dict': embedder.embedder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, f"{CHECKPOINT_DIR}/best_model.pt")
                print(f"🔥 New best model saved with validation loss: {val_loss:.4f}")
        else:
            print(f"\n✅ Epoch {epoch+1}/{EPOCHS} completed — Train Loss: {train_loss:.4f}")
        
        epoch_time = time.time() - epoch_start
        print(f"Total epoch time: {epoch_time/60:.2f} minutes")
    if os.path.exists(f"{CHECKPOINT_DIR}/best_model.pt"):
        checkpoint = torch.load(f"{CHECKPOINT_DIR}/best_model.pt")
        extractor.load_state_dict(checkpoint['extractor_state_dict'])
        embedder.embedder.load_state_dict(checkpoint['embedder_state_dict'])
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    else:
        print("Training completed! No best model saved.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    if os.name == 'nt': 
        multiprocessing.set_start_method('spawn', force=True)
    
    main()