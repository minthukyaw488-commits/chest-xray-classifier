import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from collections import Counter
from tqdm import tqdm
import os
import json

# ---------- Config ----------
DATA_DIR = "data/chest_xray"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 2
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                     else "cuda" if torch.cuda.is_available()
                     else "cpu")

print(f"Using device: {DEVICE}")

# ---------- Transforms ----------
# Conservative augmentation for chest X-rays.
# Avoid vertical flips because they create anatomically unrealistic images.
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.1
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
        shear=5
    ),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.06)),
])

# Minimal transform for validation and test
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------- Dataset ----------
train_data = datasets.ImageFolder(
    f"{DATA_DIR}/train",
    transform=train_transform
)
val_data = datasets.ImageFolder(
    f"{DATA_DIR}/val_new",
    transform=val_transform
)

print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")
print(f"Classes: {train_data.classes}")

# ---------- WeightedRandomSampler ----------
# Count samples per class
class_counts = Counter([label for _, label in train_data.samples])
print(f"\nClass counts: {dict(class_counts)}")

# Weight per class = 1 / count (rarer class gets higher weight)
class_weights = {
    cls: 1.0 / count 
    for cls, count in class_counts.items()
}
print(f"Class weights: {class_weights}")

# Assign weight to every single sample
sample_weights = [
    class_weights[label] 
    for _, label in train_data.samples
]

# Create sampler — replacement=True means oversampling allowed
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Use sampler instead of shuffle=True
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    sampler=sampler,       # sampler handles ordering
    num_workers=2,
    pin_memory=True
)
val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

print(f"\nExpected batches per epoch: {len(train_loader)}")

# ---------- Model ----------
model = models.convnext_tiny(weights="IMAGENET1K_V1")
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ---------- Loss ----------
# Standard CrossEntropy (no weighting — sampler handles imbalance now)
criterion = nn.CrossEntropyLoss()

# ---------- Optimizer ----------
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# ---------- LR Scheduler ----------
# Cosine annealing — smoothly reduces LR over training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

# ---------- Training Loop ----------
def train_one_epoch(epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct/total:.4f}"
        )
    
    return total_loss / len(train_loader), correct / total


def validate(epoch):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    # Track per-class accuracy
    class_correct = Counter()
    class_total = Counter()
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            for label, pred in zip(labels.cpu(), preds.cpu()):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1
    
    val_acc = correct / total
    
    # Print per-class accuracy
    for cls_idx, cls_name in enumerate(train_data.classes):
        cls_acc = class_correct[cls_idx] / max(class_total[cls_idx], 1)
        print(f"  {cls_name} accuracy: {cls_acc:.4f} ({class_correct[cls_idx]}/{class_total[cls_idx]})")
    
    return total_loss / len(val_loader), val_acc


# ---------- Main ----------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    best_acc = 0
    history = []
    
    print("\n" + "="*60)
    print("Starting training with WeightedRandomSampler + conservative augmentation")
    print("="*60 + "\n")
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(epoch)
        val_loss, val_acc = validate(epoch)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f}")
        print(f"  Val   Loss={val_loss:.4f} | Val   Acc={val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "results/best_model.pth")
            print(f"  Saved best model (val_acc={val_acc:.4f})")
        
        print("-" * 50)
        
        # Track history
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(current_lr, 6),
        })
    
    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")
    
    # Save training history
    with open("results/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print("Training history saved to results/training_history.json")
