import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os

# ---------- Config ----------
DATA_DIR = "data/chest_xray"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
NUM_CLASSES = 2
DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                     else "cuda" if torch.cuda.is_available() 
                     else "cpu")

print(f"Using device: {DEVICE}")

# ---------- Data ----------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_data = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
val_data = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")
print(f"Classes: {train_data.classes}")

# ---------- Model ----------
model = models.convnext_tiny(weights="IMAGENET1K_V1")
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ---------- Loss & Optimizer ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ---------- Training loop ----------
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
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=loss.item(), acc=correct/total)
    
    return total_loss / len(train_loader), correct / total

def validate(epoch):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(val_loader), correct / total

# ---------- Main ----------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    best_acc = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(epoch)
        val_loss, val_acc = validate(epoch)
        
        print(f"\nEpoch {epoch+1}: "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}\n")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "results/best_model.pth")
            print(f"✅ Saved new best model (val_acc={val_acc:.4f})")
    
    print(f"\nTraining done. Best val acc: {best_acc:.4f}")