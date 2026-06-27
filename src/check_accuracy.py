import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import json

# ---------- Config ----------
DATA_DIR = "data/chest_xray"
BATCH_SIZE = 32
NUM_CLASSES = 2
MODEL_PATH = "results/best_model.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                     else "cuda" if torch.cuda.is_available() 
                     else "cpu")

print(f"Using device: {DEVICE}")

# ---------- Data ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_data = datasets.ImageFolder(f"{DATA_DIR}/val", transform=transform)
test_data = datasets.ImageFolder(f"{DATA_DIR}/test", transform=transform)

val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# ---------- Load Model ----------
model = models.convnext_tiny(weights=None)
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print(f"Loaded model from {MODEL_PATH}\n")

# ---------- Evaluate ----------
def evaluate(loader, name):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating {name}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"{name} accuracy: {acc:.4f} ({correct}/{total})\n")
    return acc

val_acc = evaluate(val_loader, "Validation")
test_acc = evaluate(test_loader, "Test")

# ---------- Save metrics ----------
metrics = {
    "val_accuracy": round(val_acc, 4),
    "test_accuracy": round(test_acc, 4),
    "val_samples": len(val_data),
    "test_samples": len(test_data),
    "model": "ConvNeXt-Tiny",
}

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Metrics saved to results/metrics.json")