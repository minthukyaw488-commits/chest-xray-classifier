import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json

DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                     else "cuda" if torch.cuda.is_available() 
                     else "cpu")

# Load model
model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
model.load_state_dict(torch.load("results/best_model.pth", map_location=DEVICE))
model = model.to(DEVICE).eval()

# Load test data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_data = datasets.ImageFolder("data/chest_xray/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Predict
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(DEVICE))
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification report
report = classification_report(all_labels, all_preds, 
                              target_names=test_data.classes, 
                              output_dict=True)
print(classification_report(all_labels, all_preds, target_names=test_data.classes))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_data.classes,
            yticklabels=test_data.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=100)
print("\n✅ Confusion matrix saved to results/confusion_matrix.png")

# Save report
with open("results/classification_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("✅ Report saved to results/classification_report.json")