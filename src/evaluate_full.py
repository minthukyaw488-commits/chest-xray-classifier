import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os

EVAL_DIR = os.getenv("EVAL_DIR", "data/chest_xray/test")
MODEL_PATH = os.getenv("MODEL_PATH", "results/best_model.pth")
PNEUMONIA_THRESHOLD = float(os.getenv("PNEUMONIA_THRESHOLD", "0.60"))
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
MIN_PNEUMONIA_RECALL = 0.98

DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                     else "cuda" if torch.cuda.is_available() 
                     else "cpu")


def predict_with_threshold(probabilities, threshold):
    """Classify as pneumonia only when pneumonia probability reaches threshold."""
    return (probabilities[:, 1] >= threshold).astype(int)


def calculate_medical_metrics(labels, preds):
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    pneumonia_precision = tp / max(tp + fp, 1)
    normal_precision = tn / max(tn + fn, 1)

    return {
        "true_normal": int(tn),
        "false_pneumonia": int(fp),
        "false_normal": int(fn),
        "true_pneumonia": int(tp),
        "sensitivity_pneumonia_recall": float(sensitivity),
        "specificity_normal_recall": float(specificity),
        "pneumonia_precision": float(pneumonia_precision),
        "normal_precision": float(normal_precision),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def choose_threshold(threshold_metrics):
    high_recall_options = [
        item for item in threshold_metrics
        if item["sensitivity_pneumonia_recall"] >= MIN_PNEUMONIA_RECALL
    ]
    candidates = high_recall_options or threshold_metrics
    return max(
        candidates,
        key=lambda item: (
            item["specificity_normal_recall"],
            item["macro_f1"],
            item["accuracy"],
        ),
    )

# Load model
model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

# Load test data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_data = datasets.ImageFolder(EVAL_DIR, transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Predict
all_probs, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(DEVICE))
        probs = F.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
all_preds = predict_with_threshold(all_probs, PNEUMONIA_THRESHOLD)

threshold_metrics = []
for threshold in THRESHOLDS:
    preds = predict_with_threshold(all_probs, threshold)
    metrics = calculate_medical_metrics(all_labels, preds)
    metrics["threshold"] = threshold
    threshold_metrics.append(metrics)

recommended_threshold = choose_threshold(threshold_metrics)
medical_metrics = calculate_medical_metrics(all_labels, all_preds)
medical_metrics["threshold"] = PNEUMONIA_THRESHOLD
medical_metrics["roc_auc"] = float(roc_auc_score(all_labels, all_probs[:, 1]))
medical_metrics["pr_auc"] = float(average_precision_score(all_labels, all_probs[:, 1]))

# Classification report
report = classification_report(all_labels, all_preds, 
                              target_names=test_data.classes, 
                              output_dict=True,
                              zero_division=0)
report["medical_metrics"] = medical_metrics
report["recommended_threshold"] = recommended_threshold

print(f"Evaluation directory: {EVAL_DIR}")
print(f"Pneumonia threshold: {PNEUMONIA_THRESHOLD:.2f}\n")
print(classification_report(
    all_labels,
    all_preds,
    target_names=test_data.classes,
    zero_division=0,
))
print("Medical metrics:")
for key, value in medical_metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

print("\nThreshold sweep:")
for item in threshold_metrics:
    print(
        f"  threshold={item['threshold']:.2f} | "
        f"sensitivity={item['sensitivity_pneumonia_recall']:.4f} | "
        f"specificity={item['specificity_normal_recall']:.4f} | "
        f"macro_f1={item['macro_f1']:.4f}"
    )
print(
    "\nRecommended threshold "
    f"(keeps pneumonia recall >= {MIN_PNEUMONIA_RECALL:.2f} when possible): "
    f"{recommended_threshold['threshold']:.2f}"
)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_data.classes,
            yticklabels=test_data.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (threshold={PNEUMONIA_THRESHOLD:.2f})")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=100)
print("\nConfusion matrix saved to results/confusion_matrix.png")

# Save report
with open("results/classification_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("Report saved to results/classification_report.json")

with open("results/threshold_metrics.json", "w") as f:
    json.dump(threshold_metrics, f, indent=2)
print("Threshold metrics saved to results/threshold_metrics.json")
