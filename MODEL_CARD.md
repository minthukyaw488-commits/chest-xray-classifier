# Model Card: Chest X-Ray Pneumonia Classifier

## Model Details

- Architecture: ConvNeXt-Tiny (pretrained on ImageNet)
- Framework: PyTorch
- Version: 1.2.0
- Date: August 2026

## Intended Use

Educational demonstration and portfolio project for medical AI.
Not intended for clinical diagnosis.

## Training Data

- Kaggle Chest X-Ray Images (Pneumonia)
- 5,216 training images
- Classes: NORMAL, PNEUMONIA
- Imbalance ratio: 1:3

## Performance

Evaluated on the 624-image held-out test set at a 0.70 decision threshold.

- Test Accuracy: 95.0%
- PNEUMONIA Recall (Sensitivity): 98.2%
- PNEUMONIA Precision: 94.1%
- NORMAL Recall (Specificity): 89.7%
- NORMAL Precision: 96.8%
- Macro F1: 94.6%
- ROC AUC: 0.988
- PR AUC: 0.991

## Training Setup

- Class balancing: WeightedRandomSampler (oversamples minority NORMAL class)
- Augmentation: crop, horizontal flip, rotation, color jitter, affine, light blur, random erasing
- 10 epochs, AdamW (lr=1e-4, weight decay=1e-4), cosine annealing LR, gradient clipping (max_norm=1.0)
- Decision threshold tuned via sweep (0.50–0.80) to keep pneumonia recall ≥ 0.98

## Limitations

- Trained on single dataset with domain shift issues
- Not validated on external data
- Cannot distinguish bacterial vs viral pneumonia
- May not generalize across different X-ray equipment

## Ethical Considerations

This model is for research and educational use only. Real medical AI systems
require clinical validation, regulatory approval, and clinician oversight.
