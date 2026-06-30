# Model Card: Chest X-Ray Pneumonia Classifier

## Model Details

- **Model Type:** Convolutional Neural Network
- **Architecture:** ConvNeXt-Tiny
- **Pre-training:** ImageNet
- **Fine-tuning:** Chest X-ray pneumonia dataset
- **Framework:** PyTorch
- **Date:** June 2026
- **Version:** 1.1.0

## Intended Use

### Primary Use Cases
- Educational demonstration of medical AI
- Research baseline for chest X-ray classification
- Portfolio project

### Out-of-Scope Use Cases
- Clinical diagnosis or treatment decisions
- Replacement for medical professionals
- Production medical systems

## Training Data

- **Source:** Kaggle Chest X-Ray Images (Pneumonia) by Paul Mooney
- **Size:** 5,216 training images
- **Classes:** NORMAL, PNEUMONIA
- **Imbalance Ratio:** Approximately 1:3 (NORMAL:PNEUMONIA)

## Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 81% |
| PNEUMONIA Recall | 100% |
| NORMAL Recall | 50% |

## Limitations

- Trained on a single dataset from one geographic region
- Significant performance gap between validation (97%) and test (81%) indicates domain shift
- Has not been validated on external datasets
- Cannot distinguish between bacterial and viral pneumonia
- May not generalize to different X-ray equipment or imaging protocols

## Ethical Considerations

This model is for research and educational purposes only. Medical AI systems require:
- Extensive clinical validation
- Regulatory approval (FDA, CE marking)
- Ongoing monitoring for bias and drift
- Clear protocols for clinician oversight

## Contact

For questions about this model, please open an issue on the GitHub repository.
