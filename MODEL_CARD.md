# Model Card: Chest X-Ray Pneumonia Classifier

## Model Details

- Architecture: ConvNeXt-Tiny (pretrained on ImageNet)
- Framework: PyTorch
- Version: 1.1.0
- Date: June 2026

## Intended Use

Educational demonstration and portfolio project for medical AI.
Not intended for clinical diagnosis.

## Training Data

- Kaggle Chest X-Ray Images (Pneumonia)
- 5,216 training images
- Classes: NORMAL, PNEUMONIA
- Imbalance ratio: 1:3

## Performance

- Test Accuracy: 81%
- PNEUMONIA Recall: 100%
- NORMAL Recall: 50%

## Limitations

- Trained on single dataset with domain shift issues
- Not validated on external data
- Cannot distinguish bacterial vs viral pneumonia
- May not generalize across different X-ray equipment

## Ethical Considerations

This model is for research and educational use only. Real medical AI systems
require clinical validation, regulatory approval, and clinician oversight.
