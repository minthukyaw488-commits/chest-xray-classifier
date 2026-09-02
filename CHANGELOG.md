# Changelog

## [1.2.0] - 2026-08-31

### Added
- WeightedRandomSampler for class-balanced batches
- Conservative augmentation pipeline (crop, flip, rotation, color jitter, affine, blur, random erasing)
- Cosine annealing LR schedule and gradient clipping
- Threshold sweep in evaluation with ROC AUC and PR AUC
- Training history logging (results/training_history.json)

### Changed
- Replaced weighted loss with WeightedRandomSampler
- Trained for 10 epochs (was 5)
- Test accuracy improved from 81% to 95%; NORMAL recall from 50% to 90%

## [1.1.0] - 2026-06-28

### Added
- Streamlit web demo (app.py)
- Class-weighted loss for imbalance handling
- Custom validation split script
- Full evaluation with confusion matrix
- MIT License
- Model card documentation

### Changed
- Restructured README with experimentation log
- Custom 15% validation split from training data

## [1.0.0] - 2026-05-18

### Added
- Initial ConvNeXt-Tiny baseline
- Basic training pipeline
- Achieved 88.46% test accuracy
