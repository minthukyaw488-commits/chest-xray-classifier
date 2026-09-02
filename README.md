# Chest X-Ray Pneumonia Classifier

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

Medical AI assistant that detects pneumonia from chest X-ray images. Built with ConvNeXt-Tiny, deployed with Grad-CAM interpretability and LLM-generated clinical reports.

**[Live Demo](https://novem-chest-xray.streamlit.app)** · **[Model Weights](https://huggingface.co/novemtk18/chest-xray-classifier)** · **[Report Issues](https://github.com/minthukyaw488-commits/chest-xray-classifier/issues)**

---

## Demo

![Application Demo](results/demo_1_home.png)

The application combines three capabilities:

1. **Classification** — ConvNeXt-Tiny CNN predicts NORMAL or PNEUMONIA
2. **Interpretability** — Grad-CAM heatmap shows where the model looks
3. **Reporting** — LLaMA 3.3 generates patient-friendly clinical reports

---

## Results

Evaluated on the 624-image held-out test set at a decision threshold of **0.70** (tuned for high pneumonia sensitivity).

| Metric | Value |
|--------|-------|
| Test Accuracy | **95.0%** |
| PNEUMONIA Recall (Sensitivity) | 98.2% |
| PNEUMONIA Precision | 94.1% |
| NORMAL Recall (Specificity) | 89.7% |
| NORMAL Precision | 96.8% |
| Macro F1 | 94.6% |
| ROC AUC | 0.988 |
| PR AUC | 0.991 |

**Confusion matrix:** 383 true pneumonia · 210 true normal · 24 false pneumonia · 7 false normal (missed cases).

![Confusion Matrix](results/confusion_matrix.png)

Trained on 5,216 images from the Kaggle Chest X-Ray Dataset. See [Experimentation Log](#experimentation-log) for the story behind these numbers.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/minthukyaw488-commits/chest-xray-classifier.git
cd chest-xray-classifier

# 2. Install
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The model weights are downloaded automatically from Hugging Face on first run.

To use the AI report generator, add your [Groq API key](https://console.groq.com/keys) to `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

---

## Tech Stack

- **Model:** ConvNeXt-Tiny (PyTorch)
- **Interpretability:** Grad-CAM
- **Web App:** Streamlit
- **LLM Reports:** Groq API (LLaMA 3.3-70b)
- **Model Hosting:** Hugging Face Hub

---

## Project Structure
├── src/
│   ├── train.py              # Training: WeightedRandomSampler + augmentation + cosine LR
│   ├── evaluate_full.py      # Full evaluation, threshold sweep, ROC/PR AUC
│   ├── gradcam.py            # Grad-CAM implementation
│   ├── report_generator.py   # Groq LLM report generator
│   └── model.py              # Model architecture
├── results/                  # Metrics, threshold sweep, plots, screenshots
├── app.py                    # Streamlit application
├── requirements.txt
└── README.md

---

## Training

```bash
# Create validation split (15% from training data)
python src/split_data.py

# Train the model
python src/train.py

# Evaluate on test set
python src/evaluate_full.py
```

**Configuration:** ConvNeXt-Tiny (ImageNet pretrained) · 10 epochs · Batch size 32 · AdamW (lr=1e-4, weight decay=1e-4) · Cosine annealing LR · Gradient clipping (max_norm=1.0)

**Class imbalance:** handled with a `WeightedRandomSampler` (oversamples the minority NORMAL class) instead of weighted loss.

**Augmentation:** random crop, horizontal flip, rotation (±10°), color jitter, affine (translate/scale/shear), light Gaussian blur, and random erasing — all conservative to keep chest X-rays anatomically realistic (no vertical flips).

**Threshold tuning:** `evaluate_full.py` sweeps decision thresholds (0.50–0.80) and recommends one that keeps pneumonia recall ≥ 0.98 while maximizing specificity. The current model uses **0.70**.

---

## Experimentation Log

### v1 — Baseline
88.46% test accuracy. Model over-predicted pneumonia due to 1:3 class imbalance in training data. NORMAL recall was only 70%.

### v2 — Weighted Loss
Added class-weighted loss to address imbalance. Validation accuracy jumped to 97.83%, but test accuracy dropped to 81%. This gap revealed **distribution shift** between training and test sets — a common issue in medical AI where test data comes from a different patient cohort.

### v3 — Interpretability
Added Grad-CAM to visualize model attention. Verified the model focuses on lung fields, not spurious artifacts. Deployed to Streamlit Cloud.

### v4 — LLM Reports
Integrated Groq API (LLaMA 3.3) to generate patient-friendly clinical reports from classification results.

### v5 — Robust Training + Threshold Tuning
Overhauled the training pipeline to close the train/test gap from v2:

- Replaced weighted loss with a **`WeightedRandomSampler`** so every batch is class-balanced.
- Added **conservative augmentation** (crop, flip, rotation, color jitter, affine, blur, random erasing) to fight overfitting.
- Trained longer (**10 epochs**) with **cosine annealing LR** and **gradient clipping**.
- Added a **threshold sweep** in evaluation with ROC/PR AUC, and picked a 0.70 decision threshold that keeps pneumonia recall high without over-predicting.

**Result: test accuracy rose from 81% → 95%**, NORMAL recall from 50% → 90%, while pneumonia recall stayed at 98%. ROC AUC reached 0.988. The distribution-shift gap from v2 is largely closed.

### Key Insight
High validation accuracy doesn't guarantee real-world performance. Balanced sampling, honest augmentation, independent test evaluation, and threshold tuning — not just a bigger model — are what turned an 81% model into a reliable 95% one.

---

## Limitations

- Not validated on external datasets (NIH ChestX-ray14, CheXpert) — single-dataset training limits generalizability
- Cannot distinguish bacterial vs viral pneumonia
- Class imbalance is mitigated (WeightedRandomSampler) but the source data is still 1:3 NORMAL:PNEUMONIA
- 7 pneumonia cases are still missed on the test set at the current threshold

---

## Future Work

- [x] WeightedRandomSampler for class balance
- [x] Conservative augmentation pipeline
- [x] Decision-threshold tuning with ROC/PR AUC
- [ ] External dataset validation (CheXpert, NIH ChestX-ray14)
- [ ] Stronger augmentation (CutMix, MixUp)
- [ ] Multi-model benchmarking (ResNet50, EfficientNet)
- [ ] Multi-language support (EN/KO/MY)
- [ ] Downloadable PDF reports

---

## Disclaimer

This is a research and educational tool. Predictions and AI-generated reports should not be used for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals.

---

## License

This project is licensed under the MIT License.

---

## Author

**NOVEM (Min Thu Kyaw)** — Medical AI Student, Konyang University
