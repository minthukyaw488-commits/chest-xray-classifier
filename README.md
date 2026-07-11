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

<<<<<<< HEAD
### Key Features

- Real-time pneumonia detection from uploaded X-ray images
- Grad-CAM attention visualization showing where the model looks
- Severity-based clinical recommendations (High/Moderate/Low Concern)
- AI-generated medical reports powered by LLaMA 3.3
- Confidence-weighted actionable guidance

Run locally:
\`\`\`bash
streamlit run app.py
\`\`\`
---

## Project Overview

This project applies transfer learning to classify chest X-ray images as NORMAL or PNEUMONIA. Beyond classification, it emphasizes model interpretability through Grad-CAM visualization, allowing users to see which regions influenced the model's decision. This is critical for medical AI applications where trust and transparency are essential.

---

## Key Features

**Deep Learning Model**
- ConvNeXt-Tiny architecture pretrained on ImageNet
- Fine-tuned on Kaggle Chest X-Ray Dataset
- Class-weighted loss to address 1:3 data imbalance

**Interpretability**
- Grad-CAM heatmap visualization
- Shows model attention overlay on original X-ray
- Helps validate that model focuses on lung fields, not artifacts

**Clinical Recommendations**
- Severity levels: High/Moderate/Low Concern
- Confidence-based actionable guidance
- Symptoms to watch checklist
- Clear disclaimer for educational use

**Deployment**
- Live web application on Streamlit Community Cloud
- Model weights hosted on Hugging Face Hub
- No local installation required to use
=======
1. **Classification** — ConvNeXt-Tiny CNN predicts NORMAL or PNEUMONIA
2. **Interpretability** — Grad-CAM heatmap shows where the model looks
3. **Reporting** — LLaMA 3.3 generates patient-friendly clinical reports
>>>>>>> e4b598d (docs: streamline README for professional readability)

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 81% |
| PNEUMONIA Recall | 100% |
| PNEUMONIA Precision | 77% |
| NORMAL Recall | 50% |

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
│   ├── train.py              # Training with weighted loss
│   ├── evaluate_full.py      # Full evaluation + confusion matrix
│   ├── gradcam.py            # Grad-CAM implementation
│   ├── report_generator.py   # Groq LLM report generator
│   └── model.py              # Model architecture
├── results/                  # Metrics, plots, screenshots
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

**Configuration:** ConvNeXt-Tiny · 5 epochs · Batch size 32 · AdamW (lr=1e-4) · Weighted Cross Entropy

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

### Key Insight
High validation accuracy doesn't guarantee real-world performance. Independent test evaluation and interpretability tools are essential in medical AI.

---

## Limitations

- Class imbalance (1:3 NORMAL:PNEUMONIA) affects generalization
- Distribution shift between train and test sets
- Not validated on external datasets (NIH ChestX-ray14, CheXpert)
- Cannot distinguish bacterial vs viral pneumonia
- Single-dataset training limits generalizability

---

## Future Work

- [ ] External dataset validation (CheXpert, NIH ChestX-ray14)
- [ ] WeightedRandomSampler alternative to weighted loss
- [ ] Stronger augmentation (CutMix, MixUp)
- [ ] Multi-model benchmarking (ResNet50, EfficientNet)
- [ ] Multi-language support (EN/KO/MY)
- [ ] Downloadable PDF reports

---

## Disclaimer

This is a research and educational tool. Predictions and AI-generated reports should not be used for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals.

---

## Author

**NOVEM (Min Thu Kyaw)** — Medical AI Student, Konyang University

## License

<<<<<<< HEAD
This project is released under the MIT License. See [LICENSE](LICENSE) for details.
The dataset license follows the original Kaggle terms.
=======
MIT © 2026 NOVEM
>>>>>>> e4b598d (docs: streamline README for professional readability)
