# Usage Examples

## Training a Model

```bash
# 1. Create validation split
python src/split_data.py

# 2. Train the model
python src/train.py
```

## Quick Evaluation

```bash
python src/check_accuracy.py
```

## Full Evaluation with Confusion Matrix

```bash
python src/evaluate_full.py
```

## Running the Demo

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Single Image Inference

```bash
python src/inference.py --image path/to/xray.jpg
```
