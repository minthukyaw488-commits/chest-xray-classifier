import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io

# ---------- Page config ----------
st.set_page_config(
    page_title="Chest X-Ray Classifier",
    page_icon="🫁",
    layout="centered"
)

# ---------- Config ----------
import os
from huggingface_hub import hf_hub_download

HF_REPO = "novemtk18/chest-xray-classifier"
MODEL_FILENAME = "best_model.pth"
LOCAL_MODEL_DIR = "results"
MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, MODEL_FILENAME)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                     else "cuda" if torch.cuda.is_available() 
                     else "cpu")

# ---------- Load model (cached so it loads only once) ----------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        with st.spinner("Downloading model from Hugging Face..."):
            hf_hub_download(
                repo_id=HF_REPO,
                filename=MODEL_FILENAME,
                local_dir=LOCAL_MODEL_DIR,
            )
    
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model

# ---------- Preprocessing ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(image):
    """Run inference on a PIL Image."""
    # Convert grayscale X-rays to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
    
    pred_idx = probs.argmax()
    return CLASS_NAMES[pred_idx], probs

# ---------- UI ----------
st.title("🫁 Chest X-Ray Pneumonia Classifier")
st.markdown("Upload a chest X-ray image and the model will predict if it shows **pneumonia** or appears **normal**.")

with st.expander("ℹ️ About this model"):
    st.markdown("""
    - **Architecture:** ConvNeXt-Tiny (pretrained on ImageNet)
    - **Training data:** ~5,200 chest X-ray images (Kaggle)
    - **Test accuracy:** 88.46%
    - **High pneumonia recall:** 99% (rarely misses real cases)
    
    ⚠️ **Disclaimer:** This is a student project for educational purposes only. 
    Not for clinical use.
    """)

# Load model
with st.spinner("Loading model..."):
    model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG chest X-ray image"
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🎯 Prediction")
        
        with st.spinner("Analyzing..."):
            pred_class, probs = predict(image)
        
        # Show prediction with color
        if pred_class == "PNEUMONIA":
            st.error(f"### Result: {pred_class}")
        else:
            st.success(f"### Result: {pred_class}")
        
        confidence = probs.max() * 100
        st.metric("Confidence", f"{confidence:.1f}%")
    
    # Probability bars
    st.subheader("📊 Class Probabilities")
    
    for i, class_name in enumerate(CLASS_NAMES):
        prob = probs[i] * 100
        st.write(f"**{class_name}:** {prob:.2f}%")
        st.progress(float(probs[i]))
    
    # Footer info
    st.markdown("---")
    st.caption(f"Model running on: `{DEVICE}`")

else:
    st.info("👆 Upload an X-ray image to see predictions")
    
    # Show example info when no upload
    st.markdown("### 💡 Try it with sample images")
    st.markdown("""
    You can find chest X-ray images for testing from:
    - [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
    - Your own `data/chest_xray/test/` folder
    """)

# Footer
st.markdown("---")
st.markdown(
    "Built with PyTorch + Streamlit | "
    "[GitHub Repo](https://github.com/minthukyaw488-commits/chest-xray-classifier)"
)