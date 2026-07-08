import numpy as np
import sys
sys.path.append("src")
from gradcam import GradCAM, overlay_heatmap
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
from huggingface_hub import hf_hub_download
from datetime import datetime

# ---------- Page Config ----------
st.set_page_config(
    page_title="Medical AI Assistant - Chest X-Ray Analysis",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Soft blue background */
    .stApp {
        background: #e0f2fe;
    }
    
    .main {
        padding-top: 1rem;
        background: #e0f2fe;
    }
    
    /* Headings */
    h1 {
        color: #0c4a6e;
        font-weight: 700;
        letter-spacing: -0.5px;
        font-size: 2rem;
    }
    
    h2, h3 {
        color: #0c4a6e;
        font-weight: 600;
    }
    
    /* Header bar - deep blue */
    .header-bar {
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(12, 74, 110, 0.15);
    }
    
    .header-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    
    .header-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.25rem;
        color: white;
    }
    
    /* Diagnosis cards - deep blue theme */
    .diagnosis-card {
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: #0c4a6e;
        color: white;
        box-shadow: 0 4px 12px rgba(12, 74, 110, 0.2);
    }
    
    .diagnosis-normal {
        background: linear-gradient(135deg, #0c4a6e 0%, #075985 100%);
        border-left: 4px solid #22c55e;
    }
    
    .diagnosis-pneumonia {
        background: linear-gradient(135deg, #0c4a6e 0%, #075985 100%);
        border-left: 4px solid #ef4444;
    }
    
    .diagnosis-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #bae6fd;
    }
    
    .diagnosis-value {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: white;
    }
    
    .diagnosis-confidence {
        font-size: 0.95rem;
        color: #e0f2fe;
    }
    
    /* Probability bars */
    .prob-row {
        background: #0c4a6e;
        color: white;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(12, 74, 110, 0.15);
    }
    
    /* Info panels - deep blue */
    .info-panel {
        background: #0c4a6e;
        color: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(12, 74, 110, 0.15);
    }
    
    /* Severity/status blocks */
    .severity-block {
        background: #075985;
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin: 0.75rem 0;
    }
    
    .action-block {
        background: #0369a1;
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin: 0.75rem 0;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: #075985;
        color: white;
        border-left: 4px solid #fbbf24;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 1rem 0;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #bae6fd;
    }
    
    .sidebar-section {
        background: #0c4a6e;
        color: white;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    /* File uploader */
    section[data-testid="stFileUploader"] {
        background: white;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Text on background (default) */
p, li, span {
    color: #0c4a6e;
}

/* Force white text inside all dark blocks */
.info-panel, .info-panel *,
.sidebar-section, .sidebar-section *,
.prob-row, .prob-row *,
.diagnosis-card, .diagnosis-card *,
.severity-block, .severity-block *,
.action-block, .action-block *,
.disclaimer, .disclaimer * {
    color: white !important;
}

/* But keep the accent colors on labels */
.info-panel div[style*="color: #64748b"],
.sidebar-section div[style*="color: #64748b"] {
    color: #bae6fd !important;
}

/* Keep percentage colors on prob-row spans */
.prob-row span[style*="color: #dc2626"] {
    color: #fca5a5 !important;
}

.prob-row span[style*="color: #16a34a"] {
    color: #86efac !important;
}
    
    /* Footer */
    .footer {
        text-align: center;
        color: #0c4a6e;
        font-size: 0.8rem;
        padding: 1.5rem;
        border-top: 1px solid #7dd3fc;
        margin-top: 2rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- Config ----------
HF_REPO = "novemtk18/chest-xray-classifier"
MODEL_FILENAME = "best_model.pth"
LOCAL_MODEL_DIR = "results"
MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, MODEL_FILENAME)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                     else "cuda" if torch.cuda.is_available() 
                     else "cpu")

# ---------- Load Model ----------
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
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad_(True)
    
    # Forward pass
    outputs = model(img_tensor)
    probs = F.softmax(outputs, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(probs.argmax())
    
    # Generate Grad-CAM
    heatmap, _ = gradcam.generate(img_tensor, class_idx=pred_idx)
    heatmap_overlay = overlay_heatmap(image, heatmap, alpha=0.45)
    
    return CLASS_NAMES[pred_idx], probs, heatmap_overlay

# ---------- Header ----------
st.markdown(
    """
    <div class='header-bar'>
        <div class='header-title'>Medical AI Assistant</div>
        <div class='header-subtitle'>Chest X-Ray Analysis for Pneumonia Screening</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### System Information")
    
    st.markdown(
        f"""
        <div class='sidebar-section'>
            <div style='font-size: 0.75rem; color: #64748b; text-transform: uppercase; 
                        letter-spacing: 1px; font-weight: 600;'>Runtime</div>
            <div style='font-size: 1rem; font-weight: 600; color: #1e293b; margin-top: 0.25rem;'>
                {str(DEVICE).upper()}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
        <div class='sidebar-section'>
            <div style='font-size: 0.75rem; color: #64748b; text-transform: uppercase; 
                        letter-spacing: 1px; font-weight: 600;'>Session</div>
            <div style='font-size: 0.9rem; color: #1e293b; margin-top: 0.25rem;'>
                {datetime.now().strftime('%Y-%m-%d %H:%M')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("### Model Specifications")
    
    st.markdown("""
    <div class='sidebar-section'>
        <div style='font-size: 0.85rem; line-height: 1.6;'>
            <strong>Architecture:</strong> ConvNeXt-Tiny<br>
            <strong>Parameters:</strong> ~28M<br>
            <strong>Input Size:</strong> 224×224<br>
            <strong>Classes:</strong> 2 (Binary)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Performance Metrics")
    
    st.markdown("""
    <div class='sidebar-section'>
        <div style='font-size: 0.85rem; line-height: 1.8;'>
            <strong>Test Accuracy:</strong> 81.0%<br>
            <strong>PNEUMONIA Recall:</strong> 100%<br>
            <strong>PNEUMONIA Precision:</strong> 77%<br>
            <strong>NORMAL Recall:</strong> 50%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Resources")
    st.markdown("""
    <div style='font-size: 0.85rem; line-height: 1.8;'>
        <a href='https://github.com/minthukyaw488-commits/chest-xray-classifier' 
           style='color: #3b82f6; text-decoration: none;'>Source Code (GitHub)</a><br>
        <a href='https://huggingface.co/novemtk18/chest-xray-classifier' 
           style='color: #3b82f6; text-decoration: none;'>Model Weights (HuggingFace)</a>
    </div>
    """, unsafe_allow_html=True)

# ---------- Load Model ----------
model = load_model()

@st.cache_resource
def get_gradcam(_model):
    target_layer = _model.features[-1]
    return GradCAM(_model, target_layer)

gradcam = get_gradcam(model)

# ---------- Main Content ----------
col_main, col_side = st.columns([2, 1], gap="large")

with col_main:
    st.markdown("### Image Upload")
    st.markdown(
        "<div style='color: #64748b; font-size: 0.9rem; margin-bottom: 1rem;'>"
        "Upload a chest X-ray image to analyze for pneumonia indicators.</div>",
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader(
        label="Select X-ray image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    # Process image if uploaded
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        with st.spinner("Processing image and generating attention map..."):
            pred_class, probs, heatmap_img = predict(image)
        
        st.markdown("#### Image Analysis")
        
        tab1, tab2 = st.tabs(["Original X-Ray", "AI Attention Map"])
        
        with tab1:
            st.image(image, use_column_width=True)
            st.caption(f"File: {uploaded_file.name} | Size: {image.size[0]}×{image.size[1]} px")
        
        with tab2:
            st.image(heatmap_img, use_column_width=True)
            st.caption(
                "Red/yellow areas = high model attention. "
                "Blue = low attention. Shows where the model focused."
            )
            st.markdown(
                """
                <div class='info-panel'>
                    <div style='font-size: 0.85rem; line-height: 1.6;'>
                        <strong>How to interpret:</strong> The Grad-CAM heatmap reveals 
                        which regions influenced the model's decision. Ideally, the model 
                        should focus on the lung fields, not the borders or background.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

with col_side:
    if uploaded_file is not None:
        st.markdown("### Analysis Report")
        
        with st.spinner("Processing image and generating attention map..."):
             pred_class, probs, heatmap_img = predict(image)
        
        confidence = probs.max() * 100
        
        if pred_class == "PNEUMONIA":
            # Diagnosis card
            st.markdown(
                f"""
                <div class='diagnosis-card diagnosis-pneumonia'>
                    <div class='diagnosis-label'>Model Prediction</div>
                    <div class='diagnosis-value' style='color: #991b1b;'>PNEUMONIA</div>
                    <div class='diagnosis-confidence'>Confidence: {confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Severity level based on confidence
            if confidence >= 90:
                severity = "HIGH CONCERN"
                severity_color = "#991b1b"
                severity_bg = "#fef2f2"
                recommendation = "Strong indicators of pneumonia detected. Urgent medical consultation strongly advised. Do not delay seeking professional care."
                action = "Visit a doctor or emergency room today."
            elif confidence >= 70:
                severity = "MODERATE CONCERN"
                severity_color = "#c2410c"
                severity_bg = "#fff7ed"
                recommendation = "Likely pneumonia detected. Medical evaluation recommended within 24-48 hours."
                action = "Schedule a doctor visit soon. Watch for worsening symptoms."
            else:
                severity = "LOW-MODERATE CONCERN"
                severity_color = "#a16207"
                severity_bg = "#fefce8"
                recommendation = "Possible pneumonia indicators. Model is uncertain — professional review recommended."
                action = "Consult a doctor for proper diagnosis. Monitor symptoms closely."
            
            st.markdown(
                f"""
                <div style='background: {severity_bg}; border-left: 4px solid {severity_color}; 
                            padding: 1rem 1.25rem; border-radius: 6px; margin: 0.75rem 0;'>
                    <div style='font-size: 0.75rem; color: {severity_color}; font-weight: 700; 
                                text-transform: uppercase; letter-spacing: 1.5px;'>
                        Severity Level
                    </div>
                    <div style='font-size: 1.25rem; font-weight: 700; color: {severity_color}; 
                                margin: 0.25rem 0;'>
                        {severity}
                    </div>
                </div>
                
                <div class='info-panel'>
                    <div style='font-size: 0.8rem; color: #64748b; font-weight: 600; 
                                text-transform: uppercase; letter-spacing: 1px;'>
                        What This Means
                    </div>
                    <div style='font-size: 0.9rem; margin-top: 0.5rem; color: #1e293b; line-height: 1.6;'>
                        {recommendation}
                    </div>
                </div>
                
                <div style='background: #eff6ff; border-left: 4px solid #2563eb; 
                            padding: 1rem 1.25rem; border-radius: 6px; margin: 0.75rem 0;'>
                    <div style='font-size: 0.75rem; color: #1e40af; font-weight: 700; 
                                text-transform: uppercase; letter-spacing: 1.5px;'>
                        Recommended Action
                    </div>
                    <div style='font-size: 0.95rem; margin-top: 0.5rem; color: #1e293b; font-weight: 500;'>
                        {action}
                    </div>
                </div>
                
                <div class='info-panel'>
                    <div style='font-size: 0.8rem; color: #64748b; font-weight: 600; 
                                text-transform: uppercase; letter-spacing: 1px;'>
                        Symptoms to Watch
                    </div>
                    <div style='font-size: 0.85rem; margin-top: 0.5rem; color: #1e293b; line-height: 1.7;'>
                        • Difficulty breathing or shortness of breath<br>
                        • High fever (above 39°C / 102°F)<br>
                        • Persistent cough with mucus<br>
                        • Chest pain when breathing<br>
                        • Confusion (especially in elderly)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        else:
            # Diagnosis card
            st.markdown(
                f"""
                <div class='diagnosis-card diagnosis-normal'>
                    <div class='diagnosis-label'>Model Prediction</div>
                    <div class='diagnosis-value' style='color: #166534;'>NORMAL</div>
                    <div class='diagnosis-confidence'>Confidence: {confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Status based on confidence
            if confidence >= 90:
                status = "APPEARS HEALTHY"
                status_color = "#166534"
                status_bg = "#f0fdf4"
                message = "No signs of pneumonia detected. The X-ray appears normal."
                action = "No immediate medical action needed. Maintain healthy habits."
            elif confidence >= 70:
                status = "LIKELY NORMAL"
                status_color = "#15803d"
                status_bg = "#f0fdf4"
                message = "X-ray appears normal, but model has some uncertainty."
                action = "If you have symptoms, consult a doctor to confirm."
            else:
                status = "UNCERTAIN — LIKELY NORMAL"
                status_color = "#a16207"
                status_bg = "#fefce8"
                message = "X-ray leans toward normal, but confidence is low. Could be an unclear image."
                action = "If experiencing symptoms, seek medical evaluation to be safe."
            
            st.markdown(
                f"""
                <div style='background: {status_bg}; border-left: 4px solid {status_color}; 
                            padding: 1rem 1.25rem; border-radius: 6px; margin: 0.75rem 0;'>
                    <div style='font-size: 0.75rem; color: {status_color}; font-weight: 700; 
                                text-transform: uppercase; letter-spacing: 1.5px;'>
                        Status
                    </div>
                    <div style='font-size: 1.25rem; font-weight: 700; color: {status_color}; 
                                margin: 0.25rem 0;'>
                        {status}
                    </div>
                </div>
                
                <div class='info-panel'>
                    <div style='font-size: 0.8rem; color: #64748b; font-weight: 600; 
                                text-transform: uppercase; letter-spacing: 1px;'>
                        What This Means
                    </div>
                    <div style='font-size: 0.9rem; margin-top: 0.5rem; color: #1e293b; line-height: 1.6;'>
                        {message}
                    </div>
                </div>
                
                <div style='background: #eff6ff; border-left: 4px solid #2563eb; 
                            padding: 1rem 1.25rem; border-radius: 6px; margin: 0.75rem 0;'>
                    <div style='font-size: 0.75rem; color: #1e40af; font-weight: 700; 
                                text-transform: uppercase; letter-spacing: 1.5px;'>
                        Recommended Action
                    </div>
                    <div style='font-size: 0.95rem; margin-top: 0.5rem; color: #1e293b; font-weight: 500;'>
                        {action}
                    </div>
                </div>
                
                <div class='info-panel'>
                    <div style='font-size: 0.8rem; color: #64748b; font-weight: 600; 
                                text-transform: uppercase; letter-spacing: 1px;'>
                        When to Still See a Doctor
                    </div>
                    <div style='font-size: 0.85rem; margin-top: 0.5rem; color: #1e293b; line-height: 1.7;'>
                        • Persistent cough lasting over 2 weeks<br>
                        • Fever with breathing difficulty<br>
                        • Unexplained chest pain<br>
                        • Blood in cough<br>
                        • Fatigue that won't go away
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Probability breakdown
        st.markdown("#### Probability Distribution")
        
        for i, class_name in enumerate(CLASS_NAMES):
            prob = probs[i] * 100
            color = "#dc2626" if class_name == "PNEUMONIA" else "#16a34a"
            
            st.markdown(
                f"""
                <div class='prob-row'>
                    <div style='display: flex; justify-content: space-between; 
                                margin-bottom: 0.5rem;'>
                        <span style='font-weight: 600; color: #1e293b; font-size: 0.9rem;'>
                            {class_name}
                        </span>
                        <span style='font-weight: 700; color: {color}; font-size: 0.9rem;'>
                            {prob:.2f}%
                        </span>
                    </div>
                    <div style='background: #e2e8f0; height: 6px; border-radius: 3px; 
                                overflow: hidden;'>
                        <div style='background: {color}; width: {prob}%; height: 100%;'></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown("### Getting Started")
        st.markdown(
            """
            <div class='info-panel'>
                <div style='font-size: 0.9rem; line-height: 1.7; color: #1e293b;'>
                    <strong>Instructions:</strong><br>
                    1. Upload a chest X-ray image using the panel on the left<br>
                    2. The system will analyze the image automatically<br>
                    3. Results appear here with severity and recommendations<br>
                    4. Review the probability distribution
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div class='info-panel'>
                <div style='font-size: 0.8rem; color: #64748b; font-weight: 600; 
                            text-transform: uppercase; letter-spacing: 1px;'>
                    Supported Inputs
                </div>
                <div style='font-size: 0.85rem; margin-top: 0.5rem; color: #1e293b;'>
                    JPG, JPEG, PNG formats<br>
                    Frontal chest radiographs<br>
                    Maximum size: 200 MB
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------- Disclaimer ----------
st.markdown(
    """
    <div class='disclaimer'>
        <strong>Important Notice:</strong> This is a research and educational tool. 
        Predictions should not be used as the sole basis for clinical decisions. 
        All medical diagnoses must be made by qualified healthcare professionals 
        based on comprehensive clinical evaluation.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Footer ----------
st.markdown(
    """
    <div class='footer'>
        Medical AI Assistant | Developed by NOVEM | Konyang University<br>
        Built with PyTorch, Streamlit, and Hugging Face
    </div>
    """,
    unsafe_allow_html=True
)