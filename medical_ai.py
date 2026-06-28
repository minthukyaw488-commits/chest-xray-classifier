import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
import os 
load_dotenv()

# ── setup ──────────────────────────────────────────────

MODEL_PATH = "/Users/user/chest-xray-classifier/results/best_model.pth"

CLASSES      = ["NORMAL", "PNEUMONIA"]

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── load CNN model ─────────────────────────────────────
def load_model():
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# ── preprocess image ───────────────────────────────────
def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# ── predict with CNN ───────────────────────────────────
def predict(model, image_path):
    tensor = preprocess(image_path)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        pred   = torch.argmax(probs).item()
    label      = CLASSES[pred]
    confidence = probs[pred].item() * 100
    return label, confidence

# ── explain with LLM ───────────────────────────────────
def explain(label, confidence):
    prompt = f"""
A chest X-ray was analyzed by a Medical AI model.

Result: {label}
Confidence: {confidence:.1f}%

Please provide:
1. What this result means in simple terms
2. What {label} typically looks like on a chest X-ray
3. What the patient should do next
4. A reminder that this is AI analysis and a real doctor should confirm

Keep it clear and easy to understand for a non-medical person.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful Medical AI assistant specialized in chest X-ray analysis. Be clear, accurate, and always recommend consulting a real doctor."},
            {"role": "user",   "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ── follow-up Q&A ──────────────────────────────────────
def followup_chat(label, confidence):
    messages = [
        {"role": "system", "content": f"You are a Medical AI assistant. The patient's chest X-ray was classified as {label} with {confidence:.1f}% confidence. Answer their follow-up questions clearly and always recommend seeing a real doctor."}
    ]
    print("\nYou can now ask follow-up questions.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Goodbye! Please consult a real doctor for medical advice.")
            break
        if user_input.strip() == "":
            continue

        messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        print(f"\nAI: {reply}\n")

# ── main ───────────────────────────────────────────────
def main():
    print("========================================")
    print("   Chest X-Ray Medical AI Assistant     ")
    print("========================================")
    print("Built with ConvNeXt-Tiny + LLaMA 3.3\n")

    cnn_model = load_model()
    print("Model loaded successfully!\n")

    image_path = input("Enter the path to your X-ray image: ").strip()

    print("\nAnalyzing X-ray...")
    label, confidence = predict(cnn_model, image_path)

    print(f"\nCNN Result : {label}")
    print(f"Confidence : {confidence:.1f}%")
    print("\nGenerating explanation...\n")
    print("=" * 40)

    explanation = explain(label, confidence)
    print(explanation)
    print("=" * 40)

    followup_chat(label, confidence)

if __name__ == "__main__":
    main()
