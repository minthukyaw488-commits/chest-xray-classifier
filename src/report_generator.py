"""
Medical report generator using Groq LLM API.
"""
import os
from groq import Groq


def get_groq_client():
    """Initialize Groq client with API key from environment or Streamlit secrets."""
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            raise ValueError(
                "GROQ_API_KEY not found. Set it as an environment variable "
                "or in Streamlit secrets."
            )
    
    return Groq(api_key=api_key)


def generate_medical_report(prediction, confidence, normal_prob, pneumonia_prob):
    """
    Generate a plain-language medical report using Groq LLM.
    
    Args:
        prediction: "NORMAL" or "PNEUMONIA"
        confidence: Confidence score (0-100)
        normal_prob: Probability of NORMAL class (0-100)
        pneumonia_prob: Probability of PNEUMONIA class (0-100)
    
    Returns:
        dict with keys: patient_summary, clinical_findings, 
                        recommendations, technical_notes
    """
    client = get_groq_client()
    
    prompt = f"""You are a medical AI assistant helping to interpret chest X-ray classification results.

The AI model has analyzed a chest X-ray with these results:
- Prediction: {prediction}
- Confidence: {confidence:.2f}%
- Probability of NORMAL: {normal_prob:.2f}%
- Probability of PNEUMONIA: {pneumonia_prob:.2f}%

Generate a clear, structured medical report with EXACTLY these four sections. Do not add any other sections. Keep each section concise (2-4 sentences).

**PATIENT SUMMARY**
(Explain the result in simple, non-technical language a patient can understand. Avoid medical jargon.)

**CLINICAL FINDINGS**
(Describe what the AI detected. If pneumonia, mention it could be bacterial or viral pneumonia and typical patterns. If normal, describe what makes it appear healthy.)

**RECOMMENDATIONS**
(Give specific, actionable next steps based on the confidence level. Higher confidence = stronger recommendations.)

**TECHNICAL NOTES**
(Brief professional summary for healthcare providers. Include confidence interpretation and any caveats.)

Be direct, avoid excessive disclaimers within the sections (there's a general disclaimer already). Use professional but accessible language.
"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful medical AI assistant that generates clear, structured reports based on chest X-ray classification results."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
        )
        
        full_text = response.choices[0].message.content
        return parse_report(full_text)
    
    except Exception as e:
        return {
            "error": f"Report generation failed: {str(e)}",
            "patient_summary": None,
            "clinical_findings": None,
            "recommendations": None,
            "technical_notes": None,
        }


def parse_report(text):
    """Parse the LLM response into structured sections."""
    sections = {
        "patient_summary": "",
        "clinical_findings": "",
        "recommendations": "",
        "technical_notes": "",
    }
    
    section_map = {
        "PATIENT SUMMARY": "patient_summary",
        "CLINICAL FINDINGS": "clinical_findings",
        "RECOMMENDATIONS": "recommendations",
        "TECHNICAL NOTES": "technical_notes",
    }
    
    current_section = None
    current_text = []
    
    for line in text.split("\n"):
        line = line.strip()
        
        # Check if this line is a section header
        found_header = False
        for header, key in section_map.items():
            if header in line.upper():
                # Save previous section
                if current_section:
                    sections[current_section] = "\n".join(current_text).strip()
                # Start new section
                current_section = key
                current_text = []
                found_header = True
                break
        
        if not found_header and current_section and line:
            # Remove markdown asterisks
            cleaned = line.replace("**", "").strip()
            if cleaned:
                current_text.append(cleaned)
    
    # Save last section
    if current_section:
        sections[current_section] = "\n".join(current_text).strip()
    
    return sections