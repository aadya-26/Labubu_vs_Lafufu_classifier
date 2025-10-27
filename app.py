import streamlit as st
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import time
import base64
from io import BytesIO

# Page config
st.set_page_config(page_title="Labubu Detector", layout="wide", page_icon="🧸")

# Function to convert image to base64
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Load corner image
corner_img_path = r"Labubu_vs_Lafufu_classifier/labubu_wo_background.png"
corner_img_base64 = image_to_base64(corner_img_path)

# Custom CSS with corner images
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    .stApp {{
        background: linear-gradient(135deg, #4c1d95 0%, #5b21b6 30%, #6366f1 70%, #8b5cf6 100%);
        background-attachment: fixed;
    }}
    
    [data-testid="stHeader"], .css-1v3fvcr {{display: none;}}
    
    .main-container {{
        background: rgba(255, 255, 255, 0.98);
        border-radius: 30px;
        padding: 3rem 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        max-width: 1200px;
        margin: 0 auto;
    }}
    
    .corner-image {{
        position: fixed;
        width: 100px;
        height: 100px;
        z-index: 1000;
        opacity: 0.9;
        transition: all 0.3s ease;
        animation: float 3s ease-in-out infinite;
    }}
    
    .corner-image:hover {{
        opacity: 1;
        transform: scale(1.1) !important;
    }}
    
    .corner-left {{
        top: 20px;
        left: 20px;
    }}
    
    .corner-right {{
        top: 20px;
        right: 20px;
        transform: scaleX(-1);
    }}
    
    @keyframes float {{
        0%, 100% {{
            transform: translateY(0px);
        }}
        50% {{
            transform: translateY(-10px);
        }}
    }}
    
    .corner-right:hover {{
        transform: scaleX(-1) scale(1.1) !important;
    }}
    
   .title {{
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #93c5fd 0%, #60a5fa 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(147, 197, 253, 0.3);
        margin-top: 2rem;
    }}
    
    .subtitle {{
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 500;
    }}

    .analyze-btn button {{
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: #0f172a;
        border: none;
        padding: 0.9rem 2.5rem;
        font-size: 1.2rem;
        font-weight: 700;
        border-radius: 15px;
        width: 100%;
        transition: all 0.3s ease;
        margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(251, 191, 36, 0.4);
    }}

    .analyze-btn button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(251, 191, 36, 0.6);
    }}

    .result-card {{
        padding: 2rem;
        animation: slideIn 0.5s ease-out, pulse 0.3s ease-in-out 0.5s;
        transform-origin: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}

    @keyframes slideIn {{
        from {{
            opacity: 0;
            transform: translateX(30px) scale(0.9);
        }}
        to {{
            opacity: 1;
            transform: translateX(0) scale(1);
        }}
    }}

    @keyframes pulse {{
        0%, 100% {{
            transform: scale(1);
        }}
        50% {{
            transform: scale(1.03);
        }}
    }}

    .confidence-bar {{
        width: 100%;
        height: 12px;
        background: rgba(255,255,255,0.3);
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }}

    .confidence-fill {{
        height: 100%;
        background: linear-gradient(90deg, rgba(255,255,255,0.8) 0%, rgba(255,255,255,1) 100%);
        border-radius: 10px;
        transition: width 1s ease-out;
        animation: fillBar 1s ease-out;
    }}

    @keyframes fillBar {{
        from {{
            width: 0% !important;
        }}
    }}

    .stats-container {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(245, 158, 11, 0.1) 100%);
        border-radius: 20px;
    }}

    .stat-box {{
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }}

    .stat-box:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 2px solid rgba(251, 191, 36, 0.3);
    }}

    .stat-number {{
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }}

    .stat-label {{
        font-size: 1rem;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .image-box {{
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }}

    .image-box img {{
        border-radius: 15px;
        width: 100% !important;
        height: auto !important;
        max-height: 450px;
        object-fit: contain;
    }}

    .stFileUploader {{
        margin-bottom: 1.5rem;
    }}

    /* Hide default streamlit elements */
    .stColumn > div {{
        height: 100%;
    }}

</style>
""", unsafe_allow_html=True)

# Add corner images if available
if corner_img_base64:
    st.markdown(f"""
    <img src="data:image/png;base64,{corner_img_base64}" class="corner-image corner-left" alt="Labubu">
    <img src="data:image/png;base64,{corner_img_base64}" class="corner-image corner-right" alt="Labubu">
    """, unsafe_allow_html=True)

# Initialize session state
if 'real_count' not in st.session_state:
    st.session_state.real_count = 0
if 'fake_count' not in st.session_state:
    st.session_state.fake_count = 0
if 'unsure_count' not in st.session_state:
    st.session_state.unsure_count = 0

# Load model (cached)
@st.cache_resource
def load_model():
    try:
        # Load from Hugging Face Hub
        model_name = "Aadyasingh/Labubu_vs_Lafufu_Classifier"
        
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        return processor, model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please check if the model is correctly uploaded to Hugging Face.")
        return None, None, None

# Title
st.markdown('<div class="title">Labubu Authenticator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect authentic toys with precision </div>', unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader("Drop your image here", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Analyze button at top
    if st.button("🔍 Analyze Toy", key="analyze", use_container_width=True, help="Click to analyze", type="primary"):
        with st.spinner('🔄 Analyzing your toy...'):
            # Load model
            processor, model, device = load_model()
            
            # Check if model loaded successfully
            if processor is None or model is None:
                st.error("Failed to load the model. Please try again later.")
            else:
                # Process image
                inputs = processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(device)
                
                # Predict
                with torch.no_grad():
                    outputs = model(pixel_values=pixel_values)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    prediction = logits.argmax(dim=-1).item()
                    confidence = probs[0][prediction].item() * 100
                    labubu_confidence = probs[0][0].item() * 100
                    lafufu_confidence = probs[0][1].item() * 100
                
                time.sleep(0.5)  # Brief pause for effect

                # Store result in session state
                if max(labubu_confidence, lafufu_confidence) < 60:
                    st.session_state.result_type = "unsure"
                    st.session_state.unsure_count += 1
                    st.session_state.result_confidence = max(labubu_confidence, lafufu_confidence)
                elif prediction == 0:
                    st.session_state.result_type = "real"
                    st.session_state.real_count += 1
                    st.session_state.result_confidence = confidence
                else:
                    st.session_state.result_type = "fake"
                    st.session_state.fake_count += 1
                    st.session_state.result_confidence = confidence

    # Side by side layout
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        # Resize image to fit box width
        box_width = 500  # approximate width of the box
        img_width, img_height = image.size
        aspect_ratio = img_height / img_width
        new_width = box_width
        new_height = int(box_width * aspect_ratio)
        
        # Limit height if too tall
        if new_height > 450:
            new_height = 450
            new_width = int(450 / aspect_ratio)
        
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        st.image(resized_image, use_container_width=True, caption="📸 Your uploaded toy")

    with col2:
        # Display result if available
        if hasattr(st.session_state, 'result_type'):
            if st.session_state.result_type == "unsure":
                st.markdown(f"""
                <div class="result-card" style="background: linear-gradient(135deg, #fb923c 0%, #f97316 100%); border-radius:20px; text-align:center; color: white;">
                    <div style="font-size:6rem; margin-bottom: 1rem;">🤷</div>
                    <div style="font-size:2.5rem; font-weight:800; margin-bottom: 0.5rem;">Uncertain</div>
                    <div style="margin-bottom:1.5rem; font-size: 1.1rem; opacity: 0.95;">This doesn't appear to be a Labubu or Lafufu</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {st.session_state.result_confidence:.0f}%;"></div>
                    </div>
                    <p style="font-size: 1.2rem; font-weight: 700;">Confidence: {st.session_state.result_confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            elif st.session_state.result_type == "real":
                st.markdown(f"""
                <div class="result-card" style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); border-radius:20px; text-align:center; color: white;">
                    <div style="font-size:6rem; margin-bottom: 1rem;">🎉</div>
                    <div style="font-size:2.5rem; font-weight:800; margin-bottom: 0.5rem;">Authentic Labubu!</div>
                    <div style="margin-bottom:1.5rem; font-size: 1.1rem; opacity: 0.95;">This is a genuine Labubu toy</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {st.session_state.result_confidence:.0f}%;"></div>
                    </div>
                    <p style="font-size: 1.2rem; font-weight: 700;">Confidence: {st.session_state.result_confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); border-radius:20px; text-align:center; color: white;">
                    <div style="font-size:6rem; margin-bottom: 1rem;">⚠️</div>
                    <div style="font-size:2.5rem; font-weight:800; margin-bottom: 0.5rem;">Counterfeit Detected</div>
                    <div style="margin-bottom:1.5rem; font-size: 1.1rem; opacity: 0.95;">This appears to be a fake Lafufu</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {st.session_state.result_confidence:.0f}%;"></div>
                    </div>
                    <p style="font-size: 1.2rem; font-weight: 700;">Confidence: {st.session_state.result_confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%); border-radius:20px; text-align:center; color: #475569; padding: 3rem 2rem; height: 100%;">
                <div style="font-size:5rem; margin-bottom: 1rem;">🔮</div>
                <div style="font-size:2rem; font-weight:700; margin-bottom: 0.5rem;">Ready to Scan</div>
                <div style="font-size: 1.1rem;">Click the analyze button to check authenticity</div>
            </div>
            """, unsafe_allow_html=True)

# Enhanced Stats
st.markdown(f"""
<div class="stats-container">
    <div class="stat-box">
        <div class="stat-number" style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🎉 {st.session_state.real_count}</div>
        <div class="stat-label">Authentic</div>
    </div>
    <div class="stat-box">
        <div class="stat-number" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">⚠️ {st.session_state.fake_count}</div>
        <div class="stat-label">Counterfeit</div>
    </div>
    <div class="stat-box">
        <div class="stat-number" style="background: linear-gradient(135deg, #fb923c 0%, #f97316 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🤷 {st.session_state.unsure_count}</div>
        <div class="stat-label">Uncertain</div>
    </div>
</div>
""", unsafe_allow_html=True)

