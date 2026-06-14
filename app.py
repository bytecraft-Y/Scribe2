import streamlit as st
import streamlit.components.v1 as components
from faster_whisper import WhisperModel
import moviepy.editor as mp
import os
import tempfile
import gc
import time
import random

st.set_page_config(page_title="Scribe AI | Workspace", layout="wide", initial_sidebar_state="collapsed")

# 1. Advanced Creative UI Styling
st.markdown("""
<style>
    /* Lock the viewport, kill all scrollbars */
    body, .stApp { overflow: hidden !important; background-color: #F8FAFC !important; }
    
    /* Premium Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
    }
    
    /* Locked scrollable transcript area */
    .transcript-scroll-area {
        height: 60vh !important;
        overflow-y: auto !important;
        padding: 20px;
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }
    
    /* Creative Pulse Animation for the Start Button */
    @keyframes pulse-ring { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
    .pulse-btn { animation: pulse-ring 2s infinite; }
</style>
""", unsafe_allow_html=True)

# Backend logic remains the same (Model Loading)
@st.cache_resource
def load_model():
    return WhisperModel("base", device="cpu", compute_type="int8")
model = load_model()

# 2. Main Layout (Using columns as fixed panes)
st.markdown("<h2 style='color:#1E3A8A; margin-bottom:20px;'>🎙️ Scribe Pro Workspace</h2>", unsafe_allow_html=True)
col_left, col_right = st.columns([1, 1.5], gap="large")

# --- LEFT PANE: CONTROLS (Fixed) ---
with col_left:
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Media", type=["mp4", "wav", "mp3"])
        
        if uploaded_file:
            # Media logic here (omitted for brevity, same as previous)
            if st.button("🚀 Start Transcription", key="start_btn"):
                st.info("Processing...") # Add your animation logic here
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- RIGHT PANE: SCROLLABLE DIALOG (Fixed) ---
with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📄 Interactive Transcript")
    # THE SCROLLABLE AREA
    st.markdown('<div class="transcript-scroll-area">', unsafe_allow_html=True)
    # Inject your transcript segments here
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
