import streamlit as st
from faster_whisper import WhisperModel
import moviepy.editor as mp
import google.generativeai as genai
import os
import tempfile
import gc

st.set_page_config(page_title="Advanced Hinglish Transcriber", layout="wide")
st.title("🌍 Advanced Hinglish Audio Transcriber")
st.write("Two-Stage Architecture: Faster-Whisper Extraction + LLM Transliteration.")

# --- API Security Sidebar ---
with st.sidebar:
    st.header("⚙️ System Configuration")
    st.write("An API key is required to convert Hindi script to English letters (Romanized Hinglish).")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    st.markdown("[Get a free API key here](https://aistudio.google.com/app/apikey)")

# 1. Load the Audio Extraction Engine
@st.cache_resource
def load_whisper():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

whisper_model = load_whisper()

# 2. The LLM Transliteration Agent
def convert_to_roman_hinglish(raw_text, key):
    genai.configure(api_key=key)
    
    # --- CHANGED: Use the universally stable 'gemini-pro' endpoint ---
    llm = genai.GenerativeModel('gemini-pro') 
    
    system_prompt = (
        "You are a strict transliteration engine. Your job is to take mixed Hindi/English text "
        "and rewrite it completely in Roman letters (the English alphabet). "
        "DO NOT translate the Hindi words to English words. Keep the exact phonetic Hinglish vocabulary. "
        "Example Input: आज हम machine learning सीखेंगे "
        "Example Output: Aaj hum machine learning seekhenge \n\n"
        f"Text to convert:\n{raw_text}"
    )
    
    response = llm.generate_content(system_prompt)
    return response.text
# --- File Processing ---
SUPPORTED_FORMATS = ["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"]
uploaded_file = st.file_uploader("Select Media File", type=SUPPORTED_FORMATS)

if uploaded_file is not None:
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar to enable processing.")
        st.stop()

    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    tmp_media = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    tmp_media.write(uploaded_file.read())
    tmp_media_path = tmp_media.name
    tmp_media.close() 

    st.info("File uploaded. Initiating Two-Stage Pipeline...")
    tmp_audio_path = "temp_audio_processing.wav"

    try:
        # Phase 1: Audio Extraction
        with st.spinner("Stage 1: Extracting audio track..."):
            clip = mp.AudioFileClip(tmp_media_path)
            clip.write_audiofile(tmp_audio_path, fps=16000, logger=None)
            clip.close()
            
        # Phase 2: Whisper Raw Text Generation
        with st.spinner("Stage 2: Running Whisper AI Extraction..."):
            segments, _ = whisper_model.transcribe(
                tmp_audio_path, 
                beam_size=5,
                language="hi", # Force Hindi mode to capture the vocabulary perfectly
                condition_on_previous_text=True 
            )
            raw_mixed_text = " ".join([segment.text for segment in segments])
            
        # Phase 3: LLM Script Conversion
        with st.spinner("Stage 3: Transliterating Devanagari to Roman Hinglish..."):
            final_hinglish_text = convert_to_roman_hinglish(raw_mixed_text, api_key)
            
        st.success("Pipeline Execution Complete!")
        
        # Display Results
        st.text_area("Final Romanized Hinglish Output", final_hinglish_text, height=300)
        
        with st.expander("View Raw Extraction Data (Debug Mode)"):
            st.write(raw_mixed_text)
            
        st.download_button(
            label="📥 Download as TXT",
            data=final_hinglish_text,
            file_name="hinglish_transcript.txt",
            mime="text/plain"
        )
            
    except Exception as e:
        st.error(f"A processing error occurred: {str(e)}")
        
    finally:
        if os.path.exists(tmp_media_path):
            os.remove(tmp_media_path)
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        gc.collect()
