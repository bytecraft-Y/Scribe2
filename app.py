import streamlit as st
from faster_whisper import WhisperModel
import moviepy.editor as mp
import os
import tempfile
import gc

st.set_page_config(page_title="Global Media Transcriber", layout="wide")
st.title("🌍 Multilingual Transcription Assistant")
st.write("Powered by faster-whisper. Supports 99+ languages including native code-switching.")

# 1. Load the Optimized CTranslate2 Model
@st.cache_resource
def load_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

model = load_model()

# --- NEW: Language & Task Configuration UI ---
st.markdown("### Transcription Settings")
col1, col2 = st.columns(2)

with col1:
    # A mapped dictionary to pass the correct ISO language code to the model
    LANGUAGE_OPTIONS = {
        "Auto-Detect Language": None,
        "English": "en",
        "Hindi (Best for Hinglish)": "hi",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh",
        "Japanese": "ja",
        "Russian": "ru",
        "Portuguese": "pt"
    }
    selected_lang_label = st.selectbox("Audio Language", list(LANGUAGE_OPTIONS.keys()))
    lang_code = LANGUAGE_OPTIONS[selected_lang_label]

with col2:
    # Determine if the user wants the original text or translated English text
    mode_selection = st.radio(
        "Output Mode", 
        ["Transcribe (Keep original language)", "Translate (Force output to English)"]
    )
    task_code = "translate" if "Translate" in mode_selection else "transcribe"

st.markdown("---")
# ---------------------------------------------

# 2. File Upload Handling
SUPPORTED_FORMATS = ["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"]
uploaded_file = st.file_uploader("Select Media File", type=SUPPORTED_FORMATS)

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    tmp_media = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    tmp_media.write(uploaded_file.read())
    tmp_media_path = tmp_media.name
    tmp_media.close() 

    st.info("File uploaded successfully. Processing media...")
    tmp_audio_path = "temp_audio_processing.wav"

    try:
        # Phase 1: Fast Audio Extraction
        with st.spinner("Extracting audio track..."):
            clip = mp.AudioFileClip(tmp_media_path)
            clip.write_audiofile(tmp_audio_path, fps=16000, logger=None)
            clip.close()
            
        # Phase 2: High-Speed Multilingual Transcription
        with st.spinner(f"Running {task_code} engine..."):
            # Pass the user's UI selections directly into the inference engine
            segments, info = model.transcribe(
                tmp_audio_path, 
                beam_size=5,
                language=lang_code,
                task=task_code
            )
            
            # If auto-detect was used, show the user what language the AI found
            if lang_code is None:
                st.success(f"Language Detected: **{info.language.upper()}** (Probability: {info.language_probability:.2f})")
            else:
                st.success("Processing Complete!")
            
            # Compile the text
            transcript_text = " ".join([segment.text for segment in segments])
            
            st.text_area("Output", transcript_text, height=300)
            
            st.download_button(
                label="📥 Download as TXT",
                data=transcript_text,
                file_name="transcript.txt",
                mime="text/plain"
            )
            
    except Exception as e:
        st.error(f"A processing error occurred: {str(e)}")
        
    finally:
        # 3. Aggressive Server Cleanup
        if os.path.exists(tmp_media_path):
            os.remove(tmp_media_path)
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        gc.collect()
