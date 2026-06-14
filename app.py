import streamlit as st
from faster_whisper import WhisperModel
import moviepy.editor as mp
import os
import tempfile
import gc

st.set_page_config(page_title="Global Media Transcriber", layout="wide")
st.title("🌍 Universal Multilingual & Hinglish Transcriber")

@st.cache_resource
def load_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

model = load_model()

# --- UI Configuration ---
st.markdown("### Transcription Settings")
col1, col2 = st.columns(2)

with col1:
    LANGUAGE_OPTIONS = {
        "Auto-Detect Language": None,
        "Hinglish / Hindi (Phonetic Mixed)": "hi",
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de"
    }
    selected_lang_label = st.selectbox("Audio Language", list(LANGUAGE_OPTIONS.keys()))
    lang_code = LANGUAGE_OPTIONS[selected_lang_label]

with col2:
    mode_selection = st.radio(
        "Output Mode", 
        ["Transcribe (Keep original text)", "Translate (Force to English)"]
    )
    task_code = "translate" if "Translate" in mode_selection else "transcribe"

# --- NEW: Dynamic Context Input ---
st.markdown("#### Advanced Accuracy (Recommended for Hinglish)")
video_context = st.text_input(
    "What is this video about? (Optional)", 
    placeholder="e.g., Software engineering, a spiritual podcast, a gaming vlog..."
)
st.markdown("---")

# --- File Processing ---
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
        with st.spinner("Extracting audio track..."):
            clip = mp.AudioFileClip(tmp_media_path)
            clip.write_audiofile(tmp_audio_path, fps=16000, logger=None)
            clip.close()
            
        with st.spinner(f"Running {task_code} engine..."):
            
            # --- The Universal Hinglish Logic ---
            dynamic_prompt = None
            if lang_code == "hi":
                # A generic phonetic primer that tells the AI to expect mixed languages
                base_primer = "Haan, toh aaj hum baat karenge. Let's start the topic. "
                
                # If you typed a topic in the UI, it injects it to bias the vocabulary
                if video_context.strip():
                    dynamic_prompt = f"{base_primer} This audio is specifically about {video_context}."
                else:
                    dynamic_prompt = base_primer

            segments, info = model.transcribe(
                tmp_audio_path, 
                beam_size=5,
                language=lang_code,
                task=task_code,
                initial_prompt=dynamic_prompt,
                condition_on_previous_text=True 
            )
            
            if lang_code is None:
                st.success(f"Language Detected: **{info.language.upper()}** (Probability: {info.language_probability:.2f})")
            else:
                st.success("Processing Complete!")
            
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
        if os.path.exists(tmp_media_path):
            os.remove(tmp_media_path)
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        gc.collect()
