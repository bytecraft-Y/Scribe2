import streamlit as st
from faster_whisper import WhisperModel
import moviepy.editor as mp
import os
import tempfile
import gc

# 1. Page Configuration MUST be the first command
st.set_page_config(
    page_title="Scribe AI | Local Transcriber", 
    page_icon="🎙️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Inject Custom CSS for Professional Styling
st.markdown("""
<style>
    /* Hide default Streamlit branding and menus */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Center and style the main app title */
    .app-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        color: #1E3A8A; /* Deep Blue */
        text-align: center;
        margin-bottom: -10px;
    }
    
    .app-subtitle {
        text-align: center;
        color: #6B7280;
        font-size: 1.2rem;
        margin-bottom: 50px;
    }

    /* Style the Primary Download Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }

    /* Style the Text Area to look like a modern terminal */
    .stTextArea textarea {
        background-color: #F8FAFC;
        border-radius: 12px;
        border: 2px solid #E2E8F0;
        font-family: 'Courier New', Courier, monospace;
        font-size: 15px;
        color: #1E293B;
        padding: 15px;
        line-height: 1.6;
    }
    
    .stTextArea textarea:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 1px #3B82F6;
    }
    
    /* Clean up the file uploader box */
    .stFileUploader {
        border-radius: 12px;
        padding: 20px;
        background-color: #ffffff;
        border: 2px dashed #CBD5E1;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 3. Backend Model Loading
@st.cache_resource
def load_model():
    # Using the optimized base model for accuracy + speed
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

# 4. Main Dashboard UI
st.markdown("<h1 class='app-title'>🎙️ Scribe AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>Lightning-Fast Local Media Transcriber</p>", unsafe_allow_html=True)

# Wrap the uploader in a clean container centered on the page
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    SUPPORTED_FORMATS = ["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"]
    uploaded_file = st.file_uploader("Drop your media file here", type=SUPPORTED_FORMATS)

# 5. Processing Logic
if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    tmp_media = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    tmp_media.write(uploaded_file.read())
    tmp_media_path = tmp_media.name
    tmp_media.close() 

    tmp_audio_path = "temp_audio_processing.wav"
    
    st.markdown("---")
    
    # Create two columns for the output display: Left side Status, Right side Text
    status_col, output_col = st.columns([1, 2])

    try:
        with status_col:
            st.markdown("### ⚙️ Processing Status")
            # Using st.status for a clean, modern loading sequence
            with st.status("Initializing Engine...", expanded=True) as status:
                
                st.write("⏱️ Demuxing media file...")
                clip = mp.AudioFileClip(tmp_media_path)
                clip.write_audiofile(tmp_audio_path, fps=16000, logger=None)
                clip.close()
                
                st.write("🧠 Running AI Inference...")
                segments, info = model.transcribe(
                    tmp_audio_path, 
                    beam_size=7, 
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                st.write("✍️ Formatting timestamps...")
                transcript_lines = []
                for segment in segments:
                    start_min, start_sec = divmod(int(segment.start), 60)
                    end_min, end_sec = divmod(int(segment.end), 60)
                    start_time = f"{start_min:02d}:{start_sec:02d}"
                    end_time = f"{end_min:02d}:{end_sec:02d}"
                    transcript_lines.append(f"[{start_time} -> {end_time}] {segment.text.strip()}")
                
                transcript_text = "\n".join(transcript_lines)
                
                status.update(label="Transcription Complete!", state="complete", expanded=False)

        with output_col:
            st.markdown("### 📄 Results")
            st.text_area("Output", transcript_text, height=450, label_visibility="collapsed")
            
            # The download button is styled by the CSS above
            st.download_button(
                label="📥 Download Subtitles (TXT)",
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
