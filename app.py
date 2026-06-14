import streamlit as st
from faster_whisper import WhisperModel
import moviepy.editor as mp
import os
import tempfile
import gc

# 1. Page Configuration
st.set_page_config(
    page_title="Scribe AI | Workspace", 
    page_icon="🎙️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Inject Custom CSS for Fixed Workspace
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .app-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    .app-subtitle {
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 30px;
    }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }

    .stTextArea textarea {
        background-color: #F8FAFC;
        border-radius: 12px;
        border: 2px solid #E2E8F0;
        font-family: 'Courier New', Courier, monospace;
        font-size: 14px;
        color: #1E293B;
        padding: 15px;
        line-height: 1.6;
    }
    
    .stTextArea textarea:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 1px #3B82F6;
    }
    
    /* Create a distinct visual panel for the control side */
    .control-panel {
        background-color: #F1F5F9;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# 3. Backend Model Loading
@st.cache_resource
def load_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

# 4. Fixed One-Page Layout Header
st.markdown("<h1 class='app-title'>🎙️ Scribe AI Workspace</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>Lightning-Fast Local Media Transcriber</p>", unsafe_allow_html=True)

# 5. Define the Fixed Workspace Columns (1/3 for controls, 2/3 for output)
col_controls, col_output = st.columns([1, 2], gap="large")

# --- LEFT COLUMN: CONTROLS & STATUS ---
with col_controls:
    st.markdown("### 📥 Input Media")
    SUPPORTED_FORMATS = ["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"]
    uploaded_file = st.file_uploader("Upload audio or video", type=SUPPORTED_FORMATS, label_visibility="collapsed")
    
    # Placeholder variables for the processing logic
    tmp_media_path = None
    tmp_audio_path = None
    transcript_text = ""
    processing_complete = False

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        
        # Save file to temp storage
        tmp_media = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        tmp_media.write(uploaded_file.read())
        tmp_media_path = tmp_media.name
        tmp_media.close() 
        tmp_audio_path = "temp_audio_processing.wav"
        
        # Add a media player so the user can review what they uploaded
        st.markdown("**Media Preview:**")
        if file_extension.lower() in ['.mp3', '.wav']:
            st.audio(tmp_media_path)
        else:
            st.video(tmp_media_path)
        
        st.markdown("---")
        st.markdown("### ⚙️ Engine Status")
        
        try:
            with st.status("Processing Pipeline Active...", expanded=True) as status:
                st.write("⏱️ Demuxing media file...")
                clip = mp.AudioFileClip(tmp_media_path)
                clip.write_audiofile(tmp_audio_path, fps=16000, logger=None)
                clip.close()
                
                st.write("🧠 Running Base AI Inference...")
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
                processing_complete = True
                
                status.update(label="Transcription Complete!", state="complete", expanded=False)
                
        except Exception as e:
            st.error(f"A processing error occurred: {str(e)}")
            
        finally:
            # Cleanup
            if os.path.exists(tmp_media_path):
                os.remove(tmp_media_path)
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
            gc.collect()

# --- RIGHT COLUMN: OUTPUT WORKSPACE ---
with col_output:
    st.markdown("### 📄 Transcript Editor")
    
    if not processing_complete:
        # What the user sees BEFORE uploading a file
        st.info("👈 Upload a media file on the left to generate a transcript.")
        st.text_area("Output", "", height=500, disabled=True, label_visibility="collapsed")
    else:
        # What the user sees AFTER processing is done
        st.text_area("Output", transcript_text, height=500, label_visibility="collapsed")
        
        st.download_button(
            label="📥 Download Transcript as TXT",
            data=transcript_text,
            file_name="Scribe_Transcript.txt",
            mime="text/plain"
        )
