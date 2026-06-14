import streamlit as st
from faster_whisper import WhisperModel
import moviepy.editor as mp
import os
import tempfile
import gc

# 1. Page Configuration MUST be first
st.set_page_config(
    page_title="Scribe AI | Workspace", 
    page_icon="🎙️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Aggressive CSS to lock the layout and kill page scrolling
st.markdown("""
<style>
    /* Hide all default Streamlit nav and footers */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Kill main page scrolling and reduce top padding to push app to the top */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 95vw !important;
    }
    
    /* Title Styling */
    .app-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2rem;
        color: #1E3A8A;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }

    /* Style the Text Area to fill the right panel perfectly */
    .stTextArea textarea {
        background-color: #F8FAFC !important;
        border-radius: 8px;
        border: 2px solid #E2E8F0 !important;
        font-family: 'Courier New', Courier, monospace;
        font-size: 14px;
        line-height: 1.6;
        color: #0F172A !important; /* Forces dark text */
        -webkit-text-fill-color: #0F172A !important; /* Forces dark text even when disabled */
    }
    
    /* Style Download Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-weight: 600;
        width: 100%;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Backend Model Loading
@st.cache_resource
def load_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

# 4. App Header
st.markdown("<h1 class='app-title'>🎙️ Scribe AI Workspace</h1>", unsafe_allow_html=True)
st.markdown("---")

# 5. The Rigid Dual-Pane Layout
col_controls, col_output = st.columns([1, 2], gap="large")

# --- LEFT PANEL: FIXED HEIGHT CONTROLS ---
with col_controls:
    # This locks the left side to exactly 650px tall. It will scroll internally if needed.
    with st.container(height=650, border=True):
        st.markdown("### 📥 Input Media")
        SUPPORTED_FORMATS = ["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"]
        uploaded_file = st.file_uploader("Upload audio or video", type=SUPPORTED_FORMATS, label_visibility="collapsed")
        
        tmp_media_path = None
        tmp_audio_path = None
        
        if 'transcript_text' not in st.session_state:
            st.session_state.transcript_text = ""
        
        if uploaded_file is not None:
            file_extension = os.path.splitext(uploaded_file.name)[1]
            tmp_media = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            tmp_media.write(uploaded_file.read())
            tmp_media_path = tmp_media.name
            tmp_media.close() 
            tmp_audio_path = "temp_audio_processing.wav"
            
            st.markdown("**Media Preview:**")
            if file_extension.lower() in ['.mp3', '.wav']:
                st.audio(tmp_media_path)
            else:
                st.video(tmp_media_path)
            
            st.markdown("---")
            
            # Button to trigger processing so the app doesn't auto-run instantly
            if st.button("🚀 Start Transcription"):
                try:
                    with st.status("Engine Active...", expanded=True) as status:
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
                        
                        # Save to session state so it doesn't vanish if the page rerenders
                        st.session_state.transcript_text = "\n".join(transcript_lines)
                        
                        status.update(label="Transcription Complete!", state="complete", expanded=False)
                        st.rerun() # Instantly refreshes the right panel with the text
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    
                finally:
                    if os.path.exists(tmp_media_path):
                        os.remove(tmp_media_path)
                    if os.path.exists(tmp_audio_path):
                        os.remove(tmp_audio_path)
                    gc.collect()

# --- RIGHT PANEL: FIXED HEIGHT EDITOR ---
with col_output:
    # Locks the right side to exactly 650px tall to match the left.
    with st.container(height=650, border=True):
        st.markdown("### 📄 Transcript Editor")
        
        if st.session_state.transcript_text == "":
            st.info("👈 Upload a file and click 'Start Transcription' to begin.")
            # Height 500 ensures the text box fills the container
            st.text_area("Output", "", height=500, disabled=True, label_visibility="collapsed")
        else:
            st.text_area("Output", st.session_state.transcript_text, height=480, label_visibility="collapsed")
            
            st.download_button(
                label="📥 Download Transcript as TXT",
                data=st.session_state.transcript_text,
                file_name="Scribe_Transcript.txt",
                mime="text/plain"
            )
