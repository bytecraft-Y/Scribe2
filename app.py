import streamlit as st
import streamlit.components.v1 as components
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

# 2. Custom CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 95vw !important;
    }
    
    .app-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2rem;
        color: #1E3A8A;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }

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

# 5. Dual-Pane Layout
col_controls, col_output = st.columns([1, 2], gap="large")

# --- LEFT PANEL: CONTROLS & MEDIA ---
with col_controls:
    with st.container(height=650, border=True):
        st.markdown("### 📥 Input Media")
        SUPPORTED_FORMATS = ["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"]
        uploaded_file = st.file_uploader("Upload audio or video", type=SUPPORTED_FORMATS, label_visibility="collapsed")
        
        tmp_media_path = None
        tmp_audio_path = None
        
        # We now store structured data (timestamps + text) instead of a giant string
        if 'segments_data' not in st.session_state:
            st.session_state.segments_data = []
        if 'pure_text_transcript' not in st.session_state:
            st.session_state.pure_text_transcript = ""
        
        if uploaded_file is not None:
            file_extension = os.path.splitext(uploaded_file.name)[1]
            tmp_media = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            tmp_media.write(uploaded_file.read())
            tmp_media_path = tmp_media.name
            tmp_media.close() 
            tmp_audio_path = "temp_audio_processing.wav"
            
            st.markdown("**Media Preview:**")
            # Render the native Streamlit media players
            if file_extension.lower() in ['.mp3', '.wav']:
                st.audio(tmp_media_path)
            else:
                st.video(tmp_media_path)
            
            st.markdown("---")
            
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
                        
                        st.write("✍️ Formatting synchronized timestamps...")
                        
                        # Clear old data
                        st.session_state.segments_data = []
                        pure_text_lines = []
                        
                        for segment in segments:
                            start_min, start_sec = divmod(int(segment.start), 60)
                            end_min, end_sec = divmod(int(segment.end), 60)
                            start_fmt = f"{start_min:02d}:{start_sec:02d}"
                            end_fmt = f"{end_min:02d}:{end_sec:02d}"
                            
                            # Save structured dictionary for the JavaScript tracker
                            st.session_state.segments_data.append({
                                "start": segment.start,
                                "end": segment.end,
                                "text": segment.text.strip(),
                                "display_time": f"[{start_fmt} -> {end_fmt}]"
                            })
                            
                            pure_text_lines.append(f"[{start_fmt} -> {end_fmt}] {segment.text.strip()}")
                        
                        st.session_state.pure_text_transcript = "\n".join(pure_text_lines)
                        
                        status.update(label="Transcription Complete!", state="complete", expanded=False)
                        st.rerun() 
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    
                finally:
                    if os.path.exists(tmp_media_path):
                        os.remove(tmp_media_path)
                    if os.path.exists(tmp_audio_path):
                        os.remove(tmp_audio_path)
                    gc.collect()

# --- RIGHT PANEL: INTERACTIVE TRACKER ---
with col_output:
    with st.container(height=650, border=True):
        st.markdown("### 📄 Real-Time Transcript")
        
        if not st.session_state.segments_data:
            st.info("👈 Upload a file and click 'Start Transcription' to begin.")
        else:
            # 1. Build the Interactive HTML Canvas
            html_content = """
            <div id="transcript-box" style="
                height: 480px; 
                overflow-y: auto; 
                padding: 15px; 
                background-color: #F8FAFC; 
                border-radius: 8px; 
                border: 2px solid #E2E8F0; 
                font-family: 'Courier New', Courier, monospace; 
                font-size: 15px;
                line-height: 1.8; 
                color: #0F172A;">
            """
            
            # Inject spans with hidden data attributes containing the exact seconds
            for i, seg in enumerate(st.session_state.segments_data):
                html_content += (
                    f"<span class='transcript-segment' id='seg-{i}' "
                    f"data-start='{seg['start']}' data-end='{seg['end']}' "
                    f"style='padding: 2px 4px; border-radius: 4px; transition: all 0.2s ease; display: inline-block;'>"
                    f"<strong>{seg['display_time']}</strong> {seg['text']}"
                    f"</span><br>"
                )
            html_content += "</div>"
            
            # Render the HTML
            st.markdown(html_content, unsafe_allow_html=True)
            
            # 2. Inject the JavaScript Bridge to sync Video/Audio to the Text
            js_sync_code = """
            <script>
                // Reach out of the iframe to find the Streamlit parent document
                const parentDoc = window.parent.document;
                
                // Set an interval to find the media element (it sometimes loads a second later)
                let syncInterval = setInterval(() => {
                    const mediaElement = parentDoc.querySelector('video, audio');
                    const segments = parentDoc.querySelectorAll('.transcript-segment');
                    
                    if (mediaElement && segments.length > 0) {
                        clearInterval(syncInterval); // Found it, stop polling
                        
                        mediaElement.addEventListener('timeupdate', () => {
                            const currentTime = mediaElement.currentTime;
                            
                            segments.forEach(seg => {
                                const start = parseFloat(seg.getAttribute('data-start'));
                                const end = parseFloat(seg.getAttribute('data-end'));
                                
                                // If the media is currently playing this segment, highlight it
                                if (currentTime >= start && currentTime <= end) {
                                    seg.style.backgroundColor = '#DBEAFE'; // Light Blue
                                    seg.style.color = '#1D4ED8'; // Dark Blue text
                                    seg.style.fontWeight = 'bold';
                                } else {
                                    // Reset when the media passes
                                    seg.style.backgroundColor = 'transparent';
                                    seg.style.color = '#0F172A';
                                    seg.style.fontWeight = 'normal';
                                }
                            });
                        });
                    }
                }, 1000);
            </script>
            """
            # Render the invisible script
            components.html(js_sync_code, height=0, width=0)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Keep the standard text download intact
            st.download_button(
                label="📥 Download Transcript as TXT",
                data=st.session_state.pure_text_transcript,
                file_name="Scribe_Transcript.txt",
                mime="text/plain"
            )
