import streamlit as st
import streamlit.components.v1 as components
from faster_whisper import WhisperModel
import moviepy.editor as mp
import os
import tempfile
import gc
import time
import random

# 1. Page Configuration
st.set_page_config(
    page_title="Scribe | Career Pulse AI", 
    page_icon="🎙️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Aggressive CSS "Chrome Stripping" & Theming
st.markdown("""
<style>
    /* Completely hide Streamlit's native UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none !important;} /* Hides the sidebar arrow */
    
    /* The Custom Sticky Glass Navbar */
    .glass-nav {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 65px;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(226, 232, 240, 0.8);
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 40px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .nav-logo {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 800;
        font-size: 1.4rem;
        color: #0F172A;
        display: flex;
        align-items: center;
    }
    .nav-brand-sub {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
        margin-left: 8px;
        letter-spacing: 0.5px;
    }
    .nav-badges {
        display: flex;
        gap: 15px;
        align-items: center;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .badge-pro {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 4px rgba(37,99,235,0.2);
    }
    .badge-lab {
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Adjust main workspace to account for the sticky navbar */
    .block-container {
        padding-top: 90px !important;
        padding-bottom: 0rem !important;
        max-width: 95vw !important;
    }
    
    /* Make analytics metrics look like floating SaaS cards */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }

    /* Style Primary Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-weight: 600;
        width: 100%;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# 3. Inject the Custom Navbar
st.markdown("""
<div class="glass-nav">
    <div class="nav-logo">
        🎙️ Scribe <span class="nav-brand-sub">| Powered by Career Pulse AI</span>
    </div>
    <div class="nav-badges">
        <span class="badge-lab">Lab: AI, 9</span>
        <span class="badge-pro">PRO WORKSPACE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def to_srt_time(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    ms = int((s % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

def to_vtt_time(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    ms = int((s % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"

# 4. Backend Model Loading
@st.cache_resource
def load_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

# 5. Dual-Pane Layout
col_controls, col_output = st.columns([1, 2], gap="large")

# --- LEFT PANEL: MEDIA & ANALYTICS ---
with col_controls:
    with st.container(height=720, border=True):
        st.markdown("### 📥 Input Media")
        SUPPORTED_FORMATS = ["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"]
        uploaded_file = st.file_uploader("Upload audio or video", type=SUPPORTED_FORMATS, label_visibility="collapsed")
        
        tmp_media_path = None
        tmp_audio_path = None
        
        if 'segments_data' not in st.session_state:
            st.session_state.segments_data = []
        if 'pure_text' not in st.session_state:
            st.session_state.pure_text = ""
        if 'srt_text' not in st.session_state:
            st.session_state.srt_text = ""
        if 'vtt_text' not in st.session_state:
            st.session_state.vtt_text = ""
        if 'analytics' not in st.session_state:
            st.session_state.analytics = None
        
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
            
            if st.session_state.analytics:
                st.markdown("<br><b>📊 Audio Analytics</b>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Language", st.session_state.analytics['lang'])
                m2.metric("Confidence", st.session_state.analytics['conf'])
                m3.metric("Duration", st.session_state.analytics['dur'])
            
            st.markdown("---")
            
            if st.button("🚀 Start Transcription"):
                try:
                    motivational_quotes = [
                        "\"Great things are not done by impulse...\" — Vincent Van Gogh",
                        "\"The only way to do great work is to love what you do.\" — Steve Jobs",
                        "\"Words are, of course, the most powerful drug.\" — Rudyard Kipling",
                        "Extracting the signal from the noise...",
                        "Translating acoustic waves into meaning..."
                    ]
                    
                    with st.status("Initializing Engine...", expanded=True) as status:
                        st.write("⏱️ Demuxing media file...")
                        clip = mp.AudioFileClip(tmp_media_path)
                        total_duration = clip.duration
                        clip.write_audiofile(tmp_audio_path, fps=16000, logger=None)
                        clip.close()
                        
                        st.write("🧠 Running Base AI Inference...")
                        segments, info = model.transcribe(
                            tmp_audio_path, 
                            beam_size=7, 
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500)
                        )
                        
                        st.session_state.analytics = {
                            "lang": info.language.upper(),
                            "conf": f"{info.language_probability * 100:.1f}%",
                            "dur": f"{int(total_duration // 60)}m {int(total_duration % 60)}s"
                        }
                        
                        st.session_state.segments_data = []
                        pure_lines = []
                        srt_lines = []
                        vtt_lines = ["WEBVTT\n"]
                        
                        last_quote_time = time.time()
                        current_quote = random.choice(motivational_quotes)
                        status.update(label=f"⏳ {current_quote}")
                        
                        for i, segment in enumerate(segments, start=1):
                            if time.time() - last_quote_time > 4:
                                current_quote = random.choice(motivational_quotes)
                                status.update(label=f"⏳ {current_quote}")
                                last_quote_time = time.time()
                                
                            start_min, start_sec = divmod(int(segment.start), 60)
                            end_min, end_sec = divmod(int(segment.end), 60)
                            display_time = f"[{start_min:02d}:{start_sec:02d} -> {end_min:02d}:{end_sec:02d}]"
                            
                            st.session_state.segments_data.append({
                                "start": segment.start,
                                "end": segment.end,
                                "text": segment.text.strip(),
                                "display_time": display_time
                            })
                            
                            pure_lines.append(f"{display_time} {segment.text.strip()}")
                            srt_lines.append(f"{i}\n{to_srt_time(segment.start)} --> {to_srt_time(segment.end)}\n{segment.text.strip()}\n")
                            vtt_lines.append(f"{to_vtt_time(segment.start)} --> {to_vtt_time(segment.end)}\n{segment.text.strip()}\n")
                        
                        st.session_state.pure_text = "\n".join(pure_lines)
                        st.session_state.srt_text = "\n".join(srt_lines)
                        st.session_state.vtt_text = "\n".join(vtt_lines)
                        
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

# --- RIGHT PANEL: INTERACTIVE EDITOR ---
with col_output:
    with st.container(height=720, border=True):
        
        if not st.session_state.segments_data:
            st.markdown("### 📄 Real-Time Transcript")
            st.info("👈 Upload a file and click 'Start Transcription' to begin.")
        else:
            st.markdown("<input type='text' id='search-input' placeholder='🔍 Search keywords in transcript...' style='width: 100%; padding: 12px; margin-bottom: 15px; border-radius: 8px; border: 1px solid #CBD5E1; font-family: sans-serif; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
            
            html_content = """
            <div id="transcript-box" style="
                height: 500px; 
                overflow-y: auto; 
                padding: 20px; 
                background-color: #F8FAFC; 
                border-radius: 12px; 
                border: 1px solid #E2E8F0; 
                font-family: 'Courier New', Courier, monospace; 
                font-size: 15px;
                line-height: 1.8; 
                color: #0F172A;">
            """
            for i, seg in enumerate(st.session_state.segments_data):
                html_content += (
                    f"<span class='transcript-segment' id='seg-{i}' "
                    f"data-start='{seg['start']}' data-end='{seg['end']}' "
                    f"style='padding: 2px 6px; border-radius: 4px; transition: all 0.2s ease; display: inline-block;'>"
                    f"<strong style='color:#64748B;'>{seg['display_time']}</strong> {seg['text']}"
                    f"</span><br>"
                )
            html_content += "</div>"
            st.markdown(html_content, unsafe_allow_html=True)
            
            js_sync_code = """
            <script>
                const parentDoc = window.parent.document;
                let syncInterval = setInterval(() => {
                    const mediaElement = parentDoc.querySelector('video, audio');
                    const segments = parentDoc.querySelectorAll('.transcript-segment');
                    const searchInput = parentDoc.getElementById('search-input');
                    let activeSegmentId = null;

                    if (mediaElement && segments.length > 0) {
                        clearInterval(syncInterval);

                        segments.forEach(seg => {
                            seg.style.cursor = 'pointer';
                            seg.title = "Click to jump to this moment";
                            seg.addEventListener('click', () => {
                                const start = parseFloat(seg.getAttribute('data-start'));
                                mediaElement.currentTime = start;
                                mediaElement.play();
                            });
                        });

                        if (searchInput) {
                            searchInput.addEventListener('input', (e) => {
                                const term = e.target.value.toLowerCase();
                                segments.forEach(seg => {
                                    if (seg.innerText.toLowerCase().includes(term)) {
                                        seg.style.opacity = '1';
                                        if (term.length > 1) {
                                            seg.style.backgroundColor = '#FEF08A';
                                        } else {
                                            seg.style.backgroundColor = 'transparent';
                                        }
                                    } else {
                                        seg.style.opacity = '0.3';
                                        seg.style.backgroundColor = 'transparent';
                                    }
                                });
                            });
                        }

                        mediaElement.addEventListener('timeupdate', () => {
                            const currentTime = mediaElement.currentTime;
                            if (searchInput && searchInput.value.length > 0) return;

                            segments.forEach((seg, index) => {
                                const start = parseFloat(seg.getAttribute('data-start'));
                                const end = parseFloat(seg.getAttribute('data-end'));
                                
                                if (currentTime >= start && currentTime <= end) {
                                    seg.style.backgroundColor = '#DBEAFE'; 
                                    seg.style.color = '#1D4ED8'; 
                                    seg.style.fontWeight = 'bold';
                                    if (activeSegmentId !== index) {
                                        activeSegmentId = index;
                                        seg.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                    }
                                } else {
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
            components.html(js_sync_code, height=0, width=0)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_txt, col_srt, col_vtt = st.columns(3)
            with col_txt:
                st.download_button("📥 Basic (TXT)", st.session_state.pure_text, "Scribe.txt", "text/plain")
            with col_srt:
                st.download_button("🎬 Premiere (SRT)", st.session_state.srt_text, "Scribe.srt", "text/plain")
            with col_vtt:
                st.download_button("🌐 Web (VTT)", st.session_state.vtt_text, "Scribe.vtt", "text/plain")
