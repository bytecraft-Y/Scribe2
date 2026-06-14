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
    page_title="Scribe AI | Pro Workspace", 
    page_icon="🎙️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS for "Standalone Site" feel
st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; max-width: 95vw !important; }
    .app-title { font-family: 'Inter', sans-serif; font-weight: 800; font-size: 2rem; color: #1E3A8A; margin-bottom: 0px; }
    div.stButton > button:first-child { background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%); color: white; font-weight: 600; width: 100%; border-radius: 8px; border: none; transition: 0.3s; }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def to_srt_time(seconds):
    h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60); ms = int((s % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

def to_vtt_time(seconds):
    h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60); ms = int((s % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"

@st.cache_resource
def load_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

# Header
st.markdown("<h1 class='app-title'>🎙️ Scribe AI Pro Workspace</h1>", unsafe_allow_html=True)
st.markdown("---")

col_controls, col_output = st.columns([1, 2], gap="large")

# --- LEFT PANEL ---
with col_controls:
    with st.container(height=700, border=True):
        st.markdown("### 📥 Input Media")
        uploaded_file = st.file_uploader("Upload audio or video", type=["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"], label_visibility="collapsed")
        
        if 'segments_data' not in st.session_state: st.session_state.update({'segments_data': [], 'pure_text': '', 'srt_text': '', 'vtt_text': '', 'analytics': None})
        
        if uploaded_file:
            tmp_media = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
            tmp_media.write(uploaded_file.read()); tmp_media_path = tmp_media.name; tmp_media.close()
            tmp_audio_path = "temp_audio_processing.wav"
            
            st.markdown("**Media Preview:**")
            st.audio(tmp_media_path) if tmp_media_path.endswith(('.mp3', '.wav')) else st.video(tmp_media_path)
            
            if st.session_state.analytics:
                st.markdown("#### 📊 Performance Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Language", st.session_state.analytics['lang']); m2.metric("Accuracy", st.session_state.analytics['conf']); m3.metric("Duration", st.session_state.analytics['dur'])
            
            st.markdown("---")
            
            if st.button("🚀 Start Transcription"):
                loader_html = """<style> .wave-container { display: flex; justify-content: center; gap: 8px; margin-bottom: 15px;} .wave-bar { width: 8px; height: 16px; background: #3B82F6; border-radius: 8px; animation: pulse 1.2s infinite; } @keyframes pulse { 0%, 100% { transform: scaleY(1); } 50% { transform: scaleY(2.5); } } </style><div class="wave-container"><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div></div>"""
                visual_loader = st.empty(); visual_loader.markdown(loader_html, unsafe_allow_html=True)
                
                try:
                    with st.status("Initializing...", expanded=True) as status:
                        clip = mp.AudioFileClip(tmp_media_path); total_dur = clip.duration; clip.write_audiofile(tmp_audio_path, fps=16000, logger=None); clip.close()
                        segments, info = model.transcribe(tmp_audio_path, beam_size=7, vad_filter=True)
                        st.session_state.analytics = {"lang": info.language.upper(), "conf": f"{info.language_probability*100:.1f}%", "dur": f"{int(total_dur//60)}m {int(total_dur%60)}s"}
                        
                        st.session_state.segments_data, pure_lines, srt_lines, vtt_lines = [], [], [], ["WEBVTT\n"]
                        for i, seg in enumerate(segments, 1):
                            start_fmt = f"[{int(seg.start)//60:02d}:{int(seg.start)%60:02d} -> {int(seg.end)//60:02d}:{int(seg.end)%60:02d}]"
                            st.session_state.segments_data.append({"start": seg.start, "end": seg.end, "text": seg.text.strip(), "display_time": start_fmt})
                            pure_lines.append(f"{start_fmt} {seg.text.strip()}")
                            srt_lines.append(f"{i}\n{to_srt_time(seg.start)} --> {to_srt_time(seg.end)}\n{seg.text.strip()}\n")
                            vtt_lines.append(f"{to_vtt_time(seg.start)} --> {to_vtt_time(seg.end)}\n{seg.text.strip()}\n")
                        
                        st.session_state.update({'pure_text': "\n".join(pure_lines), 'srt_text': "\n".join(srt_lines), 'vtt_text': "\n".join(vtt_lines)})
                        visual_loader.empty(); st.rerun()
                except Exception as e: st.error(f"Error: {e}")
                finally: 
                    for f in [tmp_media_path, tmp_audio_path]: 
                        if os.path.exists(f): os.remove(f)
                    gc.collect()

# --- RIGHT PANEL ---
with col_output:
    with st.container(height=700, border=True):
        if not st.session_state.segments_data:
            st.markdown("### 📄 Real-Time Transcript"); st.info("👈 Upload & Transcribe to begin.")
        else:
            st.markdown("<input type='text' id='search-input' placeholder='🔍 Search keywords...' style='width: 100%; padding: 12px; margin-bottom: 10px; border-radius: 8px; border: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
            html_content = "<div id='transcript-box' style='height: 480px; overflow-y: auto; padding: 15px; background: #F8FAFC; border-radius: 8px;'>"
            for i, seg in enumerate(st.session_state.segments_data):
                html_content += f"<span class='transcript-segment' id='seg-{i}' data-start='{seg['start']}' data-end='{seg['end']}' style='display:inline-block; padding: 2px 4px; border-radius:4px;'><strong>{seg['display_time']}</strong> {seg['text']}</span><br>"
            st.markdown(html_content + "</div>", unsafe_allow_html=True)
            
            components.html("<script>const parentDoc=window.parent.document; let syncInterval=setInterval(()=>{ const media=parentDoc.querySelector('video, audio'); const segs=parentDoc.querySelectorAll('.transcript-segment'); if(media && segs.length>0){ clearInterval(syncInterval); media.addEventListener('timeupdate', ()=>{ const time=media.currentTime; segs.forEach((seg, i)=>{ const start=parseFloat(seg.getAttribute('data-start')); const end=parseFloat(seg.getAttribute('data-end')); if(time>=start && time<=end){ seg.style.backgroundColor='#DBEAFE'; seg.scrollIntoView({behavior:'smooth', block:'center'}); }else{ seg.style.backgroundColor='transparent'; } }); }); }}, 1000);</script>", height=0)
            
            st.markdown("<br>**Export Options:**", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.download_button("📥 TXT", st.session_state.pure_text, "Scribe.txt"); c2.download_button("🎬 SRT", st.session_state.srt_text, "Scribe.srt"); c3.download_button("🌐 VTT", st.session_state.vtt_text, "Scribe.vtt")
