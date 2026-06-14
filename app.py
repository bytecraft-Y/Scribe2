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

# 2. Custom CSS - "Chrome Stripped" UI
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

# --- LEFT PANEL: MEDIA & ANALYTICS ---
with col_controls:
    with st.container(height=700, border=True):
        st.markdown("### 📥 Input Media")
        uploaded_file = st.file_uploader("Upload audio or video", type=["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"], label_visibility="collapsed")
        
        # Persistent Placeholder for Media
        media_placeholder = st.empty()
        
        # Initialize States
        if 'segments_data' not in st.session_state: 
            st.session_state.update({'segments_data': [], 'pure_text': '', 'srt_text': '', 'vtt_text': '', 'analytics': None})
        
        if uploaded_file is None:
            media_placeholder.empty()
            st.session_state.analytics = None
            st.session_state.segments_data = []
            st.session_state.pure_text = ""
            st.session_state.srt_text = ""
            st.session_state.vtt_text = ""
        else:
            # Rewind the file pointer before reading
            uploaded_file.seek(0)
            
            file_extension = os.path.splitext(uploaded_file.name)[1]
            tmp_media = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            tmp_media.write(uploaded_file.read())
            tmp_media.close()
            tmp_media_path = tmp_media.name
            
            with media_placeholder.container():
                st.markdown("**Media Preview:**")
                # Seek back to 0 again for the Streamlit player to read it properly
                uploaded_file.seek(0)
                if file_extension.lower() in ['.mp3', '.wav']:
                    st.audio(uploaded_file)
                else:
                    st.video(uploaded_file)
            
            if st.session_state.analytics:
                st.markdown("#### 📊 Performance Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Lang", st.session_state.analytics['lang'])
                m2.metric("Conf", st.session_state.analytics['conf'])
                m3.metric("Dur", st.session_state.analytics['dur'])
            
            st.markdown("---")
            
            if st.button("🚀 Start Transcription"):
                # Safe HTML string assignment
                loader_html = """
                <style>
                    .wave-container { display: flex; justify-content: center; gap: 8px; margin-bottom: 15px;}
                    .wave-bar { width: 8px; height: 16px; background: #3B82F6; border-radius: 8px; animation: pulse 1.2s infinite; }
                    .wave-bar:nth-child(1) { animation-delay: -1.2s; }
                    .wave-bar:nth-child(2) { animation-delay: -1.1s; }
                    .wave-bar:nth-child(3) { animation-delay: -1.0s; }
                    .wave-bar:nth-child(4) { animation-delay: -0.9s; }
                    .wave-bar:nth-child(5) { animation-delay: -0.8s; }
                    @keyframes pulse { 0%, 100% { transform: scaleY(1); background: #93C5FD; } 50% { transform: scaleY(2.5); background: #2563EB; box-shadow: 0 0 12px rgba(37,99,235,0.6); } }
                </style>
                <div class="wave-container"><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div></div>
                """
                
                visual_loader = st.empty()
                visual_loader.markdown(loader_html, unsafe_allow_html=True)
                
                try:
                    with st.status("Initializing Engine...", expanded=True) as status:
                        tmp_audio_path = "temp_audio_processing.wav"
                        clip = mp.AudioFileClip(tmp_media_path)
                        total_dur = clip.duration
                        clip.write_audiofile(tmp_audio_path, fps=16000, logger=None)
                        clip.close()
                        
                        segments, info = model.transcribe(tmp_audio_path, beam_size=7, vad_filter=True)
                        st.session_state.analytics = {
                            "lang": info.language.upper(), 
                            "conf": f"{info.language_probability*100:.1f}%", 
                            "dur": f"{int(total_dur//60)}m {int(total_dur%60)}s"
                        }
                        
                        st.session_state.segments_data, pure_lines, srt_lines, vtt_lines = [], [], [], ["WEBVTT\n"]
                        last_quote = time.time()
                        
                        motivational_quotes = [
                            "Extracting the signal from the noise...",
                            "Translating acoustic waves into meaning...",
                            "\"The secret of getting ahead is getting started.\" — Mark Twain",
                            "\"Well begun is half done.\" — Aristotle",
                            "\"Creativity is intelligence having fun.\" — Albert Einstein",
                            "Parsing the soundscape...",
                            "Resolving ambiguity in real time...",
                            "Building understanding from waveforms..."
                        ]
                        
                        for i, seg in enumerate(segments, 1):
                            if time.time() - last_quote > 4: 
                                status.update(label=f"⏳ {random.choice(motivational_quotes)}", expanded=True)
                                last_quote = time.time()
                                
                            start_fmt = f"[{int(seg.start)//60:02d}:{int(seg.start)%60:02d} -> {int(seg.end)//60:02d}:{int(seg.end)%60:02d}]"
                            st.session_state.segments_data.append({"start": seg.start, "end": seg.end, "text": seg.text.strip(), "display_time": start_fmt})
                            pure_lines.append(f"{start_fmt} {seg.text.strip()}")
                            srt_lines.append(f"{i}\n{to_srt_time(seg.start)} --> {to_srt_time(seg.end)}\n{seg.text.strip()}\n")
                            vtt_lines.append(f"{to_vtt_time(seg.start)} --> {to_vtt_time(seg.end)}\n{seg.text.strip()}\n")
                        
                        st.session_state.update({'pure_text': "\n".join(pure_lines), 'srt_text': "\n".join(srt_lines), 'vtt_text': "\n".join(vtt_lines)})
                        visual_loader.empty()
                        st.rerun()
                        
                except Exception as e: 
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(tmp_media_path): os.remove(tmp_media_path)
                    if 'tmp_audio_path' in locals() and os.path.exists(tmp_audio_path): os.remove(tmp_audio_path)
                    gc.collect()

# --- RIGHT PANEL ---
with col_output:
    with st.container(height=700, border=True):
        if not st.session_state.segments_data:
            st.markdown("### 📄 Real-Time Transcript")
            st.info("👈 Upload & Transcribe to begin.")
        else:
            st.markdown("<input type='text' id='search-input' placeholder='🔍 Search keywords...' style='width: 100%; padding: 12px; margin-bottom: 10px; border-radius: 8px; border: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
            
            html_content = "<div id='transcript-box' style='height: 480px; overflow-y: auto; padding: 15px; background: #F8FAFC; border-radius: 8px;'>"
            for i, seg in enumerate(st.session_state.segments_data):
                html_content += f"<span class='transcript-segment' id='seg-{i}' data-start='{seg['start']}' data-end='{seg['end']}' style='display:inline-block; padding: 2px 4px; border-radius:4px; cursor:pointer;' title='Click to seek'><strong>{seg['display_time']}</strong> {seg['text']}</span><br>"
            html_content += "</div>"
            st.markdown(html_content, unsafe_allow_html=True)
            
            js_code = """
            <script>
                const parentDoc=window.parent.document; 
                let syncInterval=setInterval(()=>{ 
                    const media=parentDoc.querySelector('video, audio'); 
                    const segs=parentDoc.querySelectorAll('.transcript-segment'); 
                    const searchInput = parentDoc.getElementById('search-input');
                    let activeSegmentId = null;
                    
                    if(media && segs.length>0){ 
                        clearInterval(syncInterval); 
                        
                        segs.forEach(seg => {
                            seg.addEventListener('click', () => {
                                media.currentTime = parseFloat(seg.getAttribute('data-start'));
                                media.play();
                            });
                        });
                        
                        if (searchInput) {
                            searchInput.addEventListener('input', (e) => {
                                const term = e.target.value.toLowerCase();
                                segs.forEach(seg => {
                                    if (seg.innerText.toLowerCase().includes(term)) {
                                        seg.style.opacity = '1';
                                        seg.style.backgroundColor = term.length > 1 ? '#FEF08A' : 'transparent';
                                    } else {
                                        seg.style.opacity = '0.2';
                                        seg.style.backgroundColor = 'transparent';
                                    }
                                });
                            });
                        }
                        
                        media.addEventListener('timeupdate', ()=>{ 
                            if (searchInput && searchInput.value.length > 0) return;
                            const time=media.currentTime; 
                            segs.forEach((seg, i)=>{ 
                                const start=parseFloat(seg.getAttribute('data-start')); 
                                const end=parseFloat(seg.getAttribute('data-end')); 
                                if(time>=start && time<=end){ 
                                    seg.style.backgroundColor='#DBEAFE'; 
                                    if (activeSegmentId !== i) {
                                        activeSegmentId = i;
                                        seg.scrollIntoView({behavior:'smooth', block:'center'});
                                    }
                                }else{ 
                                    seg.style.backgroundColor='transparent'; 
                                } 
                            }); 
                        }); 
                    }
                }, 1000);
            </script>
            """
            components.html(js_code, height=0)
            
            st.markdown("<br>**Export Options:**", unsafe_allow_html=True)
            col_txt, col_srt, col_vtt = st.columns(3)
            col_txt.download_button("📥 TXT", st.session_state.pure_text, "Scribe.txt")
            col_srt.download_button("🎬 SRT", st.session_state.srt_text, "Scribe.srt")
            col_vtt.download_button("🌐 VTT", st.session_state.vtt_text, "Scribe.vtt")
