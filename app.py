import streamlit as st
import streamlit.components.v1 as components
from faster_whisper import WhisperModel
import os
import tempfile
import gc
import time
import random

# 1. Page Configuration
st.set_page_config(
    page_title="Scribe AI | Enterprise Workspace", 
    page_icon="🎙️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS - "Chrome Stripped" UI & Enterprise Layout
st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; max-width: 95vw !important; }
    .app-title { font-family: 'Inter', sans-serif; font-weight: 800; font-size: 2.2rem; color: #0F172A; margin-bottom: 0px; letter-spacing: -0.5px;}
    .app-subtitle { color: #3B82F6; font-size: 1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px; display: block; }
    div.stButton > button:first-child { background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%); color: white; font-weight: 600; width: 100%; border-radius: 8px; border: none; transition: 0.3s; }
    div[data-testid="stTabs"] button { font-weight: 600; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# Helper Functions for Subtitles
def to_srt_time(seconds):
    h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60); ms = int((s % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

def to_vtt_time(seconds):
    h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60); ms = int((s % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"

# Load AI Model Natively
@st.cache_resource
def load_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

# Header
st.markdown("<h1 class='app-title'>Scribe AI Workspace</h1><span class='app-subtitle'>Enterprise Knowledge Extraction</span>", unsafe_allow_html=True)
st.markdown("---")

col_controls, col_output = st.columns([1, 2], gap="large")

# ==========================================
# LEFT PANEL: MEDIA & ANALYTICS
# ==========================================
with col_controls:
    with st.container(height=720, border=True):
        st.markdown("### 📥 Input Media")
        uploaded_file = st.file_uploader("Upload audio or video", type=["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"], label_visibility="collapsed")
        
        # Anti-Ghosting Placeholder
        media_placeholder = st.empty()
        
        # Initialize States
        if 'segments_data' not in st.session_state: 
            st.session_state.update({'segments_data': [], 'pure_text': '', 'srt_text': '', 'vtt_text': '', 'analytics': None, 'ai_summary': None})
        
        # Hard Reset on File Removal
        if uploaded_file is None:
            media_placeholder.empty()
            st.session_state.analytics = None
            st.session_state.ai_summary = None
            st.session_state.segments_data = []
            st.session_state.pure_text = ""
            st.session_state.srt_text = ""
            st.session_state.vtt_text = ""
        else:
            # File Processing
            uploaded_file.seek(0)
            file_extension = os.path.splitext(uploaded_file.name)[1]
            tmp_media = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            tmp_media.write(uploaded_file.read())
            tmp_media.close()
            tmp_media_path = tmp_media.name
            
            with media_placeholder.container():
                st.markdown("**Media Preview:**")
                uploaded_file.seek(0)
                if file_extension.lower() in ['.mp3', '.wav']:
                    st.audio(uploaded_file)
                else:
                    st.video(uploaded_file)
            
            # Advanced Analytics Grid
            if st.session_state.analytics:
                st.markdown("#### 📊 Speaker Diagnostics")
                c1, c2 = st.columns(2)
                c3, c4 = st.columns(2)
                c1.metric("Language", st.session_state.analytics['lang'])
                c2.metric("AI Confidence", st.session_state.analytics['conf'])
                c3.metric("Duration", st.session_state.analytics['dur'])
                c4.metric("Pace (WPM)", st.session_state.analytics['wpm'])
            
            st.markdown("---")
            
            if st.button("🚀 Start Extraction Pipeline"):
                # Glowing CSS Sound Wave
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
                        # Feed the media directly to Faster-Whisper (No MoviePy needed)
                        st.write("🧠 Running Base AI Inference...")
                        segments, info = model.transcribe(tmp_media_path, beam_size=7, vad_filter=True)
                        
                        st.session_state.segments_data, pure_lines, srt_lines, vtt_lines = [], [], [], ["WEBVTT\n"]
                        last_quote = time.time()
                        total_words = 0
                        
                        motivational_quotes = [
                            "Extracting semantic data...", 
                            "Processing acoustic signals...", 
                            "Compiling transcription matrix...",
                            "\"The secret of getting ahead is getting started.\" — Mark Twain",
                            "Resolving ambiguity in real time..."
                        ]
                        
                        for i, seg in enumerate(segments, 1):
                            if time.time() - last_quote > 4: 
                                status.update(label=f"⏳ {random.choice(motivational_quotes)}", expanded=True)
                                last_quote = time.time()
                            
                            text_val = seg.text.strip()
                            total_words += len(text_val.split())
                            
                            start_fmt = f"[{int(seg.start)//60:02d}:{int(seg.start)%60:02d} -> {int(seg.end)//60:02d}:{int(seg.end)%60:02d}]"
                            st.session_state.segments_data.append({"start": seg.start, "end": seg.end, "text": text_val, "display_time": start_fmt})
                            pure_lines.append(f"{start_fmt} {text_val}")
                            srt_lines.append(f"{i}\n{to_srt_time(seg.start)} --> {to_srt_time(seg.end)}\n{text_val}\n")
                            vtt_lines.append(f"{to_vtt_time(seg.start)} --> {to_vtt_time(seg.end)}\n{text_val}\n")
                        
                        # Populate Analytics
                        total_dur = info.duration
                        duration_mins = total_dur / 60
                        calc_wpm = int(total_words / duration_mins) if duration_mins > 0 else 0
                        
                        st.session_state.analytics = {
                            "lang": info.language.upper(), 
                            "conf": f"{info.language_probability*100:.1f}%", 
                            "dur": f"{int(total_dur//60)}m {int(total_dur%60)}s",
                            "wpm": f"{calc_wpm} WPM"
                        }
                        
                        st.session_state.update({'pure_text': "\n".join(pure_lines), 'srt_text': "\n".join(srt_lines), 'vtt_text': "\n".join(vtt_lines)})
                        st.session_state.ai_summary = None 
                        
                        visual_loader.empty()
                        st.rerun()
                        
                except Exception as e: 
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(tmp_media_path): os.remove(tmp_media_path)
                    gc.collect()


# ==========================================
# RIGHT PANEL: TABS & INTERACTIVITY
# ==========================================
with col_output:
    with st.container(height=720, border=True):
        if not st.session_state.segments_data:
            st.markdown("### 📄 Workspace Output")
            st.info("👈 Upload & Run Pipeline to begin data extraction.")
        else:
            tab_transcript, tab_ai = st.tabs(["📄 Raw Transcript", "✨ AI Insights (Beta)"])
            
            # --- TAB 1: RAW TRANSCRIPT ---
            with tab_transcript:
                st.markdown("<input type='text' id='search-input' placeholder='🔍 Search exact word...' style='width: 100%; padding: 12px; margin-bottom: 10px; border-radius: 8px; border: 1px solid #E2E8F0; color: #0F172A;'>", unsafe_allow_html=True)
                
                # Dark Slate text (#0F172A) forced to override Streamlit Dark Mode
                html_content = "<div id='transcript-box' style='height: 400px; overflow-y: auto; padding: 15px; background: #F8FAFC; border-radius: 8px; color: #0F172A;'>"
                for i, seg in enumerate(st.session_state.segments_data):
                    html_content += f"<span class='transcript-segment' id='seg-{i}' data-start='{seg['start']}' data-end='{seg['end']}' style='display:inline-block; padding: 2px 4px; border-radius:4px; cursor:pointer; color: #0F172A;' title='Click to seek'><strong>{seg['display_time']}</strong> {seg['text']}</span><br>"
                html_content += "</div>"
                st.markdown(html_content, unsafe_allow_html=True)
                
                st.markdown("<br>**Export Options:**", unsafe_allow_html=True)
                col_txt, col_srt, col_vtt = st.columns(3)
                col_txt.download_button("📥 TXT", st.session_state.pure_text, "Scribe.txt")
                col_srt.download_button("🎬 SRT", st.session_state.srt_text, "Scribe.srt")
                col_vtt.download_button("🌐 VTT", st.session_state.vtt_text, "Scribe.vtt")

            # --- TAB 2: AI INSIGHTS ---
            with tab_ai:
                st.markdown("### 🧠 Automated Knowledge Extraction")
                st.write("Generate a structured executive summary and extract key questions from the raw audio data.")
                
                if st.button("✨ Generate Smart Summary"):
                    with st.spinner("Analyzing semantic structure..."):
                        time.sleep(1.5) 
                        
                        # Fallback Mock NLP Extractor
                        full_text = " ".join([s['text'] for s in st.session_state.segments_data])
                        sentences = [s.strip() for s in full_text.replace('?', '.').replace('!', '.').split('.') if len(s) > 20]
                        questions = [s.strip() + "?" for s in full_text.split('?') if len(s) > 15][:-1]
                        
                        mock_summary = "#### 📌 Executive Summary\n"
                        if sentences:
                            mock_summary += f"* **Primary Topic:** The speaker discusses concepts relating to '{sentences[0][:50]}...'\n"
                            mock_summary += f"* **Key Statement:** \"{random.choice(sentences)}\"\n"
                        else:
                            mock_summary += "* Insufficient data for summary.\n"
                            
                        if questions:
                            mock_summary += "\n#### ❓ Extracted Questions/Action Items\n"
                            for q in questions[:3]:
                                mock_summary += f"* {q}\n"
                                
                        st.session_state.ai_summary = mock_summary
                
                if st.session_state.ai_summary:
                    st.markdown("<div style='background: #F1F5F9; padding: 20px; border-radius: 8px; border-left: 4px solid #3B82F6; color: #0F172A;'>", unsafe_allow_html=True)
                    st.markdown(st.session_state.ai_summary)
                    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# JAVASCRIPT BRIDGE (Precise Regex & Auto-Scroll)
# ==========================================
if st.session_state.segments_data:
    js_code = r"""
    <script>
        const parentDoc=window.parent.document; 
        let syncInterval=setInterval(()=>{ 
            const media=parentDoc.querySelector('video, audio'); 
            const segs=parentDoc.querySelectorAll('.transcript-segment'); 
            const searchInput = parentDoc.getElementById('search-input');
            let activeSegmentId = null;
            
            if(media && segs.length>0){ 
                clearInterval(syncInterval); 
                
                // Click-to-Seek
                segs.forEach(seg => {
                    seg.addEventListener('click', () => {
                        media.currentTime = parseFloat(seg.getAttribute('data-start'));
                        media.play();
                    });
                });
                
                // Precise Regex Keyword Search
                if (searchInput) {
                    searchInput.addEventListener('input', (e) => {
                        const term = e.target.value.trim();
                        if (term.length === 0) {
                            segs.forEach(seg => { seg.style.opacity = '1'; seg.style.backgroundColor = 'transparent'; });
                            return;
                        }
                        const escapedTerm = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                        const regex = new RegExp('\\b' + escapedTerm + '\\b', 'i');
                        segs.forEach(seg => {
                            if (regex.test(seg.innerText)) {
                                seg.style.opacity = '1'; seg.style.backgroundColor = '#FEF08A';
                            } else {
                                seg.style.opacity = '0.2'; seg.style.backgroundColor = 'transparent';
                            }
                        });
                    });
                }
                
                // Cinematic Auto-Scroll
                media.addEventListener('timeupdate', ()=>{ 
                    if (searchInput && searchInput.value.trim().length > 0) return;
                    const time=media.currentTime; 
                    segs.forEach((seg, i)=>{ 
                        const start=parseFloat(seg.getAttribute('data-start')); 
                        const end=parseFloat(seg.getAttribute('data-end')); 
                        
                        if(time>=start && time<=end){ 
                            seg.style.backgroundColor='#DBEAFE'; seg.style.color='#1D4ED8'; seg.style.fontWeight='bold';
                            if (activeSegmentId !== i) {
                                activeSegmentId = i;
                                seg.scrollIntoView({behavior:'smooth', block:'center'});
                            }
                        }else{ 
                            seg.style.backgroundColor='transparent'; seg.style.color='#0F172A'; seg.style.fontWeight='normal';
                        } 
                    }); 
                }); 
            }
        }, 1000);
    </script>
    """
    components.html(js_code, height=0)
