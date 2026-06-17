import streamlit as st
import streamlit.components.v1 as components
from faster_whisper import WhisperModel
import os
import tempfile
import gc
import time
import random
import re

# 1. Page Configuration
st.set_page_config(
    page_title="Scribe AI | Transcript Extractor", 
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

# Load Speech-to-Text Model Natively
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

# Load Hugging Face Summarization Model Natively
@st.cache_resource
def load_summarizer_model():
    from transformers import pipeline
    # UPDATED: 'text2text-generation' replaces 'summarization' for v5 compatibility
    return pipeline("text2text-generation", model="facebook/bart-large-cnn", device=-1)

model = load_whisper_model()

# Header
st.markdown("<h1 class='app-title'>Scribe AI: Transcript Extractor</h1><span class='app-subtitle'>Audio-to-Text & Insight Extraction Tool</span>", unsafe_allow_html=True)
st.markdown("---")

col_controls, col_output = st.columns([1, 2], gap="large")

# ==========================================
# LEFT PANEL: MEDIA & ANALYTICS
# ==========================================
with col_controls:
    with st.container(height=720, border=True):
        st.markdown("### 📥 Source Media")
        
        # Initialize States
        if 'segments_data' not in st.session_state: 
            st.session_state.update({'segments_data': [], 'pure_text': '', 'srt_text': '', 'vtt_text': '', 'analytics': None, 'ai_summary': None})
            
        # Clean Local File Uploader
        uploaded_file = st.file_uploader("Upload audio or video for transcription", type=["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"], label_visibility="collapsed")

        # Hard Reset on File Removal
        if uploaded_file is None:
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
            
            st.markdown("**Media Preview:**")
            uploaded_file.seek(0)
            if file_extension.lower() in ['.mp3', '.wav']:
                st.audio(uploaded_file)
            else:
                st.video(uploaded_file)
            
            # Advanced Analytics Grid
            if st.session_state.analytics:
                st.markdown("#### 📊 Transcription Diagnostics")
                c1, c2 = st.columns(2)
                c3, c4 = st.columns(2)
                c1.metric("Language Detected", st.session_state.analytics['lang'])
                c2.metric("AI Confidence", st.session_state.analytics['conf'])
                c3.metric("Audio Duration", st.session_state.analytics['dur'])
                c4.metric("Pace (WPM)", st.session_state.analytics['wpm'])
            
            st.markdown("---")
            
            if st.button("🚀 Generate Transcript"):
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
                    with st.status("Initializing Transcription Engine...", expanded=True) as status:
                        st.write("🧠 Running Base AI Inference...")
                        segments, info = model.transcribe(tmp_media_path, beam_size=5, vad_filter=True)
                        
                        st.session_state.segments_data, pure_lines, srt_lines, vtt_lines = [], [], [], ["WEBVTT\n"]
                        last_quote = time.time()
                        total_words = 0
                        
                        motivational_quotes = [
                            "Extracting audio data...", 
                            "Transcribing speech to text...", 
                            "Compiling transcription matrix...",
                            "Filtering background noise...",
                            "Resolving text ambiguity..."
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
                    st.error(f"Transcription Error: {e}")
                finally:
                    if os.path.exists(tmp_media_path): os.remove(tmp_media_path)
                    gc.collect()


# ==========================================
# RIGHT PANEL: TABS & INTERACTIVITY
# ==========================================
with col_output:
    with st.container(height=720, border=True):
        if not st.session_state.segments_data:
            st.markdown("### 📄 Extraction Output")
            st.info("👈 Upload media and click 'Generate Transcript' to begin.")
        else:
            tab_transcript, tab_ai = st.tabs(["📄 Generated Transcript", "✨ Extracted Insights"])
            
            # --- TAB 1: RAW TRANSCRIPT ---
            with tab_transcript:
                st.markdown("<input type='text' id='search-input' placeholder='🔍 Search transcript for exact words...' style='width: 100%; padding: 12px; margin-bottom: 10px; border-radius: 8px; border: 1px solid #E2E8F0; color: #0F172A;'>", unsafe_allow_html=True)
                
                html_content = "<div id='transcript-box' style='height: 400px; overflow-y: auto; padding: 15px; background: #F8FAFC; border-radius: 8px; color: #0F172A;'>"
                for i, seg in enumerate(st.session_state.segments_data):
                    html_content += f"<span class='transcript-segment' id='seg-{i}' data-start='{seg['start']}' data-end='{seg['end']}' style='display:inline-block; padding: 2px 4px; border-radius:4px; cursor:pointer; color: #0F172A;' title='Click to seek'><strong>{seg['display_time']}</strong> {seg['text']}</span><br>"
                html_content += "</div>"
                st.markdown(html_content, unsafe_allow_html=True)
                
                st.markdown("<br>**Download Transcript Files:**", unsafe_allow_html=True)
                col_txt, col_srt, col_vtt = st.columns(3)
                col_txt.download_button("📥 Plain Text (.txt)", st.session_state.pure_text, "Scribe_Transcript.txt")
                col_srt.download_button("🎬 Subtitles (.srt)", st.session_state.srt_text, "Scribe_Transcript.srt")
                col_vtt.download_button("🌐 Web Subtitles (.vtt)", st.session_state.vtt_text, "Scribe_Transcript.vtt")

            # --- TAB 2: HUGGING FACE TRANSFORMER INSIGHTS ---
            with tab_ai:
                st.markdown("### 🧬 Embedded Transformer Engine")
                st.write("Utilizing an offline **BART neural network** (`facebook/bart-large-cnn`) for deep abstractive text summarization. Your data never leaves your machine.")
                
                st.write("---")
                
                if st.button("✨ Extract Insights from Transcript", type="primary"):
                    with st.spinner("Initializing Hugging Face pipeline & analyzing text... (This may take a moment on the first run)"):
                        try:
                            # 1. Load the model from cache
                            summarizer = load_summarizer_model()
                            
                            # 2. Prepare the raw text
                            raw_transcript = " ".join([s['text'] for s in st.session_state.segments_data])
                            words = raw_transcript.split()
                            
                            if len(words) < 20:
                                st.session_state.ai_summary = "⚠️ The transcript is too short to generate a meaningful summary. Please provide longer audio."
                            else:
                                # 3. Smart Chunking (BART has a token limit, so we chunk large files by ~400 words)
                                chunk_size = 400
                                text_chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
                                
                                summaries = []
                                for idx, chunk in enumerate(text_chunks):
                                    # Dynamically calculate max length for short chunks
                                    chunk_len = len(chunk.split())
                                    max_len = min(150, int(chunk_len * 0.8))
                                    min_len = min(30, int(chunk_len * 0.2))
                                    
                                    if chunk_len > 10:
                                        res = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
                                        
                                        # UPDATED: Safely handle output keys for both older and newer versions of Transformers
                                        summary_output = res[0].get('summary_text', res[0].get('generated_text', ''))
                                        summaries.append(f"* {summary_output}")
                                
                                # 4. Compile the Final Output
                                compiled_summary = "\n".join(summaries)
                                
                                output = f"### 📌 Executive Summary (BART AI)\n\n"
                                output += "The following insights were autonomously generated by the offline deep learning model:\n\n"
                                output += compiled_summary
                                
                                st.session_state.ai_summary = output
                                st.toast("Transformer Summarization Complete!", icon="🚀")
                                
                        except Exception as hf_error:
                            st.error(f"Error executing Transformer pipeline: {str(hf_error)}")

                # Render Output Card
                if st.session_state.ai_summary:
                    st.markdown("<div style='background: #F8FAFC; padding: 25px; border-radius: 8px; border-left: 5px solid #06B6D4; color: #0F172A; margin-top: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>", unsafe_allow_html=True)
                    st.markdown(st.session_state.ai_summary)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button("📥 Download Extracted Insights", st.session_state.ai_summary, "BART_Extracted_Insights.md", "text/markdown")


# ==========================================
# JAVASCRIPT BRIDGE (Buffer-Flush Ghost Buster Edition)
# ==========================================
if st.session_state.segments_data:
    js_code = r"""
    <script>
        const parentWindow = window.parent;
        const parentDoc = parentWindow.document;
        
        if (parentWindow.scribeSyncInterval) {
            clearInterval(parentWindow.scribeSyncInterval);
        }
        
        parentWindow.scribeSyncInterval = setInterval(() => {
            
            // --- THE GHOST BUSTER (BUFFER FLUSH EDITION) ---
            const allMedia = parentDoc.querySelectorAll('video, audio');
            if (allMedia.length > 1) {
                for (let i = 0; i < allMedia.length - 1; i++) {
                    let ghost = allMedia[i];
                    ghost.pause();
                    ghost.removeAttribute('src'); 
                    ghost.load(); 
                    ghost.remove();
                }
            }
            
            const media = allMedia[allMedia.length - 1]; 
            const searchInput = parentDoc.getElementById('search-input');
            const transcriptBox = parentDoc.getElementById('transcript-box');
            
            if (!media || !transcriptBox || transcriptBox.offsetHeight === 0) return;
            
            // --- A. CLICK-TO-SEEK ---
            if (!transcriptBox.dataset.clickAttached) {
                transcriptBox.addEventListener('click', (e) => {
                    const seg = e.target.closest('.transcript-segment');
                    if (seg) {
                        const start = parseFloat(seg.getAttribute('data-start'));
                        if (!isNaN(start)) {
                            media.currentTime = start;
                            media.play();
                        }
                    }
                });
                transcriptBox.dataset.clickAttached = 'true';
            }
            
            // --- B. SEARCH ENGINE ---
            if (searchInput && !searchInput.dataset.searchAttached) {
                searchInput.addEventListener('input', (e) => {
                    const term = e.target.value.trim();
                    const escapedTerm = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                    const regex = term ? new RegExp('\\b' + escapedTerm + '\\b', 'i') : null;
                    const segs = transcriptBox.querySelectorAll('.transcript-segment');
                    
                    segs.forEach(seg => {
                        if (!term) {
                            seg.style.opacity = '1'; 
                            seg.style.backgroundColor = 'transparent';
                        } else if (regex.test(seg.innerText)) {
                            seg.style.opacity = '1'; 
                            seg.style.backgroundColor = '#FEF08A';
                        } else {
                            seg.style.opacity = '0.2'; 
                            seg.style.backgroundColor = 'transparent';
                        }
                    });
                });
                searchInput.dataset.searchAttached = 'true';
            }
            
            // --- C. THE REAL-TIME TRACKER ---
            if (searchInput && searchInput.value.trim().length > 0) return;
            
            const time = media.currentTime;
            const segs = transcriptBox.querySelectorAll('.transcript-segment');
            
            segs.forEach(seg => {
                const start = parseFloat(seg.getAttribute('data-start'));
                const end = parseFloat(seg.getAttribute('data-end'));
                
                if (time >= start && time <= end) {
                    if (seg.dataset.active !== 'true') { 
                        seg.dataset.active = 'true';
                        seg.style.backgroundColor = '#DBEAFE';  
                        seg.style.color = '#1D4ED8';            
                        seg.style.fontWeight = 'bold';
                        seg.scrollIntoView({behavior: 'smooth', block: 'center'}); 
                    }
                } else {
                    if (seg.dataset.active === 'true') {
                        seg.dataset.active = 'false';
                        seg.style.backgroundColor = 'transparent';
                        seg.style.color = '#0F172A';
                        seg.style.fontWeight = 'normal';
                    }
                }
            });
            
        }, 500); 
    </script>
    """
    components.html(js_code, height=0)
