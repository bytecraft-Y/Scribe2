import streamlit as st
import streamlit.components.v1 as components
from faster_whisper import WhisperModel
import os
import tempfile
import gc
import time
import random
import re
from collections import Counter  # <--- ADD THIS LINE

# 1. Page Configuration
st.set_page_config(
    page_title="Scribe AI | Transcript Extractor", 
    page_icon="🎙️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS - Cinematic Dark Theme & Glassmorphism
st.markdown("""
<style>
    /* HIDE DEFAULT STREAMLIT ELEMENTS */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; max-width: 95vw !important; }
    
    /* 1. ADDING THE DARK BACKGROUND */
    .stApp {
        /* OPTION A: If you want to use a specific image link from Freepik */
        /* background-image: url("data:image/avif;base64,AAAAHGZ0eXBhdmlmAAAAAG1pZjFhdmlmbWlhZgAAANZtZXRhAAAAAAAAACFoZGxyAAAAAAAAAABwaWN0AAAAAAAAAAAAAAAAAAAAAA5waXRtAAAAAAABAAAAImlsb2MAAAAAREAAAQABAAAAAAD6AAEAAAAAAAABXAAAACNpaW5mAAAAAAABAAAAFWluZmUCAAAAAAEAAGF2MDEAAAAAVmlwcnAAAAA4aXBjbwAAAAxhdjFDgQQMAAAAABRpc3BlAAAAAAAAAuQAAAHuAAAAEHBpeGkAAAAAAwgICAAAABZpcG1hAAAAAAAAAAEAAQOBAgMAAAFkbWRhdBIACgoZJi4/bwQENBoQMssCRGAAUUUUQPS++8XWN++IAwOpd8kxsmoef55qpVZKHKXTZdUgT8Kw0vtDwY7NgUTW2eBtKViyc6wxSWjmtvjzvob/GjQfJOqFMUyMR06Y0QnUlFpa3GPccXPeQ5aH8YED/aXrm8aMyPzcSFmkXkeeV7OVxEohOVqVbSeAaWq1q7eYOENWm1ZUWkR8cYqehVZdr7RzEMNueD85K0qL5FwIND1golZ11mN/rTX5OW9U/5Nt5/6L/qDhh3plGCDNb5jyM6zEOIrJyuP1ZH2goH3D7fyL8bBDgRTwgzuM76C1io+ydAIvOS502LcEJUqSZIDY5LaXXEnaBXVAmODxtAJgF9+Y+3XR3WMfWSk/O1OFrBqyVod1qbgxu3Vdou2UHTU89cLIao7W/cTBhr9uPt9HCWiO6/gA8/sm7eenmTSKq01FHmUPD6kpfnswgA=="); */
        
           }

    /* 2. INVERTING TEXT COLORS FOR DARK MODE */
    .app-title { font-family: 'Inter', sans-serif; font-weight: 800; font-size: 2.2rem; color: #F8FAFC; margin-bottom: 0px; letter-spacing: -0.5px;}
    .app-subtitle { color: #38BDF8; font-size: 1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px; display: block; }
    p, span, li, h1, h2, h3, h4, h5, h6, label { color: #F1F5F9 !important; }
    
    /* 3. FROSTED GLASS PANELS (Glassmorphism) */
    /* This makes the white boxes semi-transparent dark boxes to let the background blur through */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px;
    }

    /* 4. TRANSCRIPT BOX & BUTTON STYLING */
    #transcript-box { background: rgba(0, 0, 0, 0.3) !important; color: #F8FAFC !important; border: 1px solid rgba(255, 255, 255, 0.05); }
    .transcript-segment { color: #CBD5E1 !important; }
    
    /* 5. SEARCH BAR FIX FOR DARK MODE */
    #search-input { 
        background: rgba(15, 23, 42, 0.8) !important; 
        color: #F8FAFC !important; 
        border: 1px solid rgba(255, 255, 255, 0.2) !important; 
    }
    #search-input::placeholder { color: #64748B !important; }

    div.stButton > button:first-child { background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%); color: white; font-weight: 600; width: 100%; border-radius: 8px; border: none; transition: 0.3s; }
    div[data-testid="stTabs"] button { font-weight: 600; font-size: 1.1rem; color: #94A3B8; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #38BDF8 !important; }
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

# Load Hugging Face Summarization Model (Bulletproof Direct Method)
@st.cache_resource
def load_summarizer_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    # Load the tokenizer and model directly, bypassing the pipeline registry
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    
    return tokenizer, model

model = load_whisper_model()

# Header
st.markdown("<h1 class='app-title'>Scribe AI: Transcript Extractor</h1><span class='app-subtitle'>Audio/Video-to-Text & Insight Extraction Tool</span>", unsafe_allow_html=True)
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
            tab_transcript, tab_ai = st.tabs(["📄 Generated Transcript", "✨ Extract Insights"])
            
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
# --- TAB 2: HYBRID OFFLINE ENGINE (BART + HEURISTICS) ---
            with tab_ai:
               
                
                if st.button("✨ Extract Insights from Transcript", type="primary"):
                    with st.spinner("Analyzing text... (This may take a moment)"):
                        try:
                            # 1. Load the model and tokenizer from cache
                            tokenizer, summarizer_model = load_summarizer_model()
                            
                            # 2. Prepare the raw text
                            raw_transcript = " ".join([s['text'] for s in st.session_state.segments_data])
                            words = raw_transcript.split()
                            
                            if len(words) < 20:
                                st.session_state.ai_summary = "⚠️ The transcript is too short to generate a meaningful summary. Please provide longer audio."
                            else:
                                # ==========================================
                                # PART A: BART NEURAL SUMMARIZATION
                                # ==========================================
                                chunk_size = 400
                                text_chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
                                
                                summaries = []
                                for chunk in text_chunks:
                                    if len(chunk.split()) > 10:
                                        inputs = tokenizer(chunk, max_length=1024, return_tensors="pt", truncation=True)
                                        summary_ids = summarizer_model.generate(
                                            inputs["input_ids"], 
                                            num_beams=4, 
                                            max_length=150, 
                                            min_length=30, 
                                            early_stopping=True
                                        )
                                        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                                        summaries.append(f"{summary_text}")
                                
                                compiled_summary = "\n\n".join(summaries)
                                
                                # ==========================================
                                # PART B: HEURISTIC EXTRACTION (Actions, Decisions, Tags)
                                # ==========================================
                                sentences = [s.strip() for s in re.split(r'[.!?]', raw_transcript) if len(s.strip()) > 15]
                                
                                # 1. Extract Tags (Basic Frequency)
                                stop_words = set(["the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "is", "it", "you", "that", "this", "was", "for", "on", "as", "with", "so", "we", "they", "i", "are", "be", "have", "not", "do", "will", "can", "about", "what", "how"])
                                word_counts = Counter([w.lower() for w in re.findall(r'\b\w+\b', raw_transcript) if w.lower() not in stop_words and len(w) > 3])
                                top_tags = [word for word, count in word_counts.most_common(5)]
                                
                                # 2. Extract Action Items & Decisions (Regex Heuristics)
                                action_pattern = re.compile(r'\b(need to|have to|will|should|must|task|assign|todo|action item|fix|update|build)\b', re.IGNORECASE)
                                decision_pattern = re.compile(r'\b(decided|agreed|concluded|choose|settled|resolved|instead of)\b', re.IGNORECASE)
                                
                                extracted_actions = []
                                extracted_decisions = []
                                
                                for sent in sentences:
                                    if action_pattern.search(sent) and sent not in extracted_actions and len(extracted_actions) < 5:
                                        extracted_actions.append(sent)
                                    if decision_pattern.search(sent) and sent not in extracted_decisions and len(extracted_decisions) < 4:
                                        extracted_decisions.append(sent)

                                # ==========================================
                                # PART C: COMPILE THE FINAL ENTERPRISE REPORT
                                # ==========================================
                                output = f"**🏷️ Extracted Tags:** {', '.join([k.capitalize() for k in top_tags])}\n\n---\n"
                                
                                output += f"### 📌 Executive Summary\n{compiled_summary}\n\n"
                                
                                output += "### ✅ Action Items & Tasks\n"
                                if extracted_actions:
                                    for a in extracted_actions: output += f"* {a.capitalize()}.\n"
                                else: output += "* No explicit action items detected in this transcript.\n"
                                
                                output += "\n### 💡 Key Decisions & Insights\n"
                                if extracted_decisions:
                                    for d in extracted_decisions: output += f"* {d.capitalize()}.\n"
                                else: output += "* No explicit strategic decisions flagged.\n"
                                
                                st.session_state.ai_summary = output
                                st.toast("Hybrid Extraction Complete!", icon="🚀")
                                
                        except Exception as hf_error:
                            st.error(f"Error executing AI pipeline: {str(hf_error)}")

                # Render Output Card
                if st.session_state.ai_summary:
                    # Use Streamlit's native container with a clean border
                    with st.container(border=True):
                        st.markdown(st.session_state.ai_summary)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button("📥 Download Extracted Insights", st.session_state.ai_summary, "Enterprise_Insights_Report.md", "text/markdown")
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
