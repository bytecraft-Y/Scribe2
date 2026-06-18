import streamlit as st
import streamlit.components.v1 as components
from faster_whisper import WhisperModel
import os
import tempfile
import gc
import time
import random
import re
from collections import Counter

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Scribe AI | Transcript Extractor", 
    page_icon="🎙️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================
# 2. CUSTOM CSS - Cinematic Dark Theme & Glassmorphism
# =========================================
bg_image_base64 = "" 
st.markdown(f"""
<style>
    #MainMenu {{visibility: hidden;}} footer {{visibility: hidden;}} header {{visibility: hidden;}}
    
    /* Safe layout padding */
    .block-container {{ padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 95vw !important; }}
    
    .stApp {{
        background: radial-gradient(circle at top right, #1E293B 0%, #020617 100%);
        background-image: url("{bg_image_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .app-title {{ font-family: 'Inter', sans-serif; font-weight: 800; font-size: 2.2rem; color: #FFFFFF !important; margin-bottom: 0px; letter-spacing: -0.5px;}}
    .app-subtitle {{ color: #38BDF8 !important; font-size: 1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px; display: block; }}
    p, span, li, h1, h2, h3, h4, h5, h6, label {{ color: #F1F5F9 !important; }}
    
    div[data-testid="stVerticalBlock"] > div[style*="border"] {{
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 12px;
        padding: 1rem; 
    }}

    /* Responsive Transcript Box */
    #transcript-box {{ 
        background: rgba(0, 0, 0, 0.3) !important; 
        color: #F8FAFC !important; 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        height: 50vh; 
        overflow-y: auto; 
        padding: 15px;
        border-radius: 8px;
    }}
    
    #search-input {{ 
        background: rgba(15, 23, 42, 0.8) !important; 
        color: #F8FAFC !important; 
        border: 1px solid rgba(255, 255, 255, 0.2) !important; 
    }}
    
    div.stButton > button:first-child {{ background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%); color: white; font-weight: 600; width: 100%; border-radius: 8px; border: none; transition: 0.3s; }}
    div[data-testid="stTabs"] button {{ font-weight: 600; font-size: 1.1rem; color: #94A3B8; }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{ color: #38BDF8 !important; }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HELPER FUNCTIONS & AI MODELS
# ==========================================
def to_srt_time(seconds):
    h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60); ms = int((s % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

def to_vtt_time(seconds):
    h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60); ms = int((s % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

@st.cache_resource(show_spinner=False)
def load_summarizer_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    return tokenizer, model

# ==========================================
# 4. MAIN UI HEADER
# ==========================================
st.markdown("<h1 class='app-title'>Scribe AI: Transcript Extractor</h1><span class='app-subtitle'>Audio/Video-to-Text & Insight Extraction Tool</span>", unsafe_allow_html=True)
st.markdown("---")

col_controls, col_output = st.columns([1, 2], gap="large")

# ==========================================
# LEFT PANEL: MEDIA & ANALYTICS
# ==========================================
with col_controls:
    with st.container(border=True):
        st.markdown("### 📥 Source Media")
        
        if 'segments_data' not in st.session_state: 
            st.session_state.update({'segments_data': [], 'pure_text': '', 'srt_text': '', 'vtt_text': '', 'analytics': None, 'ai_summary': None})
            
        uploaded_file = st.file_uploader("Upload audio or video for transcription", type=["mp3", "wav", "m4a", "aac", "flac", "ogg", "mp4", "ts", "mov", "mkv", "avi"], label_visibility="collapsed")

        if uploaded_file is None:
            st.session_state.analytics = None
            st.session_state.ai_summary = None
            st.session_state.segments_data = []
            st.session_state.pure_text = ""
            st.session_state.srt_text = ""
            st.session_state.vtt_text = ""
        else:
            uploaded_file.seek(0)
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            tmp_media = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            tmp_media.write(uploaded_file.read())
            tmp_media.close()
            tmp_media_path = tmp_media.name
            
            st.markdown("**Media Preview:**")
            uploaded_file.seek(0)
            
            if file_extension in ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']:
                audio_bytes = uploaded_file.read()
                st.audio(audio_bytes)
            else:
                st.video(uploaded_file)
            
            if st.session_state.analytics:
                st.markdown("#### 📊 Transcription Diagnostics")
                c1, c2 = st.columns(2)
                c3, c4 = st.columns(2)
                c1.metric("Language Detected", st.session_state.analytics['lang'])
                c2.metric("AI Confidence", st.session_state.analytics['conf'])
                c3.metric("Audio Duration", st.session_state.analytics['dur'])
                c4.metric("Pace (WPM)", st.session_state.analytics['wpm'])
            
            st.markdown("---")
            
            if st.button("Generate Transcript"):
                
                model_scene = st.empty()
                if 'whisper_loaded' not in st.session_state:
                    scene_html = """
                    <div style='padding: 30px; border-radius: 12px; background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); color: white; text-align: center; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3); margin-bottom: 20px;'>
                        <h3 style='margin-bottom: 5px; font-weight: 600; color: white;'>🧠 Booting Whisper Edge-AI...</h3>
                        <p style='color: #94A3B8; font-size: 0.95rem;'>Loading neural network weights into active memory. This happens once per session.</p>
                        <div style='margin-top: 20px; display: flex; justify-content: center;'>
                            <div style="border: 4px solid rgba(255,255,255,0.1); border-left-color: #3B82F6; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite;"></div>
                        </div>
                        <style>@keyframes spin { to { transform: rotate(360deg); } }</style>
                    </div>
                    """
                    model_scene.markdown(scene_html, unsafe_allow_html=True)
                
                model = load_whisper_model()
                st.session_state.whisper_loaded = True
                model_scene.empty()

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
                        
                        segments, info = model.transcribe(tmp_media_path, beam_size=5, vad_filter=False)
                        
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
    with st.container(border=True):
        if not st.session_state.segments_data:
            st.markdown("### 📄 Extraction Output")
            st.info("👈 Upload media and click 'Generate Transcript' to begin.")
        else:
            # NO EMOJIS IN THE STRING BELOW TO PREVENT SYNTAX ERRORS
            tab_transcript, tab_ai = st.tabs(["Transcript", "Insights"])
            
            with tab_transcript:
                st.markdown("<input type='text' id='search-input' placeholder='🔍 Search transcript for exact words...' style='width: 100%; padding: 12px; margin-bottom: 10px; border-radius: 8px; border: 1px solid #E2E8F0; color: #0F172A;'>", unsafe_allow_html=True)
                
                html_content = "<div id='transcript-box'>"
                for i, seg in enumerate(st.session_state.segments_data):
                    html_content += f"<span class='transcript-segment' id='seg-{i}' data-start='{seg['start']}' data-end='{seg['end']}' style='display:inline-block; padding: 2px 4px; border-radius:4px; cursor:pointer; color: #CBD5E1; transition: 0.2s;' title='Click to seek'><strong>{seg['display_time']}</strong> {seg['text']}</span><br>"
                html_content += "</div>"
                st.markdown(html_content, unsafe_allow_html=True)
                
                st.markdown("<br>**Download Transcript Files:**", unsafe_allow_html=True)
                col_txt, col_srt, col_vtt = st.columns(3)
                col_txt.download_button("📥 Plain Text (.txt)", st.session_state.pure_text, "Scribe_Transcript.txt")
                col_srt.download_button("🎬 Subtitles (.srt)", st.session_state.srt_text, "Scribe_Transcript.srt")
                col_vtt.download_button("🌐 Web Subtitles (.vtt)", st.session_state.vtt_text, "Scribe_Transcript.vtt")
            
            with tab_ai:
                st.markdown("### 🧬 Advanced Offline Extraction")
                st.write("Using Neural Summarization + Bigram Phrase Extraction & Contextual Heuristics.")
                st.write("---")
                
                # NO EMOJIS IN THE BUTTON STRING
                if st.button("Extract Insights from Transcript", type="primary"):
                    with st.spinner("Running Advanced NLP Pipeline..."):
                        try:
                            tokenizer, summarizer = load_summarizer_model()
                            raw_transcript = " ".join([s['text'] for s in st.session_state.segments_data])
                            words = raw_transcript.split()
                            
                            if len(words) < 20:
                                st.session_state.ai_summary = "⚠️ Transcript too short for meaningful analysis."
                            else:
                                chunk_size = 400
                                text_chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
                                summaries = []
                                for chunk in text_chunks:
                                    if len(chunk.split()) > 10:
                                        inputs = tokenizer(chunk, max_length=1024, return_tensors="pt", truncation=True)
                                        summary_ids = summarizer.generate(inputs["input_ids"], num_beams=4, max_length=120, min_length=30, early_stopping=True)
                                        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
                                compiled_summary = " ".join(summaries)
                                
                                stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "is", "it", "you", "that", "this", "was", "for", "on", "as", "with", "so", "we", "they", "i", "are", "be", "have", "not", "do", "will", "can", "about", "what", "how", "just", "like", "know", "going", "think", "really", "would", "could", "should", "very", "much", "from", "at", "by", "there", "out"}
                                
                                clean_words = [w.lower().strip(".,!?\"'()[]") for w in words]
                                valid_words = [w for w in clean_words if w not in stop_words and len(w) > 2]
                                
                                bigrams = [f"{valid_words[i]} {valid_words[i+1]}" for i in range(len(valid_words)-1)]
                                bigram_counts = Counter(bigrams)
                                
                                top_phrases = [phrase for phrase, count in bigram_counts.most_common(5) if count > 1]
                                if not top_phrases:
                                    top_phrases = [word for word, count in Counter(valid_words).most_common(5)]

                                sentences = [s.strip() for s in re.split(r'[.!?]', raw_transcript) if len(s.strip()) > 15]
                                
                                action_pattern = re.compile(r'\b(i will|we will|we need to|i need to|action item|task is|must remember to|make sure to)\b', re.IGNORECASE)
                                decision_pattern = re.compile(r'\b(we decided|agreed that|finalized|conclusion is|going forward we|instead of)\b', re.IGNORECASE)
                                question_pattern = re.compile(r'\b(how do we|what if|should we|do we know)\b', re.IGNORECASE)
                                
                                extracted_actions, extracted_decisions, open_questions = [], [], []
                                
                                for sent in sentences:
                                    if action_pattern.search(sent) and len(extracted_actions) < 5:
                                        extracted_actions.append(sent.capitalize().strip() + ".")
                                    elif decision_pattern.search(sent) and len(extracted_decisions) < 4:
                                        extracted_decisions.append(sent.capitalize().strip() + ".")
                                    elif question_pattern.search(sent) and len(open_questions) < 3:
                                        open_questions.append(sent.capitalize().strip() + "?")

                                output = f"**🏷️ Key Topics & Tags:** `{', '.join([k.title() for k in top_phrases])}`\n\n---\n"
                                output += f"### 📌 Executive Summary\n{compiled_summary}\n\n"
                                
                                output += "### ✅ Action Items & Next Steps\n"
                                if extracted_actions:
                                    for a in extracted_actions: output += f"- {a}\n"
                                else: output += "- *No explicit action items detected.*\n"
                                
                                output += "\n### 💡 Key Decisions\n"
                                if extracted_decisions:
                                    for d in extracted_decisions: output += f"- {d}\n"
                                else: output += "- *No explicit strategic decisions flagged.*\n"
                                
                                if open_questions:
                                    output += "\n### ❓ Open Questions / Parking Lot\n"
                                    for q in open_questions: output += f"- {q}\n"
                                
                                st.session_state.ai_summary = output
                                
                        except Exception as hf_error:
                            st.error(f"Pipeline Error: {str(hf_error)}")

                if st.session_state.get('ai_summary'):
                    with st.container(border=True):
                        st.markdown(st.session_state.ai_summary)
                    st.download_button("📥 Download Report", st.session_state.ai_summary, "Scribe_Insights.md", "text/markdown")

# ==========================================
# JAVASCRIPT BRIDGE (Sliding Pill Animation & Anti-Jitter)
# ==========================================
if st.session_state.get('segments_data'):
    js_code = r"""
    <script>
        const parentWindow = window.parent;
        const parentDoc = parentWindow.document;
        
        if (parentWindow.scribeSyncInterval) {
            clearInterval(parentWindow.scribeSyncInterval);
        }
        
        parentWindow.scribeSyncInterval = setInterval(() => {
            
            // --- THE GHOST BUSTER ---
            const allMedia = parentDoc.querySelectorAll('video, audio');
            if (allMedia.length > 1) {
                for (let i = 0; i < allMedia.length - 1; i++) {
                    let ghost = allMedia[i];
                    ghost.pause(); ghost.removeAttribute('src'); 
                    ghost.load(); ghost.remove();
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
                            seg.style.setProperty('background-color', 'transparent', 'important');
                            seg.style.setProperty('color', '#CBD5E1', 'important');
                        } else if (regex.test(seg.innerText)) {
                            seg.style.opacity = '1'; 
                            seg.style.setProperty('background-color', '#FEF08A', 'important'); 
                            seg.style.setProperty('color', '#0F172A', 'important'); 
                        } else {
                            seg.style.opacity = '0.2'; 
                            seg.style.setProperty('background-color', 'transparent', 'important');
                        }
                    });
                });
                searchInput.dataset.searchAttached = 'true';
            }
            
            // --- C. THE SLIDING PILL TRACKER ---
            if (searchInput && searchInput.value.trim().length > 0) return;
            
            let pill = parentDoc.getElementById('sliding-tracker-pill');
            if (!pill) {
                transcriptBox.style.position = 'relative'; 
                pill = parentDoc.createElement('div');
                pill.id = 'sliding-tracker-pill';
                pill.style.position = 'absolute';
                
                pill.style.transition = 'top 0.3s ease, left 0.3s ease, width 0.3s ease, height 0.3s ease, opacity 0.3s ease';
                pill.style.backgroundColor = 'rgba(56, 189, 248, 0.15)'; 
                pill.style.borderLeft = '3px solid #38BDF8'; 
                pill.style.borderRadius = '4px';
                pill.style.pointerEvents = 'none'; 
                pill.style.opacity = '0';
                pill.style.zIndex = '0';
                transcriptBox.appendChild(pill);
            }
            
            const time = media.currentTime;
            const segs = transcriptBox.querySelectorAll('.transcript-segment');
            let isAudioActive = false;
            
            segs.forEach(seg => {
                const start = parseFloat(seg.getAttribute('data-start'));
                const end = parseFloat(seg.getAttribute('data-end'));
                
                seg.style.position = 'relative';
                seg.style.zIndex = '1';
                
                if (time >= start && time <= end) {
                    isAudioActive = true;
                    if (seg.dataset.active !== 'true') { 
                        seg.dataset.active = 'true';
                        
                        pill.style.opacity = '1';
                        pill.style.top = seg.offsetTop + 'px';
                        pill.style.left = seg.offsetLeft + 'px';
                        pill.style.width = (seg.offsetWidth + 10) + 'px'; 
                        pill.style.height = seg.offsetHeight + 'px';
                        
                        seg.style.setProperty('color', '#38BDF8', 'important');            
                        
                        const boxHeight = transcriptBox.clientHeight;
                        const scrollPos = transcriptBox.scrollTop;
                        const elementTop = seg.offsetTop;
                        
                        if (elementTop < scrollPos + 40 || elementTop > scrollPos + boxHeight - 40) {
                            transcriptBox.scrollTo({
                                top: elementTop - (boxHeight / 2),
                                behavior: 'smooth'
                            });
                        }
                    }
                } else {
                    if (seg.dataset.active === 'true') {
                        seg.dataset.active = 'false';
                        
                        seg.style.setProperty('background-color', 'transparent', 'important');
                        seg.style.setProperty('color', '#CBD5E1', 'important');
                        seg.style.setProperty('font-weight', 'normal', 'important');
                    }
                }
            });
            
            if (!isAudioActive) {
                pill.style.opacity = '0';
            }
            
        }, 500); 
    </script>
    """
    components.html(js_code, height=0)
