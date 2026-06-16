import streamlit as st
import streamlit.components.v1 as components
import os
import tempfile
import gc
import time
import re
import math
from collections import Counter
from faster_whisper import WhisperModel

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Scribe AI | Enterprise", page_icon="🎙️", layout="wide")

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "segments_data" not in st.session_state:
    st.session_state.segments_data = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = ""

# ==========================================
# MODEL LOADING (Cached for speed)
# ==========================================
@st.cache_resource
def load_whisper_model():
    # Using int8 quantization for efficient CPU inference
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_whisper_model()

# ==========================================
# SIDEBAR: UPLOAD & PROCESSING
# ==========================================
with st.sidebar:
    st.markdown("### 📁 Media Upload")
    uploaded_file = st.file_uploader("Upload Audio/Video", type=['mp3', 'wav', 'mp4', 'mkv'])
    
    if uploaded_file:
        if st.button("▶️ Process Media", type="primary"):
            st.session_state.uploaded_file = uploaded_file
            st.session_state.ai_summary = "" # Reset summary on new upload
            
            with st.spinner("Transcribing via Edge-AI (Faster-Whisper)..."):
                try:
                    # Save uploaded file to temporary storage
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    # Execute Faster-Whisper Inference
                    segments, info = model.transcribe(tmp_path, beam_size=7, vad_filter=True)
                    
                    # Store segments in session state
                    parsed_segments = []
                    for segment in segments:
                        parsed_segments.append({
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text
                        })
                    
                    st.session_state.segments_data = parsed_segments
                    
                    # Cleanup memory and temp files
                    os.remove(tmp_path)
                    gc.collect()
                    st.success("Transcription Complete!")
                    
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

# ==========================================
# MAIN INTERFACE
# ==========================================
st.title("🎙️ Scribe AI Workspace")

# 1. GLOBAL MEDIA PLAYER (The Fix for "Double Audio")
# Rendering this ABOVE the tabs prevents Streamlit from cloning it during reruns
if st.session_state.uploaded_file is not None:
    st.audio(st.session_state.uploaded_file)

# 2. WORKSPACE TABS
if st.session_state.segments_data:
    tab_transcript, tab_ai = st.tabs(["📝 Interactive Transcript", "🧠 AI Insights"])
    
    # --- TAB 1: INTERACTIVE TRANSCRIPT ---
    with tab_transcript:
        st.markdown("""
            <style>
                .transcript-segment {
                    padding: 8px; border-radius: 5px; cursor: pointer; transition: all 0.2s ease;
                    display: inline; line-height: 1.8; font-size: 1.1rem;
                }
                .transcript-segment:hover { background-color: #E2E8F0; }
                .search-box { width: 100%; padding: 10px; margin-bottom: 20px; border-radius: 8px; border: 1px solid #CBD5E1; }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("<input type='text' id='search-input' class='search-box' placeholder='🔍 Search transcript using Regex boundaries...'>", unsafe_allow_html=True)
        
        # Build the HTML Transcript
        html_content = "<div id='transcript-box' style='padding: 20px; background: #F8FAFC; border-radius: 10px; max-height: 500px; overflow-y: auto;'>"
        for seg in st.session_state.segments_data:
            start = seg['start']
            end = seg['end']
            text = seg['text']
            html_content += f"<span class='transcript-segment' data-start='{start}' data-end='{end}'>{text} </span>"
        html_content += "</div>"
        
        st.markdown(html_content, unsafe_allow_html=True)

    # --- TAB 2: AI INSIGHTS (TF-IDF & Regex) ---
    with tab_ai:
        st.markdown("### 🧠 Edge-AI Semantic Analyzer (TF-IDF)")
        st.write("This offline engine utilizes Term Frequency-Inverse Document Frequency and Regex heuristic parsing.")
        
        if st.button("✨ Execute Local NLP Pipeline", type="primary"):
            with st.spinner("Calculating TF-IDF matrices & extracting vectors..."):
                time.sleep(0.5) 
                
                raw_text = " ".join([s['text'] for s in st.session_state.segments_data])
                sentences = [s.strip() for s in re.split(r'[.!?]', raw_text) if len(s.strip()) > 20]
                
                if not sentences:
                    st.session_state.ai_summary = "⚠️ Insufficient transcript data to analyze. Please provide a longer audio file."
                else:
                    stop_words = set(["the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "is", "it", "you", "that", "this", "was", "for", "on", "as", "with", "so", "we", "they", "i", "are", "be", "have", "will"])
                    
                    # Calculate Document Frequency (DF)
                    df = Counter()
                    sent_tokens = []
                    for sent in sentences:
                        tokens = [w.lower() for w in re.findall(r'\b\w+\b', sent) if w.lower() not in stop_words]
                        sent_tokens.append(tokens)
                        for token in set(tokens):
                            df[token] += 1
                    
                    # Calculate TF-IDF Score
                    N = len(sentences)
                    sent_scores = []
                    global_tf_idf = Counter()
                    
                    for i, tokens in enumerate(sent_tokens):
                        score = 0
                        tf = Counter(tokens)
                        if len(tokens) > 0:
                            for token, count in tf.items():
                                idf = math.log(N / (1 + df[token]))
                                tf_idf_val = (count / len(tokens)) * idf
                                score += tf_idf_val
                                global_tf_idf[token] += tf_idf_val
                        sent_scores.append((score, i, sentences[i]))
                    
                    # Extract & Sort Chronologically
                    top_scored_sents = sorted(sent_scores, key=lambda x: x[0], reverse=True)[:3]
                    chronological_summary = sorted(top_scored_sents, key=lambda x: x[1])
                    top_keywords = [word for word, score in global_tf_idf.most_common(5)]
                    
                    # Regex Heuristic Parsing
                    action_pattern = re.compile(r'\b(need to|have to|should|must|task|assign|todo|action item|fix|update)\b', re.IGNORECASE)
                    decision_pattern = re.compile(r'\b(decided|agreed|concluded|choose|settled|resolved|instead of)\b', re.IGNORECASE)
                    
                    extracted_actions = []
                    extracted_decisions = []
                    
                    for sent in sentences:
                        if action_pattern.search(sent) and sent not in extracted_actions and len(extracted_actions) < 4:
                            extracted_actions.append(sent)
                        if decision_pattern.search(sent) and sent not in extracted_decisions and len(extracted_decisions) < 3:
                            extracted_decisions.append(sent)
                    
                    # Compile Report
                    output = f"**🏷️ Auto-Generated Tags:** {', '.join([k.capitalize() for k in top_keywords])}\n\n---\n"
                    output += "### 📌 Executive Summary\n"
                    for _, _, s in chronological_summary:
                        output += f"* {s.capitalize()}.\n"
                        
                    output += "\n### ✅ Extracted Action Items\n"
                    if extracted_actions:
                        for a in extracted_actions:
                            output += f"* {a.capitalize()}.\n"
                    else:
                        output += "* No explicit action items or tasks detected.\n"
                        
                    output += "\n### 💡 Key Decisions & Core Insights\n"
                    if extracted_decisions:
                        for d in extracted_decisions:
                            output += f"* {d.capitalize()}.\n"
                    else:
                        output += "* No explicit corporate decisions flagged.\n"
                        
                    st.session_state.ai_summary = output

        # Render AI Output Card
        if st.session_state.ai_summary:
            st.markdown("<div style='background: #F8FAFC; padding: 25px; border-radius: 8px; border-left: 5px solid #06B6D4; color: #0F172A; margin-top: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>", unsafe_allow_html=True)
            st.markdown(st.session_state.ai_summary)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button("📥 Download Analysis Brief", st.session_state.ai_summary, "Local_Analysis_Brief.md", "text/markdown")


# ==========================================
# JAVASCRIPT BRIDGE (Final Boss Edition)
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
            // --- THE GHOST BUSTER FAILSAFE ---
            // Finds all media tags and ruthlessly destroys duplicates caused by Streamlit reruns
            const allMedia = parentDoc.querySelectorAll('video, audio');
            if (allMedia.length > 1) {
                allMedia.forEach((m, index) => {
                    if (index !== 0) { 
                        m.pause(); m.removeAttribute('src'); m.remove(); 
                    }
                });
            }
            
            const media = allMedia[0];
            const searchInput = parentDoc.getElementById('search-input');
            const transcriptBox = parentDoc.getElementById('transcript-box');
            
            // Abort if the transcript tab is hidden
            if (!media || !transcriptBox || transcriptBox.offsetHeight === 0) return;
            
            // --- CLICK-TO-SEEK ---
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
            
            // --- SEARCH ENGINE ---
            if (searchInput && !searchInput.dataset.searchAttached) {
                searchInput.addEventListener('input', (e) => {
                    const term = e.target.value.trim();
                    const escapedTerm = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                    const regex = term ? new RegExp('\\b' + escapedTerm + '\\b', 'i') : null;
                    const segs = transcriptBox.querySelectorAll('.transcript-segment');
                    
                    segs.forEach(seg => {
                        if (!term) {
                            seg.style.opacity = '1'; seg.style.backgroundColor = 'transparent';
                        } else if (regex.test(seg.innerText)) {
                            seg.style.opacity = '1'; seg.style.backgroundColor = '#FEF08A';
                        } else {
                            seg.style.opacity = '0.2'; seg.style.backgroundColor = 'transparent';
                        }
                    });
                });
                searchInput.dataset.searchAttached = 'true';
            }
            
            // --- REAL-TIME TRACKER ---
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
