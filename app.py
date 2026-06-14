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



# --- Helper Functions for Subtitles ---

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



# 3. Backend Model Loading

@st.cache_resource

def load_model():

    return WhisperModel("base", device="cpu", compute_type="int8")



model = load_model()



# 4. App Header

st.markdown("<h1 class='app-title'>🎙️ Scribe AI Pro Workspace</h1>", unsafe_allow_html=True)

st.markdown("---")



# 5. Dual-Pane Layout

col_controls, col_output = st.columns([1, 2], gap="large")



# --- LEFT PANEL: MEDIA & ANALYTICS ---

with col_controls:

    with st.container(height=700, border=True):

        st.markdown("### 📥 Input Media")

        SUPPORTED_FORMATS = ["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"]

        uploaded_file = st.file_uploader("Upload audio or video", type=SUPPORTED_FORMATS, label_visibility="collapsed")

        

        tmp_media_path = None

        tmp_audio_path = None

        

        # Initialize session states

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

            

            # --- FEATURE 3: VAD Analytics Dashboard ---

            if st.session_state.analytics:

                st.markdown("### 📊 Audio Analytics")

                m1, m2, m3 = st.columns(3)

                m1.metric("Language", st.session_state.analytics['lang'])

                m2.metric("Confidence", st.session_state.analytics['conf'])

                m3.metric("Duration", st.session_state.analytics['dur'])

            

            st.markdown("---")

            

            if st.button("🚀 Start Transcription"):

                

                # --- NEW: Pure CSS Animated Sound Wave Loader ---

                loader_html = """

                <style>

                    .wave-container { display: flex; justify-content: center; align-items: center; height: 60px; gap: 8px; margin-bottom: 10px;}

                    .wave-bar { width: 8px; height: 16px; background: #3B82F6; border-radius: 8px; animation: pulse 1.2s infinite ease-in-out; }

                    .wave-bar:nth-child(1) { animation-delay: -1.2s; }

                    .wave-bar:nth-child(2) { animation-delay: -1.1s; }

                    .wave-bar:nth-child(3) { animation-delay: -1.0s; }

                    .wave-bar:nth-child(4) { animation-delay: -0.9s; }

                    .wave-bar:nth-child(5) { animation-delay: -0.8s; }

                    @keyframes pulse {

                        0%, 40%, 100% { transform: scaleY(1); background: #93C5FD; box-shadow: none; }

                        20% { transform: scaleY(2.5); background: #2563EB; box-shadow: 0 0 12px rgba(37,99,235,0.6); }

                    }

                </style>

                <div class="wave-container">

                    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>

                </div>

                """

                

                try:

                    # Create a dedicated empty space for the animation

                    visual_loader = st.empty()

                    visual_loader.markdown(loader_html, unsafe_allow_html=True)

                    

                    # Motivational Quotes Array

                    motivational_quotes = [

                        "\"Great things are not done by impulse, but by a series of small things brought together.\" — Vincent Van Gogh",

                        "\"The only way to do great work is to love what you do.\" — Steve Jobs",

                        "\"Words are, of course, the most powerful drug used by mankind.\" — Rudyard Kipling",

                        "Extracting the signal from the noise...",

                        "Translating acoustic waves into meaning..."

                    

    "\"The secret of getting ahead is getting started.\" — Mark Twain",

    "\"Well begun is half done.\" — Aristotle",

    "\"Dream big and dare to fail.\" — Norman Vaughan",

    "\"What you do today can improve all your tomorrows.\" — Ralph Marston",

    "\"Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away.\" — Antoine de Saint-Exupéry",

    "\"The best preparation for tomorrow is doing your best today.\" — H. Jackson Brown Jr.",

    "\"Creativity is intelligence having fun.\" — Albert Einstein",

    "\"Make it simple, but significant.\" — Don Draper",

    "\"The harder I work, the luckier I get.\" — Samuel Goldwyn",

    "\"Focus on being productive instead of busy.\" — Tim Ferriss",

    "\"Every strike brings me closer to the next home run.\" — Babe Ruth",

    "\"Discipline is choosing between what you want now and what you want most.\" — Abraham Lincoln",

    "\"You miss 100% of the shots you don't take.\" — Wayne Gretzky",

    "\"If opportunity doesn't knock, build a door.\" — Milton Berle",

    "\"The expert in anything was once a beginner.\" — Helen Hayes",



    "\"Initializing auditory perception modules...\"",

    "\"Listening for patterns in the spectrum...\"",

    "\"Converting spoken language into structured data...\"",

    "\"Mapping frequencies to meaning...\"",

    "\"Synchronizing with incoming audio streams...\"",

    "\"Detecting linguistic signals...\"",

    "\"Aligning phonemes with context...\"",

    "\"Parsing the soundscape...\"",

    "\"Resolving ambiguity in real time...\"",

    "\"Building understanding from waveforms...\"",

    "\"Following the rhythm of human speech...\"",

    "\"Tracking intent across every syllable...\"",

    "\"Transforming audio into knowledge...\"",

    "\"Recognizing voices, extracting meaning...\"",

    "\"Calibrating language understanding systems...\"",

    "\"Reading between the pauses...\"",

    "\"Listening for what matters most...\"",

    "\"Converting noise into narrative...\"",

    "\"Analyzing speech with precision...\"",

    "\"Turning sound into searchable insight...\"",



    "\"Mining meaning from every decibel...\"",

    "\"Distilling intelligence from audio streams...\"",

    "\"Understanding begins with listening...\"",

    "\"The signal is there. Finding it...\"",

    "\"Interpreting the language hidden in sound...\"",

    "\"Every waveform tells a story...\"",

    "\"Capturing context, one word at a time...\"",

    "\"Listening deeply. Understanding accurately...\"",

    "\"Discovering intent beneath the acoustics...\"",

    "\"Making sense of the spoken world...\""

]

                    

                    

                    with st.status("Engine Active...", expanded=True) as status:

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

                        

                        # Populate Analytics Data

                        st.session_state.analytics = {

                            "lang": info.language.upper(),

                            "conf": f"{info.language_probability * 100:.1f}%",

                            "dur": f"{int(total_duration // 60)}m {int(total_duration % 60)}s"

                        }

                        

                        st.session_state.segments_data = []

                        pure_lines = []

                        srt_lines = []

                        vtt_lines = ["WEBVTT\n"]

                        

                        # Initialize Quote Engine

                        last_quote_time = time.time()

                        current_quote = random.choice(motivational_quotes)

                        status.update(label=f"⏳ {current_quote}")

                        

                        for i, segment in enumerate(segments, start=1):

                            

                            # Rotate quote every 4 seconds

                            if time.time() - last_quote_time > 4:

                                current_quote = random.choice(motivational_quotes)

                                status.update(label=f"⏳ {current_quote}")

                                last_quote_time = time.time()

                            

                            # JSON Data

                            start_min, start_sec = divmod(int(segment.start), 60)

                            end_min, end_sec = divmod(int(segment.end), 60)

                            display_time = f"[{start_min:02d}:{start_sec:02d} -> {end_min:02d}:{end_sec:02d}]"

                            

                            st.session_state.segments_data.append({

                                "start": segment.start,

                                "end": segment.end,

                                "text": segment.text.strip(),

                                "display_time": display_time

                            })

                            

                            # TXT

                            pure_lines.append(f"{display_time} {segment.text.strip()}")

                            # SRT

                            srt_lines.append(f"{i}\n{to_srt_time(segment.start)} --> {to_srt_time(segment.end)}\n{segment.text.strip()}\n")

                            # VTT

                            vtt_lines.append(f"{to_vtt_time(segment.start)} --> {to_vtt_time(segment.end)}\n{segment.text.strip()}\n")

                        

                        st.session_state.pure_text = "\n".join(pure_lines)

                        st.session_state.srt_text = "\n".join(srt_lines)

                        st.session_state.vtt_text = "\n".join(vtt_lines)

                        

                        status.update(label="Transcription Complete!", state="complete", expanded=False)

                        

                        # Destroy the animation perfectly when finished

                        visual_loader.empty()

                        

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

    with st.container(height=700, border=True):

        

        if not st.session_state.segments_data:

            st.markdown("### 📄 Real-Time Transcript")

            st.info("👈 Upload a file and click 'Start Transcription' to begin.")

        else:

            # --- FEATURE 2: Search Bar UI ---

            st.markdown("<input type='text' id='search-input' placeholder='🔍 Search keywords in transcript...' style='width: 100%; padding: 12px; margin-bottom: 10px; border-radius: 8px; border: 2px solid #E2E8F0; font-family: sans-serif;'>", unsafe_allow_html=True)

            

            # Interactive HTML Canvas

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

            for i, seg in enumerate(st.session_state.segments_data):

                html_content += (

                    f"<span class='transcript-segment' id='seg-{i}' "

                    f"data-start='{seg['start']}' data-end='{seg['end']}' "

                    f"style='padding: 2px 4px; border-radius: 4px; transition: all 0.2s ease; display: inline-block;'>"

                    f"<strong>{seg['display_time']}</strong> {seg['text']}"

                    f"</span><br>"

                )

            html_content += "</div>"

            st.markdown(html_content, unsafe_allow_html=True)

            

            # --- The Upgraded JavaScript Bridge ---

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



                        // FEATURE 2a: Click-to-Seek

                        segments.forEach(seg => {

                            seg.style.cursor = 'pointer';

                            seg.title = "Click to jump to this moment";

                            seg.addEventListener('click', () => {

                                const start = parseFloat(seg.getAttribute('data-start'));

                                mediaElement.currentTime = start;

                                mediaElement.play();

                            });

                        });



                        // FEATURE 2b: Real-Time Keyword Search

                        if (searchInput) {

                            searchInput.addEventListener('input', (e) => {

                                const term = e.target.value.toLowerCase();

                                segments.forEach(seg => {

                                    if (seg.innerText.toLowerCase().includes(term)) {

                                        seg.style.opacity = '1';

                                        if (term.length > 1) {

                                            seg.style.backgroundColor = '#FEF08A'; // Yellow highlight

                                        } else {

                                            seg.style.backgroundColor = 'transparent'; // Reset if empty

                                        }

                                    } else {

                                        seg.style.opacity = '0.2'; // Dim out non-matches

                                        seg.style.backgroundColor = 'transparent';

                                    }

                                });

                            });

                        }



                        // FEATURE 2c: Auto-Scroll

                        mediaElement.addEventListener('timeupdate', () => {

                            const currentTime = mediaElement.currentTime;

                            // Suspend auto-scroll if user is currently searching

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

            

            # --- FEATURE 1: Professional Subtitle Export UI ---

            st.markdown("**Export Options:**")

            col_txt, col_srt, col_vtt = st.columns(3)

            

            with col_txt:

                st.download_button("📥 Basic (TXT)", st.session_state.pure_text, "Scribe.txt", "text/plain")

            with col_srt:

                st.download_button("🎬 Premiere (SRT)", st.session_state.srt_text, "Scribe.srt", "text/plain")

            with col_vtt:

                st.download_button("🌐 Web (VTT)", st.session_state.vtt_text, "Scribe.vtt", "text/plain")
