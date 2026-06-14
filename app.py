import streamlit as st
from faster_whisper import WhisperModel
import moviepy.editor as mp
import os
import tempfile
import gc

st.set_page_config(page_title="Lightning Media Transcriber", layout="wide")
st.title("⚡ High-Speed Transcription Assistant")
st.write("Powered by faster-whisper and INT8 CPU quantization.")

# 1. Load the Optimized CTranslate2 Model
@st.cache_resource
def load_model():
    # Upgraded from "tiny" to "base" for higher accuracy. 
    # (If the server crashes, downgrade back to "tiny" or try "small" if running locally).
    return WhisperModel("base", device="cpu", compute_type="int8")
model = load_model()

# 2. File Upload Handling
SUPPORTED_FORMATS = ["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"]
uploaded_file = st.file_uploader("Select Media File", type=SUPPORTED_FORMATS)

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    tmp_media = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    tmp_media.write(uploaded_file.read())
    tmp_media_path = tmp_media.name
    tmp_media.close() 

    st.info("File uploaded successfully. Processing media...")
    tmp_audio_path = "temp_audio_processing.wav"

    try:
        # Phase 1: Fast Audio Extraction
        with st.spinner("Extracting audio track..."):
            clip = mp.AudioFileClip(tmp_media_path)
            # Reduce sampling rate directly during extraction to save processing time later
            clip.write_audiofile(tmp_audio_path, fps=16000, logger=None)
            clip.close()
            
        # Phase 2: High-Speed Transcription with Enhanced Accuracy
        with st.spinner("Generating highly accurate transcript..."):
            
            # Increased beam_size to 7 and enabled vad_filter
            segments, info = model.transcribe(
                tmp_audio_path, 
                beam_size=7, 
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # --- The MM:SS timestamp formatting block ---
            transcript_lines = []
            # ... (keep the rest of your timestamp loop exactly the same)
            for segment in segments:
                # Convert raw float seconds to integers and extract minutes/seconds
                start_min, start_sec = divmod(int(segment.start), 60)
                end_min, end_sec = divmod(int(segment.end), 60)
                
                # Format as padded strings (e.g., 02:05 instead of 2:5)
                start_time = f"{start_min:02d}:{start_sec:02d}"
                end_time = f"{end_min:02d}:{end_sec:02d}"
                
                transcript_lines.append(f"[{start_time} -> {end_time}] {segment.text.strip()}")
            
            transcript_text = "\n".join(transcript_lines)
            
            st.success("Transcription Complete!")
            st.text_area("Transcript", transcript_text, height=300)
            
            st.download_button(
                label="📥 Download Transcript as TXT",
                data=transcript_text,
                file_name="transcript.txt",
                mime="text/plain"
            )
            
    except Exception as e:
        st.error(f"A processing error occurred: {str(e)}")
        
    finally:
        # 3. Aggressive Server Cleanup
        if os.path.exists(tmp_media_path):
            os.remove(tmp_media_path)
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        gc.collect()
