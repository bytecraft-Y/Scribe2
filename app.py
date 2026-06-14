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
    # INT8 quantization makes the model run up to 4x faster on CPU
    return WhisperModel("tiny", device="cpu", compute_type="int8")

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
            
        # Phase 2: High-Speed Transcription with Timestamps
        with st.spinner("Generating transcript at high speed..."):
            # beam_size=5 balances speed and accuracy
            segments, info = model.transcribe(tmp_audio_path, beam_size=5)
            
            # --- The new timestamp formatting block ---
            transcript_lines = []
            for segment in segments:
                start_time = f"{segment.start:.2f}s"
                end_time = f"{segment.end:.2f}s"
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
