🎬 ReelTranscribe AI
An automated video-to-text transcription tool optimized for social media content creators. This application leverages OpenAI Whisper for high-accuracy speech recognition and Streamlit for a seamless, browser-based user experience.

🛠 Project Overview
ReelTranscribe AI allows users to upload video files (MP4, MOV, etc.), extracts the audio stream, and processes it through a neural network to provide a full, time-stamped (optional) transcription.

⚙️ Setup & Installation
1. System Requirements
You must have FFmpeg installed on your system. This is the engine that handles the video-to-audio conversion.

macOS: brew install ffmpeg

Windows: choco install ffmpeg (or download from ffmpeg.org)

Linux: sudo apt install ffmpeg

2. Python Environment
Install the required Python libraries using pip:
pip install streamlit openai-whisper moviepy


📄 The Code (app.py)
import streamlit as st
import whisper
import moviepy.editor as mp
import os
import tempfile

# Page Configuration
st.set_page_config(page_title="ReelTranscribe AI", page_icon="🎬", layout="centered")

st.title("🎬 ReelTranscribe AI")
st.info("Upload a Reel or Video to generate a full transcription.")

# 1. Model Loading (Cached for performance)
@st.cache_resource
def load_whisper_model(size):
    return whisper.load_model(size)

model_size = st.sidebar.selectbox("Select Model Size", ["tiny", "base", "small", "medium"])
model = load_whisper_model(model_size)

# 2. File Upload Interface
uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    st.video(uploaded_file)
    
    if st.button("🚀 Start Transcription"):
        with st.spinner(f"AI is listening using the '{model_size}' model..."):
            # Create a temporary file to store the upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_file.read())
                temp_video_path = tfile.name

            try:
                # Audio Extraction
                video_clip = mp.VideoFileClip(temp_video_path)
                temp_audio_path = "temp_audio_stream.mp3"
                video_clip.audio.write_audiofile(temp_audio_path, logger=None)
                
                # Inference
                result = model.transcribe(temp_audio_path)
                
                # Display Results
                st.success("✅ Transcription Complete!")
                st.subheader("Transcript Output:")
                st.text_area("Copy Text", result["text"], height=300)
                
                st.download_button(
                    label="📥 Download .txt",
                    data=result["text"],
                    file_name="transcript.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Error: {e}")
            
            finally:
                # Cleanup temporary files
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                os.remove(temp_video_path)

📊 Expected Outputs
Web Interface
Upon running streamlit run app.py, you will see:

Sidebar: Model selection (Base is the sweet spot for speed/accuracy).

Upload Box: A drag-and-drop area for your .mp4 or .mov files.

Video Player: A native preview of your uploaded content.

Transcription Box: A text area containing the full dialogue from the video.

Transcription Format
"Hey guys! In today's video, I'm going to show you how to build a web app using Python in under five minutes. Don't forget to like and subscribe for more content like this!"

🗺 Roadmap
[ ] Word-Level Highlighting: Modify transcription logic to output JSON with per-word timestamps for "Reel-style" captions.

[ ] Translation Layer: Use the task="translate" parameter in Whisper to automatically translate foreign language videos into English text.

[ ] Direct YouTube Link Support: Integrate yt-dlp to allow transcription via URL instead of just file uploads.

[ ] SRT Export: Automatically format the transcription into a .srt file for direct import into Premiere Pro or CapCut.

[ ] Audio Visualization: Add a waveform visualizer while the audio is being processed.

📜 License
MIT License - Created for personal and educational use.

