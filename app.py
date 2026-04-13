import streamlit as st
from faster_whisper import WhisperModel
from moviepy import VideoFileClip
import os
import tempfile
import subprocess
import time
import re

# Page Configuration
st.set_page_config(page_title="ReelTranscribe AI", page_icon="🎬", layout="centered")

st.title("🎬 ReelTranscribe AI")
st.info("Upload a Reel or Video to generate a full transcription and burned subtitles.")

# --- Models ---
@st.cache_resource(show_spinner="Loading Model Weights into Memory (This may take a minute on first run)...")
def load_whisper_model(size, device_choice):
    try:
        return WhisperModel(size, device=device_choice, compute_type="int8")
    except Exception:
        pass
        
    # Final fallback if int8 fails on unsupported architectures
    return WhisperModel(size, device=device_choice, compute_type="default")

st.sidebar.markdown("### Hardware")
hardware = st.sidebar.radio("Hardware Acceleration", ["CPU (Compatible)", "GPU (Requires CUDA)"], index=0)
device_val = "cuda" if "GPU" in hardware else "cpu"

model_size = st.sidebar.selectbox("Select Model Size", ["tiny", "base", "small", "medium"], index=1)
model = load_whisper_model(model_size, device_val)

# --- Subtitle UI ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Transcription Options")
language_selection = st.sidebar.selectbox("Video Language", ["English (Fastest)", "Auto-Detect", "Spanish", "French", "German"])
lang_map = {"English (Fastest)": "en", "Auto-Detect": None, "Spanish": "es", "French": "fr", "German": "de"}
selected_language_code = lang_map[language_selection]

st.sidebar.markdown("### Subtitle Options")
burn_subtitles = st.sidebar.checkbox("Burn Subtitles into Video", value=True)
font_choice = st.sidebar.selectbox("Font", ["Times New Roman", "Arial", "Courier New", "Verdana", "Impact"])
placement = st.sidebar.selectbox("Placement", ["Bottom", "Center", "Top"])

# Map placement to FFMpeg ASS alignment
align_map = {"Bottom": 2, "Center": 5, "Top": 8}
align_val = align_map[placement]

def format_timestamp(seconds: float):
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3600000
    milliseconds -= hours * 3600000
    minutes = milliseconds // 60000
    milliseconds -= minutes * 60000
    sec = milliseconds // 1000
    milliseconds -= sec * 1000
    return f"{hours:02d}:{minutes:02d}:{sec:02d},{milliseconds:03d}"

# --- Main App ---
uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    st.video(uploaded_file)
    
    if st.button("🚀 Start Processing"):
        # UI Elements for progress tracking
        status_bar = st.empty()
        progress_bar = st.progress(0.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_video_path = os.path.join(temp_dir, "input_video.mp4")
            temp_audio_path = os.path.join(temp_dir, "audio.mp3")
            temp_srt_path = os.path.join(temp_dir, "subs.srt")
            out_video_path = os.path.join(temp_dir, "output_video.mp4")

            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.read())

            try:
                # 1. Audio extraction
                status_bar.info("⬇️ Extracting audio stream...")
                progress_bar.progress(0.0)
                
                video_clip = VideoFileClip(temp_video_path)
                video_duration = video_clip.duration  # Get duration for FFMPEG progress
                video_clip.audio.write_audiofile(temp_audio_path, logger=None)
                video_clip.close()

                # 2. Transcribe with VAD Filtering & INT8
                st.toast("Starting Heavy Processing", icon="⏳")
                status_bar.info(f"🧠 Waking up AI (Running VAD & Language Detection)... Please wait.")
                progress_bar.progress(0.0)
                
                # Faster-whisper blocks here while it parses the initial audio for language and silences
                # This can take 10-30 seconds on CPU.
                # If auto-detect is off, we skip the heavy language-ID processing phase entirely
                segments, info = model.transcribe(
                    temp_audio_path,
                    language=selected_language_code,
                    vad_filter=True, 
                    vad_parameters=dict(min_silence_duration_ms=500),
                    beam_size=1 
                )
                
                total_duration = info.duration
                start_time = time.time()
                srt_content = ""
                full_text = ""
                
                for i, segment in enumerate(segments, start=1):
                    # -- Progress Update --
                    current_time = segment.end
                    progress = min(current_time / total_duration, 1.0)
                    progress_bar.progress(progress)
                    
                    elapsed = time.time() - start_time
                    eta = (elapsed / progress) - elapsed if progress > 0 else 0
                    status_bar.info(f"🗣️ Transcribing audio (Removing silence via VAD)... | ETA: {int(eta)}s")
                    
                    # -- Text Build --
                    start_ts = format_timestamp(segment.start)
                    end_ts = format_timestamp(segment.end)
                    text = segment.text.strip()
                    full_text += text + " "
                    srt_content += f"{i}\n{start_ts} --> {end_ts}\n{text}\n\n"
                
                progress_bar.progress(1.0)
                status_bar.success("✅ Transcription Complete!")
                
                st.subheader("Transcript Output:")
                st.text_area("Copy Text", full_text, height=200)
                
                st.download_button("📥 Download .txt", full_text, "transcript.txt", "text/plain")

                # 3. Burn Subtitles
                if burn_subtitles:
                    status_bar.info("⚙️ Initialising Subtitle Engine...")
                    progress_bar.progress(0.0)
                    
                    with open(temp_srt_path, "w", encoding="utf-8") as f:
                        f.write(srt_content)

                    sub_filter = f"subtitles=subs.srt:force_style='Fontname={font_choice},Alignment={align_val},FontSize=24,MarginV=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000'"
                    
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", "input_video.mp4",
                        "-vf", sub_filter,
                        "-c:a", "copy",
                        "output_video.mp4"
                    ]

                    start_time = time.time()
                    
                    process = subprocess.Popen(
                        cmd, 
                        cwd=temp_dir, 
                        stderr=subprocess.PIPE, 
                        universal_newlines=True,
                        encoding="utf-8",
                        errors="replace"
                    )
                    
                    time_regex = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")
                    
                    for line in process.stderr:
                        match = time_regex.search(line)
                        if match:
                            h, m, s = match.groups()
                            current_sec = int(h) * 3600 + int(m) * 60 + float(s)
                            progress = min(current_sec / max(video_duration, 1.0), 1.0)
                            progress_bar.progress(progress)
                            
                            elapsed = time.time() - start_time
                            eta = (elapsed / progress) - elapsed if progress > 0 else 0
                            status_bar.info(f"🔥 Burning Subtitles... {progress:.0%} | ETA: {int(eta)}s")
                    
                    process.wait()
                    
                    if process.returncode != 0:
                        st.error("❌ FFMpeg Subtitle Render Error")
                    else:
                        progress_bar.progress(1.0)
                        status_bar.success("✅ Subtitles Burned Successfully!")
                        
                        st.balloons()
                        
                        with open(out_video_path, "rb") as f:
                            video_bytes = f.read()
                        
                        st.download_button(
                            label="📥 Download Subtitled Video",
                            data=video_bytes,
                            file_name="video_with_subtitles.mp4",
                            mime="video/mp4"
                        )

            except Exception as e:
                status_bar.error("❌ Processing Halted.")
                progress_bar.empty()
                st.error(f"Application Error: {e}")
