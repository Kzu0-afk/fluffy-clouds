import streamlit as st
from faster_whisper import WhisperModel
from moviepy import VideoFileClip
import os
import tempfile
import subprocess
import time
import re

# Page Configuration
st.set_page_config(page_title="Fluffy Cloudz ver. 1.0.0", page_icon="☁️", layout="centered")

st.title("☁️ Fluffy Cloudz ver. 1.0.0")
st.info("Upload a Reel or Video to generate a full transcription and burned subtitles.")

# --- Models ---
@st.cache_resource(show_spinner="Loading Model Weights into Memory (This may take a minute on first run)...")
def load_whisper_model(size, device_choice):
    # Smart quantization: GPU gets float16 (accurate), CPU gets int8 (fast)
    compute = "float16" if device_choice == "cuda" else "int8"
    try:
        return WhisperModel(size, device=device_choice, compute_type=compute)
    except Exception:
        # Fallback: force CPU + int8 if the chosen device/precision fails
        return WhisperModel(size, device="cpu", compute_type="int8")

st.sidebar.markdown("### Hardware")
hardware = st.sidebar.radio("Hardware Acceleration", ["CPU (Compatible)", "GPU (Requires CUDA)"], index=0)
device_val = "cuda" if "GPU" in hardware else "cpu"

model_size = st.sidebar.selectbox("Select Model Size", ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"], index=2)
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

def build_srt_from_segments(segments_list):
    """Build SRT content and plain text from transcribed segments.
    
    Uses word-level timestamps when available to produce tightly-timed
    subtitle chunks (≤10 words each, split on natural pauses). Falls
    back to segment-level timing when word data is unavailable.
    """
    srt_index = 1
    srt_parts = []
    text_parts = []

    for segment in segments_list:
        words = getattr(segment, "words", None)

        if not words:
            # Fallback: use coarse segment-level timing
            start_ts = format_timestamp(segment.start)
            end_ts = format_timestamp(segment.end)
            text = segment.text.strip()
            text_parts.append(text)
            srt_parts.append(f"{srt_index}\n{start_ts} --> {end_ts}\n{text}\n")
            srt_index += 1
            continue

        # Word-level: group into natural ≤10-word chunks
        chunk_words = []
        chunk_start = None

        for idx, word in enumerate(words):
            if chunk_start is None:
                chunk_start = word.start
            chunk_words.append(word.word.strip())

            is_last = idx + 1 >= len(words)
            has_pause = (not is_last and words[idx + 1].start - word.end > 0.7)

            if len(chunk_words) >= 10 or is_last or has_pause:
                line = " ".join(chunk_words)
                text_parts.append(line)
                start_ts = format_timestamp(chunk_start)
                end_ts = format_timestamp(word.end)
                srt_parts.append(f"{srt_index}\n{start_ts} --> {end_ts}\n{line}\n")
                srt_index += 1
                chunk_words = []
                chunk_start = None

    return "\n".join(srt_parts), " ".join(text_parts)

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

                # 2. Transcribe with accuracy-first parameters
                st.toast("Starting Heavy Processing", icon="⏳")
                status_bar.info("🧠 Waking up AI (Running VAD & Language Detection)... Please wait.")
                progress_bar.progress(0.0)
                
                # Accuracy-first configuration:
                #   beam_size=5           → explore 5 paths (was 1 which caused ~20-30% accuracy loss)
                #   word_timestamps=True  → precise per-word SRT timing
                #   condition_on_previous_text=False → kill hallucination feedback loops
                #   hallucination_silence_threshold=2 → skip phantom text in ≥2s silent gaps
                segments, info = model.transcribe(
                    temp_audio_path,
                    language=selected_language_code,
                    beam_size=5,
                    vad_filter=True, 
                    vad_parameters=dict(min_silence_duration_ms=500),
                    word_timestamps=True,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    hallucination_silence_threshold=2.0,
                )
                
                total_duration = info.duration
                start_time = time.time()
                
                # Materialize segments with progress tracking
                segments_list = []
                for segment in segments:
                    segments_list.append(segment)

                    # -- Progress Update --
                    progress = min(segment.end / total_duration, 1.0)
                    progress_bar.progress(progress)
                    
                    elapsed = time.time() - start_time
                    if progress > 0.01:
                        eta = (elapsed / progress) * (1.0 - progress)
                        mins, secs = divmod(int(eta), 60)
                        eta_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                    else:
                        eta_str = "calculating..."
                    status_bar.info(f"🗣️ Transcribing... {progress:.0%} | ETA: {eta_str}")
                
                # Build SRT with word-level precision (backend logic)
                srt_content, full_text = build_srt_from_segments(segments_list)
                
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
                            if progress > 0.01:
                                eta = (elapsed / progress) * (1.0 - progress)
                                mins, secs = divmod(int(eta), 60)
                                eta_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                            else:
                                eta_str = "calculating..."
                            status_bar.info(f"🔥 Burning Subtitles... {progress:.0%} | ETA: {eta_str}")
                    
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
