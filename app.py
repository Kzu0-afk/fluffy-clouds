import streamlit as st
from faster_whisper import WhisperModel
from moviepy import VideoFileClip
import os
import tempfile
import subprocess
import time
import re

# Page Configuration
st.set_page_config(page_title="FluffyClouds", page_icon="☁️", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .stButton>button {
        height: 3.2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }
    .stRadio > div {
        background-color: transparent !important;
    }
    div[data-testid="stMainBlockContainer"] {
        padding-top: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Models ---
@st.cache_resource(show_spinner="Loading Model Weights into Memory (This may take a minute on first run)...")
def load_whisper_model(size, device_choice):
    compute = "float16" if device_choice == "cuda" else "int8"
    try:
        return WhisperModel(size, device=device_choice, compute_type=compute)
    except Exception:
        return WhisperModel(size, device="cpu", compute_type="int8")

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
    srt_index = 1
    srt_parts = []
    text_parts = []

    for segment in segments_list:
        words = getattr(segment, "words", None)

        if not words:
            start_ts = format_timestamp(segment.start)
            end_ts = format_timestamp(segment.end)
            text = segment.text.strip()
            text_parts.append(text)
            srt_parts.append(f"{srt_index}\n{start_ts} --> {end_ts}\n{text}\n")
            srt_index += 1
            continue

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

# --- Layout Setup ---
col1, col_spacer, col2 = st.columns([6, 0.5, 3.5])

with col1:
    # Header Area
    st.markdown("<h1 style='margin-bottom: 0px;'>FluffyClouds</h1>", unsafe_allow_html=True)
    st.markdown('<span style="color: #cbd5e1; font-weight: 600; font-size: 0.8rem; background-color: #334155; padding: 3px 10px; border-radius: 12px; margin-top: 5px; display: inline-block;">VERSION 1.0.0</span>', unsafe_allow_html=True)
    st.markdown("<h4 style='color: #94A3B8; font-weight: 400; margin-top: 15px;'>A premium editorial suite for high-end transcription and cinematic subtitle crafting.</h4>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Upload Area
    uploaded_file = st.file_uploader("Drop your masterpiece here\n\nSupport for high-fidelity 4K MOV, MP4 and ProRes files up to 10GB.", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file:
         st.video(uploaded_file)
         
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Bottom Status Info
    status_c1, status_c2 = st.columns(2)
    with status_c1:
        st.markdown("""
        <div style="background-color: #1E293B; padding: 20px; border-radius: 10px; border: 1px solid #334155;">
            <div style="color: #94A3B8; font-size: 14px; margin-bottom: 5px;">⟳ Last Transcription</div>
            <div style="color: white; font-weight: 500; font-size: 16px;">System_Boot_Init.mp4</div>
            <div style="height: 4px; background-color: #3182ce; width: 70%; margin-top: 15px; border-radius: 2px;"></div>
        </div>
        """, unsafe_allow_html=True)
    with status_c2:
         st.markdown("""
        <div style="background-color: #1E293B; padding: 20px; border-radius: 10px; border: 1px solid #334155;">
            <div style="color: #94A3B8; font-size: 14px; margin-bottom: 5px;">✨ AI Model Active</div>
            <div style="color: white; font-weight: 500; font-size: 16px;">Whisper Core - High Precision</div>
            <div style="color: #10B981; font-weight: 600; font-size: 12px; margin-top: 15px;">● OPTIMIZED SYSTEM</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Right Sidebar Configuration
    st.markdown("### 🎛️ Configuration")
    st.markdown("<hr style='margin-top: 0px; margin-bottom: 20px; border-color: #334155;'>", unsafe_allow_html=True)
    
    st.caption("HARDWARE ACCELERATOR")
    hardware = st.radio("Hardware", ["NVIDIA GPU", "CPU ONLY"], label_visibility="collapsed", horizontal=True)
    device_val = "cuda" if "GPU" in hardware else "cpu"
    
    st.caption("MODEL SIZE (Premium)")
    model_size = st.selectbox("Model", ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"], index=1, label_visibility="collapsed")
    
    st.caption("TRANSCRIPTION LANGUAGE")
    language_selection = st.selectbox("Language", ["English (US) - Default", "Auto-Detect", "Spanish", "French", "German"], label_visibility="collapsed")
    lang_map = {"English (US) - Default": "en", "Auto-Detect": None, "Spanish": "es", "French": "fr", "German": "de"}
    selected_language_code = lang_map[language_selection]
    
    st.caption("BURN SUBTITLES")
    burn_subtitles = st.checkbox("Enable embedded subtitling", value=True)
    
    st.caption("FONT FAMILY")
    font_choice = st.selectbox("Font", ["Inter Modern", "Manrope Bold", "Arial", "Times New Roman", "Courier New", "Verdana", "Impact"], label_visibility="collapsed")
    actual_font = "Arial" if "Inter" in font_choice else ("Arial" if "Manrope" in font_choice else font_choice)
    
    st.caption("PLACEMENT")
    placement = st.radio("Placement", ["Top", "Center", "Bottom"], index=2, horizontal=True, label_visibility="collapsed")
    align_map = {"Bottom": 2, "Center": 5, "Top": 8}
    align_val = align_map[placement]
    
    st.markdown("<br>", unsafe_allow_html=True)
    start_processing = st.button("Generate Transcription", type="primary", use_container_width=True)

# --- Core Execution Pipeline ---
if start_processing:
    if uploaded_file is None:
        st.error("Please select a file from your assets or drop a masterpiece first!")
    else:
        # Progress UI
        status_bar = st.empty()
        progress_bar = st.progress(0.0)

        model = load_whisper_model(model_size, device_val)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_video_path = os.path.join(temp_dir, "input_video.mp4")
            temp_audio_path = os.path.join(temp_dir, "audio.mp3")
            temp_srt_path = os.path.join(temp_dir, "subs.srt")
            out_video_path = os.path.join(temp_dir, "output_video.mp4")

            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.read())

            try:
                # 1. Audio extraction
                status_bar.info("⬇️ Extracting high-fidelity audio stream...")
                progress_bar.progress(0.0)
                
                video_clip = VideoFileClip(temp_video_path)
                video_duration = video_clip.duration  
                video_clip.audio.write_audiofile(temp_audio_path, logger=None)
                video_clip.close()

                # 2. Transcribe
                status_bar.info("🧠 Initializing Core AI Pipeline... Please wait.")
                progress_bar.progress(0.0)
                
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
                
                segments_list = []
                for segment in segments:
                    segments_list.append(segment)

                    progress = min(segment.end / total_duration, 1.0)
                    progress_bar.progress(progress)
                    
                    elapsed = time.time() - start_time
                    if progress > 0.01:
                        eta = (elapsed / progress) * (1.0 - progress)
                        mins, secs = divmod(int(eta), 60)
                        eta_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                    else:
                        eta_str = "calculating..."
                    status_bar.info(f"🗣️ Transcribing Masterpiece... {progress:.0%} | ETA: {eta_str}")
                
                srt_content, full_text = build_srt_from_segments(segments_list)
                
                progress_bar.progress(1.0)
                status_bar.success("✅ Transcription Phase Complete!")
                
                st.subheader("Transcript Pipeline Output:")
                st.text_area("Plain Text Format", full_text, height=200)
                
                st.download_button("📥 Download Document (.txt)", full_text, "transcript.txt", "text/plain")

                # 3. Burn Subtitles
                if burn_subtitles:
                    status_bar.info("⚙️ Rendering Cinematic Subtitles...")
                    progress_bar.progress(0.0)
                    
                    with open(temp_srt_path, "w", encoding="utf-8") as f:
                        f.write(srt_content)

                    sub_filter = f"subtitles=subs.srt:force_style='Fontname={actual_font},Alignment={align_val},FontSize=24,MarginV=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000'"
                    
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
                        st.error("❌ FFMpeg Render Error - Ensure FFMpeg is installed and on your system PATH.")
                    else:
                        progress_bar.progress(1.0)
                        status_bar.success("✅ Media Render Successfully Completed!")
                        
                        st.balloons()
                        
                        with open(out_video_path, "rb") as f:
                            video_bytes = f.read()
                        
                        st.download_button(
                            label="📥 Download Subtitled MP4",
                            data=video_bytes,
                            file_name="video_with_subtitles.mp4",
                            mime="video/mp4"
                        )

            except Exception as e:
                status_bar.error("❌ Pipeline Halted.")
                progress_bar.empty()
                st.error(f"Application Error: {e}")
