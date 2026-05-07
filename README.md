# FluffyClouds (Local Transcriber-Subtitle Web Application)

FluffyClouds (Version 1.0.0) is a premium, Streamlit-based web application designed for high-end video transcription and cinematic subtitle crafting. It provides an intuitive, AI-powered editorial suite that easily extracts audio from video files, transcribes speech with high precision, and seamlessly renders beautifully styled subtitles directly into your media.

## Summary

The platform utilizes `faster-whisper` for optimized AI-based transcription, ensuring high transcription accuracy even on both CPU and NVIDIA GPU architectures. It handles tasks ranging from straightforward text extraction to fully burnt-in cinematic subtitles in multiple languages, making it a powerful and accessible tool for content creators, video editors, and anyone needing quick, robust subtitling.

## Features and Uses

*   **High-Fidelity Media Support:** Upload and process standard to high-resolution videos, supporting formats such as MOV, MP4, AVI, and MKV (up to 10GB).
*   **AI-Powered Transcription:** Utilizes state-of-the-art Whisper models (scalable from `tiny` to `large-v3-turbo`) with Voice Activity Detection (VAD) for precise transcription while avoiding silence hallucinations.
*   **Hardware Flexibility:** Choose between NVIDIA GPU acceleration for maximum speed or CPU-only processing to accommodate different machine setups.
*   **Cinematic Subtitles:** Burn perfectly synchronized, customizable subtitles directly into your video output. Tailor the aesthetics with premium font families and adjust subtitle placement (Top, Center, Bottom).
*   **Multilingual Support:** Transcribe in specific languages (English, Spanish, French, German) or use the Auto-Detect feature.
*   **Dual Outputs:** Instantly download plain text transcripts (`.txt`) for documentation or the finalized subbed video (`.mp4`) for publishing.

## Setup Instructions

Follow these steps to set up and run FluffyClouds on your local machine:

### Prerequisites

1.  **Python 3.8+**: Ensure you have a recent version of Python installed on your system.
2.  **FFmpeg**: You must have FFmpeg installed on your operating system and properly added to your system's PATH. This is strictly required for media handling and rendering subtitles.

### Installation

1.  **Navigate to the project folder:**
    ```bash
    cd path/to/fluffy-clouds
    ```

2.  **Create a Virtual Environment (Optional but recommended):**
    ```bash
    python -m venv venv
    ```
    *   **Windows:** `venv\Scripts\activate`
    *   **macOS/Linux:** `source venv/bin/activate`

3.  **Install Python Dependencies:**
    Install the required packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

Once your dependencies and FFmpeg are successfully set up, start the Streamlit web application by running the following command in your terminal:

```bash
streamlit run app.py
```

The application will automatically launch in your default web browser (typically accessible at `http://localhost:8501`). Drop your video masterpiece in the upload area and start generating subtitles!
