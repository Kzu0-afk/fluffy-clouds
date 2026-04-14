from faster_whisper import WhisperModel
import logging
logging.basicConfig(level=logging.DEBUG)
import sys

print("Loading model...")
try:
    model = WhisperModel("tiny", device="auto", compute_type="int8")
    print("Model loaded.")
except Exception as e:
    print("Failed to load model:", e)
    sys.exit(1)

print("Attempting to transcribe with VAD...")
try:
    # Need a dummy audio file. We can just use an empty wav or mp3 if one exists, or generate one.
    import numpy as np
    import soundfile as sf
    sf.write('dummy.wav', np.zeros(16000), 16000)
    segments, info = model.transcribe('dummy.wav', vad_filter=True)
    print("Transcribe executed, info:", info)
    for seg in segments:
        print(seg.text)
    print("Done")
except Exception as e:
    print("Failed to transcribe:", e)
