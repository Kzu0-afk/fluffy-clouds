import time
from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np

print("Generating dummy audio...")
sf.write('dummy.wav', np.zeros(16000 * 30), 16000)

print("Loading model...")
t0 = time.time()
model = WhisperModel("tiny", device="cpu", compute_type="int8")
print(f"Model loaded in {time.time() - t0:.2f}s")

print("Transcribing (VAD ON, NO Lang)...")
t0 = time.time()
segments, info = model.transcribe('dummy.wav', vad_filter=True)
print(f"Initial transcribe loop hit in {time.time() - t0:.2f}s")
for s in segments:
    pass
print(f"Transcription finished in {time.time() - t0:.2f}s")

print("Transcribing (VAD ON, Lang EN)...")
t0 = time.time()
segments, info = model.transcribe('dummy.wav', vad_filter=True, language="en")
print(f"Initial transcribe loop hit in {time.time() - t0:.2f}s")
for s in segments:
    pass
print(f"Transcription finished in {time.time() - t0:.2f}s")
