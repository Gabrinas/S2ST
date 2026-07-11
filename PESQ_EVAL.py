import librosa
import soundfile as sf
from pesq import pesq

# PESQ Evaluation

# Load and resample reference audio to 16kHz
ref, sr = librosa.load("original_4.wav", sr=16000)
deg, sr = librosa.load("generated_4.wav", sr=16000)

min_len = min(len(ref), len(deg))
ref = ref[:min_len]
deg = deg[:min_len]

# Save resampled reference if you want
#sf.write("original_1_16k.wav", ref, 16000)

# Compute PESQ (wideband mode for 16kHz)
score = pesq(16000, ref, deg, 'wb')
print("PESQ score:", score)

"""
min_len = min(len(ref), len(deg))
ref = ref[:min_len]
deg = deg[:min_len]
"""
