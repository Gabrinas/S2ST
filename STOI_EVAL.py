import librosa
from pystoi import stoi

# Load reference and generated audio at 16kHz
ref, sr = librosa.load("original_4.wav", sr=16000)
deg, sr = librosa.load("generated_4.wav", sr=16000)

# Align lengths (trim to shortest)
min_len = min(len(ref), len(deg))
ref = ref[:min_len]
deg = deg[:min_len]

# Compute STOI
score = stoi(ref, deg, sr, extended=False)
print("STOI score:", score)
