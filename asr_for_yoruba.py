
"""
from transformers import pipeline
import librosa

# Load Yoruba ASR pipeline
asr = pipeline("automatic-speech-recognition", model="NCAIR1/Yoruba-ASR")


ned = "generated_s2st_@540_with_1400audios_samp187.wav"

# Load audio file (16kHz recommended)
audio, sr = librosa.load(ned, sr=16000)

# Transcribe
result = asr(audio)
print(result["text"])
"""

from transformers import pipeline

model = "/home/gabriel/.cache/huggingface/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1"
asr = pipeline("automatic-speech-recognition", model=model)
#asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

files = [
    "generated_s2st_@427_with_1400audios_samp184.wav",
    "generated_s2st_@433_with_1400audios_samp184.wav",
    "generated_s2st_@451_with_1400audios_samp184.wav",
    "generated_s2st_@455_with_1400audios_samp184.wav",
    "generated_s2st_@504_with_1400audios_samp184.wav",
    "generated_s2st_@565_with_1400audios_samp184.wav"
]

for f in files:
    result = asr(f, language="yo")
    print(f"{f}: {result['text']}")


#ned = "LIBRISPEECH_PARTA_3_SPK_000UI_6GN_YOR_184.wav"
#result = asr(ned, language="yo")   # force Yoruba transcription
#print("Transcript:", result["text"])



