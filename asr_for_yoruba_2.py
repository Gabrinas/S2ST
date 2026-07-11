from transformers import pipeline
import librosa

# Load Yoruba ASR pipeline
#model = "/home/gabriel/.cache/huggingface/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1"
model = "/home/gabriel/.cache/huggingface/hub/models--NCAIR1--Yoruba-ASR/snapshots/d1ae7b8b79c2ccd547d8761effe5057433f3fc7f"


asr = pipeline("automatic-speech-recognition", model=model)


ned = "generated_s2st_@467_with_1400audios_samp187.wav"

# Load audio file (16kHz recommended)
audio, sr = librosa.load(ned, sr=16000)

# Transcribe
result = asr(audio)
print("Transcript of " + ned + " is ", result["text"])
