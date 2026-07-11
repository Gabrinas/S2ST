# Extract discrete units for ENGLISH
import os
import glob
import torch
import torchaudio
import numpy as np
import warnings
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import joblib

# Suppress TorchCodec warnings
warnings.filterwarnings("ignore", message=".*TorchCodec.*")

# 1. Load pretrained HuBERT
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to("cuda")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

# 2. Load pretrained k-means (joblib pickle)
kmeans = joblib.load("hubert_base_ls960_L9_km500.bin")

# 3. Function to extract discrete units
def extract_units(wav_path):
    # Force sox_io backend to avoid TorchCodec warnings
    waveform, sr = torchaudio.load(wav_path, backend="sox_io")
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

    arr = waveform.squeeze().numpy()
    inputs = feature_extractor(arr, sampling_rate=16000, return_tensors="pt")

    with torch.inference_mode():
        outputs = hubert(**inputs.to("cuda"))
    features = outputs.last_hidden_state.cpu().numpy()

    features_2d = features.reshape(-1, features.shape[-1])
    units = kmeans.predict(features_2d)
    return units

# 4. Batch process all audios
input_folder_EN = "/home/gabriel/s2st_project/ENGLISH"
files_EN = glob.glob(os.path.join(input_folder_EN, "*.wav"))
print(f"Found {len(files_EN)} wav files")

all_units_EN = []
for wav_path in files_EN:
    units_EN = extract_units(wav_path)
    all_units_EN.append(units_EN)

# 5. Pad sequences to same length
max_len_EN = max(len(u) for u in all_units_EN)
padded_units_EN = np.full((len(all_units_EN), max_len_EN), fill_value=-1, dtype=np.int32)
for i, u in enumerate(all_units_EN):
    padded_units_EN[i, :len(u)] = u

print("Final padded shape:", padded_units_EN.shape)
print(padded_units_EN)

# 6. Save once
input_folder = "/home/gabriel/s2st_project/ENGLISH"
output_folder = "/home/gabriel/s2st_project/ENGLISH_UNITS"
os.makedirs(output_folder, exist_ok=True)

files = glob.glob(os.path.join(input_folder, "*.wav"))
print(f"Found {len(files)} wav files")

for wav_path in files:
    units = extract_units(wav_path)
    filename = os.path.splitext(os.path.basename(wav_path))[0]
    out_path = os.path.join(output_folder, f"{filename}.npy")
    np.save(out_path, units)
    #print(f"Saved {out_path} with {len(units)} units")

print(f"All unit files saved in {output_folder}")
