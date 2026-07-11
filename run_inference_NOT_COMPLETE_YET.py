import torch
import os
import glob
import torch
import torchaudio
import numpy as np
import warnings
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import joblib
from s2st_model import S2STModel   # import your model class
import soundfile as sf

# Suppress TorchCodec warnings
warnings.filterwarnings("ignore", message=".*TorchCodec.*")

# 1. Load pretrained HuBERT
#hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to("cuda")
#feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

model_path = "/home/gabriel/.cache/huggingface/hub/models--facebook--hubert-base-ls960/snapshots/dba3bb02fda4248b6e082697eee756de8fe8aa8a"
hubert = HubertModel.from_pretrained(model_path, local_files_only = True).to("cuda")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, local_files_only = True)

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
print(f"Found {len(files_EN)} ENGLISH wav files"); 

all_units_EN = []
for wav_path in files_EN:
    units_EN = extract_units(wav_path)
    all_units_EN.append(units_EN)

max_len_EN = max(len(u) for u in all_units_EN)
padded_units_EN = np.full((len(all_units_EN), max_len_EN), fill_value=0, dtype=np.int32)
for i, u in enumerate(all_units_EN):
    padded_units_EN[i, :len(u)] = u
padded_units_EN = padded_units_EN[:, :600]
eng_src = padded_units_EN[0: 1]
print(padded_units_EN.shape)
print(eng_src.shape)
print(eng_src)



# checking the contents of a saved checkpoint file
checkpoint = torch.load("best_model/bestmodel_78.pt")
print("Train Loss:", checkpoint['train_loss'])
print("Val Loss:", checkpoint['val_loss'])
print("Val Acc:", checkpoint['val_acc'])


model.eval()
with torch.no_grad():
    src_feats = eng_src.to(device)  # single English input
    predicted_units = model.infer(src_feats)

print(predicted_units)


import torch

#from unit_hifigan.model import UnitVocoder # to call from python environment

import soundfile as sf

"""
import sys
sys.path.append("/home/gabriel/hifi_discrete_proj/unit_hifigan/src")
from unit_hifigan.model import UnitVocoder
"""

import sys
sys.path.append("/home/gabriel/hifi_discrete_proj/unit-hifigan/src")

from unit_hifigan.model import UnitVocoder   # model.py is inside src/
vocoder = UnitVocoder.from_pretrained("runs_hubert/yoruba_vocoder/vocoder-50000")




# Convert to tensor with batch dimension
units = torch.tensor(units_list).unsqueeze(0)

# Generate audio
audio = vocoder.generate(units)

# Save to WAV
sf.write("generated_s2st.wav", audio.squeeze().cpu().numpy(), 16000)












    
