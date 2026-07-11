import torch

model = torch.hub.load("microsoft/DNSMOS", "dns_mos")
score = model("generated_1.wav")
print("Predicted MOS:", score)
