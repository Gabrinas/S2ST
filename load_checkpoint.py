import torch

# checking the contents of a saved checkpoint file
checkpoint = torch.load("best_model/bestmodel_527.pt")
print("Train Loss:", checkpoint['train_loss'])
print("Train Acc:", checkpoint['train_acc'])
print("Val Loss:", checkpoint['val_loss'])
print("Val Acc:", checkpoint['val_acc'])

# These are the discrete unit IDs you can feed into your vocoder
preds_for_vocoder = checkpoint['preds']

print(preds_for_vocoder)

# Load regular epoch checkpoint
epoch_ckpt = torch.load("train_checkpoints/epoch_618.pt")
print("Epoch 2 - Train Loss:", epoch_ckpt['train_loss'])
print("Epoch 2 - Train Acc:", epoch_ckpt['train_acc'])
print("Epoch 2 - Val Loss:", epoch_ckpt['val_loss'])
print("Epoch 2 - Val Acc:", epoch_ckpt['val_acc'])
preds_for_vocoder_epoch2 = epoch_ckpt['preds']
print(preds_for_vocoder_epoch2)

print("best model"); print("\n")
print(preds_for_vocoder[0][0])

print("checkpoint"); print("\n")
print(preds_for_vocoder_epoch2[0][0])
print(preds_for_vocoder_epoch2[1][0])
#print(val_preds)
