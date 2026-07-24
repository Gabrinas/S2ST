import os, random, shutil

# Paths
eng_dir = "ENG_TRAIN_AUGMEN"
yor_dir = "YOR_TRAIN_AUGMEN"
train_eng = "train_eng"
train_yor = "train_yor"
val_eng = "val_eng"
val_yor = "val_yor"

# Create output dirs
for d in [train_eng, train_yor, val_eng, val_yor]:
    os.makedirs(d, exist_ok=True)

# Get sorted file lists
eng_files = sorted(os.listdir(eng_dir))
yor_files = sorted(os.listdir(yor_dir))

# Ensure alignment
assert len(eng_files) == len(yor_files), "Mismatch in parallel data length"

# Shuffle indices
indices = list(range(len(eng_files)))
random.shuffle(indices)

# Split
train_idx = indices[:6400]
val_idx = indices[6400:]

# Copy files
for i in train_idx:
    shutil.copy(os.path.join(eng_dir, eng_files[i]), train_eng)
    shutil.copy(os.path.join(yor_dir, yor_files[i]), train_yor)

for i in val_idx:
    shutil.copy(os.path.join(eng_dir, eng_files[i]), val_eng)
    shutil.copy(os.path.join(yor_dir, yor_files[i]), val_yor)

print("Done: 6400 train pairs, 1600 val pairs")
