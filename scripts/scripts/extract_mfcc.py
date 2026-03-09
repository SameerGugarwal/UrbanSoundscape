import os
import librosa
import numpy as np
import pandas as pd

AUDIO_DIR = "../datasets/raw/SONYC-UST/audio"
ANNOTATIONS = "../datasets/raw/SONYC-UST/annotations.csv"
OUTPUT = "../datasets/processed/mfcc_features.csv"

os.makedirs("../datasets/processed", exist_ok=True)

df = pd.read_csv(ANNOTATIONS)

features = []
labels = []

for idx, row in df.iterrows():
    file_path = os.path.join(AUDIO_DIR, row["audio_filename"])
    
    if not os.path.exists(file_path):
        continue
    
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        features.append(mfcc_mean)
        labels.append(row[10:].values)  # skip metadata columns
    except:
        continue

X = np.array(features)
Y = np.array(labels)

final = np.concatenate([X, Y], axis=1)
pd.DataFrame(final).to_csv(OUTPUT, index=False)

print("MFCC feature file created:", OUTPUT)
print("Shape:", final.shape)
