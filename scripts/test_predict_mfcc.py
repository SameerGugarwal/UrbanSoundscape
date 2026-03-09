import requests
import numpy as np
import librosa

API_URL = "http://51.20.51.107:8000/predict_mfcc"

# change this to any wav you have on EC2
WAV_PATH = "sample_wavs/00_000066.wav"

y, sr = librosa.load(WAV_PATH, sr=22050)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
mfcc_mean = np.mean(mfcc, axis=1).tolist()

payload = {
    "mfcc": mfcc_mean,
    "threshold": 0.2,
    "top_k": 5
}

r = requests.post(API_URL, json=payload, timeout=30)
print("Status:", r.status_code)
print(r.json())
