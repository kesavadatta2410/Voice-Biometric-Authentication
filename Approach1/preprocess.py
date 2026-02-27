import os
import librosa
import soundfile as sf
import numpy as np

RAW_DATA_DIR = "/home/rohithkaki/Voice_Biometrics/data/raw"
PROCESSED_DATA_DIR = "/home/rohithkaki/Voice_Biometrics/data/processed"
SAMPLE_RATE = 16000

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def preprocess_audio(file_path, output_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = librosa.util.normalize(audio)
    sf.write(output_path, audio, SAMPLE_RATE)

for root,_,files in os.walk(RAW_DATA_DIR):
    for file in files:
        if file.endswith(".wav"):
            input_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, RAW_DATA_DIR)
            output_dir = os.path.join(PROCESSED_DATA_DIR, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            output_file_path = os.path.join(output_dir, file)
            preprocess_audio(input_file_path, output_file_path)
print("Audio preprocessing completed.")

