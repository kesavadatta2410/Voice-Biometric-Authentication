import librosa
import numpy as np
import os

PROCESSED_DATA_DIR = "/home/rohithkaki/Voice_Biometrics/data/processed"
FEATURES_DATA_DIR = "/home/rohithkaki/Voice_Biometrics/data/features"
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
os.makedirs(FEATURES_DATA_DIR, exist_ok=True)

def extract_log_mel_spectrogram(audio_path):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    log_mel_spec = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    return log_mel_spec

for root, _, files in os.walk(PROCESSED_DATA_DIR):
    for file in files:
        if file.endswith(".wav"):
            input_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, PROCESSED_DATA_DIR)
            output_dir = os.path.join(FEATURES_DATA_DIR, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            output_file_path = os.path.join(output_dir, file.replace(".wav", ".npy"))
            log_mel = extract_log_mel_spectrogram(input_file_path)
            np.save(output_file_path, log_mel)
print("Conversion to log-Mel spectrograms completed.")