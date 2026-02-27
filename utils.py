# to plot log_mel spectrograms
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Load one feature file
spec = np.load('/home/rohithkaki/Voice_Biometrics/data/features/Actor_01/03-01-08-01-01-01-01.npy')  # shape: (80, T)

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    spec,
    sr=16000,
    hop_length=160,
    x_axis="time",
    y_axis="mel"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Log-Mel Spectrogram")
plt.tight_layout()
plt.show()


# listen to audios 
from playsound3 import playsound
playsound('/home/rohithkaki/Voice_Biometrics/data/raw/Actor_01/03-01-08-01-01-01-01.wav')
playsound('/home/rohithkaki/Voice_Biometrics/data/processed/Actor_01/03-01-08-01-01-01-01.wav')


import numpy as np
data = np.load('/home/rohithkaki/Voice_Biometrics/data/features/Actor_01/03-01-08-01-01-01-01.npy')
print(data)
print(data.shape)

#plot the loss curve
import matplotlib.pyplot as plt

# Your actual data from the terminal output
epochs = list(range(1, 11))
losses = [0.2233, 0.1939, 0.1844, 0.1781, 0.1743, 0.1708, 0.1682, 0.1655, 0.1647, 0.1621]

plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker='o', linestyle='-', color='#2c3e50', linewidth=2, label='Training Loss')

# Formatting the chart
plt.title('Speaker Embedding Training Progress', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average Loss (Triplet/Contrastive)', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Annotate the start and end points
plt.annotate(f'{losses[0]}', (epochs[0], losses[0]), textcoords="offset points", xytext=(0,10), ha='center')
plt.annotate(f'{losses[-1]}', (epochs[-1], losses[-1]), textcoords="offset points", xytext=(0,10), ha='center')

plt.savefig('loss_plot.png')
print("Plot saved as loss_plot.png")
plt.show()


