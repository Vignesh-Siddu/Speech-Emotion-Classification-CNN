import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = "dataset/Actor_01/03-01-01-01-01-01-01.wav"

y, sr = librosa.load(audio_path)

mel = librosa.feature.melspectrogram(y=y, sr=sr)
mel_db = librosa.power_to_db(mel, ref=np.max)

plt.figure(figsize=(10, 4))

librosa.display.specshow(
    mel_db,
    sr=sr,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Log-Mel Spectrogram")

plt.tight_layout()

plt.savefig("screenshots/spectrogram.png", dpi=300)

plt.show()