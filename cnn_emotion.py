import os
import librosa
import numpy as np

def extract_spectrogram(path):

    audio, sr = librosa.load(path, duration=3)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    target_length = 130

    if mel_db.shape[1] < target_length:

        pad_width = target_length - mel_db.shape[1]

        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, pad_width)),
            mode='constant'
        )

    else:

        mel_db = mel_db[:, :target_length]

    return mel_db

X = []
y = []



emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

for root, dirs, files in os.walk("dataset"):

    for file in files:

        if file.endswith(".wav"):

            path = os.path.join(root, file)

            emotion_code = file.split("-")[2]

            spectrogram = extract_spectrogram(path)

            X.append(spectrogram)

            y.append(emotion_map[emotion_code])

X = np.array(X)
y = np.array(y)

print(X.shape)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X = X.reshape(1440, 128, 130, 1)

print("New Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,130,1)
    )
)

model.add(MaxPooling2D((2,2)))

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(8, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

import matplotlib.pyplot as plt

# Accuracy Plot
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("screenshots/accuracy_curve.png", dpi=300)
plt.close()

# Loss Plot
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("screenshots/loss_curve.png", dpi=300)
plt.close()

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("CNN Accuracy:", accuracy)

model.save("model/cnn_emotion_model.keras")
print("CNN Model Saved")


