# 🎙️ Speech Emotion Classification using CNN

A deep learning project that classifies human speech into **8 emotion categories** using a **Convolutional Neural Network (CNN)** trained on the **RAVDESS** dataset.

---

## 📌 Features

- Classifies speech into **8 emotions**
- Converts speech audio into **log-Mel spectrograms**
- Implements a **CNN** using TensorFlow/Keras
- Performs audio preprocessing with **Librosa**
- Trains and evaluates the model on the **RAVDESS** dataset

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Matplotlib
- Scikit-learn

---

## 📂 Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

Emotion classes:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## 🧠 Model Architecture

```
Speech Audio (.wav)
        │
        ▼
Audio Preprocessing
        │
        ▼
Log-Mel Spectrogram
        │
        ▼
CNN
        │
        ▼
Emotion Prediction
```

---

## 🚀 How to Run

Clone the repository

```bash
git clone https://github.com/Vignesh-Siddu/Speech-Emotion-Classification-CNN.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python cnn_emotion.py
```

---

## 📈 Results

- Training Accuracy: ~91%
- Validation Accuracy: ~50%

The model successfully classifies speech into eight emotion categories. The gap between training and validation accuracy indicates overfitting, which can be improved with additional data augmentation and model tuning.

---

## 🔮 Future Improvements

- Improve generalization using data augmentation
- Experiment with deeper CNN architectures
- Add real-time microphone emotion recognition
- Deploy as a web application
