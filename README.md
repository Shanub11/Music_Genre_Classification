🎵 Music Genre Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify music into genres using the GTZAN dataset. The dataset contains 1000 audio tracks, each 30 seconds long, categorized into 10 genres.

We use Mel-spectrograms as features, treating them like images, and train a CNN model to recognize the musical genre.

📂 Dataset

Source: GTZAN Dataset – Kaggle

Genres (10 classes):

Blues

Classical

Country

Disco

Hip-Hop

Jazz

Metal

Pop

Reggae

Rock

Folder Structure
gtzan/
  genres_original/
    blues/
      blues.00000.wav
      blues.00001.wav
      ...
    classical/
    country/
    disco/
    hiphop/
    jazz/
    metal/
    pop/
    reggae/
    rock/

⚙️ Setup
1. Clone the repository
git clone https://github.com/Shanub11/music-genre-classification.git
cd music-genre-classification

2. Install dependencies
pip install -r requirements.txt


If using Google Colab, dependencies will install automatically in the notebook.

3. Download the dataset

There are multiple ways to get the dataset:

Kaggle API (Recommended):

kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip -d gtzan


Or manually download from Kaggle and extract to:

./gtzan/genres_original/

🚀 How to Run
Option 1: Run as a Python script
python train_genre_cnn.py --data_dir gtzan/genres_original --epochs 20

Option 2: Run in Jupyter Notebook / Google Colab

Open the notebook:

music_genre_classification_cnn.ipynb


Update the dataset path in the Config class:

class Config:
    data_dir = "/content/gtzan/genres_original"


Run all cells step by step.

🧠 Model Architecture

Input: Mel-Spectrogram (128 Mel bands × time frames)

4 Convolutional Blocks (Conv2D + BatchNorm + ReLU + MaxPool)

Adaptive Average Pooling

Fully Connected Layers:

FC (2048 → 256)

FC (256 → 10)

Dropout (0.3)

📊 Training & Evaluation

Optimizer: Adam

Loss Function: CrossEntropyLoss

Batch Size: 32

Epochs: 20 (configurable)

Metrics

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1)

📈 Results (Example)

Validation Accuracy: ~85–90% (depending on random split and augmentations)

Example Confusion Matrix:

📌 Features & Improvements

✅ On-the-fly Mel-Spectrogram computation using Librosa

✅ Data augmentation (time-shifting, noise injection)

✅ Training/validation curves visualization

✅ Confusion matrix & detailed classification report

Future Improvements:

Add SpecAugment for robust training

Use Transfer Learning (e.g., ResNet, EfficientNet)

Deploy as a web app (Flask/Django + React/Streamlit)

📦 Requirements

Python 3.8+
PyTorch
Librosa
Numpy, Matplotlib
scikit-learn
tqdm

Install all with:

pip install torch torchvision librosa scikit-learn matplotlib tqdm



✨ This project shows how deep learning + audio preprocessing can classify music genres effectively using CNNs!
