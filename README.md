# Emotion-Driven Speech Transcription and Cross-Lingual Translation with Arabic TTS Integration

This repository contains the implementation of a multilingual and emotion-driven system designed to enhance customer service interactions in the banking domain. The system integrates **Speech Emotion Recognition (SER)**, **Automatic Speech Recognition (ASR)**, **Machine Translation (MT)**, and **Text-to-Speech (TTS)**, enabling emotion-preserving, cross-lingual communication.

---

## Background

Effective communication across languages and emotions is critical, especially in customer service industries like banking. Language barriers and the inability to recognize customer emotions can hinder the resolution of issues. This project bridges these gaps by creating an end-to-end system that:
- Recognizes emotions in speech.
- Transcribes speech to text.
- Translates text across languages.
- Synthesizes speech—all while retaining emotional nuances.

---

## Architectures

The system employs the following models:
- **Speech Emotion Recognition (SER):**
  - Convolutional Neural Network (CNN) model for recognizing emotions from audio signals.
  - Incorporates data augmentation and feature extraction techniques like MFCCs and ZCR for improved performance.
- **Automatic Speech Recognition (ASR):**
  - **Whisper** by OpenAI, a transformer-based model for robust English transcription.
- **Machine Translation (MT):**
  - **MarianMT**, a fine-tuned, pre-trained transformer-based model for English-to-Arabic translation.
- **Text-to-Speech (TTS):**
  - **MMS-TTS-Ara**, a model for generating natural Arabic speech while retaining emotional tones.

---

## Repository Structure

### Folder Details

#### 1. **`Automatic Speech Recognition (Whisper)`**
   - Contains scripts and configurations for **OpenAI’s Whisper** model.
   - Used for automatic speech recognition to transcribe English audio into text.
   - Includes:
     - Whisper configuration files.
     - Preprocessing scripts for audio input.
     - Example usage scripts for Whisper ASR.

#### 2. **`Machine Translation`**
   - Responsible for the machine translation tasks in the pipeline, specifically **English-to-Arabic translation**.
   - **Subfolders:**
     - **`Marian_MT_Best_Model_compressed/`:**
       - Compressed pre-trained and fine-tuned MarianMT model files for translation tasks.
     - **`training_data/`:**
       - Stores datasets used for fine-tuning the MarianMT model, including English-Arabic parallel sentences.
     - **`fine_tuning/`:**
       - Scripts and configurations for fine-tuning the MarianMT model using domain-specific datasets.
     - **`evaluation/`:**
       - Evaluation metrics and scripts for assessing the quality of translations using BLEU and BERT scores.

#### 3. **`Speech Emotion Recognition`**
   - Manages emotion recognition tasks using **CNN-based models**.
   - **Subfolders:**
     - **`data/`:**
       - Raw audio files and their preprocessed versions (e.g., augmented data).
       - Contains labeled datasets like RAVDESS for emotion classification.
     - **`training/`:**
       - Training scripts and configurations for the CNN model used in Speech Emotion Recognition (SER).
       - Includes hyperparameter tuning and augmentation techniques.
     - **`evaluation/`:**
       - Scripts for evaluating the SER model’s performance using precision, recall, F1 score, etc.

#### 4. **`TTS` (Text-to-Speech)**
   - Responsible for converting translated Arabic text into natural speech using **MMS-TTS-Ara**.
   - **Subfolders:**
     - **`mms_tts_ara/`:**
       - Pre-trained MMS-TTS-Ara model files and related scripts.
     - **`synthesis/`:**
       - Text-to-speech synthesis scripts to generate audio from Arabic text.

#### 5. **`utils/`**
   - Utility scripts and helper functions shared across the project.
   - May include:
     - Data preprocessing functions.
     - Model evaluation utilities.
     - Logging and visualization scripts.

#### 6. **`requirements.txt`**
   - Lists all Python dependencies required to run the project.
   - Includes libraries for machine learning (e.g., PyTorch), data processing, and evaluation.

#### 7. **`README.md`**
   - The main documentation file for the repository.
   - Provides an overview, installation instructions, and usage examples.

#### 8. **`LICENSE`**
   - Specifies the licensing terms for using the code in the repository.
   - Typically outlines permissions and restrictions for users.

### Additional Notes
- **Compressed Models:** Compressed model files under `Marian_MT_Best_Model_compressed/` are designed for users to download and uncompress before using them in translation tasks.
- **Training Data and Fine-Tuning:** Each component (SER, MT, ASR, TTS) has its own scripts for fine-tuning and training, allowing modular customization for different domains or languages.






## Installation

To set up the project, follow the steps below:

### 1. Clone the Repository
First, clone the repository to your local machine:
git clone https://github.com/besherhasan/Emotion-Driven-Speech-Transcription-and-Cross-Lingual-Translation-with-Arabic-TTS-Integration.git
cd Emotion-Driven-Speech-Transcription-and-Cross-Lingual-Translation-with-Arabic-TTS-Integration



### Install dependencies:

bash
Copy code
pip install -r requirements.txt
Install GPU-optimized PyTorch (if applicable):

bash
Copy code
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


## Training
Model Weights
Download the pre-trained and fine-tuned model weights:

Marian MT Best Model
SER CNN Model
Whisper Model
MMS-TTS-Ara
Place the downloaded weights in the appropriate directories under /weights/.


## Methodology
The pipeline follows these steps:

Emotion Detection (SER): Analyzes the customer's voice to detect emotional tone using a CNN.
Speech Transcription (ASR): Converts English speech to text using Whisper.
Text Translation (MT): Translates English text to Arabic using fine-tuned MarianMT.
Speech Synthesis (TTS): Converts translated Arabic text to speech using MMS-TTS-Ara.
Contributions
## This project introduces:

A multilingual pipeline preserving emotional context in speech translation.
Fine-tuned MarianMT for domain-specific English-Arabic translations.
Integration of SER, ASR, MT, and TTS into a seamless system.
Practical applications in the banking domain to improve customer service experience.
Challenges
## Limited Data: Domain-specific datasets for fine-tuning were limited, affecting performance.
Hardware Constraints: Training and inference required high computational power.
## Integration: Combining SER, ASR, MT, and TTS demanded significant preprocessing and optimization.
Future Directions
Dataset Expansion: Incorporate larger, diverse datasets for better performance.
Advanced Fine-Tuning: Explore techniques like LoRA or adapter layers for efficient domain adaptation.
Real-Time Processing: Optimize the system for real-time emotion-driven translation.
Additional Languages: Extend the pipeline to support more languages.






