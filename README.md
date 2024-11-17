Emotion-Driven Speech Transcription and Cross-Lingual Translation with Arabic TTS Integration
This repository contains the implementation of a multilingual and emotion-driven system designed for enhancing customer service interactions in the banking domain. The system integrates Speech Emotion Recognition (SER), Automatic Speech Recognition (ASR), Machine Translation (MT), and Text-to-Speech (TTS), enabling emotion-preserving, cross-lingual communication.

Background
Effective communication across languages and emotions is critical, especially in customer service industries like banking. Language barriers and the inability to recognize customer emotions can hinder the resolution of issues. This project bridges these gaps by creating an end-to-end system that recognizes emotions, transcribes speech, translates text, and synthesizes audio—all while retaining emotional nuances.

Architectures
The system employs the following models:

Speech Emotion Recognition (SER):
Convolutional Neural Network (CNN) model for recognizing emotions from audio signals. It utilizes data augmentation and feature extraction (MFCCs, ZCR, etc.) for improved generalization.
Automatic Speech Recognition (ASR):
OpenAI’s Whisper model for English transcription, leveraging a transformer-based architecture for robust performance.
Machine Translation (MT):
MarianMT, a pre-trained transformer-based model fine-tuned for English-to-Arabic translation with domain-specific datasets.
Text-to-Speech (TTS):
MMS-TTS-Ara, a model for generating natural Arabic speech while retaining emotional tone.
Repository Structure
