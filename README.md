# ✋🤟 American Sign Language (ASL) Recognition using MediaPipe and SVM 🤖

Welcome to the **ASL Recognition** project! This project uses **MediaPipe** for hand landmark detection and **Support Vector Machine (SVM)** for recognizing American Sign Language (ASL) gestures in real-time via a webcam feed.

## 📜 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Introduction

In this project, we developed an ASL recognition system that captures hand gestures using a webcam, detects hand landmarks via **MediaPipe**, and classifies the signs using a **trained SVM model**. The system supports both real-time prediction and batch processing of images.

## 🎯 Features

- Real-time hand gesture detection and classification using webcam 📷.
- ASL alphabets (`a-z`) and numerical digits (`0-9`) recognition 🔠.
- Recognizes special gestures like **"I Love You", "Hello", "Yes", "No"**, and more ✋.
- GUI interface using **Tkinter** for visualizing the predicted ASL signs 🖥️.
- Simple buttons for clearing text and exiting the application 🔘.

## ⚙️ Installation

Follow these steps to set up and run the project on your local machine.

### Prerequisites

Make sure you have Python 3.x installed and the following libraries:

- `opencv-python` (for video capturing)
- `mediapipe` (for hand landmark detection)
- `scikit-learn` (for SVM model)
- `numpy` (for data manipulation)
- `matplotlib` (for plotting, optional)
- `tkinter` (for GUI)

You can install them using:

```bash
pip install opencv-python mediapipe scikit-learn numpy matplotlib
