# 👋 Real-Time American Sign Language (ASL) Recognition using MediaPipe and SVM 🤖✋

## 📄 Summary
This project is a real-time American Sign Language (ASL) recognition system built using **MediaPipe** for hand tracking and **Support Vector Machines (SVM)** for gesture classification. It captures hand gestures through a webcam, processes them to extract hand landmarks, and predicts corresponding ASL characters in real-time. The system features a user-friendly interface built with **Tkinter**, enabling live ASL gesture recognition and text translation of the signs.

## 📂 Project Structure

- **`data_collection.py`**: Script to collect training data for ASL gestures from webcam feed.
- **`data_preprocessing.py`**: Extracts hand landmarks from images and prepares the dataset.
- **`model_training.py`**: Trains an SVM classifier on the extracted hand landmarks.
- **`realtime_recognition.py`**: Runs the real-time ASL recognition using the trained model with MediaPipe.
- **`README.md`**: Documentation for the project.
- **`model.p`**: The trained SVM model file for gesture prediction.
- **`data.pickle`**: Preprocessed dataset containing hand landmarks and labels for training.

## 🛠️ Dependencies

To get started, you'll need the following Python packages installed:

```bash
pip install mediapipe opencv-python scikit-learn numpy matplotlib
```

## 🚀 How to Run

1. **Collect Gesture Data**:
    - Use `data_collection.py` to collect hand gesture images for each ASL character using your webcam.
    - The script creates folders for each class and captures images of hand signs.

    ```bash
    python data_collection.py
    ```

2. **Preprocess Data**:
    - Extract hand landmarks from the collected images using `data_preprocessing.py`.

    ```bash
    python data_preprocessing.py
    ```

3. **Train the Model**:
    - Train the SVM model with the processed dataset using `model_training.py`.

    ```bash
    python model_training.py
    ```

4. **Real-Time Recognition**:
    - Run `realtime_recognition.py` to start real-time ASL recognition via webcam.
    - The predicted sign will be displayed in a graphical Tkinter interface.

    ```bash
    python realtime_recognition.py
    ```

## 💡 Features

- **Hand Gesture Detection**: Uses MediaPipe to track hand landmarks in real-time.
- **ASL Character Prediction**: Trained SVM model to predict 43 ASL signs including alphabets and common words.
- **User Interface**: Tkinter-based interface to display predicted characters and words.
- **Real-Time Feedback**: Translates gestures into text in real-time.

## 📝 How It Works

1. **Data Collection**: The system captures hand gestures via a webcam and stores the images for each ASL character.
2. **Landmark Extraction**: MediaPipe processes the hand gestures to extract key hand landmarks (x, y coordinates).
3. **Model Training**: The extracted landmarks are used to train an SVM classifier.
4. **Real-Time Prediction**: In the final step, the system captures video input, processes hand landmarks in real-time, and predicts the corresponding ASL sign using the trained SVM model.

## 📦 Model and Dataset

- The model is trained using an SVM with a Radial Basis Function (RBF) kernel.
- The dataset consists of hand landmarks for 43 different ASL gestures including alphabets and common signs.

## 🏆 Achievements
- Efficient hand landmark extraction using **MediaPipe**.
- Real-time ASL gesture recognition with **high accuracy**.
- Integrated **Tkinter** interface for seamless user interaction.

## 📧 Contact
For any queries or feedback, feel free to reach out to me at **[istiakahmed064@gmail.com](mailto:istiakahmed064@gmail.com)**.

---

**Enjoy translating ASL in real-time!** 🎉👐
