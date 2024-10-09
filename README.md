# 👋 Real-Time American Sign Language (ASL) Translation using MediaPipe and SVM 🤖✋

## 📄 Summary
This project is a real-time American Sign Language (ASL) recognition system built using **MediaPipe** for hand tracking and **Support Vector Machines (SVM)** for gesture classification. It captures hand gestures through a webcam, processes them to extract hand landmarks, and predicts corresponding ASL characters in real-time. The system features a user-friendly web application built with **Streamlit**, enabling live ASL gesture recognition and text translation of the signs.

## 📂 Project Structure

- **`Custom_dataset_generation.ipynb`**: Script to collect training data for ASL gestures from webcam feed.
- **`landmark_extraction.ipynb`**: Extracts hand landmarks from images and prepares the dataset.
- **`model_train.ipynb`**: Trains an SVM classifier on the extracted hand landmarks.
- **`app.py`**: Runs the real-time ASL recognition & translation on streamlit-based Webapp using the trained model with MediaPipe.
- **`README.md`**: Documentation for the project.
- **`model.p`**: The trained SVM model file for gesture prediction.
- **`data.pickle`**: Preprocessed dataset containing hand landmarks and labels for training.

## 🛠️ Dependencies

To get started, you'll need the following Python packages installed:

```bash
pip install mediapipe opencv-python scikit-learn numpy matplotlib streamlit
```

## 💡 Features

- **Hand Gesture Detection**: Uses MediaPipe to track hand landmarks in real-time.
- **ASL Character Prediction**: Trained SVM model to predict 43 ASL signs including alphabets and common words.
- **User Interface**: Streamlit Web framework to display predicted characters and words.
- **Real-Time Feedback**: Translates gestures into text in real-time.


## 📧 Contact
For any queries or feedback, feel free to reach out to me at **[istiakahmed064@gmail.com](mailto:istiakahmed064@gmail.com)**.

---

**Enjoy translating ASL in real-time!** 🎉👐
