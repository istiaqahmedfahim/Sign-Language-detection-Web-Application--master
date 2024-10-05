import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import uuid
import time

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Load a pre-trained model (Replace this with your actual model)
import pickle
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Define Labels
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    # More labels up to the number of classes your model supports
}

# Professional Styling
st.set_page_config(page_title="Hand Sign Detection", layout="wide")
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        color: #4CAF50;
    }
    .footer {
        text-align: center;
        padding: 20px;
        background-color: #f1f1f1;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title Section
st.markdown('<h1 class="title">Hand Sign Detection</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #555;">Using AI to detect American Sign Language (ASL) in real-time</p>', unsafe_allow_html=True)

# Layout with two columns: webcam on the left, text output on the right
col1, col2 = st.columns([3, 1])

# Placeholder for webcam in the first column
with col1:
    st.markdown("### Live Webcam Feed")
    video_placeholder = st.empty()  # Placeholder for the video feed

# Placeholder for text output in the second column
with col2:
    st.markdown("### Predicted Sign")
    text_placeholder = st.empty()

# Start button and exit button in a sidebar for convenience
st.sidebar.title("Control Panel")
start_button = st.sidebar.button('Start Webcam')
clear_button = st.sidebar.button('Clear Text')
exit_button = st.sidebar.button('Exit')

# Current text and sign storage
predicted_text = ""
last_detected_sign = None
timer = 0

# Function to run the hand sign detection
def run_detection():
    global predicted_text, last_detected_sign, timer
    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw hand landmarks and make predictions
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Extract hand landmark positions for the model
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.append(lm.x)
                    landmark_list.append(lm.y)

                # Model prediction
                data = np.array([landmark_list])
                prediction = model.predict(data)[0]
                predicted_sign = labels_dict[int(prediction)]

                # Show the predicted sign in real-time
                if predicted_sign == last_detected_sign:
                    timer += 1
                    if timer > 20:  # Stabilize prediction after 20 frames
                        predicted_text += predicted_sign
                        text_placeholder.markdown(f"**Predicted Text:** {predicted_text}")
                        timer = 0
                else:
                    last_detected_sign = predicted_sign
                    timer = 0

                # Show the predicted sign in the video frame
                cv2.putText(frame, predicted_sign, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        # Display the frame
        video_placeholder.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q') or exit_button:
            break

    cap.release()

# Clear button functionality
if clear_button:
    predicted_text = ""
    text_placeholder.empty()

# Start webcam and detection
if start_button:
    run_detection()

# Footer Section
st.markdown(
    """
    <div class="footer">
        <p>Developed by [Your Name]</p>
        <p>© 2024 All Rights Reserved</p>
    </div>
    """, 
    unsafe_allow_html=True
)
