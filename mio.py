import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")

import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, max_num_hands=1)

# Labels dictionary
labels_dict = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f',
    6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
    12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r',
    18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x',
    24: 'y', 25: 'z', 26: '0', 27: '1', 28: '2',
    29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: 'I love You', 37: 'yes', 38: 'No', 39: 'Hello', 40: 'Thanks',
    41: 'Sorry', 43: 'space'
}

# Streamlit GUI
st.title("ASL Prediction Application")

# Variable to store the previous prediction and time
last_detected_character = None
fixed_character = ""
delayCounter = 0
start_time = time.time()

# Variable to store the current text for display
current_text = ""

# Clear text button
if st.button('Clear Text'):
    current_text = ""
    st.empty()  # Clear the text

# Function to append the new predicted character to the text field
def update_text_field(text):
    global current_text
    if text == 'space':
        current_text += ' '
    else:
        current_text += text

# Function to run the video capture and ASL prediction
def run():
    global last_detected_character, fixed_character, delayCounter, start_time

    # Initialize the video capture
    cap = cv2.VideoCapture(1)  # Use 0 if you are using the primary webcam

    # Create a placeholder for the video stream in Streamlit
    video_placeholder = st.empty()

    # Create a placeholder for the "Hand-sign to Text" output
    text_output_placeholder = st.empty()

    while cap.isOpened():
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # Now, the drawing happens on the original frame
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Make prediction using the model
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Draw a rectangle and the predicted character on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

                current_time = time.time()

                # Timer logic: Check if the predicted character is the same for more than 1 second
                if predicted_character == last_detected_character:
                    if (current_time - start_time) >= 1.0:  # Class fixed after 1 second
                        fixed_character = predicted_character
                        if delayCounter == 0:  # Add character once after it stabilizes for 1 second
                            update_text_field(fixed_character)
                            delayCounter = 1
                else:
                    # Reset the timer when a new character is detected
                    start_time = current_time
                    last_detected_character = predicted_character
                    delayCounter = 0  # Reset delay counter for a new character

        # Convert the frame back to RGB (since OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the processed frame with landmarks in the Streamlit app
        video_placeholder.image(frame_rgb, channels="RGB")

        # Display the translated text (hand-sign to text) below the video
        text_output_placeholder.markdown(f"### Hand-sign to Text\n**{current_text}**")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

# Add a button to start video capture and ASL prediction
if st.button('Start Webcam'):
    run()

# Exit button
if st.button('Exit'):
    st.stop()
