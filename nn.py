import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")

import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
import uuid
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
    24: 'y', 25: 'z', 26: 'space', 27: '1', 28: '2',
    29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: 'I love You', 37: 'yes', 38: 'No', 39: 'Hello', 40: 'Thanks',
    41: 'Sorry', 43: 'space'
}
# Phonetic mapping dictionary (for demonstration)
phonetic_mapping = {
    "a": "অ", "i": "ই", "ii": "ঈ", "u": "উ", "uu": "ঊ",
    "e": "এ", "oi": "ঐ", "o": "ও", "ou": "ঔ",
    "k": "ক", "kh": "খ", "g": "গ", "gh": "ঘ", "ng": "ঙ",
    "c": "চ", "ch": "ছ", "j": "জ", "jh": "ঝ", "ny": "ঞ",
    "t": "ট", "th": "ঠ", "d": "দ", "dh": "ঢ", "n": "ন",
    "p": "প", "ph": "ফ", "b": "ব", "bh": "ভ", "m": "ম",
    "r": "র", "l": "ল", "sh": "শ", "ss": "ষ", "s": "স", "h": "হ",
    # More consonants and mappings can be added here...
}

# Diacritics mapping for consonants followed by vowels (including three-letter cases)
diacritics_mapping = {
    "ma": "মা", "ki": "কি", "kii": "কী", "ku": "কু", "kuu": "কূ","aa": "আ",
    "ke": "কে", "kai": "কৈ", "ko": "ক", "kou": "কৌ", "dho":"ধ", "nno":"ন্য" , "ba": "বা",
    "kha": "খা", "khi": "খি", "khii": "খী", "khu": "খু", "khuu": "খূ","tha":"থা","shu": "শু","nu": "নু"
    # Add diacritic mappings for other consonants...
}


# Variable to store the previous prediction and time
last_detected_character = None
fixed_character = ""
delayCounter = 0
start_time = time.time()

# Variable to store the current text for display
current_text = ""

#phonetic Transform
def phonetic_transform_word(word, mapping, diacritics_mapping):
    result = ""
    buffer = ""
    i = 0

    while i < len(word):
        char = word[i]

        buffer += char  # Add character to buffer

        # Check for three-letter diacritic combinations first (e.g., "khi")
        if i + 2 < len(word) and (buffer + word[i + 1] + word[i + 2]) in diacritics_mapping:
            buffer += word[i + 1] + word[i + 2]  # Add the next two characters to buffer
            result += diacritics_mapping[buffer]  # Add the diacritic form for "khi"
            buffer = ""  # Clear buffer after successful match
            i += 2  # Skip the next two characters since they are already processed
        # If not found, check for two-letter diacritic combinations (e.g., "ka")
        elif i + 1 < len(word) and (buffer + word[i + 1]) in diacritics_mapping:
            buffer += word[i + 1]  # Add the next vowel character to buffer
            result += diacritics_mapping[buffer]  # Add the diacritic form for "ka"
            buffer = ""  # Clear buffer after successful match
            i += 1  # Skip the next character since it's already processed
        else:
            # Check if buffer matches a single consonant or vowel
            if buffer in mapping:
                result += mapping[buffer]
                buffer = ""  # Clear buffer after match
            else:
                # Handle multi-character combinations (like 'kh', 'aa', etc.)
                if i + 1 < len(word) and (buffer + word[i + 1]) in mapping:
                    buffer += word[i + 1]
                    result += mapping[buffer]  # Add the mapped value
                    buffer = ""  # Clear the buffer
                    i += 1  # Skip the next character
                elif buffer[:-1] in mapping:
                    result += mapping[buffer[:-1]]
                    buffer = buffer[-1]  # Keep the last character in buffer

        i += 1

    # Add any remaining buffer (if any)
    if buffer in mapping:
        result += mapping[buffer]

    return result

current_word = ""  # To store the current word
transformed_word = ""
transformed_sentence = ""
def process_space(word):
    global current_text
    global transformed_word
    global transformed_sentence
    if current_text:
        last_text = current_text.split()[-1]
        # Transform the word and print the result
        transformed_word = phonetic_transform_word(last_text, phonetic_mapping, diacritics_mapping)
        transformed_sentence += transformed_word
        transformed_sentence += " "
        #transformed_sentence += transformed_word
        return transformed_sentence # Clear word after transforming

# Function to append the new predicted character to the text field
def update_text_field(text):
    global current_text
    global transformed_sentence
    global transformed_word
    if text == 'space':
        transformed_sentence =  process_space(current_text)
        current_text += ' '
    else:
        current_text += text 

    
# Streamlit GUI
st.title("ASL Prediction Application")
tabs = st.tabs(["Home", "Sign Detection", "About"])
# --- Home Tab ---
with tabs[0]:
    st.title("Welcome to Hand Sign Detection")
    st.markdown("""
    This web app detects American Sign Language (ASL) signs using your webcam in real-time.
    
    - **Use the 'Sign Detection' tab** to start recognizing hand gestures via your webcam.
    - Adjust the confidence level and processing speed dynamically.
    - Review the **About** section for more information on how the app works.
    """)

with tabs[1]:
    st.title("Real-Time Hand Sign Detection")
    st.markdown("""
    To begin detecting hand signs, click the "Start Detection" button below.
    The system will display predictions in real-time and show them in text format.
    """)

    # Webcam and detection layout
    col1, col2 = st.columns([2, 1])
    
    # Webcam Placeholder and Start Button
    with col1:
        st.markdown("### Live Webcam Feed")
        video_placeholder = st.empty()  # Placeholder for the video feed

    # Predicted text area
    with col2:
        st.markdown("### Predicted Sign & Text")
        # Initialize the text area outside the loop to avoid DuplicateWidgetID issue
        text_area_placeholder1 = st.empty()
        # Initialize the text area outside the loop to avoid DuplicateWidgetID issue
        text_area_placeholder2 = st.empty()

    # Function to run the video capture and ASL prediction
    def run():
        global current_word
        global transformed_word , transformed_sentence
        global last_detected_character, fixed_character, delayCounter, start_time
        # Initialize the video capture
        cap = cv2.VideoCapture(1)  # Use 0 if you are using the primary webcam

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
            #text_output_placeholder.markdown(f"### Hand-sign to Text\n**{current_text}**")
            # Initialize the text area outside the loop to avoid DuplicateWidgetID issue
            # Generate a custom key using uuid or a timestamp to make sure it is unique
            custom_key1 = f"text_output_{uuid.uuid4()}"
            text_area_placeholder1.text_area('English Text Conversion', current_text, key=custom_key1)
            
            custom_key2 = f"text_output_{uuid.uuid4()}"
            text_area_placeholder2.text_area('Bangla Text Conversion', transformed_sentence, key=custom_key2)
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    # Add a button to start video capture and ASL prediction
    if st.button('Start Webcam'):
        run()

    # Exit button
    if st.button('Exit'):
        st.stop()

# --- About Tab ---
with tabs[2]:
    st.title("About This App")
    with st.expander("How It Works"):
        st.write("""
        This application uses a deep learning model to detect hand signs in real-time from webcam input. 
        The system utilizes **Mediapipe** for hand landmark detection and applies a pre-trained model to predict American Sign Language (ASL) signs.
        """)

    with st.expander("Technologies Used"):
        st.markdown("""
        - **Streamlit**: Interactive web framework
        - **Mediapipe**: For real-time hand landmark tracking
        - **OpenCV**: Video feed processing
        - **Custom Deep Learning Model**: For hand sign classification
        """)

    st.write("For more information, feel free to contact the developer or refer to the [GitHub Repository](#).")

# Footer with credits
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        padding: 10px;
        font-size: small;
        color: #555;
        margin-top: 50px;
        border-top: 1px solid #ddd;
    }
    </style>
    <div class="footer">
        Developed by Istiaq Ahmed Fahim | 2024 © All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True
)