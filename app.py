import time
import threading
import pickle
import warnings
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from flask import Flask, render_template, Response, jsonify, request

# Suppress specific warnings (optional)
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize the Flask application
app = Flask(__name__)

# ---------------------------
# Load the trained sign language model from a pickle file
# ---------------------------
model_dict = pickle.load(open('model/model.p', 'rb'))
model = model_dict['model']

# ---------------------------
# Mediapipe Hands Setup
# ---------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# Initialize Mediapipe Hands for real-time processing (using one hand)
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# ---------------------------
# Text-to-Speech (TTS) Setup using pyttsx3
# ---------------------------
engine = pyttsx3.init()

def speak_text(text):
    """
    Speaks out the provided text using pyttsx3.
    Runs in a separate daemon thread so it doesn't block video processing.
    """
    def tts_thread():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=tts_thread, daemon=True).start()

# ---------------------------
# Label Mapping and Expected Features
# ---------------------------
# Maps model prediction indices to corresponding characters (alphabet, digits, space, and period)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
    11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4',
    31: '5', 32: '6', 33: '7', 34: '8', 35: '9', 36: ' ', 37: '.'
}
expected_features = 42  # (21 landmarks × 2 coordinates)

# ---------------------------
# Global Variables for Stabilization and Sentence Formation
# ---------------------------
stabilization_buffer = []  # Buffer to stabilize predictions over time
stable_char = ""           # Last stable character recognized
word_buffer = ""           # Buffer to form words
sentence = ""              # Complete sentence formed from recognized words
last_registered_time = time.time()  # To avoid rapid duplicate registrations
registration_delay = 1.5   # Minimum delay (in seconds) before registering the same character again

# Global flag for pause functionality (default: not paused)
is_paused = False

# ---------------------------
# Initialize Webcam Capture using OpenCV
# ---------------------------
cap = cv2.VideoCapture(0)
# Set camera frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

def gen_frames():
    """
    Generator function that continuously captures frames from the webcam,
    processes each frame for sign language recognition, overlays the recognized
    text on the frame, and yields the frame as a JPEG byte stream.
    """
    global stabilization_buffer, stable_char, word_buffer, sentence, last_registered_time, is_paused
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)

        # If the processing is paused, just overlay a "Paused" text
        if is_paused:
            cv2.putText(frame, "Paused", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Convert frame color space from BGR to RGB for Mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []  # To hold normalized landmark data
                    x_coords = []
                    y_coords = []
                    # Collect raw landmark coordinates
                    for lm in hand_landmarks.landmark:
                        x_coords.append(lm.x)
                        y_coords.append(lm.y)
                    # Normalize coordinates by subtracting the minimum x and y values
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_coords))
                        data_aux.append(lm.y - min(y_coords))
                    # Ensure data_aux has exactly 'expected_features' values
                    if len(data_aux) < expected_features:
                        data_aux.extend([0] * (expected_features - len(data_aux)))
                    elif len(data_aux) > expected_features:
                        data_aux = data_aux[:expected_features]
                    
                    # Predict the sign language gesture using the trained model
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    # Append prediction to the stabilization buffer
                    stabilization_buffer.append(predicted_character)
                    if len(stabilization_buffer) > 30:  # Buffer size for roughly one second
                        stabilization_buffer.pop(0)

                    # If the same prediction appears consistently, register it as stable
                    if stabilization_buffer.count(predicted_character) > 25:
                        current_time = time.time()
                        if current_time - last_registered_time > registration_delay:
                            stable_char = predicted_character
                            last_registered_time = current_time
                            # Overlay the stable character on the frame
                            cv2.putText(frame, f"Alphabet: {stable_char}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            # Handle word and sentence formation based on the character
                            if stable_char == ' ':
                                if word_buffer.strip():
                                    speak_text(word_buffer)  # Speak the complete word
                                    sentence += word_buffer + " "
                                word_buffer = ""
                            elif stable_char == '.':
                                if word_buffer.strip():
                                    speak_text(word_buffer)
                                    sentence += word_buffer + "."
                                word_buffer = ""
                            else:
                                word_buffer += stable_char
                            # Draw hand landmarks on the frame for visualization
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                                      mp_drawing_styles.get_default_hand_connections_style())

            # Overlay the current word and sentence on the frame
            cv2.putText(frame, f"Word: {word_buffer}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Sentence: {sentence}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Encode the processed frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        # Yield the frame in a multipart HTTP response format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ---------------------------
# Flask Routes and Endpoints
# ---------------------------
@app.route('/')
def index():
    """
    Renders the main HTML page.
    (The corresponding 'index.html' should be placed in the 'templates' folder.)
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route. Returns a multipart response containing the video frames.
    """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    """
    Toggles the pause state for video processing.
    Called when the user clicks a "Pause/Play" button.
    """
    global is_paused
    is_paused = not is_paused
    return jsonify({'paused': is_paused})

@app.route('/reset', methods=['POST'])
def reset():
    """
    Resets the current word and sentence.
    Called when the user clicks a "Reset Sentence" button.
    """
    global word_buffer, sentence
    word_buffer = ""
    sentence = ""
    return jsonify({'word': word_buffer, 'sentence': sentence})

@app.route('/get_text')
def get_text():
    """
    Returns the current recognized alphabet, word, and sentence as JSON.
    Can be polled by client-side JavaScript to update the UI.
    """
    return jsonify({
        'alphabet': stable_char,
        'word': word_buffer,
        'sentence': sentence
    })

@app.route('/speak_sentence', methods=['POST'])
def speak_sentence():
    """
    Triggers text-to-speech for the current sentence.
    Called when the user clicks a "Speak Sentence" button.
    """
    speak_text(sentence)
    return jsonify({'sentence': sentence})

# ---------------------------
# Run the Flask Application
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)
