import cv2 #  OpenCV is used for capturing video from the webcam and displaying output.
import mediapipe as mp # A library from Google used to detect hand landmarks.
import numpy as np
from tensorflow.keras.models import load_model # Used to load the pre-trained ASL hand landmarks model.
import pyttsx3 #  Text-to-speech library to convert recognized ASL letters into speech.

TF_ENABLE_ONEDNN_OPTS=0. # disables OneDNN optimizations

# Load the trained model
model = load_model(r'asl_hand_landmarks_model.h5')


# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) # Enables dynamic video feed processing. Detects only one hand. Minimum confidence threshold for hand detection.
mp_drawing = mp.solutions.drawing_utils # Helps draw landmarks and connections on the hand in the video feed

# Initialize text-to-speech engine
engine = pyttsx3.init() # Pyttsx3 is initialized to convert recognized ASL signs into speech.

# Mapping of class indices to ASL letters and signs
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'delete', 'space', 'nothing']
sentence = []
last_predicted_letter = ''

# Webcam capture
cap = cv2.VideoCapture(0) # Captures video from the default webcam.
while cap.isOpened(): # The loop continuously reads frames from the webcam.
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Converts the video frame from BGR (OpenCV’s default color space) to RGB (used by MediaPipe).
    result = hands.process(rgb_frame) #  Detects hand landmarks in the RGB frame.

    if result.multi_hand_landmarks: # If hand landmarks are detected, they are drawn on the video frame
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Draws landmarks and connections between them.

            # Extract landmarks and make a prediction
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten() # Extracts hand landmarks (x, y, z coordinates) and flattens them into a 1D array. These are passed as input to the trained model.
            landmarks = np.expand_dims(landmarks, axis=0)
            prediction = model.predict(landmarks) # The model predicts the class label for the current hand pose.
            predicted_class = np.argmax(prediction) #  Returns the index of the class with the highest probability (the predicted ASL letter)

            # Map prediction to ASL character
            predicted_letter = class_names[predicted_class]

            # Process predicted letters, avoid repeating and exclude 'W'
            if predicted_letter != last_predicted_letter:  # Checks if the new prediction differs from the last one to avoid repeating letters.
                if predicted_letter == 'W' and sentence:
                    # Read the full sentence without spelling out 'W'
                    full_sentence = "".join(sentence).replace('  ', ' ').strip()
                    engine.say(f"The sentence is: {full_sentence}")
                    engine.runAndWait()
                    sentence = []  # Clear the sentence after reading
                elif predicted_letter == 'space':
                    sentence.append(' ')
                    engine.say("space")
                    engine.runAndWait()
                else:
                    # Add the letter to the sentence and spell it
                    sentence.append(predicted_letter)
                    engine.say(predicted_letter)  # Spell the letter
                    engine.runAndWait()

                # Update last predicted letter
                last_predicted_letter = predicted_letter

    # Display the sentence on the video frame
    cv2.putText(frame, "Sentence: " + "".join(sentence), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # Displays the current sentence being formed on the video frame.
    cv2.imshow('ASL Sign Language Detection', frame) # Shows the video feed with the recognized sentence on the screen.

    # Break the loop on 'q' press
    if cv2.waitKey(3000) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
