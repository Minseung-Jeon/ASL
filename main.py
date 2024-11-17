import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

#Load the trained ASL model
model = tf.keras.models.load_model('asl_model.h5')

#access hands module within mediapipe
mp_hands = mp.solutions.hands
#initialize hand tracking model
hands = mp_hands.Hands()
#accessing drawing utils (provides functions for drawing landmarks)
mp_drawing = mp.solutions.drawing_utils

#initialize video capture object. 0 refers to the default webcam
cap =cv2.VideoCapture(0)


#################################################
def extract_hand_region(image, hand_landmarks):
    # Get bounding box coordinates for the hand
    x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * image.shape[1])
    x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * image.shape[1])
    y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * image.shape[0])
    y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * image.shape[0])

    # Add some padding to the bounding box
    padding = 20
    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0], y_max + padding)

      # Extract hand region from the image
    hand_region = image[y_min:y_max, x_min:x_max]

    # Convert hand region to grayscale
    hand_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

    return hand_region
#################################################


# starts a loop that continues until pressing 'q' key
while(True):
    # Capture frame-by-frame
    # ret is a boolean indicating if the frame was captured correctly
    # frame hold the image data
    ret, frame = cap.read()

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    #converts the frame from BGR to RGB
    #OpenCV uses BGR color space, but mediapipe uses RGB color space
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # checks if any hand landmarks were detected in the frame
    if results.multi_hand_landmarks:
        #loops through set of landmarks detected in the frame (in case multiple hands are detected)
        for hand_landmarks in results.multi_hand_landmarks:
            #draws hand landmarks on the frame using the drawing utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            #################################################
            # Preprocess hand region for prediction
            hand_region = extract_hand_region(frame, hand_landmarks)  # You'll need to define this function
            resized_hand = cv2.resize(hand_region, (128, 128))
            normalized_hand = resized_hand / 255.0
            input_data = np.expand_dims(normalized_hand, axis=0)

            # Make prediction using the loaded model
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)

            # Display the predicted class (ASL letter) on the frame
            cv2.putText(frame, f"Predicted: {chr(65 + predicted_class)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #################################################


    # Display the resulting frame in a window called "Webcam Feed"
    cv2.imshow('Hand Tracking', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()