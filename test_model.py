# Import necessary libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model from the .keras file
model = load_model("asl_alphabet_model.keras")

# Print the model summary to verify the input shape
model.summary()

# Check the model's expected input shape (update if the model requires a different shape)
expected_input_shape = model.input_shape  # E.g., (None, 200, 200, 3) or (None, 63)
print("Expected input shape:", expected_input_shape)

# Define image properties (same as used during training, adjust if needed)
img_height, img_width = 200, 200  # Image dimensions

# Create a label map (adjust this if needed based on your training class indices)
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: 'SPACE', 27: 'DELETE', 28: 'NOTHING'
}

# Initialize webcam for real-time prediction
cap = cv2.VideoCapture(0)  # Use default webcam (0)

print("Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Preprocess the frame for prediction
    img = cv2.resize(frame, (img_height, img_width))  # Resize frame to model input size
    img = img / 255.0  # Normalize pixel values to [0, 1]

    # Ensure the input shape matches the model's requirements
    if len(expected_input_shape) == 2:  # For example, (None, 63)
        img = img.flatten()  # Flatten the image if required
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 63)
    else:
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 200, 200, 3)

    # Make a prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability
    predicted_label = label_map[predicted_class]  # Map the index to the label

    # Display the prediction on the frame
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Translator", frame)  # Show the frame with the prediction

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
