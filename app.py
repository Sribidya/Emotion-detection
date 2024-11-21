import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model 
model = load_model('models/emotion_recognition_model_2.h5')

# Define class names 
class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def preprocess_face(face_roi):
    """
    Preprocess a face ROI for emotion detection.
    Args:
        face_roi (numpy.ndarray): The Region of Interest (ROI) of the face.
    Returns:
        tf.Tensor: Preprocessed face tensor ready for model input.
    """
    # Convert to RGB from BGR
    if face_roi.shape[-1] == 3:
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

    # Resize the face to the model's expected input size
    face_resized = cv2.resize(face_roi, (224, 224))

    # Convert the resized image to a tensor
    face_tensor = tf.convert_to_tensor(face_resized, dtype=tf.float32)

    # Add a batch dimension
    face_tensor = tf.expand_dims(face_tensor, axis=0)

    # Preprocess using ResNet50's preprocessing function
    face_tensor = tf.keras.applications.resnet50.preprocess_input(face_tensor)

    return face_tensor

def get_emotion(face_roi):
    
    face_tensor = preprocess_face(face_roi)
    
    # Predict
    predictions = model.predict(face_tensor)
    predicted_class = np.argmax(predictions)
    emotion = class_names[predicted_class]
    
    # Print for dubugging
    print(f"Predictions: {predictions}")
    print(f"Predicted Class Index: {predicted_class}")
    print(f"Predicted Emotion: {emotion}")
    
    return emotion


def detect_and_display_emotions(frame):
    """
    Detects faces in the given frame and displays their emotions.

    Args:
        frame (numpy.ndarray): The current video frame.
    """
    # Convert the frame to grayscal
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load a pre-trained Haar Cascade 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = frame[y:y + h, x:x + w]
        
        # Predict the emotion
        emotion = get_emotion(face_roi)
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Display the emotion on the frame
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 0, 255), 4)
    
    # Show the frame with emotions displayed
    cv2.imshow('Camera', frame)

def main():
    """
    Captures video from the webcam and performs real-time emotion detection.
    """
    # Initialize the video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video frame.")
            break
        # Rotatate frame horizionaly
        frame = cv2.flip(frame, 1)
        # Detect and display emotions on the current frame
        detect_and_display_emotions(frame)

        # Exit the program when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

