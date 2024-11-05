import cv2
def get_emotion():
    pass

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# Open the default camera
cam = cv2.VideoCapture(1)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()
    if not ret:
        break
    # Write the frame to the output file
    #out.write(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
   
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        # face_roi = rgb_frame[y:y + h, x:x + w]
        # Detect emotion
        # emotion = get_emotion(face_roi)
        # Draw rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Type emotion
        # cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # Display the captured frame
        cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
