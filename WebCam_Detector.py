import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam
webcam = cv2.VideoCapture(0)

# iterate forever over the frames:
while True:
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces For loop is used when multiple images are detected
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #
    cv2.imshow("Aashish's Face Detector", frame)

    # Wait here in the code and listen for a key press
    key = cv2.waitKey(10)

    # Stop if Q is pressed
    if key == 81 or key == 113:
        break



#print(face_coordinates)

print("Code Completed")
