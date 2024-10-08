import cv2
import os
import numpy as np

def open_camera():
    # Open the default camera (camera index 0)
    cap = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    
    return cap

def detect_face_and_eyes(image, face_cascade, eye_cascade, face_recognizer, label_dict):
    eyes_open = False  # Flag to track if eyes are open
    person_name = "Unknown"  # Default name if no face is recognized

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Recognize faces using the face recognizer
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face_roi)

        # If the confidence is below a certain threshold, consider it a match
        if confidence < 100:
            person_name = label_dict[label]
            cv2.putText(image, f"Person: {person_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Region of Interest (ROI) for eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(100, 100))

        if len(eyes) == 0:  # No eyes detected
            eyes_open = False
        else:
            for (ex, ey, ew, eh) in eyes:
                # Draw a rectangle around the detected eyes
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

                # Check if the eyes are closed based on the aspect ratio
                aspect_ratio = ew / eh
                if aspect_ratio < 0.2:  # Assuming closed eyes have a smaller aspect ratio
                    eyes_open = False
                else:
                    eyes_open = True

    return person_name, eyes_open


def train_face_recognizer(data_path):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Get the face images and corresponding labels
    faces, labels, label_dict = get_images_and_labels(data_path)

    # Train the face recognizer
    face_recognizer.train(faces, labels)

    return face_recognizer, label_dict

def get_images_and_labels(data_path):
    face_images = []
    labels = []
    label_dict = {}  # Map person names to labels
    label_count = 0

    # Get the directories (one directory for each person)
    for person_name in os.listdir(data_path):
        person_path = os.path.join(data_path, person_name)
        
        if os.path.isdir(person_path):
            label_dict[label_count] = person_name
            for filename in os.listdir(person_path):
                img_path = os.path.join(person_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                face_images.append(img)
                labels.append(label_count)
            label_count += 1
    
    return face_images, np.array(labels), label_dict

def main():
    # Open the camera
    cap = open_camera()

    if cap is None:
        return

    # Load pre-trained face and eye cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Load your trained face recognizer
    face_recognizer, label_dict = train_face_recognizer(r'your_data_path')

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # If the frame is not captured, break the loop
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect and recognize faces and eyes in the frame
        person_name, eyes_open = detect_face_and_eyes(frame, face_cascade, eye_cascade, face_recognizer, label_dict)

        # Display the status of eyes (open or closed) and the recognized person's name on the screen
        cv2.putText(frame, f"Eyes: {'Open' if eyes_open else 'Closed'}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, f"Person: {person_name}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
