import cv2
import os
import numpy as np
from datetime import datetime

def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    return cap

def detect_face_and_eyes(image, face_cascade, eye_cascade, face_recognizer, label_dict):
    eyes_open = False
    person_name = "Unknown"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face_roi)

        if confidence < 100:
            person_name = label_dict[label]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

            if len(eyes) == 0:  # No eyes detected
                eyes_open = False
            else:
                # Check each detected eye for their state
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                    aspect_ratio = ew / eh  # Calculate aspect ratio

                    # Adjust the aspect ratio threshold to improve detection accuracy
                    if aspect_ratio < 0.2:  # Assuming closed eyes have a smaller aspect ratio
                        eyes_open = False
                    else:
                        eyes_open = True

    return person_name, eyes_open, len(faces) > 0

def train_face_recognizer(data_path):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels, label_dict = get_images_and_labels(data_path)
    face_recognizer.train(faces, labels)
    return face_recognizer, label_dict

def get_images_and_labels(data_path):
    face_images = []
    labels = []
    label_dict = {}
    label_count = 0

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

def create_log_folder(person_name):
    log_path = os.path.join('log_data', person_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    return log_path

def update_log(log_path, event):
    log_file = os.path.join(log_path, 'log.txt')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"{timestamp}: {event}\n")

def main():
    cap = open_camera()
    if cap is None:
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    face_recognizer, label_dict = train_face_recognizer(r'your_data_path')

    current_person = None
    current_person_entered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        person_name, eyes_open, face_detected = detect_face_and_eyes(frame, face_cascade, eye_cascade, face_recognizer, label_dict)

        if person_name != current_person:
            if person_name != "Unknown":
                if not current_person_entered:
                    current_person_log_path = create_log_folder(person_name)
                    update_log(current_person_log_path, "Entered the class.")
                    current_person_entered = True
                current_person = person_name
            else:
                if current_person_entered:
                    update_log(current_person_log_path, "Not fully in frame (head down or turned).")
                current_person = None

        if current_person_entered:
            if face_detected:
                if eyes_open:
                    update_log(current_person_log_path, "Still in class, eyes open.")
                else:
                    update_log(current_person_log_path, "Still in class, eyes closed (distracted).")
            else:
                update_log(current_person_log_path, "Not fully in frame (head down or turned).")

        cv2.putText(frame, f"Eyes: {'Open' if eyes_open else 'Closed'}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if eyes_open else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{person_name}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
