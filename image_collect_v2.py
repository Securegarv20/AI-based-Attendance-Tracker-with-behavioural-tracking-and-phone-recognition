import cv2
import os

def capture_images(person_name, output_path=r'C:\Users\garvk\Desktop\pcl_final\your_data_path'):
    cap = cv2.VideoCapture(0)

    # Check if the video capture has been initialized correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create a directory for the person if it doesn't exist
    person_path = os.path.join(output_path, person_name)
    os.makedirs(person_path, exist_ok=True)

    # Find the highest existing index and start counting from there
    existing_indices = [int(filename.split('_')[-1].split('.')[0]) for filename in os.listdir(person_path) if filename.endswith('.jpg')]
    count = max(existing_indices, default=-1) + 1

    # Capture images with glasses
    print("Capturing images with glasses...")
    while count < 15:  # Capture 15 images for people with glasses
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around face
                cv2.imwrite(os.path.join(person_path, f'{person_name}_{count:03d}_with_glasses.jpg'), face_roi)
                count += 1

        cv2.imshow('Capture Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Prompt user to remove glasses and wait for confirmation
    print("Please remove your glasses and press any key to continue capturing images without glasses.")
    cv2.waitKey(0)

    # Capture images without glasses
    print("Capturing images without glasses...")
    while count < 30:  # Capture next 15 images for people without glasses
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around face
                cv2.imwrite(os.path.join(person_path, f'{person_name}_{count:03d}_without_glasses.jpg'), face_roi)
                count += 1

        cv2.imshow('Capture Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Image capture completed.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    person_name = input("Enter your name: ")
    capture_images(person_name)
