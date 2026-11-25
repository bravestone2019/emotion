from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import cv2
import torch
import torch.nn.functional as F
import numpy as np

from collections import deque

from model import EmotionCNN

# Project paths
MODEL_PATH = ROOT / "models" / "emotion_model_weighted.pth"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Load trained model
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

classes = checkpoint["classes"]

model = EmotionCNN(num_classes=len(classes)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Face detector
cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load Haar Cascade face detector.")

# Start webcam (CAP_DSHOW is useful on Windows)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open webcam with camera index 0. Check camera permissions, "
        "close other apps using the camera, or try camera index 1."
    )

print("Webcam started. Press Q to quit.")

history = deque(maxlen=10)

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        # Mirror the webcam view
        frame = cv2.flip(frame, 1)

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            # Crop detected face
            m = int(0.1 * w)
            y1, y2 = max(y - m, 0), min(y + h + m, gray.shape[0])
            x1, x2 = max(x - m, 0), min(x + w + m, gray.shape[1])
            face = gray[y1:y2, x1:x2]

            # Match model's expected input preprocessing
            face = cv2.resize(face, (48, 48))
            face = face.astype(np.float32) / 255.0
            face = (face - 0.5) / 0.5

            # Shape: [batch, channel, height, width]
            face_tensor = torch.from_numpy(face)
            face_tensor = face_tensor.unsqueeze(0).unsqueeze(0)
            face_tensor = face_tensor.to(DEVICE)

            # Predict emotion
            with torch.inference_mode():
                outputs = model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]

                history.append(probabilities.cpu().numpy())
                avg_probs = np.mean(history, axis=0)
                predicted_index = int(np.argmax(avg_probs))
                emotion = classes[predicted_index]
                score = avg_probs[predicted_index] * 100

            # Draw face box and prediction
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            label = f"{emotion}: {score:.1f}%"

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Display status
        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Real-Time Emotion Recognition", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
