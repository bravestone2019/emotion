
import cv2
import torch
import torch.nn.functional as F

from pathlib import Path

try:
    from .model import EmotionCNN
except ImportError:
    from model import EmotionCNN


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "emotion_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================
# LOAD MODEL
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

classes = checkpoint["classes"]

model = EmotionCNN(
    num_classes=len(classes)
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Emotion model loaded!")
print("Device:", DEVICE)
print("Classes:", classes)


# ==========================================
# LOAD FACE DETECTOR
# ==========================================

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load face detector")


# ==========================================
# PREDICT FACE EMOTION
# ==========================================

def predict_face(face):

    # Convert to grayscale
    gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    # Resize to model input size
    resized = cv2.resize(
        gray,
        (48, 48)
    )

    # Convert to tensor
    tensor = torch.from_numpy(
        resized.copy()
    ).float() / 255.0

    # Normalize exactly as training inference
    tensor = (tensor - 0.5) / 0.5

    tensor = tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        outputs = model(tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )[0]

    predicted_index = probabilities.argmax().item()

    emotion = classes[predicted_index]

    confidence = probabilities[predicted_index].item() * 100

    return emotion, confidence, probabilities


# ==========================================
# PROCESS IMAGE
# ==========================================

def analyze_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("\nFaces detected:", len(faces))

    for (x, y, w, h) in faces:

        face = image[y:y+h, x:x+w]

        emotion, confidence, probabilities = predict_face(face)

        label = f"{emotion}: {confidence:.1f}%"

        print(label)

        # Draw face rectangle
        cv2.rectangle(
            image,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Draw emotion label
        cv2.putText(
            image,
            label,
            (x, max(y-10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Save result
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "emotion_prediction.jpg"

    cv2.imwrite(str(output_path), image)

    print("\nResult saved:", output_path)

    return image


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    image_path = ROOT / "sample_face.jpg"

    result = analyze_image(image_path)

    cv2.imshow(
        "Facial Emotion Recognition",
        result
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
