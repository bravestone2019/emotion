
import cv2
from pathlib import Path


# Load OpenCV Haar Cascade face detector
CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise RuntimeError("Could not load Haar Cascade detector")


def detect_faces(image_path):

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    # Read image
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not read image")

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("Faces detected:", len(faces))

    # Draw bounding boxes
    for (x, y, w, h) in faces:

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Crop detected face
        face_crop = gray[y:y+h, x:x+w]

        # Resize to CNN input dimensions
        face_resized = cv2.resize(
            face_crop,
            (48, 48)
        )

    # Save output
    output_dir = image_path.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "detected_faces.jpg"

    cv2.imwrite(str(output_path), image)

    print("Output saved:", output_path)

    return faces, image


if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parent.parent

    image_path = ROOT / "sample_face.jpg"

    detect_faces(image_path)
