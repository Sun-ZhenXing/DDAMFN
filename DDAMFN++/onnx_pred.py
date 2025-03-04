import time

import cv2
import numpy as np
import onnxruntime
from PIL import Image
from torchvision import transforms


def inference(onnx_model_path: str, image_path: str):
    # Load ONNX model
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    time1 = time.time()
    # Preprocess input image
    image_transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).numpy()
    # Perform inference
    ort_inputs = {ort_session.get_inputs()[0].name: image_tensor}
    ort_outs = ort_session.run(None, ort_inputs)
    logits = np.squeeze(ort_outs[0])
    # Apply softmax
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=0)
    time2 = time.time()
    print(f"Inference time: {time2 - time1} seconds")
    return probabilities


def predict_video(model_path: str):
    # Open video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Load ONNX model
    ort_session = onnxruntime.InferenceSession(model_path)

    image_transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Emotion labels
    emotion_labels = [
        "Neutral",
        "Happiness",
        "Sadness",
        "Surprise",
        "Fear",
        "Disgust",
        "Anger",
        "Contempt",
    ]

    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess
        image_tensor = image_transform(pil_image).unsqueeze(0).numpy()

        # Inference
        ort_inputs = {ort_session.get_inputs()[0].name: image_tensor}
        start_time = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        inference_time = time.time() - start_time

        # Process results
        logits = np.squeeze(ort_outs[0])
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=0)
        predicted_class = np.argmax(probabilities)

        # Display results on frame
        cv2.putText(
            frame,
            f"Emotion: {emotion_labels[predicted_class]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Confidence: {probabilities[predicted_class]:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Time: {inference_time*1000:.0f}ms",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # Display frame
        cv2.imshow("Emotion Recognition", frame)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:  # 27 is ESC key
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def test():
    res = inference(
        "./checkpoints_ver2.0/affecnet8_epoch25_acc0.6469.onnx", "./image0000033.jpg"
    )
    print(res)


if __name__ == "__main__":
    test()
