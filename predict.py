import cv2
import mediapipe as mp
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# ==============================
# Load trained model
# ==============================
model = load_model("asl_sign_model.h5")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

# ==============================
# MediaPipe setup
# ==============================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# ==============================
# Hand connections
# ==============================
connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# ==============================
# Prediction smoothing buffer
# ==============================
pred_buffer = deque(maxlen=5)

# ==============================
# Camera
# ==============================
cap = cv2.VideoCapture(0)
timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 1

    label = ""
    confidence = 0

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape

        points = []
        features = []

        for lm in hand:
            x, y = int(lm.x*w), int(lm.y*h)
            points.append((x, y))
            features.extend([lm.x, lm.y, lm.z])

        # ===== Prediction =====
        X = np.array(features).reshape(1, -1)
        X = scaler.transform(X)

        preds = model.predict(X, verbose=0)
        pred_class = np.argmax(preds, axis=1)[0]
        confidence = np.max(preds)

        pred_buffer.append(pred_class)
        smooth_pred = max(set(pred_buffer), key=pred_buffer.count)

        label = encoder.inverse_transform([smooth_pred])[0]

        # ===== Draw skeleton =====
        for start, end in connections:
            cv2.line(frame, points[start], points[end], (200,200,200), 2)

        for (x, y) in points:
            cv2.circle(frame, (x, y), 6, (255,255,255), -1)

    # ===== UI =====
    cv2.rectangle(frame, (10,10), (300,90), (0,0,0), -1)
    cv2.putText(frame, f"Sign: {label}", (20,55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.putText(frame, f"Conf: {confidence:.2f}", (20,85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("ASL AI Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
