import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import csv
import os

# ==============================
# Settings
# ==============================
SAVE_PATH = "dataset.csv"
TARGET_SAMPLES = 200

# Only capture remaining letters
VALID_LABELS = ['q', 'j', 'z']

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
# CSV setup (append mode)
# ==============================
file_exists = os.path.exists(SAVE_PATH)
file = open(SAVE_PATH, "a", newline="")
writer = csv.writer(file)

# write header if file new
if not file_exists:
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    writer.writerow(header)

# ==============================
# Camera
# ==============================
cap = cv2.VideoCapture(0)
timestamp = 0

current_label = None
count = 0

print("Press Q, J, Z to capture those signs")
print("Press X to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 1

    if result.hand_landmarks and current_label:
        hand = result.hand_landmarks[0]
        row = []

        for lm in hand:
            row.extend([lm.x, lm.y, lm.z])

        row.append(current_label)
        writer.writerow(row)
        count += 1

    # display info
    text = f"Label: {current_label}  Count: {count}/{TARGET_SAMPLES}"
    cv2.putText(frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Dataset Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('x'):
        break

    # start capture if q, j, or z pressed
    if chr(key) in VALID_LABELS:
        current_label = chr(key).upper()
        count = 0
        print(f"Collecting {current_label}")

    if count >= TARGET_SAMPLES:
        print(f"{current_label} DONE ✅")
        current_label = None
        count = 0

cap.release()
file.close()
cv2.destroyAllWindows()
