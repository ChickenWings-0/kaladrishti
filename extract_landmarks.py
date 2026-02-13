# extract_landmarks.py
# Yeh file sirf ek baar run karni hai – reference JSON banane ke liye

import cv2
import mediapipe as mp
import json

mp_holistic = mp.solutions.holistic

# Video ka path – yeh tera media folder ke andar hai
VIDEO_PATH = "media/vid/BH1.mp4"           # agar naam alag hai to change kar dena
OUTPUT_JSON = "bhumi_reference.json"       # yeh file yahin banegi

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Video file nahi mila! Path check kar:", VIDEO_PATH)
    exit()

frame_count = 0
reference_data = []

print("Video processing shuru...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = holistic.process(image)

    frame_data = {
        "frame": frame_count,
        "time": round(frame_count / cap.get(cv2.CAP_PROP_FPS), 2),  # seconds mein
        "pose": [],
        "left_hand": [],
        "right_hand": []
    }

    # Pose landmarks
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            frame_data["pose"].append({
                "x": round(lm.x, 4),
                "y": round(lm.y, 4),
                "z": round(lm.z, 4),
                "visibility": round(lm.visibility, 4)
            })

    # Left hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            frame_data["left_hand"].append({
                "x": round(lm.x, 4),
                "y": round(lm.y, 4),
                "z": round(lm.z, 4)
            })

    # Right hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            frame_data["right_hand"].append({
                "x": round(lm.x, 4),
                "y": round(lm.y, 4),
                "z": round(lm.z, 4)
            })

    reference_data.append(frame_data)

    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
holistic.close()

# JSON file save kar do
with open(OUTPUT_JSON, 'w') as f:
    json.dump(reference_data, f, indent=2)

print(f"\nDone! {frame_count} frames ka data save ho gaya -> {OUTPUT_JSON}")
print("File size:", os.path.getsize(OUTPUT_JSON), "bytes")