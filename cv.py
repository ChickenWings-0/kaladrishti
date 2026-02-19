import cv2
import mediapipe as mp

# Load reference image
pataka_img = cv2.imread("pataka.jpg")

# Resize image (adjust size if needed)
pataka_img = cv2.resize(pataka_img, (200, 200))

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

def detect_gesture(landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]

    fingers = []

    # Thumb
    if landmarks[4].y < landmarks[3].y:
        thumb_open = 1
    else:
        thumb_open = 0

    # Other fingers
    for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    if fingers == [1, 1, 1, 1] and thumb_open == 1:
        return "PATAKA"
    elif fingers == [0, 0, 0, 0] and thumb_open == 1:
        return "SHIKHARA"
    else:
        return "UNKNOWN"


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    gesture = "UNKNOWN"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            gesture = detect_gesture(hand_landmarks.landmark)

    # Show text
    cv2.putText(frame, gesture, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # ✅ SHOW IMAGE ONLY WHEN PATAKA
    if gesture == "PATAKA":
        h, w, _ = pataka_img.shape

        # place image top-right corner
        frame[10:10+h, frame.shape[1]-w-10:frame.shape[1]-10] = pataka_img

    cv2.imshow("Hand Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
