air writing -
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
drawing = False
prev_x, prev_y = None, None
last_pinch_time = 0
pinch_debounce = 0.4  # seconds

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    now = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * 640), int(index_finger.y * 480)

            thumb = hand_landmarks.landmark[4]
            dist = np.linalg.norm([index_finger.x - thumb.x, index_finger.y - thumb.y])

            # Pinch to toggle drawing (with debounce)
            if dist < 0.05 and (now - last_pinch_time) > pinch_debounce:
                drawing = not drawing
                prev_x, prev_y = None, None
                last_pinch_time = now

            if drawing:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 4)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

    # Combine frames
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.putText(combined, "Pinch to toggle drawing. Press 'c' to clear. 's' to save.", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Air Writing", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('c'):  # Clear canvas
        canvas[:] = 0
    elif key == ord('s'):  # Save drawing
        cv2.imwrite("drawing.png", canvas)
        print("Drawing saved as drawing.png")

cap.release()
cv2.destroyAllWindows()
