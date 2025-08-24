import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Screen size
screen_w, screen_h = pyautogui.size()

# Variables
mode = "POINTER"
last_mode_switch = 0
mode_switch_cooldown = 1.5  # seconds
dragging = False
last_click_time = 0
click_debounce = 0.3  # seconds
prev_x, prev_y = None, None

def get_distance(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

def fingers_folded(hand_landmarks):
    # Check if all four fingers (excluding thumb) are folded (tips below bases)
    folded = all(
        hand_landmarks.landmark[tip].y > hand_landmarks.landmark[base].y
        for tip, base in [(8, 5), (12, 9), (16, 13), (20, 17)]
    )
    return folded

# Capture
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
            middle_finger = hand_landmarks.landmark[12]
            thumb = hand_landmarks.landmark[4]

            # Calculate pointer coordinates (clamped)
            x = min(max(int(index_finger.x * screen_w), 0), screen_w - 1)
            y = min(max(int(index_finger.y * screen_h), 0), screen_h - 1)

            # Smooth pointer movement
            if prev_x is not None and prev_y is not None:
                x = int(prev_x * 0.7 + x * 0.3)
                y = int(prev_y * 0.7 + y * 0.3)
            prev_x, prev_y = x, y

            if mode == "POINTER":
                pyautogui.moveTo(x, y)

                # Left click (thumb + index pinch) with debounce and no drag overlap
                if get_distance(thumb, index_finger) < 0.05 and (now - last_click_time) > click_debounce and not dragging:
                    pyautogui.click()
                    last_click_time = now

                # Right click (thumb + middle pinch) with debounce
                if get_distance(thumb, middle_finger) < 0.05 and (now - last_click_time) > click_debounce:
                    pyautogui.rightClick()
                    last_click_time = now

                # Drag logic with debounce
                if get_distance(thumb, index_finger) < 0.05 and not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                elif get_distance(thumb, index_finger) > 0.1 and dragging:
                    pyautogui.mouseUp()
                    dragging = False

            elif mode == "SCROLL":
                dist = get_distance(index_finger, middle_finger)
                if dist < 0.05:
                    pyautogui.scroll(-30)
                    time.sleep(0.1)  # prevent flood
                elif dist > 0.1:
                    pyautogui.scroll(30)
                    time.sleep(0.1)

            elif mode == "VOLUME":
                # Display volume percentage (visual only)
                vol = int((1 - index_finger.y) * 100)
                cv2.putText(frame, f"Volume: {vol}%", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            elif mode == "ZOOM":
                dist = get_distance(index_finger, thumb)
                if dist < 0.05:
                    pyautogui.hotkey("ctrl", "-")
                    time.sleep(0.3)  # prevent rapid repeat
                elif dist > 0.15:
                    pyautogui.hotkey("ctrl", "+")
                    time.sleep(0.3)

            # Mode switching by checking all fingers folded (fist)
            if fingers_folded(hand_landmarks) and (now - last_mode_switch > mode_switch_cooldown):
                modes = ["POINTER", "SCROLL", "VOLUME", "ZOOM"]
                mode = modes[(modes.index(mode) + 1) % len(modes)]
                last_mode_switch = now
                dragging = False  # Reset dragging on mode switch

    else:
        # Reset states if no hand detected
        dragging = False
        prev_x, prev_y = None, None

    cv2.putText(frame, f"Mode: {mode}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
