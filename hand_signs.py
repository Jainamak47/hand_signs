import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

def detect_gesture(landmarks):
    # Check each finger individually
    fingers = []
    
    # Thumb
    if landmarks[4].x > landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Index finger
    if landmarks[8].y < landmarks[6].y:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Middle finger
    if landmarks[12].y < landmarks[10].y:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Ring finger
    if landmarks[16].y < landmarks[14].y:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Pinky
    if landmarks[20].y < landmarks[18].y:
        fingers.append(1)
    else:
        fingers.append(0)
    
    print(f"Fingers: {fingers}")  # Debug to see what we get
    
    # Your specific gestures
    if fingers == [0, 1, 1, 0, 0]:
        return "Peace ✌️"  # Peace
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up 👍"  # Thumbs Up
    elif fingers == [0, 0, 0, 0, 1]:
        return "Pinky 🤙"  # Pinky
    elif fingers == [1, 1, 0, 0, 1]:
        return "Rock On 🤘"  # Rock On
    elif fingers == [0, 1, 0, 0, 0]:
        return "Point ☝️"  # Point
    elif fingers == [1, 1, 1, 1, 1]:
        return "High Five ✋"  # High Five
    elif fingers == [0, 0, 0, 0, 0]:
        return "Fist ✊"  # Fist
    elif fingers == [1, 1, 0, 0, 0]:
        return "L Sign 🤟"  # L sign
    elif fingers == [0, 1, 0, 0, 1]:
        return "Spock 🖖"  # Spock/Live long
    else:
        return f"Unknown {fingers}"  # Show the pattern

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks.landmark)
            cv2.putText(frame, gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
    
    cv2.imshow('Hand Signs', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()