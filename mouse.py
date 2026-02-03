import cv2
import numpy as np
try:
    import pyautogui
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "pyautogui"])
    import pyautogui
import time
from cvzone.HandTrackingModule import HandDetector

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Hand detector
detector = HandDetector(detectionCon=0.7, maxHands=1)

# Screen size
screen_w, screen_h = pyautogui.size()

# Variables
prev_y = 0
click_time = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)

        x1, y1 = lmList[8][0], lmList[8][1]

        # ================= SCROLL (3 fingers) =================
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            if prev_y != 0:
                diff = prev_y - y1

                if abs(diff) > 20:
                    pyautogui.scroll(int(diff * 2))

            prev_y = y1
            cv2.putText(img, "Scrolling", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # ================= LEFT CLICK (2 fingers close) =================
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
            x2, y2 = lmList[12][0], lmList[12][1]
            length = np.hypot(x2 - x1, y2 - y1)

            if length < 40 and time.time() - click_time > 0.5:
                pyautogui.click()
                click_time = time.time()
                cv2.putText(img, "Click", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ================= MOVE (Index only) =================
        elif fingers[1] == 1 and fingers[2] == 0:
            mouse_x = np.interp(x1, (0, 640), (0, screen_w))
            mouse_y = np.interp(y1, (0, 480), (0, screen_h))

            pyautogui.moveTo(mouse_x, mouse_y)
            prev_y = 0

            cv2.putText(img, "Moving", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        prev_y = 0

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
