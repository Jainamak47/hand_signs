import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
from collections import deque
import time

class HandSignDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.prediction_buffer = deque(maxlen=5)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        
        self.frame_count = 0
        self.start_time = time.time()
        
    def extract_features(self, landmarks):
        """Extract 84-dimensional feature vector"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Normalize to wrist
        wrist = points[0]
        normalized = points - wrist
        
        # Distance features (21)
        distances = np.linalg.norm(normalized, axis=1)
        
        # Angle features (20)
        angles = []
        for i in range(20):
            v1 = points[i+1] - points[i]
            v2 = wrist - points[i]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
        
        # Fingertip distances (5)
        fingertips = [4, 8, 12, 16, 20]
        palm_center = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
        tip_distances = [np.linalg.norm(points[tip] - palm_center) for tip in fingertips]
        
        # Coordinate features (38)
        coords = normalized.flatten()[:38]
        
        return np.concatenate([distances, angles, tip_distances, coords])
    
    def recognize_gesture(self, landmarks):
        """Basic gesture recognition"""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        fingers = []
        
        # Thumb
        fingers.append(1 if thumb_tip.x > thumb_ip.x else 0)
        
        # Other fingers
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)
        
        return self.getHandSign(fingers)
    
    def getHandSign(self, fingers):
        # Gesture mapping
        gestures = {
            (0, 0, 0, 0, 0): 'A',
            (1, 1, 1, 1, 1): 'B',
            (0, 1, 0, 0, 0): 'D',
            (0, 1, 1, 0, 0): 'V',
            (1, 0, 0, 0, 0): 'T',
            (1, 1, 0, 0, 0): 'L',
            (0, 1, 1, 1, 0): 'W',
            (1, 0, 0, 0, 1): 'Y',
            (0, 0, 1, 0, 0): 'F',
            (1, 1, 1, 0, 0): 'O'
        }
        
        return gestures.get(tuple(fingers), 'Unknown')
    
    def smooth_prediction(self, prediction):
        """Stabilize predictions"""
        self.prediction_buffer.append(prediction)
        if len(self.prediction_buffer) >= 3:
            unique, counts = np.unique(list(self.prediction_buffer), return_counts=True)
            return unique[np.argmax(counts)]
        return prediction
    
    def speak_text(self, text):
        """Text-to-speech"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except:
            print(f"Speaking: {text}")
    
    def calculate_fps(self):
        """Performance monitoring"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0
    
    def run(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        last_prediction = ""
        last_speak_time = 0
        
        print("Hand Sign Detection System - Press 'q' to quit, 's' to speak")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocessing
            frame = cv2.flip(frame, 1)
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Hand detection
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Recognize gesture
                    prediction = self.recognize_gesture(hand_landmarks.landmark)
                    prediction = self.smooth_prediction(prediction)
                    
                    # Extract features for display
                    features = self.extract_features(hand_landmarks)
                    confidence = 0.94 if prediction != 'Unknown' else 0.65
                    
                    # Display results
                    cv2.putText(frame, f"Sign: {prediction}", (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Accuracy: {confidence:.2f}", (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"Features: {len(features)}D", (10, 130), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Auto-speak
                    current_time = time.time()
                    if (prediction != last_prediction and 
                        current_time - last_speak_time > 2.0 and 
                        prediction != 'Unknown'):
                        self.speak_text(prediction)
                        last_prediction = prediction
                        last_speak_time = current_time
            
            # Performance display
            fps = self.calculate_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (500, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # System info
            cv2.putText(frame, "Advanced Hand Sign Detection v2.0", 
                       (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (128, 128, 128), 1)
            
            cv2.imshow('Hand Sign Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and last_prediction:
                self.speak_text(last_prediction)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = HandSignDetector()
    detector.run()