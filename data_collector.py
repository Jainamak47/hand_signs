import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime

class HandSignDataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.data_dir = "training_data"
        self.create_directories()
        
        self.current_label = ""
        self.samples_collected = 0
        self.target_samples = 100
        
    def create_directories(self):
        """Create necessary directories for data storage"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        subdirs = ['images', 'landmarks', 'annotations']
        for subdir in subdirs:
            path = os.path.join(self.data_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
    
    def extract_features(self, landmarks):
        """Extract the same advanced features as the main detector"""
        if not landmarks:
            return None
        
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        wrist = points[0]
        normalized_points = points - wrist
        
        distances = np.linalg.norm(normalized_points, axis=1)
        
        angles = []
        for i in range(len(points)-1):
            v1 = points[i+1] - points[i]
            v2 = points[0] - points[i]
            angle = np.arccos(np.clip(np.dot(v1, v2) / 
                                    (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
            angles.append(angle)
        
        fingertips = [4, 8, 12, 16, 20]
        palm_center = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
        tip_distances = [np.linalg.norm(points[tip] - palm_center) for tip in fingertips]
        
        features = np.concatenate([
            distances,
            angles,
            tip_distances,
            normalized_points.flatten()[:38]
        ])
        
        return features
    
    def collect_data(self):
        """Data collection interface"""
        cap = cv2.VideoCapture(0)
        collecting = False
        
        print("Hand Sign Data Collector")
        print("Instructions:")
        print("1. Press letter keys (A-Z) to set current label")
        print("2. Press SPACE to start/stop collecting")
        print("3. Press 'q' to quit")
        print("4. Press 'r' to reset counter")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Display info
            cv2.putText(frame, f"Label: {self.current_label}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {self.samples_collected}/{self.target_samples}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Collecting: {'ON' if collecting else 'OFF'}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if collecting else (0, 0, 255), 2)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    if collecting and self.current_label and self.samples_collected < self.target_samples:
                        # Save data
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        
                        # Save image
                        img_path = os.path.join(self.data_dir, 'images', 
                                              f"{self.current_label}_{timestamp}.jpg")
                        cv2.imwrite(img_path, frame)
                        
                        # Save landmarks
                        features = self.extract_features(hand_landmarks)
                        if features is not None:
                            landmark_path = os.path.join(self.data_dir, 'landmarks', 
                                                       f"{self.current_label}_{timestamp}.npy")
                            np.save(landmark_path, features)
                            
                            # Save annotation
                            annotation = {
                                'label': self.current_label,
                                'timestamp': timestamp,
                                'image_path': img_path,
                                'landmark_path': landmark_path,
                                'features_shape': features.shape
                            }
                            
                            ann_path = os.path.join(self.data_dir, 'annotations', 
                                                  f"{self.current_label}_{timestamp}.json")
                            with open(ann_path, 'w') as f:
                                json.dump(annotation, f)
                            
                            self.samples_collected += 1
                            print(f"Saved sample {self.samples_collected} for {self.current_label}")
            
            cv2.imshow('Data Collector', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                collecting = not collecting
                if collecting:
                    print(f"Started collecting for {self.current_label}")
                else:
                    print("Stopped collecting")
            elif key == ord('r'):
                self.samples_collected = 0
                print("Reset sample counter")
            elif key >= ord('a') and key <= ord('z'):
                self.current_label = chr(key).upper()
                self.samples_collected = 0
                print(f"Set label to: {self.current_label}")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = HandSignDataCollector()
    collector.collect_data()