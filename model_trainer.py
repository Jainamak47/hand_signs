import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self, data_dir="training_data"):
        self.data_dir = data_dir
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
    def load_training_data(self):
        """Load all collected training data"""
        features_list = []
        labels_list = []
        
        annotations_dir = os.path.join(self.data_dir, 'annotations')
        landmarks_dir = os.path.join(self.data_dir, 'landmarks')
        
        if not os.path.exists(annotations_dir):
            print("No training data found!")
            return None, None
        
        for filename in os.listdir(annotations_dir):
            if filename.endswith('.json'):
                with open(os.path.join(annotations_dir, filename), 'r') as f:
                    annotation = json.load(f)
                
                landmark_file = os.path.basename(annotation['landmark_path'])
                landmark_path = os.path.join(landmarks_dir, landmark_file)
                
                if os.path.exists(landmark_path):
                    features = np.load(landmark_path)
                    features_list.append(features)
                    labels_list.append(annotation['label'])
        
        print(f"Loaded {len(features_list)} samples")
        return np.array(features_list), np.array(labels_list)
    
    def train_model(self):
        """Train the classification model"""
        X, y = self.load_training_data()
        
        if X is None or len(X) == 0:
            print("No training data available!")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Train model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        
        # Detailed report
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, test_pred)
        
        # Save model
        model_path = "trained_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {model_path}")
        
        return True
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
    
    def evaluate_model_performance(self):
        """Detailed performance analysis"""
        X, y = self.load_training_data()
        if X is None:
            return
        
        # Cross-validation
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 6))
            plt.title("Feature Importances")
            plt.bar(range(min(20, len(importances))), importances[indices[:20]])
            plt.xlabel("Feature Index")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.show()

if __name__ == "__main__":
    trainer = ModelTrainer()
    
    print("Hand Sign Detection Model Trainer")
    print("1. Make sure you have collected training data using data_collector.py")
    print("2. Training will begin automatically...")
    
    success = trainer.train_model()
    
    if success:
        print("\nModel training completed successfully!")
        print("You can now use the trained model with advanced_hand_detection.py")
        
        # Evaluate performance
        trainer.evaluate_model_performance()
    else:
        print("Training failed. Please collect training data first.")