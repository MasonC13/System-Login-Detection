import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os


class AnomalyDetector:
    
    def __init__(self, model_type='isolation_forest', contamination=0.1):
        self.model_type = model_type
        self.contamination = contamination
        self.model = None
        self.initialize_model()
        
    def initialize_model(self):
        if self.model_type == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.model_type == 'one_class_svm':
            self.model = OneClassSVM(nu=self.contamination, kernel='rbf')
        elif self.model_type == 'lof':
            self.model = LocalOutlierFactor(contamination=self.contamination, novelty=True)
        else:
            raise ValueError(f"Unknown model: {self.model_type}")
    
    def train(self, X_train):
        print(f"Training {self.model_type}...")
        self.model.fit(X_train)
        print("Training completed!")
        return self
    
    def predict(self, X):
        predictions = self.model.predict(X)
        anomalies = (predictions == -1).astype(int)
        return anomalies
    
    def get_anomaly_scores(self, X):
        if hasattr(self.model, 'score_samples'):
            return self.model.score_samples(X)
        elif hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X)
        return None
    
    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved: {filepath}")
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded: {filepath}")


class SupervisedDetector:
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
    def train(self, X_train, y_train):
        print("Training Random Forest...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, feature_names):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance_df


class ModelEvaluator:
    
    @staticmethod
    def evaluate_predictions(y_true, y_pred, model_name="Model"):
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS - {model_name}")
        print('='*70)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack']))
        
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nMetrics:")
        print(f"  Accuracy:  {accuracy:.2%}")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall:    {recall:.2%}")
        print(f"  F1-Score:  {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }