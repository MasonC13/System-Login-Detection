import sys
sys.path.append('.')

from data_preprocessing import LoginDataPreprocessor
from anomaly_models import AnomalyDetector, SupervisedDetector, ModelEvaluator
import pandas as pd


def main():
    print("\n" + "="*70)
    print("LOGIN ANOMALY DETECTION DEMO")
    print("="*70 + "\n")
    
    DATA_FILE = 'CybersecurityIntrusionData.csv'  # Dataset
    
    print(f"Loading data from: {DATA_FILE}\n")
    
    # Step 1: Preprocessing
    preprocessor = LoginDataPreprocessor()
    data = preprocessor.preprocess_pipeline(DATA_FILE, test_size=0.2)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Step 2: Train Isolation Forest
    print("\nTraining Isolation Forest...")
    contamination = 0.3  # Adjust: 0.1 = strict, 0.4 = lenient
    
    detector = AnomalyDetector('isolation_forest', contamination=contamination)
    detector.train(X_train)
    
    # Step 3: Predict
    predictions = detector.predict(X_test)
    
    # Step 4: Evaluate
    if y_test is not None:
        results = ModelEvaluator.evaluate_predictions(y_test, predictions, "Isolation Forest")
        
        anomaly_count = predictions.sum()
        true_positives = ((predictions == 1) & (y_test.values == 1)).sum()
        false_positives = ((predictions == 1) & (y_test.values == 0)).sum()
        
        print(f"\nDetection Summary:")
        print(f"  Total anomalies detected: {anomaly_count}/{len(predictions)}")
        print(f"  True positives: {true_positives}")
        print(f"  False positives: {false_positives}")
        print(f"  Accuracy: {results['accuracy']:.2%}")
    else:
        anomaly_count = predictions.sum()
        print(f"\nAnomalies detected: {anomaly_count}/{len(predictions)}")
    
    # Step 5: Train supervised baseline
    if y_train is not None:
        print("\nTraining Random Forest...")
        supervised = SupervisedDetector()
        supervised.train(X_train, y_train)
        sup_pred = supervised.predict(X_test)
        
        sup_results = ModelEvaluator.evaluate_predictions(y_test, sup_pred, "Random Forest")
        
        # Feature importance
        importance_df = supervised.get_feature_importance(data['feature_columns'])
        print(f"\nTop 5 Most Important Features:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Comparison
        print(f"\nModel Comparison:")
        comparison = pd.DataFrame({
            'Isolation Forest': results,
            'Random Forest': sup_results
        }).T
        print(comparison[['accuracy', 'precision', 'recall', 'f1']])
    
    # Save model
    detector.save_model('./trained_model.pkl')
    
    print(f"\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
