import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class LoginDataPreprocessor:
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self, filepath):
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        # Error handling for possible missing values
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if 'id' not in col.lower():
                df[col] = df[col].fillna('Unknown')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def explore_data(self, df):
        print("\n=== DATA OVERVIEW ===")
        print(df.info())
        print("\n=== STATISTICAL SUMMARY ===")
        print(df.describe())
        
        if 'attack_detected' in df.columns:
            print("\n=== TARGET DISTRIBUTION ===")
            print(df['attack_detected'].value_counts())
            print(f"Attack rate: {df['attack_detected'].mean():.2%}")
        
        return df
    
    def engineer_features(self, df):
        df_copy = df.copy()
        
        # Risk score multiplier calculation
        df_copy['risk_score'] = (
            df_copy['failed_logins'] * 0.3 +
            df_copy['login_attempts'] * 0.2 +
            df_copy['unusual_time_access'] * 0.3 +
            (1 - df_copy['ip_reputation_score']) * 0.2
        )
        
        # High risk flag marker threshold
        df_copy['high_risk'] = (df_copy['risk_score'] > df_copy['risk_score'].quantile(0.75)).astype(int)
        
        # Encryption strength based on type
        encryption_map = {'None': 0, 'Unknown': 0, 'DES': 1, 'AES': 2}
        df_copy['encryption_strength'] = df_copy['encryption_used'].map(encryption_map).fillna(0)
        
        # Suspicious browser flag
        df_copy['suspicious_browser'] = (df_copy['browser_type'] == 'Unknown').astype(int)
        
        print(f"Features engineered: {df_copy.shape}")
        return df_copy
    
    def encode_categorical(self, df, categorical_columns):
        df_copy = df.copy()
        
        for col in categorical_columns:
            if col in df_copy.columns:
                le = LabelEncoder()
                df_copy[col + '_encoded'] = le.fit_transform(df_copy[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}: {len(le.classes_)} categories")
        
        return df_copy
    
    def prepare_features(self, df, target_column='attack_detected'):
        df_copy = df.copy()
        
        # Exclude columns not being used as features
        exclude_cols = ['session_id', target_column, 'encryption_used', 'browser_type', 'protocol_type']
        self.feature_columns = [col for col in df_copy.columns if col not in exclude_cols]
        
        X = df_copy[self.feature_columns]
        y = df_copy[target_column] if target_column in df_copy.columns else None
        
        print(f"Features: {X.shape[1]} columns, {X.shape[0]} rows")
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        X_train = X_train.fillna(0)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test = X_test.fillna(0)
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def preprocess_pipeline(self, filepath, test_size=0.2):
        print("\n" + "="*70)
        print("STARTING DATA PREPROCESSING")
        print("="*70)
        
        df = self.load_data(filepath)
        df = self.explore_data(df)
        df = self.engineer_features(df)
        
        categorical_cols = ['protocol_type', 'encryption_used', 'browser_type']
        df = self.encode_categorical(df, categorical_cols)
        
        X, y = self.prepare_features(df)
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        else:
            X_train, X_test = X, None
            y_train, y_test = None, None
            print("No labels found - unsupervised mode")
        
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETED")
        print("="*70 + "\n")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'df': df,
            'feature_columns': self.feature_columns
        }