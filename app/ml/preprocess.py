import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df: pd.DataFrame, is_training: bool = True):
    # Create a copy to avoid SettingWithCopyWarning
    data = df.copy()
    
    # Drop customerID if it exists and we are training (it's not a feature)
    if 'customerID' in data.columns:
        data = data.drop('customerID', axis=1)
        
    # Handle TotalCharges (convert to numeric, replace empty strings with median)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
    
    # Binary encoding and Label Encoding for categorical features
    # In a real production app, you'd save the encoders/transformers.
    # For this beginner project, we'll use a simple approach with Label Encoding.
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    le = LabelEncoder()
    for col in categorical_cols:
        # If 'Churn' is in columns and we are training, encode it.
        # If we are predicting, 'Churn' won't be there.
        data[col] = le.fit_transform(data[col].astype(str))
        
    return data
