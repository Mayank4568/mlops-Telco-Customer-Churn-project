import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sqlalchemy.orm import Session

from .. import crud
from ..database import SessionLocal

# Define features
CATEGORICAL_COLS = [
    'gender', 'partner', 'dependents', 'phone_service', 
    'internet_service', 'online_security', 'tech_support', 
    'paperless_billing', 'payment_method'
]
NUMERICAL_COLS = ['senior_citizen', 'tenure', 'monthly_charges', 'total_charges']
TARGET_COL = 'churn'

def get_training_data_from_db(db: Session) -> pd.DataFrame:
    """Retrieve all customer records with non-null churn outcomes from DB."""
    records = crud.get_labeled_customers(db)
    if not records:
        return pd.DataFrame()
        
    data = []
    for r in records:
        data.append({
            'gender': r.gender,
            'senior_citizen': r.senior_citizen,
            'partner': r.partner,
            'dependents': r.dependents,
            'tenure': r.tenure,
            'phone_service': r.phone_service,
            'internet_service': r.internet_service,
            'online_security': r.online_security,
            'tech_support': r.tech_support,
            'paperless_billing': r.paperless_billing,
            'payment_method': r.payment_method,
            'monthly_charges': r.monthly_charges,
            'total_charges': r.total_charges,
            'churn': r.churn
        })
    return pd.DataFrame(data)

def train_and_evaluate(db: Session) -> dict:
    """Trains a new model on labeled DB records and updates metadata."""
    df = get_training_data_from_db(db)
    if len(df) < 10:  # Require at least 10 records to train
        raise ValueError(f"Insufficient training data. Found {len(df)} records. Need at least 10.")
        
    # Split features and target
    X = df[CATEGORICAL_COLS + NUMERICAL_COLS]
    y = df[TARGET_COL]
    
    # Train-test split
    # If the dataset is too small, use stratify conditionally
    if y.value_counts().min() > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    # Build preprocessing and model pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=2))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict & Evaluate
    y_pred = pipeline.predict(X_test)
    
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    
    # Determine new version
    latest_version = crud.get_latest_model_version(db)
    new_version = latest_version + 1
    
    # Directory to store models
    models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_filename = f"model_v{new_version}.pkl"
    model_path = os.path.join(models_dir, model_filename)
    
    # Save model pipeline
    joblib.dump(pipeline, model_path)
    
    # Register in DB
    metadata = crud.create_model_metadata(
        db=db,
        version=new_version,
        accuracy=accuracy,
        f1_score=f1,
        recall=recall,
        precision=precision,
        dataset_size=len(df),
        model_path=model_path
    )
    
    # Update Prometheus gauge metrics (if registered/imported dynamically)
    try:
        from ..metrics import set_model_metrics
        set_model_metrics(new_version, accuracy, f1, precision, recall, len(df))
    except Exception as e:
        print(f"Failed to update metrics: {e}")
        
    return {
        "version": new_version,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "dataset_size": len(df),
        "model_path": model_path
    }
