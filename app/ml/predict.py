import os
import joblib
import pandas as pd
from sqlalchemy.orm import Session
from .. import crud

_LOADED_MODEL = None
_LOADED_VERSION = None

def get_loaded_model(db: Session):
    """Retrieve the cached active model, loading it if version has changed."""
    global _LOADED_MODEL, _LOADED_VERSION
    
    active_model_meta = crud.get_active_model(db)
    if not active_model_meta:
        raise ValueError("No active machine learning model found in the database. Please run retraining first.")
        
    version = active_model_meta.version
    path = active_model_meta.model_path
    
    if _LOADED_VERSION != version or _LOADED_MODEL is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file for version {version} not found at {path}")
        _LOADED_MODEL = joblib.load(path)
        _LOADED_VERSION = version
        print(f"Successfully loaded model version {version} from {path}")
        
    return _LOADED_MODEL, _LOADED_VERSION

def predict_churn_for_customer(db: Session, customer_id: str) -> dict:
    """Predicts churn for a customer by fetching their record and feeding it to the model."""
    customer = crud.get_customer_by_customer_id(db, customer_id)
    if not customer:
        raise ValueError(f"Customer with ID {customer_id} not found.")
        
    # Get model
    model, version = get_loaded_model(db)
    
    # Construct input dataframe
    input_data = pd.DataFrame([{
        'gender': customer.gender,
        'senior_citizen': customer.senior_citizen,
        'partner': customer.partner,
        'dependents': customer.dependents,
        'tenure': customer.tenure,
        'phone_service': customer.phone_service,
        'internet_service': customer.internet_service,
        'online_security': customer.online_security,
        'tech_support': customer.tech_support,
        'paperless_billing': customer.paperless_billing,
        'payment_method': customer.payment_method,
        'monthly_charges': customer.monthly_charges,
        'total_charges': customer.total_charges
    }])
    
    # Inference
    probabilities = model.predict_proba(input_data)[0]
    prediction = int(model.predict(input_data)[0])
    churn_probability = float(probabilities[1])  # Prob of churn = 1
    
    # Log prediction in DB for tracking/drift detection
    crud.log_prediction(
        db=db,
        customer_id=customer_id,
        predicted_churn=prediction,
        probability=churn_probability,
        model_version=version
    )
    
    return {
        "customer_id": customer_id,
        "predicted_churn": prediction,
        "probability": churn_probability,
        "model_version": version
    }
