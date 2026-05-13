import joblib
import pandas as pd
import os
from app.config import settings
from app.ml.preprocess import preprocess_data
from app.logger import logger

def predict_churn(input_data: dict):
    try:
        if not os.path.exists(settings.MODEL_PATH):
            logger.error("Model file not found")
            return None
        
        model = joblib.load(settings.MODEL_PATH)
        df = pd.DataFrame([input_data])
        processed_df = preprocess_data(df)
        
        # Ensure 'Churn' column is not present if it was accidentally added
        if 'Churn' in processed_df.columns:
            processed_df = processed_df.drop('Churn', axis=1)
            
        prediction = model.predict(processed_df)[0]
        probability = model.predict_proba(processed_df)[0][1]
        
        return "Yes" if prediction == 1 else "No", float(probability)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None
