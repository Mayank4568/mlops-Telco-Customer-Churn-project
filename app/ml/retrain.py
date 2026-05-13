import pandas as pd
from sqlalchemy.orm import Session
from app import models
from app.ml.train import train_model
from app.logger import logger
import os

def retrain_from_db(db: Session):
    try:
        # Fetch all customers from DB
        customers = db.query(models.Customer).all()
        if not customers:
            logger.warning("No data in database for retraining.")
            return False
        
        # Convert to DataFrame
        data = []
        for c in customers:
            d = c.__dict__.copy()
            d.pop('_sa_instance_state', None)
            d.pop('id', None)
            data.append(d)
        
        df_new = pd.DataFrame(data)
        
        # Combine with original dataset if available to have enough data
        original_data_path = "dataset/telco_churn.csv"
        if os.path.exists(original_data_path):
            df_old = pd.read_csv(original_data_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined.to_csv("dataset/combined_data.csv", index=False)
            success = train_model("dataset/combined_data.csv")
        else:
            df_new.to_csv("dataset/retrain_data.csv", index=False)
            success = train_model("dataset/retrain_data.csv")
            
        return success
    except Exception as e:
        logger.error(f"Error during retrain_from_db: {str(e)}")
        return False
