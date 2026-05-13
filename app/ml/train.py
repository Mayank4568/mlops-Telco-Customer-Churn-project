import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from app.ml.preprocess import preprocess_data
from app.config import settings
from app.logger import logger

def train_model(data_path: str = "dataset/telco_churn.csv"):
    try:
        if not os.path.exists(data_path):
            logger.error(f"Dataset not found at {data_path}")
            return False

        df = pd.read_csv(data_path)
        processed_df = preprocess_data(df)
        
        X = processed_df.drop('Churn', axis=1)
        y = processed_df['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
        joblib.dump(model, settings.MODEL_PATH)
        
        logger.info(f"Model trained and saved to {settings.MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return False

if __name__ == "__main__":
    train_model()
