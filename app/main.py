from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Base, engine, get_db
from app.logger import logger
from app.routes import customer_routes, prediction_routes, retrain_routes
from app.metrics.prometheus_metrics import instrument_app
from app.ml.train import train_model
import os

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    openapi_tags=[
        {"name": "Health", "description": "Health check operations"},
        {"name": "Customers", "description": "Operations with customers"},
        {"name": "Prediction", "description": "Machine learning prediction operations"},
        {"name": "ML Retraining", "description": "Machine learning model retraining operations"},
    ]
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument the app with Prometheus metrics
instrument_app(app)

# Include routers
app.include_router(customer_routes.router)
app.include_router(prediction_routes.router)
app.include_router(retrain_routes.router)

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.PROJECT_NAME} application...")
    # Ensure the model directory exists
    os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
    # Train model if not already present
    if not os.path.exists(settings.MODEL_PATH):
        logger.info("No model found. Training initial model...")
        if not os.path.exists("dataset/telco_churn.csv"):
            logger.error("Initial dataset not found. Please place telco_churn.csv in the dataset/ directory.")
            # Optionally exit or raise an error if model is critical
        else:
            train_model()
    else:
        logger.info("Existing model found. Skipping initial training.")

@app.get("/health", tags=["Health"])
async def health_check():
    logger.info("Health check requested.")
    return {"status": "ok", "message": "Service is healthy"}
