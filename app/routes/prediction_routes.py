from fastapi import APIRouter, HTTPException
from app.ml.predict import predict_churn
from app.schemas import PredictionInput, PredictionResponse
from app.logger import logger
from app.metrics.prometheus_metrics import PREDICTION_COUNT, FAILED_PREDICTION_COUNT

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=PredictionResponse)
def get_prediction(input_data: PredictionInput):
    logger.info(f"Received prediction request for customer: {input_data.customerID}")
    PREDICTION_COUNT.inc()
    try:
        churn_prediction, probability = predict_churn(input_data.model_dump())
        if churn_prediction is None:
            FAILED_PREDICTION_COUNT.inc()
            raise HTTPException(status_code=500, detail="Prediction failed due to internal error")
        
        logger.info(f"Prediction for customer {input_data.customerID}: {churn_prediction} with probability {probability:.2f}")
        return PredictionResponse(
            customerID=input_data.customerID,
            churn_prediction=churn_prediction,
            probability=probability
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        FAILED_PREDICTION_COUNT.inc()
        logger.error(f"Unhandled error during prediction for {input_data.customerID}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")
