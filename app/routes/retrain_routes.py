from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.ml.retrain import retrain_from_db
from app.database import get_db
from app.logger import logger
from app.metrics.prometheus_metrics import RETRAIN_COUNT

router = APIRouter(
    prefix="/retrain",
    tags=["ML Retraining"],
    responses={404: {"description": "Not found"}},
)

def _run_retraining_in_background(db: Session):
    # This function will be run in a background task
    # It needs its own DB session
    try:
        success = retrain_from_db(db)
        if success:
            logger.info("Model retraining completed successfully in background.")
        else:
            logger.error("Model retraining failed in background.")
    except Exception as e:
        logger.error(f"Error in background retraining task: {str(e)}")

@router.post("/", status_code=202)
def retrain_model(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    logger.info("Received model retraining request.")
    RETRAIN_COUNT.inc()
    
    # Run retraining in a background task to avoid blocking the API response
    background_tasks.add_task(_run_retraining_in_background, db)
    
    return {"message": "Model retraining initiated successfully. Check logs for progress."}
