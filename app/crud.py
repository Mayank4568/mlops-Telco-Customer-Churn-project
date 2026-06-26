from sqlalchemy.orm import Session
from . import models, schemas

# Customer CRUD operations
def get_customer_by_id(db: Session, customer_record_id: int):
    return db.query(models.CustomerRecord).filter(models.CustomerRecord.id == customer_record_id).first()

def get_customer_by_customer_id(db: Session, customer_id: str):
    return db.query(models.CustomerRecord).filter(models.CustomerRecord.customer_id == customer_id).first()

def get_customers(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.CustomerRecord).offset(skip).limit(limit).all()

def create_customer(db: Session, customer: schemas.CustomerCreate):
    db_customer = models.CustomerRecord(**customer.model_dump())
    db.add(db_customer)
    db.commit()
    db.refresh(db_customer)
    return db_customer

def update_customer(db: Session, customer_id: str, updates: schemas.CustomerUpdate):
    db_customer = get_customer_by_customer_id(db, customer_id)
    if not db_customer:
        return None
    
    update_data = updates.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_customer, key, value)
        
    db.commit()
    db.refresh(db_customer)
    return db_customer

def delete_customer(db: Session, customer_id: str):
    db_customer = get_customer_by_customer_id(db, customer_id)
    if not db_customer:
        return None
    db.delete(db_customer)
    db.commit()
    return db_customer

# Labeled data for training (where churn is not null)
def get_labeled_customers(db: Session):
    return db.query(models.CustomerRecord).filter(models.CustomerRecord.churn.isnot(None)).all()

# Prediction logging
def log_prediction(db: Session, customer_id: str, predicted_churn: int, probability: float, model_version: int):
    db_prediction = models.PredictionRecord(
        customer_id=customer_id,
        predicted_churn=predicted_churn,
        probability=probability,
        model_version=model_version
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

# Model Metadata operations
def get_active_model(db: Session):
    return db.query(models.ModelMetadata).filter(models.ModelMetadata.is_active == True).first()

def get_latest_model_version(db: Session):
    latest = db.query(models.ModelMetadata).order_by(models.ModelMetadata.version.desc()).first()
    return latest.version if latest else 0

def create_model_metadata(db: Session, version: int, accuracy: float, f1_score: float, recall: float, precision: float, dataset_size: int, model_path: str):
    # Deactivate other models
    db.query(models.ModelMetadata).update({models.ModelMetadata.is_active: False})
    
    db_metadata = models.ModelMetadata(
        version=version,
        accuracy=accuracy,
        f1_score=f1_score,
        recall=recall,
        precision=precision,
        dataset_size=dataset_size,
        model_path=model_path,
        is_active=True
    )
    db.add(db_metadata)
    db.commit()
    db.refresh(db_metadata)
    return db_metadata

def get_model_history(db: Session):
    return db.query(models.ModelMetadata).order_by(models.ModelMetadata.version.desc()).all()
