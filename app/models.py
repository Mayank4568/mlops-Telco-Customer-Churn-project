from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, func
from .database import Base

class CustomerRecord(Base):
    __tablename__ = "customer_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    customer_id = Column(String(50), unique=True, index=True, nullable=False)
    gender = Column(String(10), nullable=False)
    senior_citizen = Column(Integer, nullable=False)  # 0 or 1
    partner = Column(String(5), nullable=False)  # Yes or No
    dependents = Column(String(5), nullable=False)  # Yes or No
    tenure = Column(Integer, nullable=False)  # Months
    phone_service = Column(String(5), nullable=False)  # Yes or No
    internet_service = Column(String(20), nullable=False)  # DSL, Fiber optic, No
    online_security = Column(String(25), nullable=False)  # Yes, No, No internet service
    tech_support = Column(String(25), nullable=False)  # Yes, No, No internet service
    paperless_billing = Column(String(5), nullable=False)  # Yes, No
    payment_method = Column(String(50), nullable=False)
    monthly_charges = Column(Float, nullable=False)
    total_charges = Column(Float, nullable=False)
    churn = Column(Integer, nullable=True)  # 0 or 1, can be Null if not yet known (feedback loop)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    customer_id = Column(String(50), index=True, nullable=False)
    predicted_churn = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    model_version = Column(Integer, nullable=False)
    prediction_time = Column(DateTime, server_default=func.now())

class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    version = Column(Integer, primary_key=True, index=True)
    trained_at = Column(DateTime, server_default=func.now())
    accuracy = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    dataset_size = Column(Integer, nullable=False)
    model_path = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=False)
