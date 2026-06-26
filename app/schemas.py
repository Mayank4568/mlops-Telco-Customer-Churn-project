from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class CustomerBase(BaseModel):
    customer_id: str = Field(..., example="7590-VHVEG", max_length=50)
    gender: str = Field(..., example="Female")
    senior_citizen: int = Field(..., ge=0, le=1, example=0)
    partner: str = Field(..., example="Yes")
    dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=1)
    phone_service: str = Field(..., example="No")
    internet_service: str = Field(..., example="DSL")
    online_security: str = Field(..., example="No")
    tech_support: str = Field(..., example="No")
    paperless_billing: str = Field(..., example="Yes")
    payment_method: str = Field(..., example="Electronic check")
    monthly_charges: float = Field(..., ge=0.0, example=29.85)
    total_charges: float = Field(..., ge=0.0, example=29.85)
    churn: Optional[int] = Field(None, ge=0, le=1, example=0)

class CustomerCreate(CustomerBase):
    pass

class CustomerUpdate(BaseModel):
    gender: Optional[str] = None
    senior_citizen: Optional[int] = None
    partner: Optional[str] = None
    dependents: Optional[str] = None
    tenure: Optional[int] = None
    phone_service: Optional[str] = None
    internet_service: Optional[str] = None
    online_security: Optional[str] = None
    tech_support: Optional[str] = None
    paperless_billing: Optional[str] = None
    payment_method: Optional[str] = None
    monthly_charges: Optional[float] = None
    total_charges: Optional[float] = None
    churn: Optional[int] = None

class CustomerResponse(CustomerBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Prediction Schemas
class PredictRequest(BaseModel):
    customer_id: str

class PredictResponse(BaseModel):
    customer_id: str
    predicted_churn: int
    probability: float
    model_version: int

# Model Metadata Schemas
class ModelMetadataResponse(BaseModel):
    version: int
    trained_at: datetime
    accuracy: float
    f1_score: float
    recall: float
    precision: float
    dataset_size: int
    model_path: str
    is_active: bool

    class Config:
        from_attributes = True
