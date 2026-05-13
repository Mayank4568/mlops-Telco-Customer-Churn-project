from sqlalchemy import Column, Integer, String, Float, Boolean
from app.database import Base

class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    customerID = Column(String(50), unique=True, index=True)
    gender = Column(String(10))
    SeniorCitizen = Column(Integer)
    Partner = Column(String(5))
    Dependents = Column(String(5))
    tenure = Column(Integer)
    PhoneService = Column(String(5))
    MultipleLines = Column(String(20))
    InternetService = Column(String(20))
    OnlineSecurity = Column(String(20))
    OnlineBackup = Column(String(20))
    DeviceProtection = Column(String(20))
    TechSupport = Column(String(20))
    StreamingTV = Column(String(20))
    StreamingMovies = Column(String(20))
    Contract = Column(String(20))
    PaperlessBilling = Column(String(5))
    PaymentMethod = Column(String(30))
    MonthlyCharges = Column(Float)
    TotalCharges = Column(String(20)) # Using string because dataset has some empty strings
    Churn = Column(String(5))
