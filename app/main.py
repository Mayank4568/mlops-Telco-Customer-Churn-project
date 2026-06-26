import time
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.openapi.docs import get_swagger_ui_html
from sqlalchemy.orm import Session
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import os
import pandas as pd
from fastapi.responses import Response, HTMLResponse


from .database import get_db, engine, Base
from . import crud, schemas, models
from .seed import seed_database_if_empty
from .ml.predict import predict_churn_for_customer
from .ml.train import train_and_evaluate
from .metrics import CRUD_OPERATIONS, MODEL_PREDICTIONS, PREDICTION_LATENCY

app = FastAPI(
    title="Telco Customer Churn - CRUD & ML Lifecycle API",
    description="An MLOps API providing customer CRUD operations, model inference, and model retraining, instrumented with Prometheus metrics.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Interactive API Docs",
        oauth2_redirect_url=app.oauth2_redirect_url,
        swagger_css_url="https://unpkg.com/swagger-ui-themes@3.0.1/themes/3.x/theme-material.css",
    )


# Initialize database and seed if empty on startup
@app.on_event("startup")
def startup_event():
    # Ensure tables are created
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        seed_database_if_empty(db)
    finally:
        db.close()

    # Print service URLs to the console for easy access
    print("\n" + "="*60)
    print("🚀 TELCO CHURN SYSTEM IS READY!")
    print("------------------------------------------------------------")
    print("1. Main Dashboard:        http://localhost:8000")
    print("2. Swagger API Docs:      http://localhost:8000/docs")
    print("3. Prometheus Metrics:    http://localhost:9090")
    print("4. Grafana Dashboard:     http://localhost:3000")
    print("="*60 + "\n")


# Hook Prometheus instrumentator for HTTP metrics
# This handles default HTTP request duration, sizes, and status code metrics.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
INDEX_HTML_PATH = os.path.join(TEMPLATES_DIR, "index.html")

@app.get("/", response_class=HTMLResponse, tags=["General"])
def read_root():
    if os.path.exists(INDEX_HTML_PATH):
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Template file not found</h1>", status_code=404)

@app.get("/health", tags=["General"])
def health_check(db: Session = Depends(get_db)):
    try:
        # Check database health
        db.execute(models.CustomerRecord.__table__.select().limit(1))
        
        # Check model health
        active_model = crud.get_active_model(db)
        model_status = "uninitialized"
        if active_model:
            model_status = f"active (v{active_model.version})"
            
        return {
            "status": "healthy",
            "database": "connected",
            "model_version": model_status
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unhealthy service: {str(e)}"
        )

# CRUD Endpoints for Customer Records
@app.post("/api/v1/customers", response_model=schemas.CustomerResponse, status_code=status.HTTP_201_CREATED, tags=["CRUD Operations"])
def create_customer_record(customer: schemas.CustomerCreate, db: Session = Depends(get_db)):
    try:
        existing = crud.get_customer_by_customer_id(db, customer.customer_id)
        if existing:
            CRUD_OPERATIONS.labels(operation="create", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Customer record with customer_id '{customer.customer_id}' already exists."
            )
        result = crud.create_customer(db, customer)
        CRUD_OPERATIONS.labels(operation="create", status="success").inc()
        return result
    except HTTPException:
        raise
    except Exception as e:
        CRUD_OPERATIONS.labels(operation="create", status="error").inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/api/v1/customers/{customer_id}", response_model=schemas.CustomerResponse, tags=["CRUD Operations"])
def get_customer_record(customer_id: str, db: Session = Depends(get_db)):
    try:
        customer = crud.get_customer_by_customer_id(db, customer_id)
        if not customer:
            CRUD_OPERATIONS.labels(operation="read", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer record with customer_id '{customer_id}' not found."
            )
        CRUD_OPERATIONS.labels(operation="read", status="success").inc()
        return customer
    except HTTPException:
        raise
    except Exception as e:
        CRUD_OPERATIONS.labels(operation="read", status="error").inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/api/v1/customers", response_model=list[schemas.CustomerResponse], tags=["CRUD Operations"])
def list_customer_records(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        customers = crud.get_customers(db, skip=skip, limit=limit)
        CRUD_OPERATIONS.labels(operation="read_list", status="success").inc()
        return customers
    except Exception as e:
        CRUD_OPERATIONS.labels(operation="read_list", status="error").inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.put("/api/v1/customers/{customer_id}", response_model=schemas.CustomerResponse, tags=["CRUD Operations"])
def update_customer_record(customer_id: str, updates: schemas.CustomerUpdate, db: Session = Depends(get_db)):
    try:
        updated = crud.update_customer(db, customer_id, updates)
        if not updated:
            CRUD_OPERATIONS.labels(operation="update", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer record with customer_id '{customer_id}' not found."
            )
        CRUD_OPERATIONS.labels(operation="update", status="success").inc()
        return updated
    except HTTPException:
        raise
    except Exception as e:
        CRUD_OPERATIONS.labels(operation="update", status="error").inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.delete("/api/v1/customers/{customer_id}", tags=["CRUD Operations"])
def delete_customer_record(customer_id: str, db: Session = Depends(get_db)):
    try:
        deleted = crud.delete_customer(db, customer_id)
        if not deleted:
            CRUD_OPERATIONS.labels(operation="delete", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer record with customer_id '{customer_id}' not found."
            )
        CRUD_OPERATIONS.labels(operation="delete", status="success").inc()
        return {"message": f"Customer record '{customer_id}' deleted successfully."}
    except HTTPException:
        raise
    except Exception as e:
        CRUD_OPERATIONS.labels(operation="delete", status="error").inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# ML Endpoints (Prediction & Retraining)
@app.post("/api/v1/predict", response_model=schemas.PredictResponse, tags=["Machine Learning"])
def predict_churn(request: schemas.PredictRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    try:
        res = predict_churn_for_customer(db, request.customer_id)
        
        # Log prediction result counts to Prometheus
        MODEL_PREDICTIONS.labels(
            prediction_class=str(res["predicted_churn"]),
            model_version=str(res["model_version"])
        ).inc()
        
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        
        CRUD_OPERATIONS.labels(operation="predict", status="success").inc()
        return res
    except ValueError as ve:
        CRUD_OPERATIONS.labels(operation="predict", status="error").inc()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        CRUD_OPERATIONS.labels(operation="predict", status="error").inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/api/v1/retrain", tags=["Machine Learning"])
def retrain_model(db: Session = Depends(get_db)):
    try:
        # Run model retraining
        results = train_and_evaluate(db)
        CRUD_OPERATIONS.labels(operation="retrain", status="success").inc()
        return {
            "message": "Model retrained and deployed successfully.",
            "metrics": results
        }
    except ValueError as ve:
        CRUD_OPERATIONS.labels(operation="retrain", status="error").inc()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        CRUD_OPERATIONS.labels(operation="retrain", status="error").inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/api/v1/dataset/upload", tags=["Dataset Management"])
def upload_dataset_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        df = pd.read_csv(file.file)
        
        column_mapping = {
            'customerid': 'customer_id',
            'customer_id': 'customer_id',
            'gender': 'gender',
            'seniorcitizen': 'senior_citizen',
            'senior_citizen': 'senior_citizen',
            'partner': 'partner',
            'dependents': 'dependents',
            'tenure': 'tenure',
            'phoneservice': 'phone_service',
            'phone_service': 'phone_service',
            'internetservice': 'internet_service',
            'internet_service': 'internet_service',
            'onlinesecurity': 'online_security',
            'online_security': 'online_security',
            'techsupport': 'tech_support',
            'tech_support': 'tech_support',
            'paperlessbilling': 'paperless_billing',
            'paperless_billing': 'paperless_billing',
            'paymentmethod': 'payment_method',
            'payment_method': 'payment_method',
            'monthlycharges': 'monthly_charges',
            'monthly_charges': 'monthly_charges',
            'totalcharges': 'total_charges',
            'total_charges': 'total_charges',
            'churn': 'churn'
        }
        
        df.columns = [col.lower() for col in df.columns]
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        
        required_cols = [
            'customer_id', 'gender', 'senior_citizen', 'partner', 'dependents', 
            'tenure', 'phone_service', 'internet_service', 'online_security', 
            'tech_support', 'paperless_billing', 'payment_method', 
            'monthly_charges', 'total_charges'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"CSV is missing required columns: {', '.join(missing_cols)}"
            )
            
        imported_count = 0
        for _, row in df.iterrows():
            cust_id = str(row['customer_id']).strip()
            
            churn_val = None
            if 'churn' in df.columns:
                val = str(row['churn']).strip().lower()
                if val in ['1', '1.0', 'yes', 'true']:
                    churn_val = 1
                elif val in ['0', '0.0', 'no', 'false']:
                    churn_val = 0
            
            try:
                monthly = float(row['monthly_charges'])
            except:
                monthly = 0.0
                
            try:
                total = float(row['total_charges'])
            except:
                total = 0.0
                
            customer_data = schemas.CustomerCreate(
                customer_id=cust_id,
                gender=str(row['gender']).capitalize() if pd.notna(row['gender']) else 'Male',
                senior_citizen=int(row['senior_citizen']) if pd.notna(row['senior_citizen']) else 0,
                partner=str(row['partner']).capitalize() if pd.notna(row['partner']) else 'No',
                dependents=str(row['dependents']).capitalize() if pd.notna(row['dependents']) else 'No',
                tenure=int(row['tenure']) if pd.notna(row['tenure']) else 1,
                phone_service=str(row['phone_service']).capitalize() if pd.notna(row['phone_service']) else 'Yes',
                internet_service=str(row['internet_service']) if pd.notna(row['internet_service']) else 'DSL',
                online_security=str(row['online_security']) if pd.notna(row['online_security']) else 'No',
                tech_support=str(row['tech_support']) if pd.notna(row['tech_support']) else 'No',
                paperless_billing=str(row['paperless_billing']).capitalize() if pd.notna(row['paperless_billing']) else 'Yes',
                payment_method=str(row['payment_method']) if pd.notna(row['payment_method']) else 'Electronic check',
                monthly_charges=monthly,
                total_charges=total,
                churn=churn_val
            )
            
            existing = crud.get_customer_by_customer_id(db, cust_id)
            if existing:
                updates = schemas.CustomerUpdate(**customer_data.model_dump())
                crud.update_customer(db, cust_id, updates)
            else:
                crud.create_customer(db, customer_data)
            imported_count += 1
            
        retrain_results = None
        if imported_count > 0:
            try:
                retrain_results = train_and_evaluate(db)
            except Exception as train_err:
                retrain_results = {"error": f"Failed to auto-retrain: {str(train_err)}"}
                
        return {
            "message": f"Successfully processed CSV file and imported/updated {imported_count} customer records.",
            "imported_count": imported_count,
            "retrain_status": retrain_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while parsing the CSV file: {str(e)}"
        )

@app.post("/api/v1/dataset/reset", tags=["Dataset Management"])
def reset_database(db: Session = Depends(get_db)):
    try:
        # Clear predictions, model metadata and customer records
        db.query(models.PredictionRecord).delete()
        db.query(models.ModelMetadata).delete()
        db.query(models.CustomerRecord).delete()
        db.commit()
        
        # Seed database again
        seed_database_if_empty(db)
        
        return {
            "message": "Database reset and seeded successfully. Model v1 trained."
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/models/active", response_model=schemas.ModelMetadataResponse, tags=["Machine Learning"])
def get_active_model_details(db: Session = Depends(get_db)):
    active_model = crud.get_active_model(db)
    if not active_model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active model found.")
    return active_model

@app.get("/api/v1/models/history", response_model=list[schemas.ModelMetadataResponse], tags=["Machine Learning"])
def get_model_history_list(db: Session = Depends(get_db)):
    return crud.get_model_history(db)
