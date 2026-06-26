# Telco Customer Churn - CRUD & ML Lifecycle Monitoring System

This project is a complete, production-grade MLOps system developed for the Telco Customer Churn prediction task. It implements a fully-featured FastAPI REST API offering CRUD operations on customer records, real-time machine learning predictions, automatic database logging, model retraining, and prometheus-grafana monitoring.

---

## Architecture & Tech Stack

1. **FastAPI Application (`app/`)**: Handles CRUD operations and exposes model serving and retraining endpoints.
2. **Database Support (SQLite/MySQL)**: Uses SQLAlchemy. Defaults to SQLite (`telco_churn.db`) for lightweight local runs, but connects to MySQL automatically via Docker Compose.
3. **ML Lifecycle (scikit-learn & joblib)**:
   - Categorical and numerical column preprocessing using a robust `ColumnTransformer`.
   - Classification using a `RandomForestClassifier` pipeline.
   - Dynamic model loader that reloads the model into memory only when a new version is trained.
   - Automated model performance tracking (Accuracy, F1-score, Recall, Precision, Dataset Size, Version) registered to database metadata.
4. **Monitoring (Prometheus & Grafana)**:
   - Custom metrics tracking CRUD operations, model predictions, prediction probability, and training dataset.
   - Provisioned dashboard in Grafana configured to load out of the box.

---

## Project Structure

```
mlops-Telco-Customer-Churn-project/
├── app/
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── train.py          # Model training pipeline & evaluation
│   │   ├── predict.py        # Model inference & prediction logging
│   │   └── saved_models/     # Directory where trained models are saved (.pkl)
│   ├── __init__.py
│   ├── database.py           # SQLAlchemy setup (handles SQLite/MySQL dynamically)
│   ├── models.py             # SQLAlchemy schemas (CustomerRecord, PredictionRecord, ModelMetadata)
│   ├── schemas.py            # Pydantic validation schemas
│   ├── crud.py               # Database transaction layer
│   ├── metrics.py            # Prometheus custom metrics definition
│   ├── seed.py               # Synthetic churn data generator & DB initializer
│   └── main.py               # FastAPI router, Prometheus middleware, & startup tasks
├── config/
│   ├── prometheus.yml        # Prometheus configuration & targets
│   └── grafana/              # Grafana dashboard & datasource provisioning
│       └── provisioning/
│           ├── dashboards/
│           │   ├── dashboard.yml
│           │   └── telco_churn_dashboard.json
│           └── datasources/
│               └── datasource.yml
├── Dockerfile                # FastAPI container builder
├── docker-compose.yml        # Multi-container conductor (DB, App, Prometheus, Grafana)
├── requirements.txt          # Python package requirements
└── README.md                 # This guide
```

---

## Quick Start (Two Ways to Run)

### Method 1: Using Docker Compose (Recommended - Starts everything)

This launches the FastAPI application, a MySQL Database, Prometheus, and Grafana simultaneously. The database is automatically seeded, and model version 1 is trained on startup!

1. Make sure you have Docker installed and running.
2. Run the following command in the `m1` directory:
   ```bash
   docker-compose up --build
   ```
3. Access the services:
   - **Main Web UI Dashboard (Frontend)**: [http://localhost:8000](http://localhost:8000)
   - **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs) *(now styled with a modern Material theme)*
   - **Prometheus UI**: [http://localhost:9090](http://localhost:9090)
   - **Grafana Dashboard**: [http://localhost:3000](http://localhost:3000) (Anonymous Admin is enabled; no login required!)

---

### Method 2: Running Locally (Fast development)

This method uses SQLite, avoiding Docker overhead.

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the FastAPI App**:
   ```bash
   uvicorn app.main:app --reload
   ```
   *Note: On startup, the app creates `telco_churn.db`, generates synthetic customer records, and trains the first model version (`model_v1.pkl` in `app/ml/saved_models/`).*
3. Access the services:
   - **Main Web UI Dashboard (Frontend)**: [http://localhost:8000](http://localhost:8000) or [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs) or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)


---

## API Endpoints Reference

### CRUD Operations
- **`POST /api/v1/customers`**: Add a new customer record.
- **`GET /api/v1/customers/{customer_id}`**: Fetch details of a specific customer.
- **`GET /api/v1/customers`**: Retrieve a paginated list of customer records.
- **`PUT /api/v1/customers/{customer_id}`**: Update features or label (churn status) of an existing customer.
- **`DELETE /api/v1/customers/{customer_id}`**: Remove a customer record.

### Machine Learning & MLOps
- **`POST /api/v1/predict`**: Predict Churn for a customer.
  - *Request Body*: `{ "customer_id": "CUSTOMER-ID" }`
  - *Details*: Fetches customer data from the database, runs model inference, returns predicted class (0/1) and churn probability, and logs prediction details in the database.
- **`POST /api/v1/retrain`**: Triggers model retraining.
  - *Details*: Queries the database for all records with a non-null `churn` value, performs a train/test split, trains a new random forest model, evaluates metrics (Accuracy, F1, Precision, Recall), saves the model pipeline, sets the new version as active, and updates Prometheus gauges.
- **`GET /api/v1/models/active`**: Get metadata for the currently active model.
- **`GET /api/v1/models/history`**: Get performance metrics of all historical models.

### Metrics & Diagnostics
- **`GET /health`**: Returns application database and model health status.
- **`GET /metrics`**: Exposes standard HTTP and custom MLOps/CRUD metrics for Prometheus.

---

## Verifying the ML Lifecycle & Monitoring

To test the system fully, run these sample HTTP requests (using the Swagger UI or `curl`):

### 1. Create a Customer Record
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/customers' \
  -H 'Content-Type: application/json' \
  -d '{
  "customer_id": "9999-NEWCUST",
  "gender": "Male",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "Yes",
  "tenure": 24,
  "phone_service": "Yes",
  "internet_service": "Fiber optic",
  "online_security": "Yes",
  "tech_support": "Yes",
  "paperless_billing": "Yes",
  "payment_method": "Credit card (automatic)",
  "monthly_charges": 85.50,
  "total_charges": 2052.00,
  "churn": null
}'
```

### 2. Predict Churn
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "customer_id": "9999-NEWCUST"
}'
```
*Note: This logs a prediction record in the database and increments the `model_predictions_total` metric.*

### 3. Add Feedback & Retrain
1. Update the customer's churn status to register new feedback:
   ```bash
   curl -X 'PUT' \
     'http://localhost:8000/api/v1/customers/9999-NEWCUST' \
     -H 'Content-Type: application/json' \
     -d '{
     "churn": 1
   }'
   ```
2. Call the retrain endpoint to update the model with this new data:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/api/v1/retrain'
   ```
   This will train a new model version (V2), evaluate it, and dynamically load V2 for all subsequent prediction requests!
