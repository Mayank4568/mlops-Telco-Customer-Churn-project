from prometheus_client import Gauge, Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# Custom Metrics
RETRAIN_COUNT = Counter("retrain_total", "Total number of model retraining events")
PREDICTION_COUNT = Counter("prediction_total", "Total number of churn prediction requests")
FAILED_PREDICTION_COUNT = Counter("failed_prediction_total", "Total number of failed churn prediction requests")

# CRUD Metrics (Counters)
CUSTOMER_CRUD_OPERATIONS = Counter(
    "customer_crud_operations_total",
    "Total number of CRUD operations on customers",
    ["operation_type"] # Labels for operation type (create, read, update, delete)
)

# Function to instrument FastAPI app
def instrument_app(app):
    Instrumentator().instrument(app).expose(app)

    # Add custom metrics for CRUD operations
    @app.middleware("http")
    async def add_crud_metrics(request, call_next):
        response = await call_next(request)
        if request.scope['type'] == 'http':
            method = request.method
            path = request.url.path

            if path.startswith("/customers"):
                if method == "POST":
                    CUSTOMER_CRUD_OPERATIONS.labels(operation_type="create").inc()
                elif method == "GET" and "/customers/" not in path: # Exclude specific customer get
                    CUSTOMER_CRUD_OPERATIONS.labels(operation_type="read_all").inc()
                elif method == "GET" and "/customers/" in path:
                    CUSTOMER_CRUD_OPERATIONS.labels(operation_type="read_one").inc()
                elif method == "PUT":
                    CUSTOMER_CRUD_OPERATIONS.labels(operation_type="update").inc()
                elif method == "DELETE":
                    CUSTOMER_CRUD_OPERATIONS.labels(operation_type="delete").inc()

        return response

