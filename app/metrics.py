from prometheus_client import Counter, Histogram, Gauge

# CRUD Operation Metrics
CRUD_OPERATIONS = Counter(
    "crud_operations_total",
    "Total count of CRUD database operations",
    ["operation", "status"]  # operation: create, read, update, delete; status: success, error
)

# ML Prediction Metrics
MODEL_PREDICTIONS = Counter(
    "model_predictions_total",
    "Total number of model churn predictions",
    ["prediction_class", "model_version"]  # prediction_class: 0 (No), 1 (Yes)
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Time taken to run model inference",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
)

# Model Quality Metrics (Gauges for MLOps tracking)
MODEL_VERSION = Gauge(
    "model_version",
    "Current active version of the model"
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Accuracy of the active model"
)

MODEL_F1_SCORE = Gauge(
    "model_f1_score",
    "F1 score of the active model"
)

MODEL_PRECISION = Gauge(
    "model_precision",
    "Precision of the active model"
)

MODEL_RECALL = Gauge(
    "model_recall",
    "Recall of the active model"
)

MODEL_DATASET_SIZE = Gauge(
    "model_training_dataset_size",
    "Number of customer samples used to train the active model"
)

def set_model_metrics(version: int, accuracy: float, f1: float, precision: float, recall: float, dataset_size: int):
    """Updates Gauges with the latest model evaluation metrics."""
    MODEL_VERSION.set(version)
    MODEL_ACCURACY.set(accuracy)
    MODEL_F1_SCORE.set(f1)
    MODEL_PRECISION.set(precision)
    MODEL_RECALL.set(recall)
    MODEL_DATASET_SIZE.set(dataset_size)
