import random
import string
from sqlalchemy.orm import Session
from . import crud, models, schemas
from .database import engine, Base
from .ml.train import train_and_evaluate

def generate_random_customer_id() -> str:
    parts = [
        "".join(random.choices(string.digits, k=4)),
        "".join(random.choices(string.ascii_uppercase, k=5))
    ]
    return "-".join(parts)

def generate_synthetic_customers(n: int = 150) -> list[schemas.CustomerCreate]:
    customers = []
    
    genders = ["Female", "Male"]
    yes_no = ["Yes", "No"]
    internet_services = ["DSL", "Fiber optic", "No"]
    payment_methods = [
        "Electronic check", "Mailed check", 
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    
    for _ in range(n):
        cust_id = generate_random_customer_id()
        gender = random.choice(genders)
        senior = 1 if random.random() < 0.15 else 0
        partner = random.choice(yes_no)
        dependents = random.choice(yes_no)
        tenure = random.randint(1, 72)
        phone = random.choice(yes_no)
        
        internet = random.choice(internet_services)
        if internet == "No":
            security = "No internet service"
            support = "No internet service"
        else:
            security = random.choice(yes_no)
            support = random.choice(yes_no)
            
        paperless = random.choice(yes_no)
        payment = random.choice(payment_methods)
        
        monthly = round(random.uniform(18.90, 118.75), 2)
        if internet == "No":
            monthly = round(random.uniform(18.90, 25.00), 2)
        elif internet == "Fiber optic":
            monthly = round(random.uniform(70.00, 118.75), 2)
            
        total = round(tenure * monthly, 2)
        
        # Churn heuristic (to make the dataset learnable)
        # Higher monthly charges, low tenure, and no tech support/security increase churn probability
        churn_score = 0.0
        if tenure < 12:
            churn_score += 0.4
        if monthly > 80.0:
            churn_score += 0.3
        if support == "No":
            churn_score += 0.2
        if security == "No":
            churn_score += 0.1
        if internet == "Fiber optic":
            churn_score += 0.15
            
        churn_prob = min(max(churn_score, 0.05), 0.95)
        churn = 1 if random.random() < churn_prob else 0
        
        customers.append(schemas.CustomerCreate(
            customer_id=cust_id,
            gender=gender,
            senior_citizen=senior,
            partner=partner,
            dependents=dependents,
            tenure=tenure,
            phone_service=phone,
            internet_service=internet,
            online_security=security,
            tech_support=support,
            paperless_billing=paperless,
            payment_method=payment,
            monthly_charges=monthly,
            total_charges=total,
            churn=churn
        ))
        
    return customers

def seed_database_if_empty(db: Session):
    """Seed the database with initial records and train the V1 model if empty."""
    # Create tables if not exist
    Base.metadata.create_all(bind=engine)
    
    # Check if there are customer records
    count = db.query(models.CustomerRecord).count()
    if count == 0:
        print("Database is empty. Generating and inserting synthetic churn data...")
        synthetic_records = generate_synthetic_customers(150)
        for r in synthetic_records:
            crud.create_customer(db, r)
        print(f"Successfully seeded database with {len(synthetic_records)} records.")
        
        # Train initial model
        print("Training initial machine learning model (V1)...")
        res = train_and_evaluate(db)
        print(f"Initial model trained: Version {res['version']} with Accuracy {res['accuracy']:.4f}")
    else:
        # If DB exists, update metrics for the active model in Prometheus (if one exists)
        active_model = crud.get_active_model(db)
        if active_model:
            try:
                from .metrics import set_model_metrics
                set_model_metrics(
                    active_model.version, 
                    active_model.accuracy, 
                    active_model.f1_score, 
                    active_model.precision, 
                    active_model.recall, 
                    active_model.dataset_size
                )
                print(f"Restored metrics for active model version {active_model.version}")
            except Exception as e:
                print(f"Failed to restore active model metrics: {e}")
