import random
import string
import os
import pandas as pd
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
        csv_path = "/workspace/telco_churn.csv"
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "telco_churn.csv")
            
        if os.path.exists(csv_path):
            print(f"Database is empty. Loading dataset from {csv_path}...")
            df = pd.read_csv(csv_path)
            
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
            
            imported = 0
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
                    tc_str = str(row['total_charges']).strip()
                    if not tc_str or tc_str == 'nan':
                        total = 0.0
                    else:
                        total = float(tc_str)
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
                
                crud.create_customer(db, customer_data)
                imported += 1
                if imported % 1000 == 0:
                    print(f"Imported {imported} records...")
                    
            print(f"Successfully loaded {imported} customer records.")
            print("Training model version 1 on CSV dataset...")
            res = train_and_evaluate(db)
            print(f"Model V1 trained successfully! Accuracy: {res['accuracy']:.4f}, F1: {res['f1_score']:.4f}")
        else:
            print(f"Database is empty, and telco_churn.csv was not found at {csv_path}. Cannot seed database.")
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
