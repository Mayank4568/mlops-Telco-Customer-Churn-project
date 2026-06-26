import os
import pandas as pd
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app import crud, schemas, models
from app.ml.train import train_and_evaluate

def main():
    csv_path = "/workspace/telco_churn.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    print("Reading CSV dataset...")
    df = pd.read_csv(csv_path)
    
    # Standardize column names (lowercase with underscores)
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
        print(f"Error: CSV is missing required columns: {missing_cols}")
        return
        
    print(f"Found {len(df)} rows. Importing to database...")
    db = SessionLocal()
    try:
        # Clear existing customer records & prediction logs to have a clean start
        print("Clearing database tables...")
        db.query(models.PredictionRecord).delete()
        db.query(models.ModelMetadata).delete()
        db.query(models.CustomerRecord).delete()
        db.commit()
        
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
        print("Training model version 1 on new dataset...")
        res = train_and_evaluate(db)
        print(f"Model V1 trained successfully! Accuracy: {res['accuracy']:.4f}, F1: {res['f1_score']:.4f}")
        
    except Exception as e:
        db.rollback()
        print(f"Database error during load: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()
