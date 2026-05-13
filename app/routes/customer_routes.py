from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app import crud, schemas, models
from app.database import get_db
from app.logger import logger
from app.metrics.prometheus_metrics import CUSTOMER_CRUD_OPERATIONS

router = APIRouter(
    prefix="/customers",
    tags=["Customers"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=schemas.CustomerResponse, status_code=201)
def create_customer(customer: schemas.CustomerCreate, db: Session = Depends(get_db)):
    logger.info(f"Creating customer with ID: {customer.customerID}")
    db_customer = crud.get_customer_by_cid(db, customerID=customer.customerID)
    if db_customer:
        logger.warning(f"Customer with ID {customer.customerID} already exists.")
        raise HTTPException(status_code=400, detail="Customer with this ID already exists")
    try:
        new_customer = crud.create_customer(db=db, customer=customer)
        logger.info(f"Customer {new_customer.customerID} created successfully.")
        return new_customer
    except Exception as e:
        logger.error(f"Error creating customer {customer.customerID}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/", response_model=List[schemas.CustomerResponse])
def read_customers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    logger.info("Fetching all customers.")
    customers = crud.get_customers(db, skip=skip, limit=limit)
    return customers

@router.get("/{customer_id}", response_model=schemas.CustomerResponse)
def read_customer(customer_id: int, db: Session = Depends(get_db)):
    logger.info(f"Fetching customer with internal ID: {customer_id}")
    db_customer = crud.get_customer(db, customer_id=customer_id)
    if db_customer is None:
        logger.warning(f"Customer with internal ID {customer_id} not found.")
        raise HTTPException(status_code=404, detail="Customer not found")
    return db_customer

@router.put("/{customer_id}", response_model=schemas.CustomerResponse)
def update_customer(customer_id: int, customer: schemas.CustomerUpdate, db: Session = Depends(get_db)):
    logger.info(f"Updating customer with internal ID: {customer_id}")
    db_customer = crud.update_customer(db, customer_id=customer_id, customer=customer)
    if db_customer is None:
        logger.warning(f"Customer with internal ID {customer_id} not found for update.")
        raise HTTPException(status_code=404, detail="Customer not found")
    logger.info(f"Customer with internal ID {customer_id} updated successfully.")
    return db_customer

@router.delete("/{customer_id}", status_code=204)
def delete_customer(customer_id: int, db: Session = Depends(get_db)):
    logger.info(f"Deleting customer with internal ID: {customer_id}")
    success = crud.delete_customer(db, customer_id=customer_id)
    if not success:
        logger.warning(f"Customer with internal ID {customer_id} not found for deletion.")
        raise HTTPException(status_code=404, detail="Customer not found")
    logger.info(f"Customer with internal ID {customer_id} deleted successfully.")
    return {"ok": True}
