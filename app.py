
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load best model
best_model = joblib.load("best_model.pkl")

class PurchaseInput(BaseModel):
    total_amount: float
    product_category: str
    payment_type: str
    delivery_location: str
    days_since_last_purchase: int
    prev_purchases: int
    total_spent_historical: float
    avg_order_value: float
    preferred_category: str

@app.post("/predict")
async def predict(input: PurchaseInput):
    try:
        df = pd.DataFrame([input.dict()])
        prediction = best_model.predict(df)[0]
        prediction_proba = best_model.predict_proba(df)[0]
        return {
            "predicted_class": "Repeat Purchase" if prediction == 1 else "No Repeat Purchase",
            "probability_no_repeat": float(prediction_proba[0]),
            "probability_repeat": float(prediction_proba[1])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Customer Purchase Prediction API"}
