"""
Phase 7: FastAPI Backend for Dynamic Pricing Predictions
GlobalMart Tide Detergent Pricing Strategy API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, date
import uvicorn
import os
import requests
from azure.identity import DefaultAzureCredential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dynamic Pricing API",
    description="AI-powered dynamic pricing API for GlobalMart Tide detergent",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
feature_names = []

# -------------------
# Azure ML endpoint config
# -------------------
ENDPOINT_URL = os.environ.get("AZUREML_ENDPOINT_URL")
AZURE_RESOURCE = "https://ml.azure.com"

def get_token():
    credential = DefaultAzureCredential()
    token = credential.get_token(AZURE_RESOURCE + "/.default")
    return token.token

# Pydantic models
class PricingFeatures(BaseModel):
    MRP: float = Field(..., ge=0)
    NoPromoPrice: float = Field(..., ge=0)
    SellingPrice: float = Field(..., ge=0)
    CTR: float = Field(0.02, ge=0, le=1)
    AbandonedCartRate: float = Field(0.2, ge=0, le=1)
    BounceRate: float = Field(0.3, ge=0, le=1)
    IsMetro: bool = Field(True)
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    dayofweek: int = Field(..., ge=1, le=7)
    quarter: int = Field(..., ge=1, le=4)
    competitor_price: float = Field(0, ge=0)

class PredictionResponse(BaseModel):
    predicted_units_sold: float
    confidence_score: float
    pricing_recommendation: str
    timestamp: datetime

class BatchPredictionRequest(BaseModel):
    predictions: List[PricingFeatures]

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    summary: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: datetime
    version: str

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        timestamp=datetime.now(),
        version="1.0.0"
    )

def prepare_payload(features: PricingFeatures) -> Dict:
    return {
        "input_data": [features.dict()]
    }

def generate_pricing_recommendation(predicted_units: float, features: PricingFeatures) -> str:
    current_price = features.SellingPrice
    competitor_price = features.competitor_price
    mrp = features.MRP

    if predicted_units < 30:
        if current_price > competitor_price:
            return "DECREASE PRICE - Low demand predicted, price higher than competitor"
        else:
            return "INCREASE MARKETING - Low demand despite competitive pricing"
    elif predicted_units > 80:
        if current_price < mrp * 0.9:
            return "INCREASE PRICE - High demand allows for price optimization"
        else:
            return "MAINTAIN PRICE - Good demand at current pricing"
    else:
        return "MAINTAIN PRICE - Balanced demand and pricing"

@app.post("/predict", response_model=PredictionResponse)
async def predict_units_sold(features: PricingFeatures):
    try:
        token = get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        payload = prepare_payload(features)
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
        response.raise_for_status()
        prediction = response.json()["predicted_units"]

        confidence = min(0.95, 0.6 + (prediction / 100) * 0.3)
        recommendation = generate_pricing_recommendation(prediction, features)

        return PredictionResponse(
            predicted_units_sold=round(prediction, 2),
            confidence_score=round(confidence, 3),
            pricing_recommendation=recommendation,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    try:
        token = get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        payload = {
            "input_data": [f.dict() for f in request.predictions]
        }
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
        response.raise_for_status()
        predictions = response.json()["predicted_units_list"]

        results = []
        predictions_sum = 0

        for features, pred in zip(request.predictions, predictions):
            prediction = max(0, pred)
            predictions_sum += prediction
            confidence = min(0.95, 0.6 + (prediction / 100) * 0.3)
            recommendation = generate_pricing_recommendation(prediction, features)

            results.append(PredictionResponse(
                predicted_units_sold=round(prediction, 2),
                confidence_score=round(confidence, 3),
                pricing_recommendation=recommendation,
                timestamp=datetime.now()
            ))

        summary = {
            "total_predictions": len(results),
            "average_units_predicted": round(predictions_sum / len(results), 2),
            "total_units_predicted": round(predictions_sum, 2),
            "avg_confidence": round(sum(r.confidence_score for r in results) / len(results), 3)
        }

        return BatchPredictionResponse(results=results, summary=summary)

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
