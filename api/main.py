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
import mlflow
import mlflow.sklearn
import logging
from datetime import datetime, date
import uvicorn
import os

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

# Global variables for model and features
model = None
feature_names = []

# Pydantic models for request/response
class PricingFeatures(BaseModel):
    """Input features for pricing prediction"""
    MRP: float = Field(..., description="Maximum Retail Price", ge=0)
    NoPromoPrice: float = Field(..., description="Price without promotion", ge=0)
    SellingPrice: float = Field(..., description="Current selling price", ge=0)
    CTR: float = Field(0.02, description="Click-through rate", ge=0, le=1)
    AbandonedCartRate: float = Field(0.2, description="Cart abandonment rate", ge=0, le=1)
    BounceRate: float = Field(0.3, description="Bounce rate", ge=0, le=1)
    IsMetro: bool = Field(True, description="Is metro city location")
    month: int = Field(..., description="Month (1-12)", ge=1, le=12)
    day: int = Field(..., description="Day of month (1-31)", ge=1, le=31)
    dayofweek: int = Field(..., description="Day of week (1-7)", ge=1, le=7)
    quarter: int = Field(..., description="Quarter (1-4)", ge=1, le=4)
    competitor_price: float = Field(0, description="Competitor price", ge=0)

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_units_sold: float
    confidence_score: float
    pricing_recommendation: str
    timestamp: datetime

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    predictions: List[PricingFeatures]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    results: List[PredictionResponse]
    summary: Dict[str, float]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: datetime
    version: str

# Startup event to load model
@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model, feature_names
    
    try:
        # In a real deployment, load from Azure ML or MLflow model registry
        model_path = "/models/dynamic_pricing_model.pkl"
        
        # For demo purposes, create a simple model if file doesn't exist
        if not os.path.exists(model_path):
            logger.warning("Model file not found, creating demo model")
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            # Create synthetic training data for demo
            np.random.seed(42)
            X_demo = np.random.rand(1000, 12)
            y_demo = 50 + X_demo[:, 0] * 30 + X_demo[:, 1] * 20 + np.random.normal(0, 5, 1000)
            y_demo = np.maximum(y_demo, 0)  # Ensure positive values
            
            model.fit(X_demo, y_demo)
            
            feature_names = [
                'MRP', 'NoPromoPrice', 'SellingPrice', 'CTR', 'AbandonedCartRate',
                'BounceRate', 'IsMetro', 'month', 'day', 'dayofweek', 'quarter', 'competitor_price'
            ]
        else:
            model = joblib.load(model_path)
            # Load feature names
            with open("/models/feature_names.txt", "r") as f:
                feature_names = [line.strip() for line in f.readlines()]
        
        logger.info("Model loaded successfully")
        logger.info(f"Feature names: {feature_names}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None

def prepare_features(features: PricingFeatures) -> np.ndarray:
    """Prepare features for model prediction"""
    
    # Create feature dictionary
    feature_dict = {
        'MRP': features.MRP,
        'NoPromoPrice': features.NoPromoPrice,
        'SellingPrice': features.SellingPrice,
        'CTR': features.CTR,
        'AbandonedCartRate': features.AbandonedCartRate,
        'BounceRate': features.BounceRate,
        'IsMetro': 1 if features.IsMetro else 0,
        'month': features.month,
        'day': features.day,
        'dayofweek': features.dayofweek,
        'quarter': features.quarter,
        'competitor_price': features.competitor_price
    }
    
    # Add derived features
    feature_dict['discount_rate'] = (features.MRP - features.SellingPrice) / features.MRP * 100
    feature_dict['price_diff'] = features.SellingPrice - features.competitor_price
    feature_dict['price_ratio'] = features.SellingPrice / max(features.competitor_price, 1)
    feature_dict['conversion_efficiency'] = features.CTR * (1 - features.AbandonedCartRate)
    feature_dict['stock_utilization'] = 0.8  # Default value
    feature_dict['stockout_risk'] = 0.2  # Default value
    feature_dict['fulfillment_rate'] = 0.9  # Default value
    feature_dict['engagement_score'] = 100.0  # Default value
    feature_dict['funnel_efficiency'] = 0.7  # Default value
    
    # Create feature array in correct order
    if feature_names:
        feature_array = np.array([feature_dict.get(name, 0) for name in feature_names])
    else:
        # Use default order if feature names not loaded
        feature_array = np.array([
            features.MRP, features.NoPromoPrice, features.SellingPrice,
            features.CTR, features.AbandonedCartRate, features.BounceRate,
            1 if features.IsMetro else 0, features.month, features.day,
            features.dayofweek, features.quarter, features.competitor_price
        ])
    
    return feature_array.reshape(1, -1)

def generate_pricing_recommendation(predicted_units: float, features: PricingFeatures) -> str:
    """Generate pricing recommendation based on prediction"""
    
    # Simple recommendation logic
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

# API Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_units_sold(features: PricingFeatures):
    """Predict units sold based on pricing and market features"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        feature_array = prepare_features(features)
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        prediction = max(0, prediction)  # Ensure non-negative
        
        # Calculate confidence score (simplified)
        # In a real implementation, use prediction intervals or ensemble variance
        confidence = min(0.95, 0.6 + (prediction / 100) * 0.3)
        
        # Generate recommendation
        recommendation = generate_pricing_recommendation(prediction, features)
        
        return PredictionResponse(
            predicted_units_sold=round(prediction, 2),
            confidence_score=round(confidence, 3),
            pricing_recommendation=recommendation,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        predictions_sum = 0
        
        for features in request.predictions:
            # Prepare features
            feature_array = prepare_features(features)
            
            # Make prediction
            prediction = model.predict(feature_array)[0]
            prediction = max(0, prediction)
            predictions_sum += prediction
            
            # Calculate confidence
            confidence = min(0.95, 0.6 + (prediction / 100) * 0.3)
            
            # Generate recommendation
            recommendation = generate_pricing_recommendation(prediction, features)
            
            results.append(PredictionResponse(
                predicted_units_sold=round(prediction, 2),
                confidence_score=round(confidence, 3),
                pricing_recommendation=recommendation,
                timestamp=datetime.now()
            ))
        
        # Calculate summary statistics
        summary = {
            "total_predictions": len(results),
            "average_units_predicted": round(predictions_sum / len(results), 2),
            "total_units_predicted": round(predictions_sum, 2),
            "avg_confidence": round(sum(r.confidence_score for r in results) / len(results), 3)
        }
        
        return BatchPredictionResponse(
            results=results,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_info = {
        "model_type": type(model).__name__,
        "features": feature_names if feature_names else "Not available",
        "feature_count": len(feature_names) if feature_names else 0,
        "loaded_at": datetime.now(),
        "status": "active"
    }
    
    # Add model-specific information
    if hasattr(model, 'n_estimators'):
        model_info["n_estimators"] = model.n_estimators
    if hasattr(model, 'feature_importances_'):
        model_info["has_feature_importance"] = True
    
    return model_info

@app.get("/features/importance")
async def get_feature_importance():
    """Get feature importance from the model"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not hasattr(model, 'feature_importances_'):
        raise HTTPException(status_code=400, detail="Model does not support feature importance")
    
    if not feature_names:
        raise HTTPException(status_code=400, detail="Feature names not available")
    
    importance_data = [
        {"feature": name, "importance": float(importance)}
        for name, importance in zip(feature_names, model.feature_importances_)
    ]
    
    # Sort by importance
    importance_data.sort(key=lambda x: x["importance"], reverse=True)
    
    return {
        "feature_importance": importance_data,
        "total_features": len(importance_data)
    }

@app.post("/optimize/price")
async def optimize_price(features: PricingFeatures, price_range: List[float] = [50, 150]):
    """Find optimal price within a range for maximum predicted units sold"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        min_price, max_price = price_range
        optimal_price = features.SellingPrice
        max_units = 0
        
        # Test different prices
        for test_price in np.arange(min_price, max_price + 1, 1):
            test_features = features.copy()
            test_features.SellingPrice = test_price
            
            feature_array = prepare_features(test_features)
            predicted_units = model.predict(feature_array)[0]
            predicted_units = max(0, predicted_units)
            
            if predicted_units > max_units:
                max_units = predicted_units
                optimal_price = test_price
        
        return {
            "optimal_price": optimal_price,
            "predicted_units_at_optimal": round(max_units, 2),
            "current_price": features.SellingPrice,
            "current_predicted_units": round(max(0, model.predict(prepare_features(features))[0]), 2),
            "price_range_tested": price_range,
            "improvement": round(max_units - max(0, model.predict(prepare_features(features))[0]), 2)
        }
        
    except Exception as e:
        logger.error(f"Price optimization error: {e}")
        raise HTTPException(status_code=500, detail=f"Price optimization failed: {str(e)}")

@app.get("/metrics/model")
async def get_model_metrics():
    """Get model performance metrics (mock implementation)"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # In a real implementation, these would come from model validation
    metrics = {
        "r2_score": 0.85,
        "rmse": 12.5,
        "mae": 8.3,
        "last_trained": "2024-06-01T10:00:00",
        "training_samples": 10000,
        "validation_samples": 2000,
        "test_samples": 2000
    }
    
    return metrics

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
