# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2: Model Training and Azure ML Deployment
# MAGIC ## Dynamic Pricing Model for GlobalMart Tide Detergent

# COMMAND ----------

# MAGIC %pip install mlflow pandas scikit-learn xgboost azureml-sdk azure-ai-ml

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime
import joblib

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Processed Data

# COMMAND ----------

# Load the processed data
data_path = "/dbfs/FileStore/processed/processed_pricing_data.csv"
df = pd.read_csv(data_path)

print(f"Loaded data shape: {df.shape}")
print("\nData info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Preparation for Modeling

# COMMAND ----------

# Separate features and target
target_col = 'UnitsSold'
feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols]
y = df[target_col]

print(f"Features: {len(feature_cols)}")
print(f"Target variable: {target_col}")
print(f"Feature columns: {feature_cols}")

# Check for any remaining missing values
print(f"\nMissing values in features: {X.isnull().sum().sum()}")
print(f"Missing values in target: {y.isnull().sum()}")

# COMMAND ----------

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Further split training data for validation
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"Training set: {X_train_final.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Model Training and Hyperparameter Tuning

# COMMAND ----------

# Set MLflow experiment
mlflow.set_experiment("/Users/databricks/dynamic_pricing_experiment")

def train_and_evaluate_model(model, model_name, X_train, X_val, y_train, y_val, params=None):
    """Train model and log results to MLflow"""
    
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Hyperparameter tuning if params provided
        if params:
            grid_search = GridSearchCV(model, params, cv=3, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            mlflow.log_params(grid_search.best_params_)
        else:
            best_model = model
            best_model.fit(X_train, y_train)
            mlflow.log_params(model.get_params())
        
        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        # Log metrics
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("val_r2", val_r2)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("val_mae", val_mae)
        
        # Log model
        if 'xgb' in model_name.lower():
            mlflow.xgboost.log_model(best_model, "model")
        else:
            mlflow.sklearn.log_model(best_model, "model")
        
        # Log feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
        
        print(f"{model_name} Results:")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Val R²: {val_r2:.4f}")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Val RMSE: {val_rmse:.4f}")
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Val MAE: {val_mae:.4f}")
        
        return best_model, val_r2

# COMMAND ----------

# Define models and their hyperparameters
models_config = {
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
    },
    "XGBoost": {
        "model": xgb.XGBRegressor(random_state=42),
        "params": {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    },
    "Linear Regression": {
        "model": LinearRegression(),
        "params": None
    }
}

# Train all models
trained_models = {}
model_scores = {}

for model_name, config in models_config.items():
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    model, score = train_and_evaluate_model(
        config["model"], 
        model_name, 
        X_train_final, 
        X_val, 
        y_train_final, 
        y_val, 
        config["params"]
    )
    
    trained_models[model_name] = model
    model_scores[model_name] = score

# COMMAND ----------

# Select best model
best_model_name = max(model_scores, key=model_scores.get)
best_model = trained_models[best_model_name]
best_score = model_scores[best_model_name]

print(f"\n{'='*50}")
print(f"BEST MODEL: {best_model_name}")
print(f"Validation R² Score: {best_score:.4f}")
print(f"{'='*50}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Final Model Evaluation on Test Set

# COMMAND ----------

# Retrain best model on full training set
best_model.fit(X_train, y_train)

# Final evaluation on test set
y_test_pred = best_model.predict(X_test)

test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"Final Test Results for {best_model_name}:")
print(f"Test R²: {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Model Registration and Deployment Preparation

# COMMAND ----------

# Register the best model with MLflow
with mlflow.start_run(run_name=f"final_{best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    
    # Log final metrics
    mlflow.log_metric("final_test_r2", test_r2)
    mlflow.log_metric("final_test_rmse", test_rmse)
    mlflow.log_metric("final_test_mae", test_mae)
    
    # Log model parameters
    mlflow.log_params(best_model.get_params())
    
    # Register model
    if 'xgb' in best_model_name.lower():
        model_uri = mlflow.xgboost.log_model(
            best_model, 
            "model",
            registered_model_name="dynamic_pricing_model"
        )
    else:
        model_uri = mlflow.sklearn.log_model(
            best_model, 
            "model",
            registered_model_name="dynamic_pricing_model"
        )
    
    # Log feature names
    feature_names = list(X_train.columns)
    mlflow.log_text(str(feature_names), "feature_names.txt")
    
    # Save model locally for API
    model_path = "/dbfs/FileStore/models/dynamic_pricing_model.pkl"
    joblib.dump(best_model, model_path)
    
    # Save feature names
    with open("/dbfs/FileStore/models/feature_names.txt", "w") as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print(f"Model registered successfully!")
    print(f"Model URI: {model_uri}")
    print(f"Model saved to: {model_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Feature Importance Analysis

# COMMAND ----------

if hasattr(best_model, 'feature_importances_'):
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance_df.head(10))
    
    # Save feature importance
    feature_importance_df.to_csv("/dbfs/FileStore/models/feature_importance.csv", index=False)

# COMMAND ----------

# Create prediction function for deployment
def predict_units_sold(features_dict):
    """
    Predict units sold based on input features
    """
    # Load model
    model = joblib.load("/dbfs/FileStore/models/dynamic_pricing_model.pkl")
    
    # Create feature array in correct order
    feature_order = X_train.columns.tolist()
    features_array = np.array([features_dict.get(feature, 0) for feature in feature_order]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features_array)[0]
    
    return prediction

# Test the prediction function
sample_features = X_test.iloc[0].to_dict()
sample_prediction = predict_units_sold(sample_features)
actual_value = y_test.iloc[0]

print(f"Sample prediction test:")
print(f"Predicted: {sample_prediction:.2f}")
print(f"Actual: {actual_value:.2f}")
print(f"Error: {abs(sample_prediction - actual_value):.2f}")

print("\nPhase 2 Model Training and Deployment Preparation completed successfully!")
