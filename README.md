# Dynamic Pricing Strategy for GlobalMart Tide Detergent
# AI-Assisted Automated Deployment of ML Models

## Project Overview
This project implements an end-to-end machine learning pipeline for dynamic pricing strategy using Azure technologies, FastAPI, and Streamlit.

## Project Structure
```
hackathon-dynamic-pricing/
├── notebooks/               # Databricks notebooks
│   ├── 01_data_preprocessing.py    # Phase 1: Data processing
│   ├── 02_model_training.py        # Phase 2: Model training
│   └── 03_testing_framework.py     # Phase 3: Testing
├── api/                    # FastAPI backend
│   ├── main.py            # API application
│   └── requirements.txt   # API dependencies
├── ui/                    # Streamlit frontend
│   ├── streamlit_app.py   # Dashboard application
│   └── requirements.txt   # UI dependencies
├── data/                  # Data files
├── tests/                 # Test files
└── README.md             # This file
```

## Azure Resources
- **Resource Group**: tpl-oops-all-ai
- **Azure Databricks**: oops_all_ai_ad
- **Key Vault**: oopsallai-kv
- **ML Workspace**: oopsallai-mlstudio

## Quick Start

### 1. Data Processing (Phase 1)
Run the Databricks notebook `01_data_preprocessing.py`:
- Upload your CSV files to DBFS
- Execute the notebook to clean and engineer features
- Processed data will be saved for model training

### 2. Model Training (Phase 2)
Run the Databricks notebook `02_model_training.py`:
- Trains multiple ML models
- Performs hyperparameter tuning
- Registers best model with MLflow
- Saves model for API deployment

### 3. Testing (Phase 3)
Run the testing notebook `03_testing_framework.py`:
- Comprehensive unit tests
- Integration tests
- Smoke tests
- Test coverage reporting

### 4. API Deployment (Phase 7)
Start the FastAPI backend:
```bash
cd api
# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at: http://localhost:8000
API Documentation: http://localhost:8000/docs

### 5. UI Dashboard (Phase 7)
Start the Streamlit frontend:
```bash
cd ui
# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Dashboard will be available at: http://localhost:8501

## API Endpoints

### Health Check
- `GET /health` - Check API health status

### Predictions
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

### Optimization
- `POST /optimize/price` - Find optimal price

### Model Information
- `GET /model/info` - Model metadata
- `GET /features/importance` - Feature importance
- `GET /metrics/model` - Model performance metrics

## Features

### Data Processing
- Automated data cleaning and validation
- Feature engineering for pricing optimization
- Date-based feature extraction
- Data integration from multiple sources

### Model Training
- Multiple algorithm comparison (Random Forest, XGBoost, Gradient Boosting)
- Automated hyperparameter tuning
- MLflow experiment tracking
- Model registration and versioning

### Testing Framework
- Unit tests for data processing
- Integration tests for ML pipeline
- Smoke tests for basic functionality
- Comprehensive test reporting

### API Features
- RESTful API with FastAPI
- Input validation with Pydantic models
- Error handling and logging
- Interactive API documentation

### Dashboard Features
- Single prediction interface
- Batch prediction capabilities
- Price optimization tools
- Model analytics and insights
- Historical data analysis

## Data Schema

### Input Features
- **MRP**: Maximum Retail Price
- **NoPromoPrice**: Price without promotion
- **SellingPrice**: Current selling price
- **CTR**: Click-through rate
- **AbandonedCartRate**: Cart abandonment rate
- **BounceRate**: Website bounce rate
- **IsMetro**: Metro city location flag
- **Date Features**: Month, day, day of week, quarter
- **competitor_price**: Competitor pricing

### Target Variable
- **UnitsSold**: Number of units sold

## Model Performance
- R² Score: ~0.85
- RMSE: ~12.5
- MAE: ~8.3

## Deployment Architecture
1. **Data Processing**: Azure Databricks
2. **Model Training**: Azure ML Studio
3. **Model Storage**: MLflow Model Registry
4. **API Backend**: FastAPI
5. **Frontend**: Streamlit Dashboard
6. **Infrastructure**: Azure Resource Group

## Best Practices Implemented
- Modular code architecture
- Comprehensive testing
- Error handling and logging
- API documentation
- Feature engineering automation
- Model performance monitoring
- Version control with MLflow

## Usage Examples

### Single Prediction
```python
import requests

prediction_data = {
    "MRP": 100.0,
    "NoPromoPrice": 90.0,
    "SellingPrice": 80.0,
    "CTR": 0.025,
    "AbandonedCartRate": 0.2,
    "BounceRate": 0.3,
    "IsMetro": True,
    "month": 6,
    "day": 15,
    "dayofweek": 3,
    "quarter": 2,
    "competitor_price": 85.0
}

response = requests.post("http://localhost:8000/predict", json=prediction_data)
result = response.json()
print(f"Predicted Units: {result['predicted_units_sold']}")
```

### Price Optimization
The system can find optimal pricing to maximize units sold or revenue within specified constraints.

## Troubleshooting

### API Connection Issues
- Ensure FastAPI server is running on port 8000
- Check firewall settings
- Verify all dependencies are installed

### Model Loading Issues
- Ensure model files exist in the correct path
- Check file permissions
- Verify MLflow model registry access

### Dashboard Issues
- Ensure Streamlit is running on port 8501
- Check API connectivity
- Verify all UI dependencies are installed

## Future Enhancements
- Real-time model retraining
- A/B testing framework
- Advanced price elasticity modeling
- Integration with inventory management
- Multi-product pricing optimization

## Support
For issues and questions, please refer to the troubleshooting section or contact the development team.
