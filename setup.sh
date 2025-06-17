#!/bin/bash

# setup.sh - Project Setup Script for Dynamic Pricing Strategy

echo "🚀 Setting up Dynamic Pricing Strategy Project..."

# Create virtual environment for API
echo "📦 Setting up API environment..."
cd api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
cd ..

# Create virtual environment for UI
echo "🖥️ Setting up UI environment..."
cd ui
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
cd ..

# Create sample data directory
echo "📊 Creating sample data..."
mkdir -p data/sample

# Create sample CSV files for testing
python << EOF
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample sales data
dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='D')
sales_data = []

for date in dates:
    for _ in range(np.random.randint(5, 15)):  # 5-15 transactions per day
        sales_data.append({
            'TransactionDate': date.strftime('%Y-%m-%d'),
            'MRP': np.random.uniform(90, 120),
            'NoPromoPrice': np.random.uniform(80, 110),
            'SellingPrice': np.random.uniform(70, 100),
            'UnitsSold': np.random.randint(20, 100)
        })

sales_df = pd.DataFrame(sales_data)
sales_df.to_csv('data/sample/sales_data.csv', index=False)
print(f"Generated {len(sales_df)} sales records")

# Generate competitor data
competitor_data = []
for date in dates:
    competitor_data.append({
        'Date': date.strftime('%Y-%m-%d'),
        'Brand': 'CompetitorA',
        'MRP': np.random.uniform(95, 125),
        'DiscountRate': np.random.uniform(5, 25),
        'BasePrice': np.random.uniform(80, 110),
        'FinalPrice': np.random.uniform(75, 105)
    })

competitor_df = pd.DataFrame(competitor_data)
competitor_df.to_csv('data/sample/competitor_data.csv', index=False)
print(f"Generated {len(competitor_df)} competitor records")

# Generate customer behavior data
customer_data = []
for date in dates:
    customer_data.append({
        'Date': date.strftime('%Y-%m-%d'),
        'CTR': np.random.uniform(0.01, 0.05),
        'AbandonedCartRate': np.random.uniform(0.1, 0.3),
        'BounceRate': np.random.uniform(0.2, 0.5),
        'FunnelDrop_ViewToCart': np.random.uniform(0.1, 0.4),
        'FunnelDrop_CartToCheckout': np.random.uniform(0.1, 0.3),
        'ReturningVisitorRatio': np.random.uniform(0.2, 0.6),
        'AvgSessionDuration_sec': np.random.uniform(60, 300)
    })

customer_df = pd.DataFrame(customer_data)
customer_df.to_csv('data/sample/customer_behavior_data.csv', index=False)
print(f"Generated {len(customer_df)} customer behavior records")

# Generate inventory data
inventory_data = []
for date in dates:
    for fc in ['FC001', 'FC002', 'FC003']:
        inventory_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'FC_ID': fc,
            'IsMetro': fc in ['FC001', 'FC002'],
            'StockStart': np.random.randint(100, 500),
            'Demand': np.random.randint(50, 200),
            'DemandFulfilled': np.random.randint(40, 180),
            'Backorders': np.random.randint(0, 20),
            'StockEnd': np.random.randint(50, 300),
            'ReorderPoint': 100,
            'OrderPlaced': np.random.choice([0, 1]),
            'OrderQty': np.random.randint(0, 100),
            'LeadTimeFloat': np.random.uniform(1, 7),
            'SafetyStock': 50
        })

inventory_df = pd.DataFrame(inventory_data)
inventory_df.to_csv('data/sample/inventory_data.csv', index=False)
print(f"Generated {len(inventory_df)} inventory records")

print("✅ Sample data generation completed!")
EOF

echo "📝 Creating configuration files..."

# Create .env file for API
cat > api/.env << EOF
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Model Configuration
MODEL_PATH=/models/dynamic_pricing_model.pkl
FEATURE_NAMES_PATH=/models/feature_names.txt

# Azure Configuration (replace with your values)
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_RESOURCE_GROUP=tpl-oops-all-ai
AZURE_ML_WORKSPACE=oopsallai-mlstudio

# MLflow Configuration
MLFLOW_TRACKING_URI=databricks://your_databricks_workspace
EOF

# Create .env file for UI
cat > ui/.env << EOF
# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# API Configuration
API_BASE_URL=http://localhost:8000
EOF

echo "🔧 Creating startup scripts..."

# Create API startup script
cat > start_api.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Dynamic Pricing API..."

cd api
source venv/bin/activate

echo "📡 Starting FastAPI server on http://localhost:8000"
echo "📚 API Documentation will be available at http://localhost:8000/docs"

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
EOF

# Create UI startup script
cat > start_ui.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Dynamic Pricing Dashboard..."

cd ui
source venv/bin/activate

echo "🖥️ Starting Streamlit dashboard on http://localhost:8501"

streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
EOF

# Create combined startup script
cat > start_all.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Complete Dynamic Pricing System..."

# Function to start API in background
start_api() {
    echo "📡 Starting API server..."
    cd api
    source venv/bin/activate
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
    API_PID=$!
    cd ..
    echo "API started with PID: $API_PID"
}

# Function to start UI
start_ui() {
    echo "🖥️ Starting UI dashboard..."
    sleep 5  # Wait for API to start
    cd ui
    source venv/bin/activate
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
    UI_PID=$!
    cd ..
    echo "UI started with PID: $UI_PID"
}

# Cleanup function
cleanup() {
    echo "🛑 Shutting down services..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null
        echo "API server stopped"
    fi
    if [ ! -z "$UI_PID" ]; then
        kill $UI_PID 2>/dev/null
        echo "UI dashboard stopped"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start services
start_api
start_ui

echo ""
echo "✅ System is running!"
echo "📡 API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🖥️ Dashboard: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for services
wait
EOF

# Make scripts executable
chmod +x start_api.sh
chmod +x start_ui.sh
chmod +x start_all.sh

echo "📋 Creating project documentation..."

# Create development guide
cat > DEVELOPMENT.md << 'EOF'
# Development Guide

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment support

### Setup
1. Run the setup script:
   ```bash
   bash setup.sh
   ```

2. This will:
   - Create virtual environments for API and UI
   - Install all required dependencies
   - Generate sample data
   - Create configuration files

### Running the System

#### Option 1: Start Everything (Recommended)
```bash
bash start_all.sh
```

#### Option 2: Start Components Separately

**API Only:**
```bash
bash start_api.sh
```

**UI Only:**
```bash
bash start_ui.sh
```

### Accessing the System
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501

## Development Workflow

### 1. Data Processing (Databricks)
- Upload the notebooks to your Databricks workspace
- Upload sample data to DBFS
- Run `01_data_preprocessing.py` to process data
- Run `02_model_training.py` to train models
- Run `03_testing_framework.py` to validate

### 2. API Development
- API code is in `api/main.py`
- Models are loaded from `/models/` directory
- Add new endpoints by extending the FastAPI app

### 3. UI Development
- Dashboard code is in `ui/streamlit_app.py`
- Add new pages by extending the main() function
- New visualizations can be added using Plotly

### 4. Testing
- Unit tests are included in the testing notebook
- API testing can be done via the `/docs` endpoint
- UI testing through the dashboard interface

## Configuration

### API Configuration (api/.env)
- `API_HOST`: API host address
- `API_PORT`: API port number
- `MODEL_PATH`: Path to trained model
- Azure and MLflow settings

### UI Configuration (ui/.env)
- `STREAMLIT_SERVER_PORT`: Dashboard port
- `API_BASE_URL`: API endpoint URL

## Troubleshooting

### Common Issues

1. **Port already in use**
   - Kill existing processes: `pkill -f uvicorn` or `pkill -f streamlit`
   - Use different ports in configuration

2. **Model not found**
   - Ensure model training was completed
   - Check model path in configuration
   - Use demo mode for development

3. **API connection failed**
   - Verify API is running on correct port
   - Check firewall settings
   - Ensure no proxy blocking connections

### Logs
- API logs: Check terminal output where API was started
- UI logs: Check terminal output where dashboard was started

## Project Structure
```
├── api/                    # FastAPI backend
├── ui/                     # Streamlit frontend  
├── notebooks/              # Databricks notebooks
├── data/                   # Data files
├── tests/                  # Test files
├── start_*.sh             # Startup scripts
└── README.md              # Project documentation
```
EOF

echo "🔒 Creating security guidelines..."

cat > SECURITY.md << 'EOF'
# Security Guidelines

## Environment Variables
- Never commit `.env` files to version control
- Use Azure Key Vault for production secrets
- Rotate API keys regularly

## API Security
- Enable HTTPS in production
- Implement authentication for production use
- Validate all input data
- Use rate limiting

## Data Protection
- Encrypt sensitive data at rest
- Use secure data transmission
- Implement proper access controls
- Regular security audits

## Azure Security
- Use managed identities when possible
- Follow Azure Security Center recommendations
- Enable monitoring and alerting
- Regular security updates
EOF

echo "✅ Project setup completed!"
echo ""
echo "📋 Next Steps:"
echo "1. Review the generated configuration files"
echo "2. Update Azure credentials in .env files"
echo "3. Upload notebooks to Databricks workspace"
echo "4. Run the setup: bash setup.sh"
echo "5. Start the system: bash start_all.sh"
echo ""
echo "🔗 Quick Links:"
echo "- API: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
echo "- Dashboard: http://localhost:8501"
echo ""
echo "📚 Documentation:"
echo "- README.md - Project overview"
echo "- DEVELOPMENT.md - Development guide"
echo "- SECURITY.md - Security guidelines"
