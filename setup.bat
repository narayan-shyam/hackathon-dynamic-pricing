@echo off
echo 🚀 Setting up Dynamic Pricing Strategy Project...

REM Create virtual environment for API
echo 📦 Setting up API environment...
cd api
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
call venv\Scripts\deactivate
cd ..

REM Create virtual environment for UI
echo 🖥️ Setting up UI environment...
cd ui
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
call venv\Scripts\deactivate
cd ..

REM Create sample data directory
echo 📊 Creating sample data...
if not exist "data\sample" mkdir data\sample

REM Generate sample data using Python
echo Generating sample CSV files...
python -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample sales data
dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='D')
sales_data = []

for date in dates:
    for _ in range(np.random.randint(5, 15)):
        sales_data.append({
            'TransactionDate': date.strftime('%%Y-%%m-%%d'),
            'MRP': np.random.uniform(90, 120),
            'NoPromoPrice': np.random.uniform(80, 110),
            'SellingPrice': np.random.uniform(70, 100),
            'UnitsSold': np.random.randint(20, 100)
        })

sales_df = pd.DataFrame(sales_data)
sales_df.to_csv('data/sample/sales_data.csv', index=False)
print(f'Generated {len(sales_df)} sales records')

# Generate competitor data
competitor_data = []
for date in dates:
    competitor_data.append({
        'Date': date.strftime('%%Y-%%m-%%d'),
        'Brand': 'CompetitorA',
        'MRP': np.random.uniform(95, 125),
        'DiscountRate': np.random.uniform(5, 25),
        'BasePrice': np.random.uniform(80, 110),
        'FinalPrice': np.random.uniform(75, 105)
    })

competitor_df = pd.DataFrame(competitor_data)
competitor_df.to_csv('data/sample/competitor_data.csv', index=False)
print(f'Generated {len(competitor_df)} competitor records')

# Generate customer behavior data
customer_data = []
for date in dates:
    customer_data.append({
        'Date': date.strftime('%%Y-%%m-%%d'),
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
print(f'Generated {len(customer_df)} customer behavior records')

# Generate inventory data
inventory_data = []
for date in dates:
    for fc in ['FC001', 'FC002', 'FC003']:
        inventory_data.append({
            'Date': date.strftime('%%Y-%%m-%%d'),
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
print(f'Generated {len(inventory_df)} inventory records')

print('✅ Sample data generation completed!')
"

echo 📝 Creating configuration files...

REM Create .env file for API
echo # API Configuration > api\.env
echo API_HOST=0.0.0.0 >> api\.env
echo API_PORT=8000 >> api\.env
echo DEBUG=True >> api\.env
echo. >> api\.env
echo # Model Configuration >> api\.env
echo MODEL_PATH=/models/dynamic_pricing_model.pkl >> api\.env
echo FEATURE_NAMES_PATH=/models/feature_names.txt >> api\.env
echo. >> api\.env
echo # Azure Configuration (replace with your values) >> api\.env
echo AZURE_SUBSCRIPTION_ID=your_subscription_id >> api\.env
echo AZURE_RESOURCE_GROUP=tpl-oops-all-ai >> api\.env
echo AZURE_ML_WORKSPACE=oopsallai-mlstudio >> api\.env

REM Create .env file for UI
echo # Streamlit Configuration > ui\.env
echo STREAMLIT_SERVER_PORT=8501 >> ui\.env
echo STREAMLIT_SERVER_ADDRESS=0.0.0.0 >> ui\.env
echo. >> ui\.env
echo # API Configuration >> ui\.env
echo API_BASE_URL=http://localhost:8000 >> ui\.env

echo 🔧 Creating startup scripts...

REM Create API startup script
echo @echo off > start_api.bat
echo echo 🚀 Starting Dynamic Pricing API... >> start_api.bat
echo cd api >> start_api.bat
echo call venv\Scripts\activate >> start_api.bat
echo echo 📡 Starting FastAPI server on http://localhost:8000 >> start_api.bat
echo echo 📚 API Documentation will be available at http://localhost:8000/docs >> start_api.bat
echo uvicorn main:app --host 0.0.0.0 --port 8000 --reload >> start_api.bat

REM Create UI startup script
echo @echo off > start_ui.bat
echo echo 🚀 Starting Dynamic Pricing Dashboard... >> start_ui.bat
echo cd ui >> start_ui.bat
echo call venv\Scripts\activate >> start_ui.bat
echo echo 🖥️ Starting Streamlit dashboard on http://localhost:8501 >> start_ui.bat
echo streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 >> start_ui.bat

echo ✅ Project setup completed!
echo.
echo 📋 Next Steps:
echo 1. Review the generated configuration files
echo 2. Update Azure credentials in .env files
echo 3. Upload notebooks to Databricks workspace
echo 4. Start API: start_api.bat
echo 5. Start UI: start_ui.bat
echo.
echo 🔗 Quick Links:
echo - API: http://localhost:8000
echo - API Docs: http://localhost:8000/docs
echo - Dashboard: http://localhost:8501

pause
