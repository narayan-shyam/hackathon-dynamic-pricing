"""
Project Structure Creation with Production-Ready Templates
"""

from pathlib import Path
from typing import Dict, Any
import logging

# Update to absolute imports or provide stubs if not present
# from exceptions import DirectoryCreationError
# from utilities import DataValidator, retry

logger = logging.getLogger(__name__)


class DirectoryCreationError(Exception):
    pass

def retry(max_attempts=3, delay=0.5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

class DataValidator:
    def validate(self, *args, **kwargs):
        return True

class ProjectStructureManager:
    """Manages creation of production-ready ML project structure with all necessary components"""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.validator = DataValidator()

    def create_project_structure(self):
        """Create comprehensive ML project directory structure with all templates"""

        project_structure = {
            "api": {
                "main.py": "",
                "requirements.txt": "",
                "Dockerfile": "",
                "venv/": {},
            },
            "ui": {
                "streamlit_app.py": "",
                "requirements.txt": "",
                "Dockerfile": "",
                "venv/": {},
            },
            "notebooks": {
                "01_data_preprocessing.py": "",
                "02_model_training.py": "",
                "03_testing_framework.py": "",
            },
            "monitoring": {
                "04_monitoring_and_logging.py": "",
            },
            "retraining": {
                "05_automated_retraining.py": "",
            },
            "data": {
                "sample": {},
                "processed": {},
            },
            "models": {},
            "tests": {},
            "docs": {},
            ".github": {
                "workflows": {},
            },
            "generate_sample_data.py": "",
            "requirements.txt": self._get_requirements_template(),
            "setup.sh": "",
            "setup.bat": "",
            "docker-compose.yml": self._get_docker_compose_template(),
            "docker-compose.prod.yml": "",
            ".gitignore": self._get_gitignore_template(),
            "README.md": self._get_readme_template(),
            "PROJECT_SUMMARY.md": "",
        }

        self._create_structure_recursive(self.base_path, project_structure)
        logger.info(f"Project structure created successfully at: {self.base_path}")

    @retry(max_attempts=3, delay=0.5)
    def _create_structure_recursive(self, current_path: Path, structure: Dict[str, Any]):
        """Recursively create directory structure with error handling"""
        try:
            current_path.mkdir(parents=True, exist_ok=True)
            for name, content in structure.items():
                item_path = current_path / name
                if isinstance(content, dict):
                    self._create_structure_recursive(item_path, content)
                else:
                    # Only create file if content is not empty
                    if content != "" or not item_path.exists():
                        with open(item_path, 'w', encoding='utf-8') as f:
                            f.write(content)
        except Exception as e:
            logger.error(f"Failed to create structure at {current_path}: {str(e)}")
            raise DirectoryCreationError(f"Directory creation failed: {str(e)}")

    def _get_requirements_template(self) -> str:
        return '''# Core ML libraries
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
mlflow-skinny>=2.0.0

# Data processing
PyYAML>=6.0
joblib>=1.3.0

# Azure integration
azure-keyvault-secrets>=4.7.0
azure-identity>=1.13.0
azure-monitor-opentelemetry>=1.0.0

# Development tools
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0

# API and web
fastapi>=0.100.0
uvicorn>=0.23.0
requests>=2.31.0

# Utilities
click>=8.1.0
tqdm>=4.65.0
python-dotenv>=1.0.0
'''

    def _get_docker_compose_template(self) -> str:
        return '''version: '3.8'

services:
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
      - AZURE_SUBSCRIPTION_ID=${AZURE_SUBSCRIPTION_ID}
      - AZURE_RESOURCE_GROUP=${AZURE_RESOURCE_GROUP}
      - AZURE_CLIENT_ID=${AZURE_CLIENT_ID}
      - AZURE_CLIENT_SECRET=${AZURE_CLIENT_SECRET}
      - AZURE_TENANT_ID=${AZURE_TENANT_ID}
      - MODEL_PATH=/app/models/dynamic_pricing_model.pkl
      - LOG_LEVEL=INFO
      - ENVIRONMENT=development
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs

  ui:
    build:
      context: ./ui
      dockerfile: Dockerfile
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://api:8000
    depends_on:
      - api

volumes:
  models:
  logs:
'''

    def _get_gitignore_template(self) -> str:
        return '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# Environment variables
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# MLflow
mlruns/
mlartifacts/

# Model files
*.pkl
*.joblib
*.h5
*.pb

# Data files
*.csv
*.xlsx
*.json
data/raw/
data/processed/

# Logs
*.log
logs/

# Streamlit
.streamlit/

# OS
.DS_Store
Thumbs.db

# Azure
.azure/

# Temporary files
*.tmp
*.temp
temp/
'''

    def _get_readme_template(self) -> str:
        return '''# Dynamic Pricing Strategy for GlobalMart Tide Detergent

## Overview
Production-ready machine learning project for pricing prediction with comprehensive MLOps practices.

## Features
- ✅ End-to-end ML pipeline (preprocessing, training, testing, monitoring, retraining)
- ✅ FastAPI backend and Streamlit dashboard
- ✅ Azure ML, MLflow, Docker, and CI/CD integration
- ✅ Automated testing and monitoring

## Setup

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Azure account (for cloud deployment)

### Installation
```bash
git clone <repository-url>
cd hackathon-dynamic-pricing
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## Usage

### Local Development
```bash
python generate_sample_data.py
python notebooks/01_data_preprocessing.py
python notebooks/02_model_training.py
python notebooks/03_testing_framework.py
```

### Docker
```bash
docker-compose up --build
```

## Access Points

- **UI (Streamlit Dashboard)**: https://oopsallaiui.azurewebsites.net/
- **API (FastAPI Docs)**: https://oopsallaiapi.azurewebsites.net/docs/

## License
MIT License - see LICENSE file for details.
'''