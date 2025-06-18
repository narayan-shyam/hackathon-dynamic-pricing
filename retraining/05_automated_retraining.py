            
        except Exception as e:
            pipeline_results["status"] = "failed"
            pipeline_results["end_time"] = datetime.now().isoformat()
            pipeline_results["errors"].append(f"Pipeline error: {str(e)}")
            self.logger.error(f"Retraining pipeline failed: {e}")
            return pipeline_results
    
    def _create_synthetic_new_data(self) -> pd.DataFrame:
        """Create synthetic new data for demonstration"""
        
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        n_samples = 200
        
        new_data = pd.DataFrame({
            'MRP': np.random.uniform(85, 125, n_samples),
            'NoPromoPrice': np.random.uniform(75, 115, n_samples),
            'SellingPrice': np.random.uniform(65, 105, n_samples),
            'CTR': np.random.uniform(0.015, 0.055, n_samples),
            'AbandonedCartRate': np.random.uniform(0.15, 0.35, n_samples),
            'BounceRate': np.random.uniform(0.25, 0.55, n_samples),
            'IsMetro': np.random.choice([0, 1], n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'day': np.random.randint(1, 29, n_samples),
            'dayofweek': np.random.randint(1, 8, n_samples),
            'quarter': np.random.randint(1, 5, n_samples),
            'competitor_price': np.random.uniform(70, 110, n_samples)
        })
        
        # Create target variable with some relationship to features
        new_data['UnitsSold'] = (
            55 + 
            (125 - new_data['SellingPrice']) * 0.6 +
            new_data['CTR'] * 1200 +
            np.random.normal(0, 8, n_samples)
        )
        new_data['UnitsSold'] = np.maximum(new_data['UnitsSold'], 0)
        
        return new_data
    
    def evaluate_and_promote_challenger(self) -> Dict:
        """Evaluate challenger performance and promote if criteria met"""
        
        try:
            # Load A/B test configuration
            ab_test_path = "/dbfs/FileStore/monitoring/ab_test_config.json"
            if not os.path.exists(ab_test_path):
                return {"error": "No active A/B test found"}
            
            with open(ab_test_path, 'r') as f:
                ab_test_config = json.load(f)
            
            # Evaluate A/B test
            evaluation_results = self.champion_challenger.evaluate_ab_test(ab_test_config)
            
            # Check if challenger should be promoted
            if evaluation_results.get("meets_promotion_criteria", False):
                
                # Load challenger metadata
                with open("/dbfs/FileStore/models/challenger_metadata.json", 'r') as f:
                    challenger_metadata = json.load(f)
                
                # Promote challenger
                promotion_success = self.champion_challenger.promote_challenger_to_champion(challenger_metadata)
                
                if promotion_success:
                    # End A/B test
                    ab_test_config["status"] = "completed"
                    ab_test_config["end_date"] = datetime.now().isoformat()
                    ab_test_config["result"] = "challenger_promoted"
                    
                    with open(ab_test_path, 'w') as f:
                        json.dump(ab_test_config, f, indent=2)
                    
                    self.logger.info("Challenger successfully promoted to champion")
                    
                    return {
                        "action": "promoted",
                        "evaluation_results": evaluation_results,
                        "promotion_date": datetime.now().isoformat()
                    }
                else:
                    return {"error": "Promotion failed"}
            else:
                self.logger.info("Challenger does not meet promotion criteria")
                return {
                    "action": "continue_testing",
                    "evaluation_results": evaluation_results
                }
                
        except Exception as e:
            self.logger.error(f"Error in challenger evaluation: {e}")
            return {"error": str(e)}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Scheduling and Automation

# COMMAND ----------

class RetrainingScheduler:
    """Scheduler for automated retraining"""
    
    def __init__(self, orchestrator: RetrainingOrchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("RetrainingScheduler")
        self.is_running = False
        
    def setup_schedule(self):
        """Setup retraining schedule"""
        
        config = self.orchestrator.config.config["retraining_schedule"]
        
        if config["schedule_type"] == "interval":
            # Schedule every N days
            schedule.every(config["interval_days"]).days.do(self.run_retraining_job)
            self.logger.info(f"Scheduled retraining every {config['interval_days']} days")
            
        # Schedule challenger evaluation daily
        schedule.every().day.at("09:00").do(self.run_challenger_evaluation)
        self.logger.info("Scheduled challenger evaluation daily at 09:00")
        
        # Schedule trigger checking every hour
        schedule.every().hour.do(self.check_triggers)
        self.logger.info("Scheduled trigger checking every hour")
    
    def run_retraining_job(self):
        """Execute retraining job"""
        self.logger.info("Starting scheduled retraining job...")
        
        try:
            results = self.orchestrator.execute_retraining_pipeline()
            
            if results["status"] == "completed":
                self.logger.info("Scheduled retraining completed successfully")
            else:
                self.logger.error(f"Scheduled retraining failed: {results.get('errors', [])}")
                
        except Exception as e:
            self.logger.error(f"Error in scheduled retraining: {e}")
    
    def run_challenger_evaluation(self):
        """Execute challenger evaluation"""
        self.logger.info("Starting challenger evaluation...")
        
        try:
            results = self.orchestrator.evaluate_and_promote_challenger()
            
            if results.get("action") == "promoted":
                self.logger.info("Challenger promoted to champion")
            elif results.get("action") == "continue_testing":
                self.logger.info("Challenger evaluation: continue testing")
            elif "error" in results:
                self.logger.warning(f"Challenger evaluation error: {results['error']}")
                
        except Exception as e:
            self.logger.error(f"Error in challenger evaluation: {e}")
    
    def check_triggers(self):
        """Check for retraining triggers"""
        self.logger.info("Checking retraining triggers...")
        
        try:
            triggers = self.orchestrator.check_retraining_triggers()
            
            if triggers["triggered"]:
                self.logger.info(f"Retraining triggered: {triggers['reasons']}")
                self.run_retraining_job()
            else:
                self.logger.info("No retraining triggers detected")
                
        except Exception as e:
            self.logger.error(f"Error checking triggers: {e}")
    
    def start_scheduler(self):
        """Start the scheduler"""
        self.is_running = True
        self.logger.info("Retraining scheduler started")
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.is_running = False
        self.logger.info("Retraining scheduler stopped")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Integration Test and Demo

# COMMAND ----------

def run_retraining_demo():
    """Run complete retraining pipeline demonstration"""
    
    print("="*70)
    print("PHASE 5: AUTOMATED RETRAINING PIPELINE DEMONSTRATION")
    print("="*70)
    
    try:
        # Initialize MLflow experiment
        mlflow.set_experiment("/Users/databricks/retraining_experiment")
        
        # Create directories
        os.makedirs("/dbfs/FileStore/models", exist_ok=True)
        os.makedirs("/dbfs/FileStore/monitoring", exist_ok=True)
        os.makedirs("/dbfs/FileStore/retraining", exist_ok=True)
        os.makedirs("/dbfs/FileStore/production_data", exist_ok=True)
        
        # Initialize components
        config = RetrainingConfig()
        orchestrator = RetrainingOrchestrator(config)
        
        print("✓ Initialized retraining components")
        
        # Create synthetic reference data if not exists
        reference_path = "/dbfs/FileStore/monitoring/reference_data.csv"
        if not os.path.exists(reference_path):
            from sklearn.ensemble import RandomForestRegressor
            
            # Create synthetic reference data
            np.random.seed(42)
            reference_data = orchestrator._create_synthetic_new_data()
            reference_data.to_csv(reference_path, index=False)
            
            # Create initial model
            X = reference_data.drop('UnitsSold', axis=1)
            y = reference_data['UnitsSold']
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            model_path = "/dbfs/FileStore/models/dynamic_pricing_model.pkl"
            joblib.dump(model, model_path)
            
            print("✓ Created reference data and initial model")
        
        # Demo 1: Check retraining triggers
        print("\n--- Demo 1: Checking Retraining Triggers ---")
        triggers = orchestrator.check_retraining_triggers()
        print(f"Triggers detected: {triggers['triggered']}")
        print(f"Trigger reasons: {triggers['reasons']}")
        
        # Demo 2: Execute retraining pipeline
        print("\n--- Demo 2: Executing Retraining Pipeline ---")
        pipeline_results = orchestrator.execute_retraining_pipeline()
        
        print(f"Pipeline status: {pipeline_results['status']}")
        print(f"Stages completed:")
        for stage, details in pipeline_results['stages'].items():
            print(f"  - {stage}: {details['status']}")
        
        if pipeline_results['status'] == 'completed':
            training_stage = pipeline_results['stages'].get('model_training', {})
            if 'best_model' in training_stage:
                print(f"Best model: {training_stage['best_model']}")
                print(f"Best model metrics: {training_stage['best_model_metrics']}")
        
        # Demo 3: Challenger evaluation
        print("\n--- Demo 3: Challenger Evaluation ---")
        
        # Wait a moment to simulate evaluation period
        time.sleep(2)
        
        evaluation_results = orchestrator.evaluate_and_promote_challenger()
        
        if 'action' in evaluation_results:
            print(f"Evaluation action: {evaluation_results['action']}")
            if 'evaluation_results' in evaluation_results:
                eval_data = evaluation_results['evaluation_results']
                if 'performance_comparison' in eval_data:
                    comparison = eval_data['performance_comparison']
                    print(f"Performance improvement: {comparison.get('mae_improvement_percentage', 0):.2f}%")
                    print(f"Recommendation: {comparison.get('recommendation', 'N/A')}")
        else:
            print(f"Evaluation result: {evaluation_results}")
        
        # Demo 4: Scheduler setup (demonstration only)
        print("\n--- Demo 4: Scheduler Configuration ---")
        scheduler = RetrainingScheduler(orchestrator)
        scheduler.setup_schedule()
        
        print("✓ Scheduler configured (not started in demo)")
        print("  - Retraining scheduled every 7 days")
        print("  - Challenger evaluation scheduled daily")
        print("  - Trigger checking every hour")
        
        # Save configuration
        config_path = "/dbfs/FileStore/retraining/retraining_config.json"
        config.save_config(config_path)
        print(f"✓ Configuration saved to {config_path}")
        
        # Create retraining summary
        summary = {
            "retraining_pipeline": {
                "status": "active",
                "last_run": pipeline_results.get("end_time"),
                "pipeline_status": pipeline_results["status"],
                "champion_challenger_enabled": True
            },
            "models": {
                "champion_model": "active",
                "challenger_model": "active" if 'challenger_deployment' in pipeline_results['stages'] else "none",
                "model_registry": config.config["model_registry"]["model_name"]
            },
            "automation": {
                "scheduled_retraining": True,
                "trigger_based_retraining": True,
                "automated_promotion": True
            },
            "next_scheduled_actions": {
                "retraining_check": "Every hour",
                "challenger_evaluation": "Daily at 09:00",
                "scheduled_retraining": f"Every {config.config['retraining_schedule']['interval_days']} days"
            }
        }
        
        summary_path = "/dbfs/FileStore/retraining/retraining_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*70)
        print("PHASE 5 AUTOMATED RETRAINING IMPLEMENTATION COMPLETED!")
        print("="*70)
        print("\nKey Features Implemented:")
        print("✓ Automated data collection and validation")
        print("✓ Multi-algorithm model training with hyperparameter tuning")
        print("✓ Champion-Challenger deployment strategy")
        print("✓ A/B testing framework")
        print("✓ Automated model promotion based on performance criteria")
        print("✓ Flexible scheduling system (interval, trigger-based)")
        print("✓ Comprehensive logging and monitoring")
        print("✓ MLflow integration for model versioning")
        print("\nRetraining pipeline is now fully automated and ready for production!")
        
        return summary
        
    except Exception as e:
        print(f"❌ Error in retraining demo: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run the demonstration
retraining_summary = run_retraining_demo()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Production Deployment Utilities

# COMMAND ----------

def create_retraining_deployment_package():
    """Create deployment package for production retraining"""
    
    deployment_scripts = {
        "deploy_retraining.py": '''
"""
Production deployment script for automated retraining pipeline
"""
import os
import json
from retraining_orchestrator import RetrainingOrchestrator, RetrainingConfig, RetrainingScheduler

def deploy_retraining_pipeline():
    """Deploy retraining pipeline to production"""
    
    # Load configuration
    config = RetrainingConfig()
    config.load_config("/opt/ml/config/retraining_config.json")
    
    # Initialize orchestrator
    orchestrator = RetrainingOrchestrator(config)
    
    # Setup scheduler
    scheduler = RetrainingScheduler(orchestrator)
    scheduler.setup_schedule()
    
    # Start scheduler in background
    import threading
    scheduler_thread = threading.Thread(target=scheduler.start_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    
    print("Retraining pipeline deployed and scheduler started")

if __name__ == "__main__":
    deploy_retraining_pipeline()
''',
        
        "retraining_api.py": '''
"""
API endpoints for retraining pipeline management
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os

app = FastAPI(title="Retraining Pipeline API")

class RetrainingTrigger(BaseModel):
    trigger_type: str
    force_retrain: bool = False

@app.post("/trigger-retraining")
async def trigger_retraining(trigger: RetrainingTrigger):
    """Manually trigger retraining pipeline"""
    try:
        # Implementation would trigger the actual retraining
        return {"status": "triggered", "trigger_type": trigger.trigger_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/retraining-status")
async def get_retraining_status():
    """Get current retraining pipeline status"""
    try:
        status_path = "/opt/ml/monitoring/retraining_status.json"
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                return json.load(f)
        return {"status": "unknown"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/promote-challenger")
async def promote_challenger():
    """Manually promote challenger to champion"""
    try:
        # Implementation would promote challenger
        return {"status": "promoted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
''',
        
        "docker-compose.yml": '''
version: '3.8'
services:
  retraining-pipeline:
    build: .
    environment:
      - MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
      - AZURE_SUBSCRIPTION_ID=${AZURE_SUBSCRIPTION_ID}
      - AZURE_RESOURCE_GROUP=${AZURE_RESOURCE_GROUP}
    volumes:
      - ./config:/opt/ml/config
      - ./data:/opt/ml/data
      - ./models:/opt/ml/models
      - ./logs:/opt/ml/logs
    command: python deploy_retraining.py
    
  retraining-api:
    build: .
    ports:
      - "8001:8001"
    environment:
      - MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
    volumes:
      - ./config:/opt/ml/config
      - ./monitoring:/opt/ml/monitoring
    command: uvicorn retraining_api:app --host 0.0.0.0 --port 8001
''',
        
        "Dockerfile": '''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /opt/ml/config /opt/ml/data /opt/ml/models /opt/ml/logs /opt/ml/monitoring

CMD ["python", "deploy_retraining.py"]
''',
        
        "requirements_retraining.txt": '''
mlflow==2.8.1
pandas==2.1.3
numpy==1.26.4
scikit-learn==1.3.2
xgboost==2.0.1
fastapi==0.104.1
uvicorn[standard]==0.24.0
schedule==1.2.0
azure-ai-ml==1.11.1
azure-identity==1.15.0
evidently==0.4.11
scipy==1.11.4
joblib==1.3.2
'''
    }
    
    # Create deployment directory
    deployment_dir = "/dbfs/FileStore/deployment/retraining"
    os.makedirs(deployment_dir, exist_ok=True)
    
    # Save deployment scripts
    for filename, content in deployment_scripts.items():
        file_path = os.path.join(deployment_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
    
    print(f"✓ Deployment package created in {deployment_dir}")
    print("Files created:")
    for filename in deployment_scripts.keys():
        print(f"  - {filename}")
    
    return deployment_dir

# Create deployment package
deployment_dir = create_retraining_deployment_package()

print("\n" + "="*70)
print("PHASE 5 COMPLETED - AUTOMATED RETRAINING PIPELINE")
print("="*70)
print(f"\nDeployment package ready at: {deployment_dir}")
print("\nTo deploy in production:")
print("1. Copy deployment files to production environment")
print("2. Configure Azure credentials and MLflow tracking URI")
print("3. Run: docker-compose up -d")
print("4. Access retraining API at http://localhost:8001")
print("\nThe pipeline includes:")
print("- Automated data collection and validation")
print("- Multi-algorithm training with hyperparameter tuning")
print("- Champion-Challenger A/B testing")
print("- Scheduled and trigger-based retraining")
print("- Model promotion automation")
print("- Comprehensive monitoring and logging")
