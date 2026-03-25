#!/usr/bin/env python3
"""
Test MLflow connection and create experiment if needed
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.sklearn
from src.utils.mlflow_utils import MLflowTracker
from config.bias_config import get_config

def test_mlflow_connection():
    """Test MLflow connection and setup"""
    
    print("🔍 Testing MLflow Connection...")
    print(f"MLflow Version: {mlflow.__version__}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Test experiment creation
    config = get_config('development')
    tracker = MLflowTracker(config, experiment_name="bias-mitigation-aiops")
    
    # Test run creation
    try:
        run_id = tracker.start_pipeline_run("test_connection")
        print(f"✅ Successfully created test run: {run_id}")
        
        # Log some test metrics
        mlflow.log_metric("test_metric", 0.85)
        mlflow.log_param("test_param", "test_value")
        
        # End run
        tracker.end_pipeline_run()
        print("✅ Successfully ended test run")
        
        # Get experiment URL
        url = tracker.get_experiment_url()
        print(f"📊 MLflow UI: {url}")
        
        return True
        
    except Exception as e:
        print(f"❌ MLflow connection failed: {e}")
        return False

def start_mlflow_ui():
    """Start MLflow UI server"""
    print("\n🚀 Starting MLflow UI...")
    print("Run this command in a separate terminal:")
    print(f"cd {os.getcwd()} && source env/bin/activate && mlflow ui --port 5000")
    print("Then open: http://localhost:5000")

if __name__ == "__main__":
    success = test_mlflow_connection()
    
    if success:
        start_mlflow_ui()
        print("\n✅ MLflow is ready to use!")
    else:
        print("\n❌ Please check your MLflow installation")
