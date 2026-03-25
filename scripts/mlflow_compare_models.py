#!/usr/bin/env python3
"""
Compare models using MLflow experiment tracking
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import pandas as pd
from src.utils.mlflow_utils import MLflowTracker, MLflowModelRegistry
from config.bias_config import get_config

def compare_models_mlflow():
    """Compare all models using MLflow experiment data"""
    
    print("🔍 Comparing models using MLflow...")
    
    # Initialize MLflow tracker
    config = get_config('development')
    tracker = MLflowTracker(config)
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name("bias-mitigation-aiops")
    if not experiment:
        print("❌ No experiment found. Run the pipeline first.")
        return
    
    # Get all runs
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment.experiment_id)
    
    # Extract model results
    results = []
    for run in runs:
        run_name = run.data.tags.get('mlflow.runName', 'unknown')
        
        # Skip pipeline runs, only include model runs
        if run_name.startswith('pipeline_'):
            continue
            
        # Extract metrics
        metrics = {
            'model': run_name,
            'run_id': run.info.run_id,
            'accuracy': run.data.metrics.get('accuracy', 0),
            'disparate_impact': run.data.metrics.get('disparate_impact', 0),
            'statistical_parity': run.data.metrics.get('statistical_parity_difference', 0),
            'equal_opportunity': run.data.metrics.get('equal_opportunity_difference', 0),
            'fairness_gate': run.data.metrics.get('fairness_gate_passed', 0),
            'mitigation_strategy': run.data.tags.get('mitigation_strategy', 'unknown'),
            'fairness_status': run.data.tags.get('fairness_status', 'UNKNOWN')
        }
        
        results.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        print("❌ No model runs found. Run the training pipeline first.")
        return
    
    # Sort by accuracy, then by fairness
    df = df.sort_values(['fairness_gate', 'disparate_impact', 'accuracy'], 
                       ascending=[False, False, False])
    
    print("\n📊 Model Comparison Results:")
    print("=" * 80)
    
    for _, row in df.iterrows():
        status = "✅ FAIR" if row['fairness_gate'] else "❌ BIASED"
        print(f"\n{row['model']}")
        print(f"  Status: {status}")
        print(f"  Accuracy: {row['accuracy']:.3f}")
        print(f"  Disparate Impact: {row['disparate_impact']:.3f}")
        print(f"  Statistical Parity: {abs(row['statistical_parity']):.3f}")
        print(f"  Equal Opportunity: {abs(row['equal_opportunity']):.3f}")
        print(f"  Strategy: {row['mitigation_strategy']}")
    
    # Find best model
    fair_models = df[df['fairness_gate'] == 1]
    if not fair_models.empty:
        best_model = fair_models.loc[fair_models['accuracy'].idxmax()]
        print(f"\n🏆 Best Fair Model: {best_model['model']}")
        print(f"   Accuracy: {best_model['accuracy']:.3f}")
        print(f"   Disparate Impact: {best_model['disparate_impact']:.3f}")
    else:
        # If no fair models, find least biased
        best_model = df.loc[df['disparate_impact'].idxmax()]
        print(f"\n⚠️  No models passed fairness gate")
        print(f"🏆 Least Biased Model: {best_model['model']}")
        print(f"   Accuracy: {best_model['accuracy']:.3f}")
        print(f"   Disparate Impact: {best_model['disparate_impact']:.3f}")
    
    # Log comparison to MLflow
    with mlflow.start_run(run_name="model_comparison", nested=True):
        mlflow.log_param("num_models_compared", len(df))
        mlflow.log_param("num_fair_models", len(fair_models))
        
        if not fair_models.empty:
            mlflow.log_metric("best_fair_accuracy", best_model['accuracy'])
            mlflow.log_metric("best_fair_di", best_model['disparate_impact'])
        
        # Log comparison table
        comparison_file = "/tmp/model_comparison.csv"
        df.to_csv(comparison_file, index=False)
        mlflow.log_artifact(comparison_file, "reports")
        
        print(f"\n📝 Comparison logged to MLflow")
        print(f"🔗 View results: {tracker.get_experiment_url()}")
    
    return df

def register_best_model():
    """Register the best fair model in MLflow Model Registry"""
    
    print("\n📋 Registering best model...")
    
    # Get comparison results
    df = compare_models_mlflow()
    if df is None:
        return
    
    # Find best fair model
    fair_models = df[df['fairness_gate'] == 1]
    if fair_models.empty:
        print("❌ No fair models to register")
        return
    
    best_model = fair_models.loc[fair_models['accuracy'].idxmax()]
    
    # Register model
    result = MLflowModelRegistry.register_production_model(
        model_name=best_model['model'],
        run_id=best_model['run_id'],
        stage='Staging'
    )
    
    if result:
        print(f"✅ Model registered: {best_model['model']} v{result.version}")
    else:
        print("❌ Model registration failed")

if __name__ == "__main__":
    # Compare models
    compare_models_mlflow()
    
    # Optionally register best model
    register_best_model()