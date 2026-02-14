"""
MLflow integration utilities
Tracks experiments, models, and fairness metrics
"""

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from datetime import datetime
import json
import os


class MLflowTracker:
    """
    Handles all MLflow tracking operations
    Integrates seamlessly with the bias mitigation pipeline
    """
    
    def __init__(self, config, experiment_name="bias-mitigation-aiops"):
        """
        Initialize MLflow tracker
        
        Args:
            config: BiasConfig object
            experiment_name: Name of the MLflow experiment
        """
        self.config = config
        self.experiment_name = experiment_name
        
        # Set tracking URI from environment or use default
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        mlflow.set_experiment(experiment_name)
        
        print(f"✓ MLflow tracking initialized")
        print(f"  Experiment: {experiment_name}")
        print(f"  Tracking URI: {tracking_uri}")
    
    def start_pipeline_run(self, run_name=None):
        """
        Start a parent run for the entire pipeline
        
        Returns:
            run_id: MLflow run ID
        """
        if run_name is None:
            run_name = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.start_run(run_name=run_name)
        
        # Log pipeline configuration
        mlflow.log_params({
            'dataset': self.config.DATASET_NAME,
            'protected_attribute': self.config.PROTECTED_ATTRIBUTE,
            'di_threshold': self.config.DISPARATE_IMPACT_THRESHOLD,
            'spd_threshold': self.config.STATISTICAL_PARITY_THRESHOLD,
            'random_state': self.config.RANDOM_STATE,
            'test_size': self.config.TEST_SIZE
        })
        
        # Add tags
        mlflow.set_tags({
            'pipeline_version': '1.0',
            'stage': 'development',
            'framework': 'aif360'
        })
        
        return mlflow.active_run().info.run_id
    
    def log_model_run(self, model_name, audit_result, model_obj=None, dataset_info=None):
        """
        Log a single model's results as a child run
        
        Args:
            model_name: Name of the model
            audit_result: Audit result dict from BiasAuditor
            model_obj: The trained model object (optional)
            dataset_info: Dataset information (optional)
        """
        with mlflow.start_run(run_name=model_name, nested=True):
            # Log model type
            mlflow.log_param('model_name', model_name)
            mlflow.log_param('model_type', audit_result.get('audit_type', 'model'))
            
            # Log all fairness metrics
            metrics = audit_result.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, float(metric_value))
            
            # Log gate status
            gate_passed = audit_result.get('gate_passed', False)
            mlflow.log_metric('fairness_gate_passed', 1.0 if gate_passed else 0.0)
            
            # Log model artifact if provided
            if model_obj is not None:
                self._log_model_artifact(model_obj, model_name)
            
            # Log dataset info
            if dataset_info:
                mlflow.log_params({
                    f'dataset_{k}': v for k, v in dataset_info.items()
                    if isinstance(v, (str, int, float, bool))
                })
            
            # Add tags
            mlflow.set_tag('fairness_status', 'PASSED' if gate_passed else 'FAILED')
            mlflow.set_tag('mitigation_strategy', self._get_mitigation_strategy(model_name))
            
            return mlflow.active_run().info.run_id
    
    def _log_model_artifact(self, model_obj, model_name):
        """Log the actual model artifact"""
        try:
            # For sklearn models
            if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'predict'):
                mlflow.sklearn.log_model(
                    model_obj.model,
                    artifact_path=f"models/{model_name}",
                    registered_model_name=f"bias_mitigated_{model_name}"
                )
            # For TensorFlow models (Adversarial Debiasing)
            elif hasattr(model_obj, 'sess'):
                # TensorFlow models need special handling
                mlflow.tensorflow.log_model(
                    tf_saved_model_dir=f"models/{model_name}",
                    tf_meta_graph_tags=['serve'],
                    tf_signature_def_key='serving_default',
                    artifact_path=f"models/{model_name}"
                )
        except Exception as e:
            print(f"  ⚠️  Could not log model artifact: {e}")
    
    def _get_mitigation_strategy(self, model_name):
        """Determine mitigation strategy from model name"""
        model_name_lower = model_name.lower()
        if 'baseline' in model_name_lower:
            return 'none'
        elif 'reweigh' in model_name_lower:
            return 'preprocessing'
        elif 'adversarial' in model_name_lower:
            return 'inprocessing'
        elif 'prejudice' in model_name_lower:
            return 'inprocessing'
        else:
            return 'unknown'
    
    def log_comparison_plot(self, fig, filename='fairness_comparison.png'):
        """Log matplotlib figure as artifact"""
        try:
            # Save to temp file
            temp_path = f"/tmp/{filename}"
            fig.savefig(temp_path, dpi=150, bbox_inches='tight')
            
            # Log to MLflow
            mlflow.log_artifact(temp_path, artifact_path="plots")
            
            print(f"  ✓ Comparison plot logged to MLflow")
        except Exception as e:
            print(f"  ⚠️  Could not log plot: {e}")
    
    def log_audit_report(self, audit_history, filename='audit_report.json'):
        """Log audit report as JSON artifact"""
        try:
            # Save to temp file
            temp_path = f"/tmp/{filename}"
            with open(temp_path, 'w') as f:
                json.dump(audit_history, f, indent=2)
            
            # Log to MLflow
            mlflow.log_artifact(temp_path, artifact_path="reports")
            
            print(f"  ✓ Audit report logged to MLflow")
        except Exception as e:
            print(f"  ⚠️  Could not log report: {e}")
    
    def log_best_model_selection(self, best_model_info):
        """
        Log the final selected model information
        
        Args:
            best_model_info: Dict with best model details
        """
        mlflow.log_params({
            'selected_model': best_model_info.get('name', 'unknown'),
            'selection_reason': 'highest_accuracy_among_fair' if best_model_info.get('passed_gate') else 'best_available'
        })
        
        # Log deployment recommendation
        deploy_ready = best_model_info.get('passed_gate', False)
        mlflow.log_metric('deployment_ready', 1.0 if deploy_ready else 0.0)
        mlflow.set_tag('deployment_status', 'READY' if deploy_ready else 'NOT_READY')
    
    def end_pipeline_run(self):
        """End the parent pipeline run"""
        mlflow.end_run()
        print("✓ MLflow tracking completed")
    
    def get_experiment_url(self):
        """Get the MLflow UI URL for this experiment"""
        tracking_uri = mlflow.get_tracking_uri()
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if experiment and not tracking_uri.startswith('file://'):
            return f"{tracking_uri}/#/experiments/{experiment.experiment_id}"
        else:
            return "Run 'mlflow ui' to view results"


class MLflowModelRegistry:
    """
    Handles model registration and versioning
    """
    
    @staticmethod
    def register_production_model(model_name, run_id, stage='Production'):
        """
        Register a model for production use
        
        Args:
            model_name: Name to register model under
            run_id: MLflow run ID
            stage: Model stage ('Staging', 'Production', 'Archived')
        """
        try:
            # Get model URI
            model_uri = f"runs:/{run_id}/models/{model_name}"
            
            # Register model
            result = mlflow.register_model(model_uri, model_name)
            
            # Transition to stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=result.version,
                stage=stage
            )
            
            print(f"✓ Model registered: {model_name} v{result.version} ({stage})")
            return result
            
        except Exception as e:
            print(f"⚠️  Model registration failed: {e}")
            return None
    
    @staticmethod
    def load_production_model(model_name):
        """Load the current production model"""
        try:
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✓ Loaded production model: {model_name}")
            return model
        except Exception as e:
            print(f"⚠️  Could not load production model: {e}")
            return None
    
    @staticmethod
    def compare_models(model_names, metric='accuracy'):
        """
        Compare multiple models based on a metric
        
        Args:
            model_names: List of model names
            metric: Metric to compare on
        
        Returns:
            DataFrame with comparison results
        """
        import pandas as pd
        
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name("bias-mitigation-aiops")
        
        results = []
        runs = client.search_runs(experiment.experiment_id)
        
        for run in runs:
            if run.data.tags.get('mlflow.runName') in model_names:
                results.append({
                    'model': run.data.tags.get('mlflow.runName'),
                    'run_id': run.info.run_id,
                    metric: run.data.metrics.get(metric, 0),
                    'disparate_impact': run.data.metrics.get('disparate_impact', 0),
                    'fairness_gate': run.data.metrics.get('fairness_gate_passed', 0)
                })
        
        return pd.DataFrame(results)


def log_to_mlflow_decorator(func):
    """
    Decorator to automatically log function execution to MLflow
    """
    def wrapper(*args, **kwargs):
        with mlflow.start_run():
            # Log function name
            mlflow.log_param('function', func.__name__)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # If result is dict with metrics, log them
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
            
            return result
    
    return wrapper