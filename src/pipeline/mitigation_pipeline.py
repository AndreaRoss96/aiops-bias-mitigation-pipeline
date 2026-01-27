import sys
sys.path.append('..')

from src.models.baseline import BaselineModel
from src.fairness.auditor import BiasAuditor
from src.fairness.preprocessing import PreprocessingMitigation
from src.fairness.inprocessing import InprocessingMitigation
from src.utils.mlflow_utils import MLflowTracker
import matplotlib.pyplot as plt
import seaborn as sns
import os


class BiasMitigationPipeline:
    """
    Automated pipeline that:
    1. Trains baseline model
    2. Detects if bias exists (fairness gate)
    3. Automatically applies mitigation if needed
    4. Compares mitigation strategies
    5. Selects best fair model
    6. Logs everything to MLflow
    """

    def __init__(self, config, enable_mlflow=True):
        self.config = config
        self.auditor = BiasAuditor(config)
        self.preprocessing = PreprocessingMitigation(config)
        self.inprocessing = InprocessingMitigation(config)
        
        self.enable_mlflow = enable_mlflow and self._check_mlflow_available()
        self.mlflow_tracker = None
        
        if self.enable_mlflow:
            try:
                self.mlflow_tracker = MLflowTracker(config)
                print("MLflow tracking enabled")
            except Exception as e:
                print(f"MLflow initialization failed: {e}")
                print("   Continuing without MLflow tracking")
                self.enable_mlflow = False
        
        self.results = []
        self.best_model = None
        self.best_audit = None
        self.pipeline_run_id = None
    
    def run_baseline(self, dataset_train, dataset_test):
        """
        Step 1: Train and audit baseline model
        """
        print("\n" + "="*70)
        print("="*70)
        
        # Train baseline
        print("Training Baseline")
        baseline = BaselineModel(model_type='logistic', config=self.config)
        baseline.train(dataset_train)
        
        # Get predictions
        print("Predictions")
        baseline_pred = baseline.predict(dataset_test)
        
        # Audit fairness
        print("Fairness")
        audit_result, passed = self.auditor.audit_model(
            dataset_test, 
            baseline_pred, 
            model_name="Baseline"
        )
        
        # Log to MLflow
        if self.enable_mlflow:
            self.mlflow_tracker.log_model_run(
                model_name="Baseline",
                audit_result=audit_result,
                model_obj=baseline
            )
        
        self.results.append({
            'name': 'Baseline',
            'model': baseline,
            'predictions': baseline_pred,
            'audit': audit_result,
            'passed_gate': passed
        })
        
        return passed
    
    def run_reweighing(self, dataset_train, dataset_test):
        pass 