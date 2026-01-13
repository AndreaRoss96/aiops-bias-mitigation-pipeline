from .metrics import FairnessMetrics
import json
from datetime import datetime

class BiasAuditor:
    """
    Handles bias measurement, reporting, and fairness gate enforcement
    """
    
    def __init__(self, config):
        self.config = config
        self.metrics_calculator = FairnessMetrics(config)
        self.audit_history = []
    
    def audit_dataset(self, dataset, dataset_name="Dataset"):
        """
        Audit a dataset for inherent bias
        """
        print(f"\n{'='*70}")
        print(f"DATASET BIAS AUDIT: {dataset_name}")
        print(f"{'='*70}")
        
        metrics = self.metrics_calculator.compute_dataset_metrics(dataset)
        constraints = self.metrics_calculator.evaluate_fairness_constraints(metrics)
        
        audit_result = {
            'timestamp': datetime.now().isoformat(),
            'audit_type': 'dataset',
            'dataset_name': dataset_name,
            'metrics': metrics,
            'constraints': constraints
        }
        
        self.audit_history.append(audit_result)
        self._print_dataset_report(metrics, constraints)
        
        return audit_result
    
    def audit_model(self, dataset_true, dataset_pred, model_name="Model"):
        """
        Audit a model's predictions for bias
        """
        print(f"\n{'='*70}")
        print(f"MODEL BIAS AUDIT: {model_name}")
        print(f"{'='*70}")
        
        metrics = self.metrics_calculator.compute_classification_metrics(
            dataset_true, dataset_pred
        )
        constraints = self.metrics_calculator.evaluate_fairness_constraints(metrics)
        
        # Combine metrics and constraints
        full_metrics = {**metrics, **constraints}
        
        audit_result = {
            'timestamp': datetime.now().isoformat(),
            'audit_type': 'model',
            'model_name': model_name,
            'metrics': full_metrics,
            'gate_passed': constraints['fairness_gate_passed']
        }
        
        self.audit_history.append(audit_result)
        passed = self._print_model_report(full_metrics)
        
        return audit_result, passed
    
    def _print_dataset_report(self, metrics, constraints):
        """
        Print dataset bias report
        """
        print(f"\nDataset Fairness Metrics:")
        print(f"   Disparate Impact: {metrics['disparate_impact']:.3f}")
        print(f"   Statistical Parity Diff: {metrics['statistical_parity_difference']:.3f}")
        print(f"\nBase Rates:")
        print(f"   Unprivileged group: {metrics['base_rate_unprivileged']:.3f}")
        print(f"   Privileged group: {metrics['base_rate_privileged']:.3f}")
        print(f"\n{'='*70}\n")
    
    def _print_model_report(self, metrics):
        """
        Print detailed model audit report
        """
        print(f"\nFairness Metrics:")
        print(f"   Disparate Impact: {metrics['disparate_impact']:.3f} "
              f"{'PASS' if metrics.get('disparate_impact_pass') else '❌ FAIL'} "
              f"(threshold: {self.config.DISPARATE_IMPACT_THRESHOLD})")
        
        print(f"   Statistical Parity Difference: {metrics['statistical_parity_difference']:.3f} "
              f"{'PASS' if metrics.get('statistical_parity_pass') else '❌ FAIL'} "
              f"(threshold: ±{self.config.STATISTICAL_PARITY_THRESHOLD})")
        
        print(f"   Equal Opportunity Difference: {metrics['equal_opportunity_difference']:.3f}")
        print(f"   Average Odds Difference: {metrics['average_odds_difference']:.3f}")
        
        print(f"\nModel Performance:")
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
        
        print(f"\nTrue Positive Rates:")
        print(f"   Unprivileged group: {metrics['tpr_unprivileged']:.3f}")
        print(f"   Privileged group: {metrics['tpr_privileged']:.3f}")
        print(f"   Difference: {abs(metrics['tpr_unprivileged'] - metrics['tpr_privileged']):.3f}")
        
        print(f"\nFalse Positive Rates:")
        print(f"   Unprivileged group: {metrics['fpr_unprivileged']:.3f}")
        print(f"   Privileged group: {metrics['fpr_privileged']:.3f}")
        
        # THE GATE - This is the critical part
        gate_passed = metrics.get('fairness_gate_passed', False)
        if gate_passed:
            print(f"\nFAIRNESS GATE: PASSED")
            print(f"   Model meets all fairness requirements")
        else:
            print(f"\nFAIRNESS GATE: FAILED")
            print(f"    Model does NOT meet fairness requirements")
            print(f"    Build should FAIL (like a failing unit test)")
            print(f"    Apply mitigation before proceeding")
        
        print(f"{'='*70}\n")
        
        return gate_passed
    
    def enforce_gate(self, audit_result, fail_on_violation=True):
        """
        Enforce fairness gate - can be used in CI/CD
        
        Args:
            audit_result: Result from audit_model()
            fail_on_violation: If True, raises exception on gate failure
        
        Returns:
            bool: True if passed, False otherwise
        
        Raises:
            FairnessGateException: If gate fails and fail_on_violation=True
        """
        passed = audit_result.get('gate_passed', False)
        
        if not passed and fail_on_violation:
            raise FairnessGateException(
                f"Fairness gate failed for {audit_result['model_name']}\n"
                f"Disparate Impact: {audit_result['metrics']['disparate_impact']:.3f} "
                f"(threshold: {self.config.DISPARATE_IMPACT_THRESHOLD})\n"
                f"Apply bias mitigation before deployment."
            )
        
        return passed
    
    def save_audit_report(self, filepath):
        """Save audit history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.audit_history, f, indent=2)
        print(f"✓ Audit report saved to {filepath}")
    
    def compare_models(self, audit_results):
        """
        Compare multiple model audit results
        Returns best model based on fairness-accuracy trade-off
        """
        print(f"\n{'='*70}")
        print("MODEL COMPARISON")
        print(f"{'='*70}\n")
        
        # Filter for models that passed fairness gate
        fair_models = [r for r in audit_results if r.get('gate_passed', False)]
        
        if not fair_models:
            print(" WARNING: No models passed the fairness gate!")
            print("   Selecting best available model...")
            # Select model with highest disparate impact
            best = max(audit_results, 
                      key=lambda x: x['metrics']['disparate_impact'])
        else:
            print(f"{len(fair_models)} model(s) passed fairness gate")
            # Among fair models, select highest accuracy
            best = max(fair_models, 
                      key=lambda x: x['metrics']['accuracy'])
        
        print(f"\nSELECTED MODEL: {best['model_name']}")
        print(f"   Accuracy: {best['metrics']['accuracy']:.3f}")
        print(f"   Disparate Impact: {best['metrics']['disparate_impact']:.3f}")
        print(f"   Gate Status: {'✅ PASSED' if best.get('gate_passed') else '❌ FAILED'}")
        print(f"{'='*70}\n")
        
        return best


class FairnessGateException(Exception):
    """Exception raised when fairness gate check fails"""
    pass