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
        
        # Visualization
        from src.utils.visualization import TrainingVisualizer
        self.visualizer = TrainingVisualizer(output_dir=config.PLOTS_DIR)
        
        # MLflow integration
        self.enable_mlflow = enable_mlflow and self._check_mlflow_available()
        self.mlflow_tracker = None
        
        if self.enable_mlflow:
            try:
                self.mlflow_tracker = MLflowTracker(config)
                print("✓ MLflow tracking enabled")
            except Exception as e:
                print(f"⚠️  MLflow initialization failed: {e}")
                print("   Continuing without MLflow tracking")
                self.enable_mlflow = False
        
        self.results = []
        self.best_model = None
        self.best_audit = None
        self.pipeline_run_id = None
    
    def _check_mlflow_available(self):
        """Check if MLflow is installed"""
        try:
            import mlflow
            return True
        except ImportError:
            print("⚠️  MLflow not installed. Install with: pip install mlflow")
            return False
    
    def run_baseline(self, dataset_train, dataset_test):
        """
        Step 1: Train and audit baseline model
        """
        print("\n" + "="*70)
        print("STEP 1: BASELINE MODEL")
        print("="*70)
        
        # Train baseline
        baseline = BaselineModel(model_type='logistic', config=self.config)
        baseline.train(dataset_train)
        
        # Get predictions
        baseline_pred = baseline.predict(dataset_test)
        
        # Audit fairness
        audit_result, passed = self.auditor.audit_model(
            dataset_test, 
            baseline_pred, 
            model_name="Baseline (No Mitigation)"
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
        """
        Step 2a: Apply reweighing (pre-processing)
        """
        print("\n" + "="*70)
        print("STEP 2a: REWEIGHING MITIGATION")
        print("="*70)
        
        # Apply reweighing
        dataset_train_rw = self.preprocessing.apply_reweighing(dataset_train)
        
        # Train model on reweighed data
        rw_model = BaselineModel(model_type='logistic', config=self.config)
        rw_model.train(dataset_train_rw)
        
        # Get predictions
        rw_pred = rw_model.predict(dataset_test)
        
        # Audit fairness
        audit_result, passed = self.auditor.audit_model(
            dataset_test,
            rw_pred,
            model_name="Reweighing + Logistic Regression"
        )
        
        # Log to MLflow
        if self.enable_mlflow:
            self.mlflow_tracker.log_model_run(
                model_name="Reweighing",
                audit_result=audit_result,
                model_obj=rw_model
            )
        
        self.results.append({
            'name': 'Reweighing',
            'model': rw_model,
            'predictions': rw_pred,
            'audit': audit_result,
            'passed_gate': passed
        })
        
        return passed
    
    def run_adversarial_debiasing(self, dataset_train, dataset_test):
        """
        Step 2b: Apply adversarial debiasing (in-processing)
        """
        print("\n" + "="*70)
        print("STEP 2b: ADVERSARIAL DEBIASING")
        print("="*70)
        
        # Train adversarial debiasing model
        adv_pred, adv_model = self.inprocessing.train_adversarial_debiasing(
            dataset_train, 
            dataset_test
        )
        
        # Audit fairness
        audit_result, passed = self.auditor.audit_model(
            dataset_test,
            adv_pred,
            model_name="Adversarial Debiasing"
        )
        
        # Log to MLflow
        if self.enable_mlflow:
            self.mlflow_tracker.log_model_run(
                model_name="Adversarial",
                audit_result=audit_result,
                model_obj=adv_model
            )
        
        self.results.append({
            'name': 'Adversarial',
            'model': adv_model,
            'predictions': adv_pred,
            'audit': audit_result,
            'passed_gate': passed
        })
        
        # Cleanup TensorFlow session
        self.inprocessing.cleanup()
        
        return passed
    
    def run_full_pipeline(self, dataset_train, dataset_test):
        """
        Run complete automated mitigation pipeline with MLflow tracking
        
        This is the main entry point for Phase 3
        """
        print("\n" + "="*70)
        print("AUTOMATED BIAS MITIGATION PIPELINE")
        print("="*70)
        print("\nThis pipeline will:")
        print("  1. Train baseline model")
        print("  2. Check fairness gate")
        print("  3. Apply mitigation strategies if needed")
        print("  4. Compare all models")
        print("  5. Select best fair model")
        if self.enable_mlflow:
            print("  6. Track everything in MLflow")
        
        # Start MLflow parent run
        if self.enable_mlflow:
            self.pipeline_run_id = self.mlflow_tracker.start_pipeline_run()
            print(f"\n✓ MLflow run started: {self.pipeline_run_id}")
        
        try:
            # Step 1: Baseline
            baseline_passed = self.run_baseline(dataset_train, dataset_test)
            
            # Step 2: Apply mitigation (always, for comparison)
            print("\n💡 Applying mitigation strategies for comparison...")
            
            self.run_reweighing(dataset_train, dataset_test)
            self.run_adversarial_debiasing(dataset_train, dataset_test)
            
            # Step 3: Model selection
            self.select_best_model()
            
            # Step 4: Generate report
            fig = self.generate_comparison_report()
            
            # Log comparison plot to MLflow
            if self.enable_mlflow:
                self.mlflow_tracker.log_comparison_plot(fig)
            
            # Log audit history to MLflow
            if self.enable_mlflow:
                self.mlflow_tracker.log_audit_report(self.auditor.audit_history)
            
            # Log best model selection
            if self.enable_mlflow and self.best_model:
                self.mlflow_tracker.log_best_model_selection(self.best_model)
            
            # Generate training curves visualization
            print("\n📊 Generating training visualizations...")
            self._generate_training_plots()
            
            # Show MLflow UI link
            if self.enable_mlflow:
                url = self.mlflow_tracker.get_experiment_url()
                print(f"\n📊 View results in MLflow UI: {url}")
        
        finally:
            # End MLflow run
            if self.enable_mlflow:
                self.mlflow_tracker.end_pipeline_run()
        
        return self.best_model, self.best_audit
    
    def select_best_model(self):
        """
        Select best model based on fairness-accuracy trade-off
        
        Selection criteria:
        1. Must pass fairness gate (if any model does)
        2. Among fair models, select highest accuracy
        3. If no models pass, select model closest to passing
        """
        print("\n" + "="*70)
        print("MODEL SELECTION")
        print("="*70)
        
        # Get models that passed fairness gate
        fair_models = [r for r in self.results if r['passed_gate']]
        
        if not fair_models:
            print("\n⚠️  WARNING: No models passed the fairness gate!")
            print("   Selecting model with best fairness metrics...")
            
            # Select model with highest disparate impact
            best = max(
                self.results,
                key=lambda x: x['audit']['metrics']['disparate_impact']
            )
            print("   ⚠️  Selected model still FAILS fairness requirements")
        else:
            print(f"\n✅ {len(fair_models)} model(s) passed the fairness gate")
            print("   Selecting model with highest accuracy...")
            
            # Among fair models, select highest accuracy
            best = max(
                fair_models,
                key=lambda x: x['audit']['metrics']['accuracy']
            )
        
        self.best_model = best
        self.best_audit = best['audit']
        
        print(f"\n🏆 SELECTED MODEL: {best['name']}")
        print(f"   Accuracy: {best['audit']['metrics']['accuracy']:.3f}")
        print(f"   Disparate Impact: {best['audit']['metrics']['disparate_impact']:.3f}")
        print(f"   Statistical Parity Diff: {best['audit']['metrics']['statistical_parity_difference']:.3f}")
        print(f"   Fairness Gate: {'✅ PASSED' if best['passed_gate'] else '❌ FAILED'}")
        
        if best['passed_gate']:
            print(f"\n✅ READY FOR DEPLOYMENT")
        else:
            print(f"\n⚠️  NOT READY FOR DEPLOYMENT")
            print(f"   Further mitigation required")
        
        print("="*70 + "\n")
        
        return best
    
    def _generate_training_plots(self):
        """Generate training curve visualizations"""
        # Collect model data for visualization
        models_data = []
        
        for r in self.results:
            model_obj = r.get('model')
            metrics = r['audit']['metrics']
            
            # Get training history if available
            if hasattr(model_obj, 'training_history'):
                hist = model_obj.training_history
            else:
                hist = None
            
            models_data.append({
                'name': r['name'],
                'accuracy': metrics.get('accuracy', 0),
                'loss': 1 - metrics.get('accuracy', 0),  # Approximate loss
                'disparate_impact': metrics.get('disparate_impact', 0),
                'history': hist
            })
        
        # Plot comparison summary
        self.visualizer.plot_comparison_summary(models_data, save=True)
        
        print("   ✓ Training visualizations saved")
    
    def generate_comparison_report(self):
        """
        Generate visual comparison of all models
        """
        print("\n📊 Generating comparison visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        model_names = [r['name'] for r in self.results]
        accuracies = [r['audit']['metrics']['accuracy'] for r in self.results]
        disparate_impacts = [r['audit']['metrics']['disparate_impact'] for r in self.results]
        spd = [r['audit']['metrics']['statistical_parity_difference'] for r in self.results]
        eod = [r['audit']['metrics']['equal_opportunity_difference'] for r in self.results]
        
        colors = ['red' if not r['passed_gate'] else 'green' for r in self.results]
        
        # Plot 1: Accuracy
        axes[0, 0].bar(model_names, accuracies, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylim([0.7, 0.9])
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Disparate Impact
        axes[0, 1].bar(model_names, disparate_impacts, color=colors, alpha=0.7)
        axes[0, 1].axhline(y=0.8, color='red', linestyle='--', 
                          label=f'Threshold ({self.config.DISPARATE_IMPACT_THRESHOLD})')
        axes[0, 1].set_ylabel('Disparate Impact')
        axes[0, 1].set_title('Fairness: Disparate Impact (Higher is Better)')
        axes[0, 1].set_ylim([0, 1.2])
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Statistical Parity Difference
        axes[1, 0].bar(model_names, [abs(s) for s in spd], color=colors, alpha=0.7)
        axes[1, 0].axhline(y=self.config.STATISTICAL_PARITY_THRESHOLD, 
                          color='red', linestyle='--', 
                          label=f'Threshold ({self.config.STATISTICAL_PARITY_THRESHOLD})')
        axes[1, 0].set_ylabel('|Statistical Parity Difference|')
        axes[1, 0].set_title('Fairness: Statistical Parity (Lower is Better)')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Equal Opportunity Difference
        axes[1, 1].bar(model_names, [abs(e) for e in eod], color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('|Equal Opportunity Difference|')
        axes[1, 1].set_title('Fairness: Equal Opportunity (Lower is Better)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(self.config.PLOTS_DIR, exist_ok=True)
        filepath = f"{self.config.PLOTS_DIR}/fairness_comparison.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"   ✓ Comparison plot saved: {filepath}")
        
        return fig
    
    def get_deployment_recommendation(self):
        """
        Get deployment recommendation based on results
        """
        if self.best_model is None:
            return "Run pipeline first using run_full_pipeline()"
        
        if self.best_model['passed_gate']:
            return {
                'deploy': True,
                'model': self.best_model['name'],
                'reason': 'Model passes all fairness requirements',
                'metrics': self.best_audit['metrics'],
                'mlflow_run_id': self.pipeline_run_id if self.enable_mlflow else None
            }
        else:
            return {
                'deploy': False,
                'model': self.best_model['name'],
                'reason': 'No models meet fairness requirements',
                'action_required': 'Apply additional mitigation or adjust thresholds',
                'metrics': self.best_audit['metrics'],
                'mlflow_run_id': self.pipeline_run_id if self.enable_mlflow else None
            }