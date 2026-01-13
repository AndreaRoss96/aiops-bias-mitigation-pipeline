from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

class FairnessMetrics:
    """
    Compute and manage fairness metrics
    """
    
    def __init__(self, config):
        self.config = config
    
    def compute_dataset_metrics(self, dataset):
        """
        Compute fairness metrics on dataset (before modeling)
        Used to understand inherent bias in the data
        """
        metric = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=self.config.UNPRIVILEGED_GROUPS,
            privileged_groups=self.config.PRIVILEGED_GROUPS
        )
        
        metrics = {
            'disparate_impact': metric.disparate_impact(),
            'statistical_parity_difference': metric.statistical_parity_difference(),
            'consistency': metric.consistency()[0] if hasattr(metric, 'consistency') else None,
            'base_rate_unprivileged': metric.base_rate(privileged=False),
            'base_rate_privileged': metric.base_rate(privileged=True),
            'num_positives_unprivileged': metric.num_positives(privileged=False),
            'num_positives_privileged': metric.num_positives(privileged=True),
        }
        
        return metrics
    
    def compute_classification_metrics(self, dataset_true, dataset_pred):
        """
        Compute fairness metrics on model predictions
        Used to evaluate model fairness
        """
        metric = ClassificationMetric(
            dataset_true,
            dataset_pred,
            unprivileged_groups=self.config.UNPRIVILEGED_GROUPS,
            privileged_groups=self.config.PRIVILEGED_GROUPS
        )
        
        metrics = {
            # Group fairness metrics
            'disparate_impact': metric.disparate_impact(),
            'statistical_parity_difference': metric.statistical_parity_difference(),
            
            # Equal opportunity metrics
            'equal_opportunity_difference': metric.equal_opportunity_difference(),
            'average_odds_difference': metric.average_odds_difference(),
            'average_abs_odds_difference': metric.average_abs_odds_difference(),
            
            # Predictive parity
            'theil_index': metric.theil_index(),
            
            # Performance metrics
            'accuracy': metric.accuracy(),
            'balanced_accuracy': (metric.true_positive_rate() + metric.true_negative_rate()) / 2,
            
            # Group-specific metrics
            'tpr_unprivileged': metric.true_positive_rate(privileged=False),
            'tpr_privileged': metric.true_positive_rate(privileged=True),
            'tnr_unprivileged': metric.true_negative_rate(privileged=False),
            'tnr_privileged': metric.true_negative_rate(privileged=True),
            'fpr_unprivileged': metric.false_positive_rate(privileged=False),
            'fpr_privileged': metric.false_positive_rate(privileged=True),
            'fnr_unprivileged': metric.false_negative_rate(privileged=False),
            'fnr_privileged': metric.false_negative_rate(privileged=True),
            
            # Precision metrics
            'ppv_unprivileged': metric.positive_predictive_value(privileged=False),
            'ppv_privileged': metric.positive_predictive_value(privileged=True),
        }
        
        return metrics
    
    def evaluate_fairness_constraints(self, metrics):
        """
        Evaluate if metrics meet fairness constraints
        Returns dict with pass/fail for each constraint
        """
        constraints = {}
        
        # Disparate Impact constraint: should be >= threshold
        if 'disparate_impact' in metrics:
            constraints['disparate_impact_pass'] = (
                metrics['disparate_impact'] >= self.config.DISPARATE_IMPACT_THRESHOLD
            )
        
        # Statistical Parity constraint: absolute difference should be small
        if 'statistical_parity_difference' in metrics:
            constraints['statistical_parity_pass'] = (
                abs(metrics['statistical_parity_difference']) <= self.config.STATISTICAL_PARITY_THRESHOLD
            )
        
        # Equal Opportunity constraint
        if 'equal_opportunity_difference' in metrics:
            constraints['equal_opportunity_pass'] = (
                abs(metrics['equal_opportunity_difference']) <= self.config.EQUAL_OPPORTUNITY_THRESHOLD
            )
        
        # Overall fairness gate
        constraints['fairness_gate_passed'] = all([
            constraints.get('disparate_impact_pass', False),
            constraints.get('statistical_parity_pass', False)
        ])
        
        return constraints
    
    def get_metric_interpretation(self, metric_name, value):
        """
        Get human-readable interpretation of a metric
        """
        interpretations = {
            'disparate_impact': (
                f"{'Fair' if value >= 0.8 else 'Biased'}: "
                f"Unprivileged group has {value:.1%} the selection rate of privileged group"
            ),
            'statistical_parity_difference': (
                f"{'Fair' if abs(value) <= 0.1 else 'Biased'}: "
                f"{abs(value):.1%} difference in selection rates between groups"
            ),
            'equal_opportunity_difference': (
                f"{'Fair' if abs(value) <= 0.1 else 'Biased'}: "
                f"{abs(value):.1%} difference in true positive rates"
            ),
            'accuracy': f"Model accuracy: {value:.1%}",
        }
        
        return interpretations.get(metric_name, f"{metric_name}: {value:.3f}")