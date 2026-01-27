
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover

"""
Pre-processing bias mitigation algorithms
"""

class PreprocessingMitigation:
    """
    Pre-processing mitigation strategies
    These modify the training data before model training
    """
    
    def __init__(self, config):
        self.config = config
    
    def apply_reweighing(self, dataset_train):
        """
        Reweighing Algorithm
        
        Assigns different weights to training examples to reduce bias.
        Works by giving higher weights to instances that reduce discrimination.
        
        Args:
            dataset_train: AIF360 BinaryLabelDataset
        
        Returns:
            dataset_transformed: Reweighed dataset
        """
        print("\n🔧 Applying Reweighing (Pre-processing)...")
        
        # Initialize reweighing algorithm
        RW = Reweighing(
            unprivileged_groups=self.config.UNPRIVILEGED_GROUPS,
            privileged_groups=self.config.PRIVILEGED_GROUPS
        )
        
        # Transform dataset
        dataset_transformed = RW.fit_transform(dataset_train)
        
        print("   ✓ Training data reweighed to reduce bias")
        print(f"   ✓ Weights adjusted for {len(dataset_train.labels)} samples")
        
        # Show weight statistics
        weights = dataset_transformed.instance_weights
        print(f"   ✓ Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"   ✓ Mean weight: {weights.mean():.3f}")
        
        return dataset_transformed
    
    def apply_disparate_impact_remover(self, dataset_train, repair_level=1.0):
        """
        Disparate Impact Remover
        
        Edits feature values to increase fairness while preserving rank-ordering.
        
        Args:
            dataset_train: AIF360 BinaryLabelDataset
            repair_level: Level of repair (0.0 = no repair, 1.0 = full repair)
        
        Returns:
            dataset_transformed: Transformed dataset
        """
        print(f"\n🔧 Applying Disparate Impact Remover (repair_level={repair_level})...")
        
        # Initialize DIR algorithm
        DIR = DisparateImpactRemover(
            repair_level=repair_level,
            sensitive_attribute=self.config.PROTECTED_ATTRIBUTE
        )
        
        # Transform dataset
        dataset_transformed = DIR.fit_transform(dataset_train)
        
        print("   ✓ Features modified to reduce disparate impact")
        print(f"   ✓ Repair level: {repair_level:.1%}")
        
        return dataset_transformed
    
    def compare_preprocessing_methods(self, dataset_train):
        """
        Compare different preprocessing methods
        Returns a dict of transformed datasets
        """
        print("\n" + "="*70)
        print("PREPROCESSING COMPARISON")
        print("="*70)
        
        results = {
            'original': dataset_train,
            'reweighing': self.apply_reweighing(dataset_train),
            'dir_full': self.apply_disparate_impact_remover(dataset_train, repair_level=1.0),
            'dir_half': self.apply_disparate_impact_remover(dataset_train, repair_level=0.5)
        }
        
        print("\n✓ Preprocessing comparison complete")
        print(f"  {len(results)} variants created")
        
        return results