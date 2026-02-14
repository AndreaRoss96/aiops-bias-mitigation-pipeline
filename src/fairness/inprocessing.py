
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

"""
In-processing bias mitigation algorithms
"""

class InprocessingMitigation:
    """
    In-processing mitigation strategies
    These modify the model training process itself
    """
    
    def __init__(self, config):
        self.config = config
        self.sess = None
    
    def train_adversarial_debiasing(self, dataset_train, dataset_test=None):
        """
        Adversarial Debiasing
        
        Learns a classifier to maximize prediction accuracy and simultaneously 
        reduce an adversary's ability to determine the protected attribute from 
        the predictions.
        
        This is the "model that ignores sensitive attributes" approach.
        
        Args:
            dataset_train: AIF360 BinaryLabelDataset
            dataset_test: Optional test dataset for validation
        
        Returns:
            dataset_pred: Predictions on test set (if provided)
            model: Trained adversarial debiasing model
        """
        print("\n🔧 Training Adversarial Debiasing Model...")
        
        # Create new TensorFlow session
        if self.sess is not None:
            self.sess.close()
        
        self.sess = tf.Session()
        
        # Initialize adversarial debiasing
        debiaser = AdversarialDebiasing(
            privileged_groups=self.config.PRIVILEGED_GROUPS,
            unprivileged_groups=self.config.UNPRIVILEGED_GROUPS,
            scope_name='adversarial_debiasing',
            debias=True,
            sess=self.sess,
            num_epochs=50,
            batch_size=128,
            classifier_num_hidden_units=200,
            adversary_loss_weight=0.1
        )
        
        # Train model
        print(" Training (this may take a minute)...")
        debiaser.fit(dataset_train)
        print(" Adversarial debiasing model trained")
        
        # Make predictions if test set provided
        if dataset_test is not None:
            dataset_pred = debiaser.predict(dataset_test)
            print(" Predictions generated on test set")
            return dataset_pred, debiaser
        
        return debiaser
    
    def train_prejudice_remover(self, dataset_train, dataset_test=None, eta=1.0):
        """
        Prejudice Remover
        
        Adds a discrimination-aware regularization term to the learning objective.
        
        Args:
            dataset_train: AIF360 BinaryLabelDataset
            dataset_test: Optional test dataset
            eta: Fairness penalty parameter (higher = more fairness emphasis)
        
        Returns:
            dataset_pred: Predictions on test set (if provided)
            model: Trained prejudice remover model
        """
        print(f"\nTraining Prejudice Remover (eta={eta})...")
        
        # Get the index of the sensitive attribute
        sensitive_attr_idx = dataset_train.feature_names.index(
            self.config.PROTECTED_ATTRIBUTE
        )
        
        # Initialize prejudice remover
        pr = PrejudiceRemover(
            sensitive_attr=self.config.PROTECTED_ATTRIBUTE,
            eta=eta
        )
        
        # Train model
        print(" Training...")
        pr.fit(dataset_train)
        print("   ✓ Prejudice remover trained")
        
        # Make predictions if test set provided
        if dataset_test is not None:
            dataset_pred = pr.predict(dataset_test)
            print("   ✓ Predictions generated on test set")
            return dataset_pred, pr
        
        return pr
    
    def cleanup(self):
        """Clean up TensorFlow session"""
        if self.sess is not None:
            self.sess.close()
            self.sess = None
            print("   ✓ TensorFlow session closed")
    
    def __del__(self):
        """Ensure session is closed when object is destroyed"""
        self.cleanup()