from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()


class FairModel:
    """Base class for fairness-aware models"""
    
    def __init__(self, config):
        self.config = config
        self.is_trained = False
        self.model = None
    
    def train(self, dataset_train):
        """Train the model - to be implemented by subclasses"""
        raise NotImplementedError
    
    def predict(self, dataset):
        """Make predictions - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_model_info(self):
        """Return model information"""
        return {
            'type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'config': self.config.__class__.__name__
        }


class AdversarialDebiasingModel(FairModel):
    """
    Adversarial Debiasing Model
    
    Uses adversarial learning to train a classifier that maximizes prediction 
    accuracy while minimizing an adversary's ability to predict protected attributes.
    
    Best for: Strong fairness requirements, when preprocessing isn't enough
    """
    
    def __init__(self, config, num_epochs=50, batch_size=128, 
                 adversary_loss_weight=0.1):
        super().__init__(config)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.adversary_loss_weight = adversary_loss_weight
        self.sess = None
        self.scope_name = f'adversarial_debiasing_{id(self)}'
    
    def train(self, dataset_train):
        """
        Train adversarial debiasing model
        
        Args:
            dataset_train: AIF360 BinaryLabelDataset
        
        Returns:
            self: Trained model
        """
        print(f"\nTraining Adversarial Debiasing Model...")
        print(f"   Epochs: {self.num_epochs}, Batch size: {self.batch_size}")
        
        # Create TensorFlow session
        if self.sess is not None:
            self.sess.close()
        
        self.sess = tf.Session()
        
        # Initialize model
        self.model = AdversarialDebiasing(
            privileged_groups=self.config.PRIVILEGED_GROUPS,
            unprivileged_groups=self.config.UNPRIVILEGED_GROUPS,
            scope_name=self.scope_name,
            debias=True,
            sess=self.sess,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            adversary_loss_weight=self.adversary_loss_weight
        )
        
        # Train
        print("   Training (this may take 1-2 minutes)...")
        self.model.fit(dataset_train)
        
        self.is_trained = True
        print(" ✓ Training complete")
        
        return self
    
    def predict(self, dataset):
        """
        Make predictions
        
        Args:
            dataset: AIF360 BinaryLabelDataset
        
        Returns:
            dataset_pred: Dataset with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        dataset_pred = self.model.predict(dataset)
        return dataset_pred
    
    def cleanup(self):
        """Clean up TensorFlow session"""
        if self.sess is not None:
            self.sess.close()
            self.sess = None
            print("   ✓ TensorFlow session closed")
    
    def __del__(self):
        """Ensure session is closed when object is destroyed"""
        self.cleanup()


class PrejudiceRemoverModel(FairModel):
    """
    Prejudice Remover
    
    Adds a discrimination-aware regularization term to the learning objective.
    Works by penalizing models that produce biased predictions.
    
    Best for: Moderate fairness requirements, faster training than adversarial
    """
    
    def __init__(self, config, eta=1.0):
        """
        Args:
            config: BiasConfig object
            eta: Fairness penalty parameter (higher = more fairness emphasis)
                 Range: 0.0 (no fairness) to 25.0 (strong fairness)
        """
        super().__init__(config)
        self.eta = eta
    
    def train(self, dataset_train):
        """
        Train prejudice remover model
        
        Args:
            dataset_train: AIF360 BinaryLabelDataset
        
        Returns:
            self: Trained model
        """
        print(f"\nTraining Prejudice Remover (eta={self.eta})...")
        
        # Initialize model
        self.model = PrejudiceRemover(
            sensitive_attr=self.config.PROTECTED_ATTRIBUTE,
            eta=self.eta
        )
        
        # Train
        print("   Training...")
        self.model.fit(dataset_train)
        
        self.is_trained = True
        print(" ✓ Training complete")
        
        return self
    
    def predict(self, dataset):
        """
        Make predictions
        
        Args:
            dataset: AIF360 BinaryLabelDataset
        
        Returns:
            dataset_pred: Dataset with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        dataset_pred = self.model.predict(dataset)
        return dataset_pred


class ReweighingModel(FairModel):
    """
    Reweighing Model (Preprocessing + Baseline Classifier)
    
    Combines reweighing preprocessing with a standard classifier.
    This provides a cleaner interface than manually applying reweighing.
    
    Best for: First attempt at bias mitigation, works well in most cases
    """
    
    def __init__(self, config, base_classifier='logistic'):
        """
        Args:
            config: BiasConfig object
            base_classifier: 'logistic' or 'random_forest'
        """
        super().__init__(config)
        self.base_classifier_type = base_classifier
        self.reweighing = None
        self.base_model = None
    
    def train(self, dataset_train):
        """
        Apply reweighing and train model
        
        Args:
            dataset_train: AIF360 BinaryLabelDataset
        
        Returns:
            self: Trained model
        """
        from src.fairness.preprocessing import PreprocessingMitigation
        from src.models.baseline import BaselineModel
        
        print(f"\nTraining Reweighing Model...")
        
        # Apply reweighing
        preprocessing = PreprocessingMitigation(self.config)
        dataset_reweighed = preprocessing.apply_reweighing(dataset_train)
        
        # Train base classifier on reweighed data
        self.base_model = BaselineModel(
            model_type=self.base_classifier_type,
            config=self.config
        )
        self.base_model.train(dataset_reweighed)
        
        self.is_trained = True
        print(" ✓ Training complete")
        
        return self
    
    def predict(self, dataset):
        """
        Make predictions
        
        Args:
            dataset: AIF360 BinaryLabelDataset
        
        Returns:
            dataset_pred: Dataset with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.base_model.predict(dataset)


class DisparateImpactRemovalModel(FairModel):
    """
    Disparate Impact Removal Model
    
    Edits feature values to increase fairness while preserving rank-ordering.
    Then trains a standard classifier on the modified features.
    
    Best for: When you want to preserve feature interpretability
    """
    
    def __init__(self, config, repair_level=1.0, base_classifier='logistic'):
        """
        Args:
            config: BiasConfig object
            repair_level: Level of repair (0.0 = no repair, 1.0 = full repair)
            base_classifier: 'logistic' or 'random_forest'
        """
        super().__init__(config)
        self.repair_level = repair_level
        self.base_classifier_type = base_classifier
        self.dir_transformer = None
        self.base_model = None
    
    def train(self, dataset_train):
        """
        Apply DIR and train model
        
        Args:
            dataset_train: AIF360 BinaryLabelDataset
        
        Returns:
            self: Trained model
        """
        from src.fairness.preprocessing import PreprocessingMitigation
        from src.models.baseline import BaselineModel
        
        print(f"\nTraining DIR Model (repair_level={self.repair_level})...")
        
        # Apply DIR
        preprocessing = PreprocessingMitigation(self.config)
        dataset_transformed = preprocessing.apply_disparate_impact_remover(
            dataset_train, 
            repair_level=self.repair_level
        )
        
        # Train base classifier
        self.base_model = BaselineModel(
            model_type=self.base_classifier_type,
            config=self.config
        )
        self.base_model.train(dataset_transformed)
        
        self.is_trained = True
        print(" ✓ Training complete")
        
        return self
    
    def predict(self, dataset):
        """
        Make predictions
        
        Args:
            dataset: AIF360 BinaryLabelDataset
        
        Returns:
            dataset_pred: Dataset with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.base_model.predict(dataset)


# Factory function for easy model creation
def create_fair_model(model_type, config, **kwargs):
    """
    Factory function to create fair models
    
    Args:
        model_type: 'adversarial', 'prejudice_remover', 'reweighing', 'dir'
        config: BiasConfig object
        **kwargs: Additional parameters for specific models
    
    Returns:
        FairModel instance
    """
    models = {
        'adversarial': AdversarialDebiasingModel,
        'prejudice_remover': PrejudiceRemoverModel,
        'reweighing': ReweighingModel,
        'dir': DisparateImpactRemovalModel,
        'disparate_impact_removal': DisparateImpactRemovalModel
    }
    
    model_class = models.get(model_type.lower())
    if model_class is None:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(models.keys())}"
        )
    
    return model_class(config, **kwargs)