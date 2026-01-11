"""
Baseline models without bias mitigation
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


class BaselineModel:
    """
    Baseline models trained without any bias mitigation
    Used to establish the starting point for fairness metrics
    """
    
    def __init__(self, model_type='logistic', config=None):
        """
        Args:
            model_type: 'logistic' or 'random_forest'
            config: BiasConfig object
        """
        self.model_type = model_type
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def _init_model(self):
        """Initialize the underlying sklearn model"""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.config.RANDOM_STATE if self.config else 42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE if self.config else 42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, dataset_train):
        """
        Train baseline model on AIF360 dataset
        
        Args:
            dataset_train: AIF360 BinaryLabelDataset
        
        Returns:
            self: Trained model
        """
        print(f"\n🔧 Training {self.model_type.title()} model...")
        
        # Extract features and labels
        X_train = dataset_train.features
        y_train = dataset_train.labels.ravel()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize and train model
        self._init_model()
        self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Calculate training accuracy
        train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        
        print(f"   ✓ Training completed")
        print(f"   ✓ Training accuracy: {train_acc:.3f}")
        
        return self
    
    def predict(self, dataset):
        """
        Make predictions on AIF360 dataset
        
        Args:
            dataset: AIF360 BinaryLabelDataset
        
        Returns:
            dataset_pred: Copy of dataset with predictions as labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract and scale features
        X = dataset.features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Create a copy of the dataset with predictions
        dataset_pred = dataset.copy()
        dataset_pred.labels = predictions.reshape(-1, 1)
        
        return dataset_pred
    
    def predict_proba(self, dataset):
        """
        Get prediction probabilities
        
        Args:
            dataset: AIF360 BinaryLabelDataset
        
        Returns:
            probabilities: numpy array of shape (n_samples, n_classes)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = dataset.features
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, dataset_test):
        """
        Evaluate model on test dataset
        
        Args:
            dataset_test: AIF360 BinaryLabelDataset
        
        Returns:
            dict: Performance metrics
        """
        dataset_pred = self.predict(dataset_test)
        
        y_true = dataset_test.labels.ravel()
        y_pred = dataset_pred.labels.ravel()
        
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\nModel Performance:")
        print(f"   Test Accuracy: {accuracy:.3f}")
        print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
        
        return {
            'accuracy': accuracy,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance (for tree-based models) or coefficients
        
        Returns:
            dict: Feature names mapped to importance/coefficient values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if self.model_type == 'random_forest':
            importances = self.model.feature_importances_
        elif self.model_type == 'logistic':
            importances = np.abs(self.model.coef_[0])
        else:
            return None
        
        if feature_names is not None:
            return dict(zip(feature_names, importances))
        
        return importances
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        import pickle
        
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type
            }, f)
        
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath, config=None):
        """Load trained model from disk"""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(model_type=data['model_type'], config=config)
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.is_trained = True
        
        print(f"✓ Model loaded from {filepath}")
        return instance