
from aif360.datasets import AdultDataset, CompasDataset
import pandas as pd
"""
Data loading utilities
"""


class DatasetLoader:
    """
    Load and prepare datasets for bias mitigation
    """
    
    def __init__(self, config):
        self.config = config
    
    def load_adult_dataset(self):
        """
        Load UCI Adult Income dataset
        Protected attribute: sex (gender bias)
        """
        dataset = AdultDataset(
            protected_attribute_names=[self.config.PROTECTED_ATTRIBUTE],
            privileged_classes=[['Male']],
            categorical_features=[],
            features_to_keep=self.config.FEATURES_TO_KEEP
        )
        
        print(f"✓ Adult dataset loaded")
        print(f"  - Total samples: {len(dataset.labels)}")
        print(f"  - Protected attribute: {self.config.PROTECTED_ATTRIBUTE}")
        print(f"  - Features: {len(self.config.FEATURES_TO_KEEP)}")
        
        return dataset
    
    def load_compas_dataset(self):
        """
        Load COMPAS Recidivism dataset
        Protected attribute: race (racial bias)
        """
        dataset = CompasDataset(
            protected_attribute_names=['race'],
            privileged_classes=[['Caucasian']],
            categorical_features=['age_cat', 'c_charge_degree', 'c_charge_desc'],
            features_to_keep=['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 
                            'juv_misd_count', 'juv_other_count', 'priors_count',
                            'c_charge_degree', 'c_charge_desc', 'two_year_recid']
        )
        
        print(f"✓ COMPAS dataset loaded")
        print(f"  - Total samples: {len(dataset.labels)}")
        print(f"  - Protected attribute: race")
        
        return dataset
    
    def load_dataset(self, dataset_name=None):
        """
        Load dataset based on configuration
        """
        dataset_name = dataset_name or self.config.DATASET_NAME
        
        loaders = {
            'adult': self.load_adult_dataset,
            'compas': self.load_compas_dataset
        }
        
        loader = loaders.get(dataset_name.lower())
        if not loader:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return loader()
    
    def split_dataset(self, dataset, test_size=None, shuffle=True):
        """
        Split dataset into train and test sets
        """
        test_size = test_size or self.config.TEST_SIZE
        train_size = 1 - test_size
        
        dataset_train, dataset_test = dataset.split(
            [train_size], 
            shuffle=shuffle, 
            seed=self.config.RANDOM_STATE
        )
        
        print(f"✓ Dataset split completed")
        print(f"  - Training samples: {len(dataset_train.labels)}")
        print(f"  - Test samples: {len(dataset_test.labels)}")
        
        return dataset_train, dataset_test