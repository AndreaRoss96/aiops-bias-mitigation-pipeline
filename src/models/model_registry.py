
import pickle
import json
import os
from datetime import datetime
from pathlib import Path

"""
Model Registry for managing trained models
Handles saving, loading, and versioning of models
"""

class ModelRegistry:
    """
    Central registry for managing trained models
    
    Features:
    - Save/load models with metadata
    - Version control
    - Model comparison
    - Production model tracking
    """
    
    def __init__(self, registry_dir="models"):
        """
        Initialize model registry
        
        Args:
            registry_dir: Root directory for storing models
        """
        self.registry_dir = Path(registry_dir)
        self.baseline_dir = self.registry_dir / "baseline"
        self.fair_dir = self.registry_dir / "fair"
        self.production_dir = self.registry_dir / "production"
        
        # Create directories
        for directory in [self.baseline_dir, self.fair_dir, self.production_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.registry_dir / "registry_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load registry metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'models': {},
            'production_model': None,
            'version_counter': 0
        }
    
    def _save_metadata(self):
        """Save registry metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_version(self):
        """Generate next version number"""
        self.metadata['version_counter'] += 1
        version = self.metadata['version_counter']
        return f"v{version}"
    
    def save_model(self, model, model_name, model_type='fair', 
                   fairness_metrics=None, performance_metrics=None, 
                   description=None):
        """
        Save a model with metadata
        
        Args:
            model: Trained model object
            model_name: Name for the model
            model_type: 'baseline' or 'fair'
            fairness_metrics: Dict of fairness metrics
            performance_metrics: Dict of performance metrics
            description: Optional description
        
        Returns:
            dict: Model metadata including file path and version
        """
        # Generate version
        version = self._generate_version()
        
        # Determine save directory
        save_dir = self.fair_dir if model_type == 'fair' else self.baseline_dir
        
        # Create model-specific directory
        model_dir = save_dir / f"{model_name}_{version}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Create metadata
            metadata = {
                'model_name': model_name,
                'version': version,
                'model_type': model_type,
                'saved_at': datetime.now().isoformat(),
                'file_path': str(model_path),
                'fairness_metrics': fairness_metrics or {},
                'performance_metrics': performance_metrics or {},
                'description': description or f"{model_type.title()} model: {model_name}",
                'model_class': model.__class__.__name__
            }
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update registry
            model_id = f"{model_name}_{version}"
            self.metadata['models'][model_id] = metadata
            self._save_metadata()
            
            print(f"✓ Model saved: {model_id}")
            print(f"  Path: {model_path}")
            
            return metadata
            
        except Exception as e:
            print(f"✗ Error saving model: {e}")
            raise
    
    def load_model(self, model_name, version=None):
        """
        Load a model
        
        Args:
            model_name: Name of the model
            version: Specific version (e.g., 'v1'), or None for latest
        
        Returns:
            tuple: (model_object, metadata)
        """
        # Find model
        if version:
            model_id = f"{model_name}_{version}"
        else:
            # Get latest version
            model_versions = [
                k for k in self.metadata['models'].keys()
                if k.startswith(model_name)
            ]
            if not model_versions:
                raise ValueError(f"Model '{model_name}' not found in registry")
            
            # Sort by version number
            model_versions.sort(key=lambda x: int(x.split('_v')[-1]))
            model_id = model_versions[-1]
        
        # Get metadata
        if model_id not in self.metadata['models']:
            raise ValueError(f"Model '{model_id}' not found in registry")
        
        metadata = self.metadata['models'][model_id]
        model_path = Path(metadata['file_path'])
        
        # Load model
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            print(f"✓ Model loaded: {model_id}")
            print(f"  Saved at: {metadata['saved_at']}")
            
            return model, metadata
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def set_production_model(self, model_name, version):
        """
        Set a model as the production model
        
        Args:
            model_name: Name of the model
            version: Version to promote
        """
        model_id = f"{model_name}_{version}"
        
        if model_id not in self.metadata['models']:
            raise ValueError(f"Model '{model_id}' not found")
        
        # Copy to production directory
        source_metadata = self.metadata['models'][model_id]
        source_path = Path(source_metadata['file_path'])
        
        prod_path = self.production_dir / f"{model_name}_production.pkl"
        
        import shutil
        shutil.copy(source_path, prod_path)
        
        # Update metadata
        self.metadata['production_model'] = {
            'model_id': model_id,
            'model_name': model_name,
            'version': version,
            'promoted_at': datetime.now().isoformat(),
            'path': str(prod_path)
        }
        self._save_metadata()
        
        print(f"✓ Model promoted to production: {model_id}")
        print(f"  Path: {prod_path}")
    
    def get_production_model(self):
        """
        Get the current production model
        
        Returns:
            tuple: (model_object, metadata)
        """
        if not self.metadata['production_model']:
            raise ValueError("No production model set")
        
        prod_info = self.metadata['production_model']
        prod_path = Path(prod_info['path'])
        
        with open(prod_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✓ Production model loaded: {prod_info['model_id']}")
        
        return model, prod_info
    
    def list_models(self, model_type=None, sort_by='saved_at'):
        """
        List all registered models
        
        Args:
            model_type: Filter by 'baseline' or 'fair' (None = all)
            sort_by: Sort by 'saved_at', 'accuracy', 'disparate_impact'
        
        Returns:
            list: List of model metadata dicts
        """
        models = list(self.metadata['models'].values())
        
        # Filter by type
        if model_type:
            models = [m for m in models if m['model_type'] == model_type]
        
        # Sort
        if sort_by == 'saved_at':
            models.sort(key=lambda x: x['saved_at'], reverse=True)
        elif sort_by in ['accuracy', 'disparate_impact']:
            models.sort(
                key=lambda x: x.get('fairness_metrics', {}).get(sort_by, 0),
                reverse=True
            )
        
        return models
    
    def compare_models(self, model_ids):
        """
        Compare multiple models
        
        Args:
            model_ids: List of model IDs to compare
        
        Returns:
            DataFrame: Comparison table
        """
        import pandas as pd
        
        data = []
        for model_id in model_ids:
            if model_id in self.metadata['models']:
                meta = self.metadata['models'][model_id]
                
                row = {
                    'Model ID': model_id,
                    'Type': meta['model_type'],
                    'Saved At': meta['saved_at'][:10],  # Just date
                }
                
                # Add fairness metrics
                row.update(meta.get('fairness_metrics', {}))
                
                # Add performance metrics
                row.update(meta.get('performance_metrics', {}))
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def get_best_fair_model(self, min_disparate_impact=0.8):
        """
        Get the best model that meets fairness constraints
        
        Args:
            min_disparate_impact: Minimum acceptable disparate impact
        
        Returns:
            tuple: (model_object, metadata) or None if no fair models
        """
        fair_models = [
            (k, v) for k, v in self.metadata['models'].items()
            if v.get('fairness_metrics', {}).get('disparate_impact', 0) >= min_disparate_impact
        ]
        
        if not fair_models:
            print("⚠️  No models meet fairness constraints")
            return None
        
        # Sort by accuracy
        fair_models.sort(
            key=lambda x: x[1].get('performance_metrics', {}).get('accuracy', 0),
            reverse=True
        )
        
        best_id, best_meta = fair_models[0]
        model, _ = self.load_model(best_meta['model_name'], best_meta['version'])
        
        print(f"✓ Best fair model: {best_id}")
        print(f"  Accuracy: {best_meta['performance_metrics'].get('accuracy', 'N/A')}")
        print(f"  Disparate Impact: {best_meta['fairness_metrics'].get('disparate_impact', 'N/A')}")
        
        return model, best_meta
    
    def delete_model(self, model_id):
        """
        Delete a model from the registry
        
        Args:
            model_id: Model ID to delete
        """
        if model_id not in self.metadata['models']:
            raise ValueError(f"Model '{model_id}' not found")
        
        # Get path and delete files
        metadata = self.metadata['models'][model_id]
        model_dir = Path(metadata['file_path']).parent
        
        import shutil
        shutil.rmtree(model_dir)
        
        # Remove from metadata
        del self.metadata['models'][model_id]
        self._save_metadata()
        
        print(f"✓ Model deleted: {model_id}")
    
    def export_model(self, model_id, export_path):
        """
        Export a model to a specific path
        
        Args:
            model_id: Model ID to export
            export_path: Path to export to
        """
        model, metadata = self.load_model(
            metadata['model_name'],
            metadata['version']
        )
        
        # Save model
        with open(export_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata alongside
        metadata_path = str(export_path).replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model exported to: {export_path}")
        print(f"✓ Metadata exported to: {metadata_path}")