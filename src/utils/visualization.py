"""
Visualization utilities for training history and model performance
"""

import matplotlib.pyplot as plt
import numpy as np
import os


class TrainingVisualizer:
    """
    Visualize training history for models
    Tracks and plots accuracy, loss, and fairness metrics over epochs
    """
    
    def __init__(self, output_dir="outputs/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.history = {}
    
    def add_training_history(self, model_name, history):
        """
        Add training history for a model
        
        Args:
            model_name: Name of the model
            history: Dict with keys like 'loss', 'accuracy', 'val_loss', 'val_accuracy'
                    or list of epoch losses/metrics
        """
        self.history[model_name] = history
    
    def plot_training_curves(self, model_name=None, save=True):
        """
        Plot training curves (loss and accuracy over epochs)
        
        Args:
            model_name: Specific model to plot, or None for all models
            save: Whether to save the plot
        """
        if model_name:
            models_to_plot = {model_name: self.history[model_name]}
        else:
            models_to_plot = self.history
        
        if not models_to_plot:
            print("⚠️  No training history to plot")
            return None
        
        n_models = len(models_to_plot)
        fig, axes = plt.subplots(n_models, 2, figsize=(14, 5*n_models))
        
        # Handle single model case
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, hist) in enumerate(models_to_plot.items()):
            # Plot loss
            ax_loss = axes[idx, 0]
            ax_acc = axes[idx, 1]
            
            if isinstance(hist, dict):
                # Dictionary format (e.g., from Keras)
                epochs = range(1, len(hist.get('loss', [])) + 1)
                
                if 'loss' in hist:
                    ax_loss.plot(epochs, hist['loss'], 'b-', label='Training Loss', linewidth=2)
                if 'val_loss' in hist:
                    ax_loss.plot(epochs, hist['val_loss'], 'r--', label='Validation Loss', linewidth=2)
                
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss')
                ax_loss.set_title(f'{name} - Loss')
                ax_loss.legend()
                ax_loss.grid(True, alpha=0.3)
                
                # Plot accuracy
                if 'accuracy' in hist:
                    ax_acc.plot(epochs, hist['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
                if 'val_accuracy' in hist:
                    ax_acc.plot(epochs, hist['val_accuracy'], 'r--', label='Validation Accuracy', linewidth=2)
                
                ax_acc.set_xlabel('Epoch')
                ax_acc.set_ylabel('Accuracy')
                ax_acc.set_title(f'{name} - Accuracy')
                ax_acc.legend()
                ax_acc.grid(True, alpha=0.3)
            
            elif isinstance(hist, list):
                # List format (simple loss values)
                epochs = range(1, len(hist) + 1)
                ax_loss.plot(epochs, hist, 'b-', linewidth=2)
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss')
                ax_loss.set_title(f'{name} - Training Loss')
                ax_loss.grid(True, alpha=0.3)
                
                ax_acc.text(0.5, 0.5, 'Accuracy history not available', 
                           ha='center', va='center', transform=ax_acc.transAxes)
                ax_acc.set_title(f'{name} - Accuracy')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'training_curves.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Training curves saved: {filepath}")
        
        return fig
    
    def plot_adversarial_training(self, classifier_losses, adversary_losses, 
                                  model_name="Adversarial Debiasing", save=True):
        """
        Plot adversarial training curves (classifier vs adversary)
        
        Args:
            classifier_losses: List of classifier losses per epoch
            adversary_losses: List of adversary losses per epoch
            model_name: Name of the model
            save: Whether to save the plot
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs = range(1, len(classifier_losses) + 1)
        
        # Plot 1: Classifier Loss
        ax1.plot(epochs, classifier_losses, 'b-', linewidth=2, label='Classifier Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name} - Classifier Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Adversary Loss
        ax2.plot(epochs, adversary_losses, 'r-', linewidth=2, label='Adversary Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'{model_name} - Adversary Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Both losses together
        ax3.plot(epochs, classifier_losses, 'b-', linewidth=2, label='Classifier Loss')
        ax3.plot(epochs, adversary_losses, 'r-', linewidth=2, label='Adversary Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title(f'{model_name} - Combined View')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'{model_name.lower().replace(" ", "_")}_training.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Adversarial training plot saved: {filepath}")
        
        return fig
    
    def plot_comparison_summary(self, models_data, save=True):
        """
        Plot final comparison of all models (accuracy, loss, fairness)
        
        Args:
            models_data: List of dicts with keys: 'name', 'accuracy', 'loss', 'disparate_impact'
            save: Whether to save the plot
        """
        if not models_data:
            print("⚠️  No model data to plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        names = [m['name'] for m in models_data]
        accuracies = [m.get('accuracy', 0) for m in models_data]
        losses = [m.get('loss', 0) for m in models_data]
        dis_impacts = [m.get('disparate_impact', 0) for m in models_data]
        
        # Color code by fairness gate
        colors = ['green' if di >= 0.8 else 'red' for di in dis_impacts]
        
        # Plot 1: Final Accuracy
        axes[0, 0].bar(names, accuracies, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Final Model Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Final Loss
        axes[0, 1].bar(names, losses, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Final Model Loss (Lower is Better)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Disparate Impact
        axes[1, 0].bar(names, dis_impacts, color=colors, alpha=0.7)
        axes[1, 0].axhline(y=0.8, color='red', linestyle='--', label='Fairness Threshold')
        axes[1, 0].set_ylabel('Disparate Impact')
        axes[1, 0].set_title('Fairness: Disparate Impact')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Accuracy vs Disparate Impact scatter
        axes[1, 1].scatter(dis_impacts, accuracies, c=colors, s=150, alpha=0.6, edgecolors='black')
        axes[1, 1].axvline(x=0.8, color='red', linestyle='--', alpha=0.3, label='Fairness Threshold')
        for i, name in enumerate(names):
            axes[1, 1].annotate(name, (dis_impacts[i], accuracies[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('Disparate Impact')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Accuracy vs Fairness Trade-off')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'model_comparison_summary.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Comparison summary saved: {filepath}")
        
        return fig


class TrainingCallback:
    """
    Callback to track training progress
    Compatible with sklearn and custom training loops
    """
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.batch_losses = []
        
    def on_epoch_end(self, epoch, loss=None, accuracy=None):
        """Record metrics at end of epoch"""
        if loss is not None:
            self.epoch_losses.append(loss)
        if accuracy is not None:
            self.epoch_accuracies.append(accuracy)
    
    def on_batch_end(self, batch, loss=None):
        """Record batch loss"""
        if loss is not None:
            self.batch_losses.append(loss)
    
    def get_history(self):
        """Get training history as dict"""
        history = {}
        if self.epoch_losses:
            history['loss'] = self.epoch_losses
        if self.epoch_accuracies:
            history['accuracy'] = self.epoch_accuracies
        return history
    
    def plot_learning_curve(self, save=True, output_dir="outputs/plots"):
        """Plot learning curve from batch losses"""
        if not self.batch_losses:
            print("⚠️  No batch losses recorded")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        batches = range(1, len(self.batch_losses) + 1)
        ax.plot(batches, self.batch_losses, 'b-', alpha=0.3, linewidth=1)
        
        # Add smoothed curve
        window_size = min(50, len(self.batch_losses) // 10)
        if window_size > 1:
            smoothed = np.convolve(self.batch_losses, 
                                  np.ones(window_size)/window_size, 
                                  mode='valid')
            ax.plot(range(window_size, len(self.batch_losses)+1), smoothed, 
                   'r-', linewidth=2, label='Smoothed')
        
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{self.model_name} - Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, 
                                   f'{self.model_name.lower().replace(" ", "_")}_learning_curve.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Learning curve saved: {filepath}")
        
        return fig