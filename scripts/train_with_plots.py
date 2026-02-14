import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import re
import io
from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt

from config.bias_config import BiasConfig
from src.data.loaders import DatasetLoader
from src.models.baseline import BaselineModel
from src.models.fair_models import AdversarialDebiasingModel, ReweighingModel
from src.fairness.auditor import BiasAuditor
from src.utils.visualization import TrainingVisualizer


def parse_adversarial_output(output_text):
    """
    Parse adversarial debiasing training output to extract losses
    
    Returns:
        classifier_losses: List of classifier losses per iteration
        adversary_losses: List of adversary losses per iteration
        epochs: List of epoch numbers
    """
    classifier_losses = []
    adversary_losses = []
    epochs = []
    
    # Pattern: "epoch X; iter: Y; batch classifier loss: Z; batch adversarial loss: W"
    pattern = r'epoch (\d+); iter: (\d+); batch classifier loss: ([0-9.]+); batch adversarial loss: ([0-9.]+)'
    
    for line in output_text.split('\n'):
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            clf_loss = float(match.group(3))
            adv_loss = float(match.group(4))
            
            epochs.append(epoch)
            classifier_losses.append(clf_loss)
            adversary_losses.append(adv_loss)
    
    return classifier_losses, adversary_losses, epochs


def train_and_plot_adversarial(config, dataset_train, dataset_test):
    """Train adversarial debiasing and capture detailed training curves"""
    print("\n" + "="*70)
    print("ADVERSARIAL DEBIASING - DETAILED TRAINING")
    print("="*70)
    
    # Capture training output
    f = io.StringIO()
    with redirect_stdout(f):
        model = AdversarialDebiasingModel(config, num_epochs=50)
        model.train(dataset_train)
        predictions = model.predict(dataset_test)
    
    output = f.getvalue()
    print(output)  # Print to console as well
    
    # Parse training output
    clf_losses, adv_losses, epochs = parse_adversarial_output(output)
    
    if clf_losses:
        print(f"\n✓ Captured {len(clf_losses)} training iterations")
        
        # Create detailed plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        iterations = range(len(clf_losses))
        unique_epochs = sorted(set(epochs))
        
        # Plot 1: Classifier loss over iterations
        ax = axes[0, 0]
        ax.plot(iterations, clf_losses, 'b-', alpha=0.3, linewidth=1, label='Raw')
        # Add smoothed curve
        if len(clf_losses) > 50:
            window = 100
            smoothed = np.convolve(clf_losses, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(clf_losses)), smoothed, 'b-', linewidth=2, label='Smoothed')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Classifier Loss')
        ax.set_title('Classifier Loss Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Adversary loss over iterations
        ax = axes[0, 1]
        ax.plot(iterations, adv_losses, 'r-', alpha=0.3, linewidth=1, label='Raw')
        if len(adv_losses) > 50:
            window = 100
            smoothed = np.convolve(adv_losses, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(adv_losses)), smoothed, 'r-', linewidth=2, label='Smoothed')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Adversary Loss')
        ax.set_title('Adversary Loss Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Both losses together (smoothed)
        ax = axes[1, 0]
        if len(clf_losses) > 50:
            window = 100
            clf_smoothed = np.convolve(clf_losses, np.ones(window)/window, mode='valid')
            adv_smoothed = np.convolve(adv_losses, np.ones(window)/window, mode='valid')
            iters_smooth = range(window-1, len(clf_losses))
            ax.plot(iters_smooth, clf_smoothed, 'b-', linewidth=2, label='Classifier')
            ax.plot(iters_smooth, adv_smoothed, 'r-', linewidth=2, label='Adversary')
        else:
            ax.plot(iterations, clf_losses, 'b-', linewidth=2, label='Classifier')
            ax.plot(iterations, adv_losses, 'r-', linewidth=2, label='Adversary')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Combined Loss View (Smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Average loss per epoch
        ax = axes[1, 1]
        epoch_clf_losses = []
        epoch_adv_losses = []
        
        for epoch in unique_epochs:
            epoch_indices = [i for i, e in enumerate(epochs) if e == epoch]
            epoch_clf_losses.append(np.mean([clf_losses[i] for i in epoch_indices]))
            epoch_adv_losses.append(np.mean([adv_losses[i] for i in epoch_indices]))
        
        ax.plot(unique_epochs, epoch_clf_losses, 'bo-', linewidth=2, label='Classifier', markersize=6)
        ax.plot(unique_epochs, epoch_adv_losses, 'ro-', linewidth=2, label='Adversary', markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Loss')
        ax.set_title('Average Loss Per Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        import os
        os.makedirs('outputs/plots', exist_ok=True)
        filepath = 'outputs/plots/adversarial_training_detailed.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\n✓ Detailed training plot saved: {filepath}")
        
        # Print training summary
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Total iterations: {len(clf_losses)}")
        print(f"Total epochs: {len(unique_epochs)}")
        print(f"\nClassifier Loss:")
        print(f"  Initial: {clf_losses[0]:.3f}")
        print(f"  Final: {clf_losses[-1]:.3f}")
        print(f"  Reduction: {((clf_losses[0] - clf_losses[-1]) / clf_losses[0] * 100):.1f}%")
        print(f"\nAdversary Loss:")
        print(f"  Initial: {adv_losses[0]:.3f}")
        print(f"  Final: {adv_losses[-1]:.3f}")
        print(f"  Change: {((adv_losses[-1] - adv_losses[0]) / adv_losses[0] * 100):.1f}%")
        
        plt.show()
    else:
        print("⚠️  Could not parse training output")
    
    # Audit the model
    auditor = BiasAuditor(config)
    audit_result, passed = auditor.audit_model(dataset_test, predictions, "Adversarial Debiasing")
    
    model.cleanup()
    
    return model, audit_result


def train_and_compare_all(config, dataset_train, dataset_test):
    """Train all models and create comparison plot"""
    print("\n" + "="*70)
    print("TRAINING ALL MODELS WITH DETAILED TRACKING")
    print("="*70)
    
    visualizer = TrainingVisualizer()
    auditor = BiasAuditor(config)
    results = []
    
    # 1. Baseline
    print("\n[1/3] Training Baseline...")
    baseline = BaselineModel(model_type='logistic', config=config)
    baseline.train(dataset_train)
    baseline_pred = baseline.predict(dataset_test)
    baseline_audit, _ = auditor.audit_model(dataset_test, baseline_pred, "Baseline")
    
    results.append({
        'name': 'Baseline',
        'accuracy': baseline_audit['metrics']['accuracy'],
        'loss': 1 - baseline_audit['metrics']['accuracy'],
        'disparate_impact': baseline_audit['metrics']['disparate_impact']
    })
    
    # 2. Reweighing
    print("\n[2/3] Training Reweighing...")
    rw_model = ReweighingModel(config=config, base_classifier='logistic')
    rw_model.train(dataset_train)
    rw_pred = rw_model.predict(dataset_test)
    rw_audit, _ = auditor.audit_model(dataset_test, rw_pred, "Reweighing")
    
    results.append({
        'name': 'Reweighing',
        'accuracy': rw_audit['metrics']['accuracy'],
        'loss': 1 - rw_audit['metrics']['accuracy'],
        'disparate_impact': rw_audit['metrics']['disparate_impact']
    })
    
    # 3. Adversarial (with detailed tracking)
    print("\n[3/3] Training Adversarial Debiasing...")
    adv_model, adv_audit = train_and_plot_adversarial(config, dataset_train, dataset_test)
    
    results.append({
        'name': 'Adversarial',
        'accuracy': adv_audit['metrics']['accuracy'],
        'loss': 1 - adv_audit['metrics']['accuracy'],
        'disparate_impact': adv_audit['metrics']['disparate_impact']
    })
    
    # Create comparison summary
    visualizer.plot_comparison_summary(results, save=True)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train with detailed plots')
    parser.add_argument('--dataset', type=str, default='adult',
                       choices=['adult', 'compas'],
                       help='Dataset to use')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'adversarial', 'baseline', 'reweighing'],
                       help='Which model to train')
    
    args = parser.parse_args()
    
    # Setup
    config = BiasConfig()
    loader = DatasetLoader(config)
    
    print("="*70)
    print("DETAILED TRAINING VISUALIZATION")
    print("="*70)
    print(f"\nDataset: {args.dataset}")
    print(f"Model: {args.model}")
    
    # Load data
    dataset = loader.load_dataset(args.dataset)
    dataset_train, dataset_test = loader.split_dataset(dataset)
    
    # Train based on selection
    if args.model == 'all':
        results = train_and_compare_all(config, dataset_train, dataset_test)
    elif args.model == 'adversarial':
        model, audit = train_and_plot_adversarial(config, dataset_train, dataset_test)
    elif args.model == 'baseline':
        auditor = BiasAuditor(config)
        model = BaselineModel(model_type='logistic', config=config)
        model.train(dataset_train)
        pred = model.predict(dataset_test)
        audit, _ = auditor.audit_model(dataset_test, pred, "Baseline")
    elif args.model == 'reweighing':
        auditor = BiasAuditor(config)
        model = ReweighingModel(config=config)
        model.train(dataset_train)
        pred = model.predict(dataset_test)
        audit, _ = auditor.audit_model(dataset_test, pred, "Reweighing")
    
    print("\n✅ Training complete! Check outputs/plots/ for visualizations.")


if __name__ == "__main__":
    main()