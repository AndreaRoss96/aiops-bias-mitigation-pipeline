"""
Compare all available fair models
This script trains and compares:
- Baseline (no mitigation)
- Reweighing
- Adversarial Debiasing
- Prejudice Remover
- Disparate Impact Removal (multiple repair levels)

Usage: python scripts/compare_all_models.py --dataset adult
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from config.bias_config import BiasConfig
from src.data.loaders import DatasetLoader
from src.models.baseline import BaselineModel
from src.models.fair_models import (
    AdversarialDebiasingModel,
    PrejudiceRemoverModel,
    ReweighingModel,
    DisparateImpactRemovalModel
)
from src.fairness.auditor import BiasAuditor
from src.models.model_registry import ModelRegistry
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate


def compare_all_models(dataset_name='adult', save_models=True, enable_mlflow=False):
    """
    Train and compare all available models
    """
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    # Setup
    config = BiasConfig()
    loader = DatasetLoader(config)
    auditor = BiasAuditor(config)
    registry = ModelRegistry() if save_models else None
    
    # Load data
    print("\n📦 Loading dataset...")
    dataset = loader.load_dataset(dataset_name)
    dataset_train, dataset_test = loader.split_dataset(dataset)
    
    # Store results
    results = []
    
    # ========================================================================
    # 1. Baseline Models
    # ========================================================================
    print("\n" + "="*80)
    print("BASELINE MODELS (No Mitigation)")
    print("="*80)
    
    for model_type in ['logistic', 'random_forest']:
        print(f"\n[{len(results)+1}] Training {model_type.title()}...")
        
        model = BaselineModel(model_type=model_type, config=config)
        model.train(dataset_train)
        predictions = model.predict(dataset_test)
        
        audit_result, passed = auditor.audit_model(
            dataset_test,
            predictions,
            model_name=f"Baseline ({model_type.title()})"
        )
        
        results.append({
            'name': f"Baseline_{model_type}",
            'category': 'Baseline',
            'model': model,
            'audit': audit_result,
            'passed': passed
        })
        
        if save_models:
            registry.save_model(
                model=model,
                model_name=f"baseline_{model_type}",
                model_type='baseline',
                fairness_metrics=audit_result['metrics'],
                description=f"Baseline {model_type} without bias mitigation"
            )
    
    # ========================================================================
    # 2. Reweighing (Preprocessing)
    # ========================================================================
    print("\n" + "="*80)
    print("PREPROCESSING: REWEIGHING")
    print("="*80)
    
    for base_clf in ['logistic', 'random_forest']:
        print(f"\n[{len(results)+1}] Training Reweighing + {base_clf.title()}...")
        
        model = ReweighingModel(config=config, base_classifier=base_clf)
        model.train(dataset_train)
        predictions = model.predict(dataset_test)
        
        audit_result, passed = auditor.audit_model(
            dataset_test,
            predictions,
            model_name=f"Reweighing ({base_clf.title()})"
        )
        
        results.append({
            'name': f"Reweighing_{base_clf}",
            'category': 'Preprocessing',
            'model': model,
            'audit': audit_result,
            'passed': passed
        })
        
        if save_models:
            registry.save_model(
                model=model,
                model_name=f"reweighing_{base_clf}",
                model_type='fair',
                fairness_metrics=audit_result['metrics'],
                description=f"Reweighing preprocessing with {base_clf}"
            )
    
    # ========================================================================
    # 3. Disparate Impact Removal (Preprocessing)
    # ========================================================================
    print("\n" + "="*80)
    print("PREPROCESSING: DISPARATE IMPACT REMOVAL")
    print("="*80)
    
    for repair_level in [0.5, 0.8, 1.0]:
        print(f"\n[{len(results)+1}] Training DIR (repair={repair_level})...")
        
        model = DisparateImpactRemovalModel(
            config=config,
            repair_level=repair_level,
            base_classifier='logistic'
        )
        model.train(dataset_train)
        predictions = model.predict(dataset_test)
        
        audit_result, passed = auditor.audit_model(
            dataset_test,
            predictions,
            model_name=f"DIR (repair={repair_level})"
        )
        
        results.append({
            'name': f"DIR_{repair_level}",
            'category': 'Preprocessing',
            'model': model,
            'audit': audit_result,
            'passed': passed
        })
        
        if save_models:
            registry.save_model(
                model=model,
                model_name=f"dir_{repair_level}",
                model_type='fair',
                fairness_metrics=audit_result['metrics'],
                description=f"Disparate Impact Removal (repair={repair_level})"
            )
    
    # ========================================================================
    # 4. Adversarial Debiasing (In-processing)
    # ========================================================================
    print("\n" + "="*80)
    print("IN-PROCESSING: ADVERSARIAL DEBIASING")
    print("="*80)
    
    for adv_weight in [0.1, 0.5]:
        print(f"\n[{len(results)+1}] Training Adversarial (weight={adv_weight})...")
        
        model = AdversarialDebiasingModel(
            config=config,
            num_epochs=50,
            adversary_loss_weight=adv_weight
        )
        model.train(dataset_train)
        predictions = model.predict(dataset_test)
        
        audit_result, passed = auditor.audit_model(
            dataset_test,
            predictions,
            model_name=f"Adversarial (weight={adv_weight})"
        )
        
        results.append({
            'name': f"Adversarial_{adv_weight}",
            'category': 'In-processing',
            'model': model,
            'audit': audit_result,
            'passed': passed
        })
        
        if save_models:
            registry.save_model(
                model=model,
                model_name=f"adversarial_{adv_weight}",
                model_type='fair',
                fairness_metrics=audit_result['metrics'],
                description=f"Adversarial Debiasing (adversary_weight={adv_weight})"
            )
        
        # Cleanup TF session
        model.cleanup()
    
    # ========================================================================
    # 5. Prejudice Remover (In-processing)
    # ========================================================================
    print("\n" + "="*80)
    print("IN-PROCESSING: PREJUDICE REMOVER")
    print("="*80)
    
    for eta in [1.0, 10.0]:
        print(f"\n[{len(results)+1}] Training Prejudice Remover (eta={eta})...")
        
        model = PrejudiceRemoverModel(config=config, eta=eta)
        model.train(dataset_train)
        predictions = model.predict(dataset_test)
        
        audit_result, passed = auditor.audit_model(
            dataset_test,
            predictions,
            model_name=f"Prejudice Remover (eta={eta})"
        )
        
        results.append({
            'name': f"PrejudiceRemover_{eta}",
            'category': 'In-processing',
            'model': model,
            'audit': audit_result,
            'passed': passed
        })
        
        if save_models:
            registry.save_model(
                model=model,
                model_name=f"prejudice_remover_{eta}",
                model_type='fair',
                fairness_metrics=audit_result['metrics'],
                description=f"Prejudice Remover (eta={eta})"
            )
    
    # ========================================================================
    # Generate Comparison Report
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Create comparison table
    comparison_data = []
    for r in results:
        metrics = r['audit']['metrics']
        comparison_data.append({
            'Model': r['name'],
            'Category': r['category'],
            'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
            'Disparate Impact': f"{metrics.get('disparate_impact', 0):.3f}",
            'SPD': f"{metrics.get('statistical_parity_difference', 0):.3f}",
            'EOD': f"{metrics.get('equal_opportunity_difference', 0):.3f}",
            'Gate': '✅' if r['passed'] else '❌'
        })
    
    df = pd.DataFrame(comparison_data)
    print("\n" + tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Find best models
    fair_models = [r for r in results if r['passed']]
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    print(f"\nTotal models trained: {len(results)}")
    print(f"Models passing fairness gate: {len(fair_models)}/{len(results)}")
    
    if fair_models:
        # Best by accuracy
        best_acc = max(fair_models, key=lambda x: x['audit']['metrics']['accuracy'])
        print(f"\n🏆 Best Accurate Fair Model: {best_acc['name']}")
        print(f"   Accuracy: {best_acc['audit']['metrics']['accuracy']:.3f}")
        print(f"   Disparate Impact: {best_acc['audit']['metrics']['disparate_impact']:.3f}")
        
        # Best by fairness
        best_fair = max(fair_models, key=lambda x: x['audit']['metrics']['disparate_impact'])
        print(f"\n⚖️  Most Fair Model: {best_fair['name']}")
        print(f"   Disparate Impact: {best_fair['audit']['metrics']['disparate_impact']:.3f}")
        print(f"   Accuracy: {best_fair['audit']['metrics']['accuracy']:.3f}")
        
        # Promote best to production
        if save_models:
            print("\n" + "="*80)
            print("MODEL REGISTRY")
            print("="*80)
            
            # Set best accurate model as production
            best_meta = [m for m in registry.list_models() 
                        if best_acc['name'].lower() in m['model_name'].lower()]
            
            if best_meta:
                best_meta = best_meta[0]
                registry.set_production_model(
                    best_meta['model_name'],
                    best_meta['version']
                )
    else:
        print("\n⚠️  No models passed the fairness gate")
        best_available = max(results, 
                           key=lambda x: x['audit']['metrics']['disparate_impact'])
        print(f"\nBest available (not fair): {best_available['name']}")
        print(f"   Disparate Impact: {best_available['audit']['metrics']['disparate_impact']:.3f}")
    
    # Generate visualization
    print("\n📊 Generating comparison visualization...")
    generate_comprehensive_plot(results)
    
    return results, registry


def generate_comprehensive_plot(results):
    """Generate comprehensive comparison plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Extract data
    names = [r['name'] for r in results]
    categories = [r['category'] for r in results]
    accuracies = [r['audit']['metrics']['accuracy'] for r in results]
    dis_impacts = [r['audit']['metrics']['disparate_impact'] for r in results]
    spds = [abs(r['audit']['metrics']['statistical_parity_difference']) for r in results]
    eods = [abs(r['audit']['metrics']['equal_opportunity_difference']) for r in results]
    gates = [r['passed'] for r in results]
    
    colors = ['green' if g else 'red' for g in gates]
    
    # Plot 1: Accuracy by category
    ax = axes[0, 0]
    category_order = ['Baseline', 'Preprocessing', 'In-processing']
    for cat in category_order:
        cat_indices = [i for i, c in enumerate(categories) if c == cat]
        cat_names = [names[i] for i in cat_indices]
        cat_accs = [accuracies[i] for i in cat_indices]
        cat_colors = [colors[i] for i in cat_indices]
        ax.bar(cat_names, cat_accs, color=cat_colors, alpha=0.7)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Model')
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylim([0.7, 0.9])
    
    # Plot 2: Disparate Impact
    ax = axes[0, 1]
    ax.bar(names, dis_impacts, color=colors, alpha=0.7)
    ax.axhline(y=0.8, color='red', linestyle='--', label='Threshold (0.8)')
    ax.set_ylabel('Disparate Impact')
    ax.set_title('Disparate Impact (Higher = More Fair)')
    ax.tick_params(axis='x', rotation=90)
    ax.legend()
    
    # Plot 3: Statistical Parity Difference
    ax = axes[0, 2]
    ax.bar(names, spds, color=colors, alpha=0.7)
    ax.axhline(y=0.1, color='red', linestyle='--', label='Threshold (0.1)')
    ax.set_ylabel('|SPD|')
    ax.set_title('Statistical Parity Difference (Lower = More Fair)')
    ax.tick_params(axis='x', rotation=90)
    ax.legend()
    
    # Plot 4: Equal Opportunity Difference
    ax = axes[1, 0]
    ax.bar(names, eods, color=colors, alpha=0.7)
    ax.set_ylabel('|EOD|')
    ax.set_title('Equal Opportunity Difference (Lower = More Fair)')
    ax.tick_params(axis='x', rotation=90)
    
    # Plot 5: Accuracy vs Disparate Impact (scatter)
    ax = axes[1, 1]
    ax.scatter(dis_impacts, accuracies, c=colors, s=100, alpha=0.6)
    ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('Disparate Impact')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Fairness Trade-off')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
    SUMMARY STATISTICS
    
    Total Models: {len(results)}
    Passing Gate: {sum(gates)}/{len(results)}
    
    ACCURACY RANGE
    Min: {min(accuracies):.3f}
    Max: {max(accuracies):.3f}
    Avg: {sum(accuracies)/len(accuracies):.3f}
    
    DISPARATE IMPACT RANGE
    Min: {min(dis_impacts):.3f}
    Max: {max(dis_impacts):.3f}
    Avg: {sum(dis_impacts)/len(dis_impacts):.3f}
    
    CATEGORIES
    Baseline: {categories.count('Baseline')}
    Preprocessing: {categories.count('Preprocessing')}
    In-processing: {categories.count('In-processing')}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    # Save
    import os
    os.makedirs('outputs/plots', exist_ok=True)
    filepath = 'outputs/plots/comprehensive_comparison.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"   ✓ Plot saved: {filepath}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Compare all fair models')
    parser.add_argument('--dataset', type=str, default='adult',
                       choices=['adult', 'compas'],
                       help='Dataset to use')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save models to registry')
    parser.add_argument('--mlflow', action='store_true',
                       help='Enable MLflow tracking')
    
    args = parser.parse_args()
    
    results, registry = compare_all_models(
        dataset_name=args.dataset,
        save_models=not args.no_save,
        enable_mlflow=args.mlflow
    )
    
    print("\n✅ Comparison complete!")
    
    if not args.no_save:
        print("\n💾 Models saved to registry. List with:")
        print("   from src.models.model_registry import ModelRegistry")
        print("   registry = ModelRegistry()")
        print("   print(registry.list_models())")


if __name__ == "__main__":
    main()