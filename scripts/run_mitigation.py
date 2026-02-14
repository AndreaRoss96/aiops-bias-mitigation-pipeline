"""
Main script to run the bias mitigation pipeline with MLflow integration
Usage: python scripts/run_mitigation.py --dataset adult --env development --mlflow
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.bias_config import get_config
from src.data.loaders import DatasetLoader
from src.pipeline.mitigation_pipeline import BiasMitigationPipeline
import matplotlib.pyplot as plt


def main():
    """Main execution function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run bias mitigation pipeline')
    parser.add_argument('--dataset', type=str, default='adult', 
                       choices=['adult', 'compas'],
                       help='Dataset to use')
    parser.add_argument('--env', type=str, default='development',
                       choices=['development', 'production'],
                       help='Environment configuration')
    parser.add_argument('--save-report', action='store_true',
                       help='Save audit report to file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--mlflow', action='store_true', default=False,
                       help='Enable MLflow tracking (default: enabled)')
    parser.add_argument('--no-mlflow', dest='mlflow', action='store_false',
                       help='Disable MLflow tracking')
    parser.add_argument('--mlflow-uri', type=str, default=None,
                       help='MLflow tracking URI (default: mlruns/)')
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI if provided
    if args.mlflow_uri:
        os.environ['MLFLOW_TRACKING_URI'] = args.mlflow_uri
    
    # Initialize configuration
    print("="*70)
    print("AIOPS BIAS MITIGATION PIPELINE")
    print("Automated Fairness Checks & Mitigation")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Environment: {args.env}")
    print(f"  Output: {args.output_dir}")
    print(f"  MLflow: {'✓ Enabled' if args.mlflow else '✗ Disabled'}")
    if args.mlflow and args.mlflow_uri:
        print(f"  MLflow URI: {args.mlflow_uri}")
    
    config = get_config(args.env)
    config.OUTPUT_DIR = args.output_dir
    config.REPORTS_DIR = f"{args.output_dir}/reports"
    config.PLOTS_DIR = f"{args.output_dir}/plots"
    
    # Create output directories
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    # Phase 1: Load dataset
    print("\n" + "="*70)
    print("PHASE 1: DATASET LOADING")
    print("="*70)
    
    loader = DatasetLoader(config)
    dataset = loader.load_dataset(args.dataset)
    
    # Split into train/test
    dataset_train, dataset_test = loader.split_dataset(dataset)
    
    # Phase 2 & 3: Run mitigation pipeline
    print("\n" + "="*70)
    print("PHASE 2 & 3: BIAS AUDIT & MITIGATION")
    print("="*70)
    
    try:
        pipeline = BiasMitigationPipeline(config, enable_mlflow=args.mlflow)
        best_model, best_audit = pipeline.run_full_pipeline(dataset_train, dataset_test)
        
        # Get deployment recommendation
        recommendation = pipeline.get_deployment_recommendation()
        
        print("\n" + "="*70)
        print("DEPLOYMENT RECOMMENDATION")
        print("="*70)
        print(f"\nDeploy: {' YES' if recommendation['deploy'] else ' NO'}")
        print(f"Selected Model: {recommendation['model']}")
        print(f"Reason: {recommendation['reason']}")
        
        if not recommendation['deploy']:
            print(f"Action Required: {recommendation['action_required']}")
        
        # Display key metrics from best model
        if best_audit:
            print("\nBest Model Metrics:")
            print(f"   Accuracy: {best_audit['metrics']['accuracy']:.3f}")
            print(f"   Disparate Impact: {best_audit['metrics']['disparate_impact']:.3f}")
            print(f"   Statistical Parity Diff: {best_audit['metrics']['statistical_parity_difference']:.3f}")
        
        if args.mlflow and recommendation.get('mlflow_run_id'):
            print(f"\nMLflow Run ID: {recommendation['mlflow_run_id']}")
            print("   View results with: mlflow ui")
            print("   Then navigate to: http://localhost:5000")
        
        # Save audit report if requested
        if args.save_report:
            report_path = f"{config.REPORTS_DIR}/audit_report.json"
            pipeline.auditor.save_audit_report(report_path)
            print(f"\n✓ Audit report saved to {report_path}")
        
        # Show plots
        print("\n✓ Pipeline execution complete!")
        print(f"✓ Results saved to {args.output_dir}/")
        
        # Print MLflow instructions
        if args.mlflow:
            print("\n" + "="*70)
            print("MLFLOW TRACKING")
            print("="*70)
            print("\nTo view results in MLflow UI:")
            print("  1. Run: mlflow ui")
            print("  2. Open: http://localhost:5000")
            print("  3. Navigate to 'bias-mitigation-aiops' experiment")
            print("\nTo compare runs:")
            print("  - Select multiple runs and click 'Compare'")
            print("  - View metrics, parameters, and artifacts")
        
        # Return exit code based on fairness gate
        if recommendation['deploy']:
            print("\n✅ FAIRNESS GATE: PASSED")
            return 0
        else:
            print("\n❌ FAIRNESS GATE: FAILED")
            return 1
            
    except Exception as e:
        print("\n❌ ERROR: Pipeline execution failed")
        print(f"Error details: {str(e)}")
        
        # Print stack trace for debugging
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)