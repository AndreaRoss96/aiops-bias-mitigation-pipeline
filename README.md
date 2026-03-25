# AIOps Bias Mitigation Pipeline

Automating bias mitigation as part of AIOps using AI Fairness 360 (AIF360) and MLflow for experiment tracking.

## Problem Statement

Machine learning models can perpetuate and amplify societal biases present in training data, leading to unfair outcomes across different demographic groups. This project implements an automated bias mitigation pipeline that:

1. **Detects bias** in ML models using established fairness metrics
2. **Automatically applies mitigation techniques** when bias is detected
3. **Tracks experiments** and compares different approaches using MLflow
4. **Provides continuous monitoring** for production models

The pipeline follows a four-phase approach: Discovery & Setup, Baseline & Bias Audit, Automated Mitigation, and Production Monitoring.

## Project Structure

```
├── config/                    # Configuration files
│   ├── bias_config.py        # Bias thresholds and dataset settings
│   └── __init__.py
├── outputs/                   # Generated outputs
│   ├── plots/               # Visualization results
│   │   ├── adversarial_training_detailed.png
│   │   ├── fairness_comparison.png
│   │   └── model_comparison_summary.png
│   └── reports/             # Audit reports
├── scripts/                  # Execution scripts
│   ├── compare_all_models.py    # Compare all trained models
│   ├── mlflow_compare_models.py # MLflow model comparison
│   ├── run_mitigation.py         # Run full mitigation pipeline
│   ├── train_with_plots.py      # Train models with visualization
│   └── test_mlflow_connection.py # Test MLflow connection
├── src/                      # Source code modules
│   ├── data/                # Data loading utilities
│   │   ├── loaders.py
│   │   └── __init__.py
│   ├── fairness/            # Fairness metrics and mitigation
│   │   ├── auditor.py       # Bias audit functionality
│   │   ├── inprocessing.py # In-processing mitigation
│   │   ├── metrics.py       # Fairness metric calculations
│   │   ├── preprocessing.py # Pre-processing mitigation
│   │   └── __init__.py
│   ├── models/              # Model implementations
│   │   ├── baseline.py      # Baseline (unbiased) models
│   │   ├── fair_models.py   # Fairness-aware models
│   │   ├── model_registry.py # Model registry management
│   │   └── __init__.py
│   ├── pipeline/            # Pipeline orchestration
│   │   ├── mitigation_pipeline.py # Main pipeline logic
│   │   └── __init__.py
│   └── utils/               # Utility functions
│       ├── mlflow_utils.py  # MLflow integration
│       ├── visualization.py # Plotting utilities
│       └── __init__.py
├── tests/                    # Test suite
├── requirements.txt          # Python dependencies
├── setup_env.sh            # Environment setup script
└── README.md               # This file
```

## Module Descriptions

### Configuration (`config/`)
- **bias_config.py**: Central configuration for datasets, protected attributes, and fairness thresholds
- Supports multiple environments (development, production) with different strictness levels

### Data Handling (`src/data/`)
- **loaders.py**: Utilities for loading and preprocessing datasets (Adult Income, COMPAS)
- Handles AIF360 dataset format conversion

### Fairness Module (`src/fairness/`)
- **auditor.py**: Main bias audit orchestrator
- **metrics.py**: Comprehensive fairness metric calculations (Disparate Impact, Statistical Parity, Equal Opportunity)
- **preprocessing.py**: Pre-processing mitigation (Reweighing)
- **inprocessing.py**: In-processing mitigation (Adversarial Debiasing)

### Models (`src/models/`)
- **baseline.py**: Standard ML models without bias mitigation
- **fair_models.py**: Fairness-aware model implementations
- **model_registry.py**: Model versioning and management

### Pipeline (`src/pipeline/`)
- **mitigation_pipeline.py**: End-to-end pipeline orchestration
- Automates the entire bias detection and mitigation workflow

### Utilities (`src/utils/`)
- **mlflow_utils.py**: MLflow experiment tracking and model registry
- **visualization.py**: Plotting and visualization utilities

## Fairness Thresholds Evaluation

Based on the experimental results (see plots in `outputs/plots/`), the selected thresholds are:

### Current Thresholds
- **Disparate Impact (DI) ≥ 0.8**: Industry standard for 80% rule
- **Statistical Parity Difference ≤ 0.1**: Maximum 10% difference in selection rates
- **Equal Opportunity Difference ≤ 0.1**: Maximum 10% difference in true positive rates

### Threshold Analysis

**✅ REASONABLE** - The thresholds are well-justified:

1. **Disparate Impact (0.8)**: 
   - Follows the "four-fifths rule" from US employment law
   - Results show Adversarial_w1.0 achieves DI ≈ 0.79, very close to threshold
   - Baseline model fails with DI ≈ 0.28, demonstrating significant bias

2. **Statistical Parity (0.1)**:
   - All mitigation techniques pass this threshold
   - Reweighing achieves ≈ 0.09, Adversarial achieves ≈ 0.03-0.07
   - Baseline fails with ≈ 0.19

3. **Equal Opportunity (0.1)**:
   - More challenging threshold - all models fail
   - May need adjustment based on use case requirements
   - Consider relaxing to 0.15 for this dataset

### Trade-off Analysis
The plots show a clear accuracy-fairness trade-off:
- **Baseline**: Highest accuracy (≈0.845) but worst fairness
- **Adversarial_w1.0**: Best fairness (DI≈0.79) but lowest accuracy
- **Reweighing**: Good balance with moderate accuracy and fairness

## Installation & Setup

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd aiops-bias-mitigation-pipeline

# Run setup script
chmod +x setup_env.sh
./setup_env.sh

# Activate virtual environment
source env/bin/activate
```

### 2. MLflow Configuration
```bash
# Test MLflow connection
python scripts/test_mlflow_connection.py

# Start MLflow UI (in separate terminal)
mlflow ui --port 5000
# Open: http://localhost:5000
```

## Usage

### Running the Full Pipeline
```bash
# Activate environment
source env/bin/activate

# Run complete bias mitigation pipeline
python scripts/run_mitigation.py

# Compare all models with visualization
python scripts/compare_all_models.py
```

### Individual Components

#### 1. Train Models with Plots
```bash
python scripts/train_with_plots.py
```
- Trains baseline, reweighing, and adversarial models
- Generates training progress plots
- Saves models to `outputs/`

#### 2. Model Comparison
```bash
python scripts/compare_all_models.py
```
- Compares all trained models on fairness metrics
- Generates comparison plots
- Selects best model based on fairness-accuracy trade-off

#### 3. MLflow Integration
```bash
# Compare models using MLflow
python scripts/mlflow_compare_models.py

# View experiments
mlflow ui --port 5000
```

## Results

### Model Performance Summary

Based on the generated plots in `outputs/plots/`:

#### 1. Model Comparison Summary (`model_comparison_summary.png`)
- **Baseline Model**: Highest accuracy (≈0.845) but fails fairness gates
- **Reweighing**: Balanced performance with moderate accuracy and fairness
- **Adversarial Models**: Progressive fairness improvement with accuracy trade-off

#### 2. Fairness Comparison (`fairness_comparison.png`)
- **Disparate Impact**: Only Adversarial_w1.0 approaches the 0.8 threshold
- **Statistical Parity**: All mitigation techniques pass the 0.1 threshold
- **Equal Opportunity**: All models fail - threshold may need adjustment

#### 3. Adversarial Training Details (`adversarial_training_detailed.png`)
- **Classifier Loss**: Rapid convergence within 20 iterations
- **Adversary Loss**: Stable oscillation indicating effective adversarial pressure
- **Combined Loss**: Shows the training dynamics over epochs

### Key Findings

1. **Bias Detection**: Baseline model shows significant bias (DI=0.28)
2. **Mitigation Effectiveness**: 
   - Reweighing: Moderate improvement with minimal accuracy loss
   - Adversarial: Strong fairness improvement with notable accuracy trade-off
3. **Threshold Appropriateness**: Current thresholds are reasonable but Equal Opportunity may need adjustment

## MLflow Integration

### Experiment Tracking
- **Experiment Name**: `bias-mitigation-aiops`
- **Tracking URI**: `sqlite:///mlflow.db`
- **UI Access**: `http://localhost:5000`

### Logged Information
- **Parameters**: Dataset, protected attributes, thresholds
- **Metrics**: Accuracy, DI, Statistical Parity, Equal Opportunity
- **Artifacts**: Model files, plots, audit reports
- **Tags**: Model type, fairness status, mitigation strategy

### Model Registry
- Automatic model registration for fair models
- Version tracking and stage management
- Production deployment recommendations

## Configuration

### Environment-Specific Settings

#### Development (`config/bias_config.py`)
```python
DISPARATE_IMPACT_THRESHOLD = 0.8
STATISTICAL_PARITY_THRESHOLD = 0.1
EQUAL_OPPORTUNITY_THRESHOLD = 0.1
```

#### Production (Stricter)
```python
DISPARATE_IMPACT_THRESHOLD = 0.85
STATISTICAL_PARITY_THRESHOLD = 0.08
EQUAL_OPPORTUNITY_THRESHOLD = 0.08
```

### Dataset Configuration
- **Default**: Adult Income dataset (gender bias)
- **Alternative**: COMPAS dataset (racial bias)
- **Protected Attributes**: Sex, Race, Age

## Production Monitoring

### Fairness Drift Detection
The pipeline includes automated monitoring for:
1. **Daily fairness checks** on production data
2. **Automatic alerts** when metrics drop below thresholds
3. **Rollback triggers** for fairness degradation
4. **Model retraining** recommendations

### Deployment Pipeline
1. **Fairness Gate**: Models must pass fairness checks
2. **Model Registry**: Version-controlled deployment
3. **Monitoring**: Continuous fairness tracking
4. **Alerting**: Automated notifications for drift

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all fairness tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [AI Fairness 360](https://aif360.mybluemix.net/)
- [MLflow](https://mlflow.org/)
- [What-If Tool](https://pair-code.github.io/what-if-tool/)
- [Four-Fifths Rule](https://en.wikipedia.org/wiki/Disparate_impact#The_80%_rule)

---

**Note**: This pipeline is designed for research and development purposes. Production deployment should include additional validation and monitoring specific to your use case.
