class BiasConfig:
    """Base configuration for bias mitigation pipeline"""
    
    # Dataset configuration
    DATASET_NAME = "adult"
    PROTECTED_ATTRIBUTE = "sex"
    
    # Protected groups definition
    PRIVILEGED_GROUPS = [{'sex': 1}]  # Male
    UNPRIVILEGED_GROUPS = [{'sex': 0}]  # Female
    
    # Alternative configurations for different protected attributes
    RACE_PRIVILEGED_GROUPS = [{'race': 1}]  # White
    RACE_UNPRIVILEGED_GROUPS = [{'race': 0}]  # Non-White
    
    AGE_PRIVILEGED_GROUPS = [{'age': 1}]  # Above 25
    AGE_UNPRIVILEGED_GROUPS = [{'age': 0}]  # Below 25
    
    # Fairness thresholds
    DISPARATE_IMPACT_THRESHOLD = 0.8
    STATISTICAL_PARITY_THRESHOLD = 0.1
    EQUAL_OPPORTUNITY_THRESHOLD = 0.1
    
    # Model configuration
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    
    # Features to keep from Adult dataset
    FEATURES_TO_KEEP = [
        'age', 
        'education-num', 
        'capital-gain', 
        'capital-loss', 
        'hours-per-week'
    ]
    
    # Output paths
    OUTPUT_DIR = "outputs"
    REPORTS_DIR = "outputs/reports"
    PLOTS_DIR = "outputs/plots"
    LOGS_DIR = "outputs/logs"
    MODELS_DIR = "models"


class DevelopmentConfig(BiasConfig):
    """Development environment configuration"""
    DISPARATE_IMPACT_THRESHOLD = 0.8
    DEBUG = True


class ProductionConfig(BiasConfig):
    """Production environment configuration - stricter thresholds"""
    DISPARATE_IMPACT_THRESHOLD = 0.85
    STATISTICAL_PARITY_THRESHOLD = 0.08
    DEBUG = False


class COMPASConfig(BiasConfig):
    """Configuration for COMPAS dataset"""
    DATASET_NAME = "compas"
    PROTECTED_ATTRIBUTE = "race"
    PRIVILEGED_GROUPS = [{'race': 1}]
    UNPRIVILEGED_GROUPS = [{'race': 0}]


def get_config(env='development'):
    """Factory function to get configuration based on environment"""
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'compas': COMPASConfig
    }
    return configs.get(env, DevelopmentConfig)
    