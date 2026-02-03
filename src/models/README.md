# Models Module - ML Training, Registry, Scoring & Monitoring

This module implements a production-grade ML engineering pipeline for churn prediction, including model training, evaluation, registry, batch scoring, and monitoring.

## Architecture

```
Feature Pipeline → Training Pipeline → Evaluation Gate → Model Registry → Batch Scoring
                         ↓                                                      ↓
                   MLflow Tracking                                       Monitoring
```

## Installation

First, install the MLOps dependencies (not included by default):

```bash
pip install -e ".[mlops]"  # Installs mlflow, bentoml
pip install shap  # For model explainability (SHAP values)
```

## Quick Start

### 1. Train a Model

```bash
# Train XGBoost model with default settings
python -m src.models.build_models train

# Train with custom settings
python -m src.models.build_models train \
    --model-type xgboost \
    --experiment churn_v2 \
    --register
```

### 2. Evaluate the Model

```bash
# Evaluate the latest registered model
python -m src.models.build_models evaluate --version 1
```

### 3. Promote to Production

```bash
# Promote model version 1 to production
python -m src.models.build_models promote --version 1
```

### 4. Run Batch Scoring

```bash
# Score all active accounts with production model
python -m src.models.build_models score
```

### 5. Monitor for Drift

```bash
# Check for data and prediction drift
python -m src.models.build_models monitor --save
```

### 6. Model History & Comparison

```bash
# View all model versions
python -m src.models.build_models history

# Compare two versions
python -m src.models.build_models compare --version1 1 --version2 2
```

## Module Structure

### Configuration (`config.py`)

Defines all configuration dataclasses:

- **ModelConfig**: Model type, hyperparameters, training settings
- **TrainingConfig**: MLflow tracking, metrics, SHAP settings
- **EvaluationConfig**: Quality gate thresholds for promotion
- **ScoringConfig**: Batch scoring settings, risk tiers
- **MonitoringConfig**: Drift detection methods and thresholds

Example:

```python
from src.models import ModelConfig, ModelType

config = ModelConfig(
    model_type=ModelType.XGBOOST,
    xgboost_params={
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": 3,  # Handle class imbalance
    },
)
```

### Training Pipeline (`train.py`)

**ModelTrainer** class implements the two-model pattern:
- Always trains both primary model (XGBoost/LightGBM) and baseline (Logistic Regression)
- Performs stratified train/val/test split (60/20/20)
- Runs k-fold cross-validation
- Logs everything to MLflow
- Computes SHAP values for explainability

Example:

```python
from src.models import ModelTrainer, ModelConfig, TrainingConfig
from src.features import create_training_dataset

# Load data
X, y = create_training_dataset(engine)

# Train
trainer = ModelTrainer(ModelConfig(), TrainingConfig())
result = trainer.train(X, y)

print(f"AUC: {result.metrics['auc']:.4f}")
print(f"Beats baseline by: {result.improvement_over_baseline:.4f}")
```

### Evaluation Gate (`evaluate.py`)

**EvaluationGate** implements automated quality checks:

1. **Performance threshold**: AUC, precision, recall minimums
2. **Improvement over baseline**: Must beat simple model
3. **Improvement over production**: Must match/exceed current production
4. **Calibration check**: Predicted probabilities should match observed rates
5. **Prediction distribution**: Output shouldn't shift dramatically
6. **Feature stability**: Top features shouldn't change wildly

Example:

```python
from src.models import EvaluationGate, EvaluationConfig

gate = EvaluationGate(EvaluationConfig(min_auc=0.70))
result = gate.evaluate(training_result, X_test, y_test)

if result.recommendation == "promote":
    print("✓ Safe to promote to production")
else:
    print(f"✗ Issues: {len(result.get_failed_checks())} failed checks")
```

### Model Registry (`registry.py`)

**ModelRegistry** wraps MLflow Model Registry:

- Register trained models with metadata
- Promote models through stages (Staging → Production)
- Load production models for scoring
- Compare model versions
- Track model history

Example:

```python
from src.models import ModelRegistry

registry = ModelRegistry()

# Register a trained model
metadata = registry.register_model(
    training_result,
    description="XGBoost with improved feature engineering",
)

# Promote to production
registry.promote_model(version=1, stage="Production")

# Load for inference
model = registry.load_production_model()
```

### Batch Scoring (`score.py`)

**BatchScorer** generates predictions for accounts:

- Loads production model from registry
- Runs feature pipeline for current data
- Classifies accounts into risk tiers (high/medium/low)
- Writes predictions to PostgreSQL
- Logs scoring runs for monitoring

Example:

```python
from src.models import score_active_accounts

result = score_active_accounts(engine)

print(f"Scored {result.n_accounts_scored} accounts")
print(f"High risk: {result.risk_distribution['high']}")
```

Risk tiers (configurable):
- **High risk**: Probability ≥ 0.70 (immediate intervention needed)
- **Medium risk**: 0.40 ≤ Probability < 0.70 (monitor closely)
- **Low risk**: Probability < 0.40 (healthy customers)

### Model Monitoring (`monitoring.py`)

**ModelMonitor** tracks model health over time:

**Data Drift Detection**:
- Uses Population Stability Index (PSI)
- PSI < 0.1: Healthy ✓
- 0.1 ≤ PSI < 0.25: Warning ⚠️
- PSI ≥ 0.25: Critical ✗

**Prediction Drift**:
- Tracks mean prediction, std, risk distribution
- Alerts if predictions shift significantly

**Performance Drift** (when labels available):
- Compares actual vs predicted churn rates
- Tracks AUC degradation over time

Example:

```python
from src.models import ModelMonitor, MonitoringConfig

monitor = ModelMonitor(engine, MonitoringConfig())
report = monitor.check_drift(current_features, reference_features)

if report.health_status == HealthStatus.CRITICAL:
    print("⚠️ Model needs retraining!")
    print("Recommended actions:", report.recommended_actions)
```

## Database Tables

The system creates two PostgreSQL tables:

### `predictions` table:
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL,
    churn_probability FLOAT NOT NULL,
    risk_tier VARCHAR(20) NOT NULL,
    model_version INTEGER NOT NULL,
    scored_at TIMESTAMP NOT NULL
);
```

### `monitoring_log` table:
```sql
CREATE TABLE monitoring_log (
    id SERIAL PRIMARY KEY,
    report_date TIMESTAMP NOT NULL,
    health_status VARCHAR(20) NOT NULL,
    n_drifted_features INTEGER,
    n_critical_features INTEGER,
    prediction_drift_score FLOAT,
    performance_degradation FLOAT,
    recommended_actions TEXT
);
```

## MLflow Integration

All training runs are tracked in MLflow:

```bash
# View MLflow UI
mlflow ui

# Then open http://localhost:5000
```

Logged artifacts include:
- Hyperparameters
- Metrics (AUC, precision, recall, F1)
- Cross-validation scores
- Feature importances
- SHAP values (if enabled)
- Trained models

## Testing

Run the test suite:

```bash
# All tests
pytest tests/test_models.py -v

# Specific test
pytest tests/test_models.py::test_model_training_end_to_end -v

# With coverage
pytest tests/test_models.py --cov=src.models
```

## Hyperparameter Defaults

The default XGBoost hyperparameters are tuned for churn prediction:

- `max_depth=6`: Moderate depth to prevent overfitting
- `learning_rate=0.1`: Standard learning rate
- `n_estimators=100`: Boosting rounds
- `scale_pos_weight=3`: Upweight minority class (churned users)
- `subsample=0.8`: Row sampling for generalization
- `colsample_bytree=0.8`: Feature sampling

Override these by modifying `ModelConfig.xgboost_params`.

## Design Decisions

### Why the "Two Model" Pattern?

Always training a logistic regression baseline provides:
- **Sanity check**: Complex model should beat simple model
- **Cost-benefit analysis**: Is complexity justified?
- **Debugging**: If XGBoost barely beats LR, investigate features

### Why Evaluation Gates?

Automated quality checks prevent bad models from reaching production:
- **Consistency**: Same standards applied to every model
- **Documentation**: Clear criteria for promotion decisions
- **Safety**: Catch regressions before they impact users

### Why Population Stability Index (PSI)?

PSI is industry-standard for drift detection because it's:
- **Interpretable**: Clear thresholds (0.1, 0.25)
- **Robust**: Works for any feature distribution
- **Proven**: Used in credit scoring, fraud detection

## Common Workflows

### Weekly Retraining

```bash
# 1. Train new model
python -m src.models.build_models train --register

# 2. Evaluate against production
python -m src.models.build_models evaluate --version 2

# 3. If passed, promote
python -m src.models.build_models promote --version 2

# 4. Score accounts
python -m src.models.build_models score

# 5. Monitor for drift
python -m src.models.build_models monitor --save
```

### Experiment with Different Models

```bash
# Try XGBoost
python -m src.models.build_models train \
    --model-type xgboost \
    --experiment xgb_tuning

# Try LightGBM
python -m src.models.build_models train \
    --model-type lightgbm \
    --experiment lgbm_tuning

# Compare results in MLflow UI
mlflow ui
```

### A/B Test Models

```bash
# Promote candidate to Staging
python -m src.models.build_models promote --version 3 --stage Staging

# Score with both models
python -m src.models.build_models score --model-version 2  # Production
python -m src.models.build_models score --model-version 3  # Staging

# Compare predictions and decide
python -m src.models.build_models compare --version1 2 --version2 3
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'mlflow'"

Install MLOps dependencies:
```bash
pip install -e ".[mlops]"
```

### "SHAP computation failed"

SHAP is optional. Either:
1. Install it: `pip install shap`
2. Disable it: `--no-shap` flag

### "Model registry not found"

The model must be registered first:
```bash
python -m src.models.build_models train --register
```

### "No production model exists"

Promote a model first:
```bash
python -m src.models.build_models promote --version 1
```

## Next Steps

1. **Set up automated retraining**: Use Prefect/Airflow to run weekly
2. **Add experiment tracking**: Log A/B test results
3. **Implement model serving**: Add REST API with BentoML
4. **Set up alerting**: Integrate monitoring with Slack/PagerDuty
5. **Add more models**: Random Forest, Neural Networks, etc.

## References

- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- SHAP Documentation: https://shap.readthedocs.io/
- XGBoost Documentation: https://xgboost.readthedocs.io/
