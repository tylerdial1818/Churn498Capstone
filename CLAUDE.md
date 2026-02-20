# CLAUDE.md - Retain App Development Guide

## Project Overview

Retain is a customer lifecycle management platform for streaming services. It predicts subscriber churn using ML models trained on account, subscription, payment, support, and streaming engagement data stored in PostgreSQL.

**Python 3.11+** required. Uses modern syntax: `str | None`, `list[str]`, `match` statements.

## Architecture

```
src/
├── data/       # Database connections (SQLAlchemy), CSV loading, schema management
├── features/   # Feature engineering pipeline (transformers, validation, export)
├── models/     # ML training, evaluation, registry, scoring, monitoring
├── agents/     # LangGraph multi-agent AI workflows (DDP pipeline, intervention drafter)
└── app/        # Web application (FastAPI API + agent-backed routes)
```

Each module follows the same structure: `config.py` (dataclass configs) → core logic → `build_*.py` (CLI entry point) → `__init__.py` (public API via `__all__`).

## Quick Reference Commands

```bash
# Setup
make install-dev              # Install with dev dependencies
pip install -e ".[mlops]"     # MLflow/BentoML extras
pip install -e ".[agents]"    # LangChain/LangGraph extras

# Database
make db-init                  # Create tables
make db-load                  # Load CSVs into PostgreSQL
make db-check                 # Verify connection and row counts
make db-reset                 # Drop and recreate all tables

# Features
python -m src.features.build_features                    # Build features
python -m src.features.build_features --export-analytics # With analyst CSV

# Models
python -m src.models.build_models train --register       # Train + register
python -m src.models.build_models evaluate --version 1   # Evaluate model
python -m src.models.build_models promote --version 1    # Promote to production
python -m src.models.build_models score                  # Batch scoring
python -m src.models.build_models monitor --save         # Drift detection

# Agents
python -m src.agents.pipelines.ddp_pipeline                # Run full DDP pipeline
python -m src.agents.intervention.drafter --help           # Draft retention emails
python -m src.agents.intervention.drafter --account-id ACC_00000001 --churn-driver disengagement

# App
uvicorn src.app.main:app --reload              # Start API server (dev)
python -m src.agents.early_warning.detector     # Run early warning detection
python -m src.agents.analysis.analyzer --page executive_summary  # Generate dashboard narrative

# Testing & Quality
make test                     # pytest tests/ -v
make test-cov                 # With coverage report
make lint                     # ruff check + mypy
make format                   # black + isort + ruff fix
```

## Code Conventions

### Style & Formatting
- **Line length**: 88 (configured in black, ruff, isort)
- **Formatter**: black + isort (profile="black")
- **Linter**: ruff (rules: E, W, F, I, B, C4, UP; ignores E501)
- **Type checker**: mypy (ignore_missing_imports=true)
- **Import order**: stdlib → third-party → local (relative imports within modules)

### Naming
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`
- Test functions: `test_<what>_<scenario>`

### Patterns Used Throughout
- **Dataclass configs**: All configuration via `@dataclass` with typed fields and defaults. Mutable defaults use `field(default_factory=lambda: ...)`. Some use `frozen=True`.
- **Result types**: Operations return dataclasses with `.success`, `.errors` fields (e.g., `PipelineResult`, `TrainingResult`, `ScoringResult`).
- **Enums**: `ModelType`, `FeatureType`, `DriftMethod`, `CheckStatus`, `HealthStatus` for type-safe configuration.
- **Registry pattern**: `TRANSFORMER_REGISTRY` maps names to classes for dynamic instantiation.
- **Context objects**: `TransformerContext` bundles engine, date, windows, account_ids.
- **Module-level defaults**: `DEFAULT_MODEL_CONFIG = ModelConfig()` for common presets.

### Type Hints
- Modern 3.11+ syntax everywhere: `str | None`, `list[str]`, `dict[str, Any]`
- All function parameters and return types annotated
- All dataclass fields typed
- Uses `from typing import Any` but avoids `Optional`, `List`, `Dict`

### CLI Pattern
- `src/features/build_features.py` and `src/models/build_models.py` use **argparse** + rich output
- `src/data/database.py` and `src/data/load.py` use **click**
- All CLIs return integer exit codes
- Rich console for formatted tables, progress spinners, colored output

### Database Access
- SQLAlchemy engine via `create_engine()`, connections via `get_engine()` from `src.data.database`
- Parameterized queries: `text(query)` with `:param_name` placeholders
- Read pattern: `pd.read_sql(text(query), conn, params={...})`
- Write pattern: `df.to_sql(table, conn, if_exists="append", method="multi")`
- Context managers for all connections: `with engine.connect() as conn:`
- Environment config: `DATABASE_URL` env var or individual `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD`

### Logging
- `logger = logging.getLogger(__name__)` in each module
- RichHandler for console output: `logging.basicConfig(handlers=[RichHandler(...)])`
- Levels: DEBUG (verbose mode), INFO (default)

### Error Handling
- Accumulate errors in lists, don't raise immediately (see `PipelineResult.errors`)
- `try/except` around each transformer; failures logged, pipeline continues
- `RuntimeError` for critical failures in convenience functions
- CLI functions catch exceptions and print rich-formatted errors

## Database Schema

PostgreSQL with 6 tables. Load order respects FK constraints:

1. **accounts** (PK: account_id) - demographics, signup_date, country
2. **content_catalog** (PK: content_id) - genre, content_type, duration
3. **subscriptions** (PK: subscription_id, FK: account_id) - plan_type, status, cancel_reason
4. **payments** (PK: payment_id, FK: account_id) - amount, method, status
5. **support_tickets** (PK: ticket_id, FK: account_id) - category, priority, resolution
6. **streaming_events** (PK: event_id, FK: account_id, content_id) - watch duration, device, location

Schema file: `sql/schema.sql`

## Module Details

### src/data/
- `database.py`: `get_engine()`, `get_session()`, `get_connection()`, `init_database()`, `reset_database()`
- `load.py`: CSV loading with `LOAD_ORDER` for FK dependency ordering, chunked inserts (10K rows)

### src/features/
- `config.py`: `FeatureConfig`, `FeatureSpec`, `FeatureType` enum, `TIME_WINDOWS`, `FEATURE_SPECS` registry (23 features)
- `transformers.py`: `BaseTransformer` (ABC) with 7 concrete transformers (Engagement, Behavioral, Financial, Support, Demographic, Temporal, Target). All registered in `TRANSFORMER_REGISTRY`.
- `pipeline.py`: `FeaturePipeline` orchestrator, `PipelineResult` dataclass, `create_training_dataset()` convenience function
- `validation.py`: `FeatureValidator` checks missing values, ranges, infinities, constant features, cardinality, target leakage
- `export.py`: `export_analytics_csv()` writes stable file + timestamped snapshot to `data/processed/`

### src/models/
- `config.py`: `ModelConfig` (XGBoost/LightGBM/LogisticRegression params), `TrainingConfig` (MLflow), `EvaluationConfig` (quality gates), `ScoringConfig` (risk tiers), `MonitoringConfig` (PSI thresholds)
- `train.py`: `ModelTrainer` with two-model pattern (primary + baseline), 60/20/20 split, CV, SHAP, MLflow tracking
- `evaluate.py`: `EvaluationGate` with 6 checks (performance, baseline, production, calibration, distribution, feature stability)
- `registry.py`: `ModelRegistry` wrapping MLflow (register, promote stages, load, compare, history)
- `score.py`: `BatchScorer` for predictions, risk tiers (high >= 0.70, medium >= 0.40, low < 0.40), writes to `predictions` table
- `monitoring.py`: `ModelMonitor` with PSI drift detection (warning: 0.1, critical: 0.25), writes to `monitoring_log` table

### src/agents/
- `config.py`: `AgentConfig` dataclass (model, thresholds, context limits)
- `state.py`: `RetainAgentState` TypedDict with LangGraph annotations (`add_messages` reducer, `operator.add` for errors)
- `prompts.py`: System prompts for 7 roles: `SUPERVISOR_PROMPT`, `DETECTION_AGENT_PROMPT`, `DIAGNOSIS_AGENT_PROMPT`, `PRESCRIPTION_AGENT_PROMPT`, `INTERVENTION_DRAFTER_PROMPT`, `EARLY_WARNING_AGENT_PROMPT`, `ANALYSIS_AGENT_PROMPT`
- `tools.py`: 22 LangChain `@tool` wrappers (scoring, features, SHAP, DB queries, cohort clustering, monitoring, KPIs, risk transitions, engagement/support/payment trends, cohort analysis). Lazy imports inside function bodies to avoid import-time failures. `ALL_TOOLS`, `DETECTION_TOOLS`, `DIAGNOSIS_TOOLS`, `PRESCRIPTION_TOOLS`, `EARLY_WARNING_TOOLS`, `ANALYSIS_TOOLS` registries.
- `utils.py`: `format_dataframe_as_markdown()`, `safe_json_serialize()`, `validate_account_id()`, `truncate_for_context()`
- `pipelines/ddp_pipeline.py`: LangGraph `StateGraph` with 5 nodes (supervisor, detect, diagnose, prescribe, review). Supervisor routes via `match` statement. Human-in-the-loop via `interrupt_before=["review"]`.
- `pipelines/ddp_nodes.py`: Node implementations with `_run_agent_loop()` inner tool-calling loop, `_extract_json_from_response()` JSON parser (raw, code fence, brace extraction).
- `intervention/strategies.py`: 5 `InterventionStrategy` definitions in `STRATEGY_REGISTRY`, deterministic `select_strategy()` with business rules (no discounts < 30 days tenure, payment_failed always gets payment_recovery).
- `intervention/drafter.py`: Standalone LangGraph agent with `DrafterState` TypedDict, 4 nodes (diagnose_driver, select_strategy, draft_email, human_review). `draft_intervention()` convenience function.
- `intervention/email_renderer.py`: `render_as_markdown()`, `render_as_html()` (inline styles only), `render_as_plaintext()`, `render_comparison()` for A/B variants.
- `early_warning/detector.py`: LangGraph `StateGraph` with 3 nodes (score_comparison, investigate_escalations, group_and_report). `score_comparison` and `group_and_report` are deterministic; `investigate_escalations` uses LLM. `run_early_warning()` convenience function.
- `early_warning/alerts.py`: `RiskTransition`, `AlertGroup`, `EarlyWarningReport` dataclasses. `classify_transitions()`, `compute_alert_priority()`, `format_alert_report_markdown()` — all deterministic.
- `analysis/analyzer.py`: LangGraph `StateGraph` with 2 nodes (gather_data, generate_narrative). `gather_data` is deterministic (calls tools directly per page_context); `generate_narrative` uses LLM. `run_analysis()` convenience function with `PageContext` enum.
- `analysis/narratives.py`: `PageContext` enum, `KPIDefinition` with `KPI_REGISTRY`, `NarrativeSection`, `AnalysisNarrative` dataclasses. `assess_kpi_health()`, `compute_overall_sentiment()` — all deterministic.

### src/app/
- `main.py`: FastAPI app factory with CORS. `create_app()` includes all routers under `/api`.
- `dependencies.py`: `get_agent_config()` (cached), `get_db_engine()` for FastAPI dependency injection.
- `agent_schemas.py`: Pydantic v2 response models: `DashboardResponse`, `AtRiskResponse`, `EarlyWarningAlertResponse`, `AnalyticsResponse`, `AnalysisNarrativeResponse`, `PrescriptionResponse`, `InterventionDraftResponse`, `ExportResponse`.
- `routes/dashboard.py`: `GET /api/dashboard` — Analysis Agent in executive_summary mode.
- `routes/at_risk.py`: `GET /api/at-risk` — high-risk accounts + Early Warning Agent, `GET /api/at-risk/{account_id}` — single account detail.
- `routes/analytics.py`: `GET /api/analytics` — Analysis Agent in analytics_deep_dive mode, returns raw data for charts + narrative.
- `routes/prescriptions.py`: `GET /api/prescriptions` — deterministic strategy matching + Analysis Agent narrative.
- `routes/interventions.py`: `POST /api/interventions/draft` — Intervention Drafter, `POST /api/interventions/export` — render + integration payloads (HubSpot, Salesforce, Marketo, Braze, email), `GET /api/interventions/integrations` — supported integrations list.

## Testing

- Framework: **pytest** with `-v --tb=short`
- Test files: `tests/test_data.py`, `tests/test_features.py`, `tests/test_models.py`, `tests/test_agents_foundation.py`, `tests/test_ddp_pipeline.py`, `tests/test_intervention_drafter.py`, `tests/test_early_warning.py`, `tests/test_analysis_agent.py`, `tests/test_app_agents.py`
- Integration tests: `scripts/test_integration.py`
- Fixtures use `seed=42` for reproducibility, `make_classification()` for synthetic ML data
- Mocking: `unittest.mock` (MagicMock, patch, patch.dict) for DB and LLM dependencies. Patch at source module (e.g., `src.data.database.get_engine`, `src.models.score.BatchScorer`), not at consumer module.
- Test classes group related tests: `TestFeatureConfig`, `TestFeatureValidator`, `TestEdgeCases`
- Coverage config: source=`["src"]`, omits tests and `__init__.py`

## Key Dependencies

| Category | Packages |
|----------|----------|
| Database | psycopg2-binary, sqlalchemy |
| Data | pandas, numpy, pyarrow |
| ML | scikit-learn, xgboost, lightgbm |
| CLI/UI | click (data module), argparse + rich (features/models) |
| Env | python-dotenv |
| Agents | langchain, langchain-anthropic, langgraph (`[agents]`) |
| Optional | mlflow, bentoml (`[mlops]`), shap (install separately) |
| Dev | pytest, ruff, black, isort, mypy |

## Environment Setup

Database connection reads from `.env`:
```
DATABASE_URL=postgresql://postgres:password@localhost:5432/retain_dev
```

MLflow tracking defaults to `./mlruns` (local directory). Override with `MLFLOW_TRACKING_URI`.

## Files to Never Edit

- `data/raw/*.csv` - source data
- `.venv/` - virtual environment
- `retain.egg-info/` - build artifacts
- `mlruns/` - MLflow tracking data (managed by MLflow)
