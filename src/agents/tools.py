"""
LangChain tool wrappers around the Retain ML backend.

Each tool accepts only simple types, catches all exceptions, and returns
string results. Tools call get_engine() inside the body — never at import time.
"""

import json
import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from .config import AgentConfig
from .utils import (
    format_dataframe_as_markdown,
    safe_json_serialize,
    truncate_for_context,
    validate_account_id,
)

logger = logging.getLogger("retain.agents.tools")

# Default config used by tools when not overridden
_DEFAULT_CONFIG = AgentConfig()


# =============================================================================
# Scoring Tools
# =============================================================================


@tool
def get_account_risk_scores(
    account_ids: list[str] | None = None,
    risk_tier: str | None = None,
) -> str:
    """Get latest churn risk scores for accounts.

    Retrieves the most recent churn probability and risk tier for each account.
    Can filter by specific account IDs or by risk tier (high/medium/low).

    Args:
        account_ids: Optional list of account IDs to look up. If None, returns all.
        risk_tier: Optional filter — one of 'high', 'medium', 'low'.

    Returns:
        Markdown table of risk scores, or error message.
    """
    try:
        from src.data.database import get_engine
        from src.models.config import ScoringConfig
        from src.models.score import BatchScorer

        engine = get_engine()
        scorer = BatchScorer(engine, ScoringConfig())
        df = scorer.get_latest_scores(
            account_ids=account_ids, risk_tier=risk_tier
        )

        if df.empty:
            return "No risk scores found for the specified criteria."

        return truncate_for_context(format_dataframe_as_markdown(df))

    except Exception as e:
        logger.error(f"get_account_risk_scores failed: {e}")
        return f"Error retrieving risk scores: {e}"


@tool
def get_account_scoring_history(account_id: str) -> str:
    """Get historical churn risk scores for a single account.

    Shows how an account's churn probability has changed over time,
    useful for identifying risk trends (increasing, stable, declining).

    Args:
        account_id: Account ID (format: ACC_XXXXXXXX).

    Returns:
        Markdown table of scoring history, or error message.
    """
    try:
        if not validate_account_id(account_id):
            return f"Invalid account ID format: {account_id}. Expected ACC_XXXXXXXX."

        from src.data.database import get_engine
        from src.models.config import ScoringConfig
        from src.models.score import BatchScorer

        engine = get_engine()
        scorer = BatchScorer(engine, ScoringConfig())
        df = scorer.get_scoring_history(account_id)

        if df.empty:
            return f"No scoring history found for {account_id}."

        return format_dataframe_as_markdown(df)

    except Exception as e:
        logger.error(f"get_account_scoring_history failed: {e}")
        return f"Error retrieving scoring history: {e}"


@tool
def run_batch_scoring(reference_date: str = "2025-12-01") -> str:
    """Score all active accounts with the production churn model.

    Runs the feature pipeline and batch scorer to generate churn probabilities
    for every active account. Returns summary statistics.

    Args:
        reference_date: Reference date for feature generation (YYYY-MM-DD).

    Returns:
        JSON summary with total scored, risk distribution, and mean probability.
    """
    try:
        from src.data.database import get_engine
        from src.features import FeatureConfig, FeaturePipeline
        from src.models.config import ScoringConfig
        from src.models.score import BatchScorer

        engine = get_engine()

        # Generate features
        logger.info(f"Running feature pipeline with reference_date={reference_date}")
        feature_config = FeatureConfig(reference_date=reference_date)
        pipeline = FeaturePipeline(feature_config, engine)
        feature_result = pipeline.run(include_target=False)

        if not feature_result.success:
            return f"Feature pipeline failed: {feature_result.errors}"

        # Score
        scorer = BatchScorer(engine, ScoringConfig())
        scoring_result = scorer.score(feature_result.features)

        summary = scoring_result.get_summary()
        return json.dumps(safe_json_serialize(summary), indent=2)

    except Exception as e:
        logger.error(f"run_batch_scoring failed: {e}")
        return f"Error running batch scoring: {e}"


# =============================================================================
# Feature & Explainability Tools
# =============================================================================


@tool
def get_account_features(account_ids: list[str]) -> str:
    """Get the full feature vector for specified accounts.

    Returns all engineered features (engagement, behavioral, financial,
    support, demographic, temporal) for the given accounts.

    Args:
        account_ids: List of account IDs to retrieve features for.

    Returns:
        Markdown table of feature values, or error message.
    """
    try:
        from src.data.database import get_engine
        from src.features import FeatureConfig, create_inference_features

        engine = get_engine()
        config = FeatureConfig(reference_date=_DEFAULT_CONFIG.reference_date)
        features = create_inference_features(engine, account_ids, config)

        if features.empty:
            return f"No features found for accounts: {account_ids}"

        return truncate_for_context(format_dataframe_as_markdown(features))

    except Exception as e:
        logger.error(f"get_account_features failed: {e}")
        return f"Error retrieving features: {e}"


@tool
def explain_account_prediction(account_id: str) -> str:
    """Explain why an account has its current churn risk score using SHAP.

    Uses SHAP TreeExplainer for the production model to show the top
    features pushing the account toward or away from churn. Falls back
    to global feature importances if SHAP computation fails.

    Args:
        account_id: Account ID (format: ACC_XXXXXXXX).

    Returns:
        Text explanation with top positive and negative risk contributors.
    """
    try:
        if not validate_account_id(account_id):
            return f"Invalid account ID format: {account_id}. Expected ACC_XXXXXXXX."

        from src.data.database import get_engine
        from src.features import FeatureConfig, create_inference_features
        from src.models.registry import ModelRegistry

        engine = get_engine()
        registry = ModelRegistry()

        # Load production model
        try:
            model = registry.load_production_model()
        except Exception as e:
            return f"No production model available: {e}"

        # Get features for this account
        config = FeatureConfig(reference_date=_DEFAULT_CONFIG.reference_date)
        features = create_inference_features(engine, [account_id], config)

        if features.empty:
            return f"No features found for account {account_id}."

        # Try SHAP explanation
        try:
            import shap

            # Get the underlying model from MLflow wrapper
            raw_model = model
            if hasattr(model, "_model_impl"):
                raw_model = model._model_impl
            if hasattr(raw_model, "python_model"):
                raw_model = raw_model.python_model
            if hasattr(raw_model, "model"):
                raw_model = raw_model.model

            explainer = shap.TreeExplainer(raw_model)
            shap_values = explainer.shap_values(features)

            # Handle binary classification output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Get feature names and SHAP values for this account
            feature_names = features.columns.tolist()
            account_shap = shap_values[0] if len(shap_values.shape) > 1 else shap_values

            # Create sorted feature-SHAP pairs
            feature_shap = sorted(
                zip(feature_names, account_shap),
                key=lambda x: abs(x[1]),
                reverse=True,
            )

            # Top positive (pushing toward churn) and negative (protecting)
            positive = [(f, v) for f, v in feature_shap if v > 0][:5]
            negative = [(f, v) for f, v in feature_shap if v < 0][:5]

            lines = [f"## SHAP Explanation for {account_id}\n"]
            lines.append("### Top Risk Factors (pushing toward churn):")
            for feature, value in positive:
                feat_val = features[feature].iloc[0]
                lines.append(f"- **{feature}** = {feat_val:.4f} (SHAP: +{value:.4f})")

            lines.append("\n### Protective Factors (reducing churn risk):")
            for feature, value in negative:
                feat_val = features[feature].iloc[0]
                lines.append(f"- **{feature}** = {feat_val:.4f} (SHAP: {value:.4f})")

            return "\n".join(lines)

        except Exception as shap_error:
            logger.warning(f"SHAP failed for {account_id}, using global importances: {shap_error}")

            # Fallback to global feature importances
            try:
                prod_metrics = registry.get_production_metrics()
                run_id = None
                prod_versions = registry.client.get_latest_versions(
                    registry.model_name, stages=["Production"]
                )
                if prod_versions:
                    run_id = prod_versions[0].run_id

                if run_id:
                    from mlflow.tracking import MlflowClient
                    client = MlflowClient()
                    run = client.get_run(run_id)
                    # Extract feature importance params
                    params = run.data.params
                    importance_params = {
                        k: v for k, v in params.items() if k.startswith("top_feature_")
                    }

                    lines = [
                        f"## Feature Importance for {account_id} (global fallback)\n",
                        "**Note:** SHAP computation failed. Showing global feature importances instead.\n",
                    ]
                    for key in sorted(importance_params.keys()):
                        lines.append(f"- {importance_params[key]}")

                    return "\n".join(lines)
            except Exception:
                pass

            return (
                f"SHAP explanation failed for {account_id}: {shap_error}. "
                "Global feature importance fallback also unavailable."
            )

    except Exception as e:
        logger.error(f"explain_account_prediction failed: {e}")
        return f"Error explaining prediction for {account_id}: {e}"


@tool
def get_feature_importance_global() -> str:
    """Get the global feature importance ranking from the production model.

    Shows which features have the most influence on churn predictions
    across all accounts, ranked by importance score.

    Returns:
        Ranked list of features with importance scores.
    """
    try:
        from src.models.registry import ModelRegistry

        registry = ModelRegistry()

        # Try to get importances from the MLflow run
        prod_versions = registry.client.get_latest_versions(
            registry.model_name, stages=["Production"]
        )
        if not prod_versions:
            return "No production model found in registry."

        run_id = prod_versions[0].run_id
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        run = client.get_run(run_id)

        params = run.data.params
        importance_params = {
            k: v for k, v in params.items() if k.startswith("top_feature_")
        }

        if not importance_params:
            return "No feature importance data found for production model."

        lines = ["## Global Feature Importance (Production Model)\n"]
        for key in sorted(importance_params.keys()):
            lines.append(f"- {importance_params[key]}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"get_feature_importance_global failed: {e}")
        return f"Error retrieving feature importances: {e}"


# =============================================================================
# Database Query Tools
# =============================================================================


@tool
def get_account_profile(account_id: str) -> str:
    """Get the full profile for an account — demographics, subscription, plan, status.

    Joins accounts and subscriptions tables to provide a complete picture
    of who this customer is and their current subscription state.

    Args:
        account_id: Account ID (format: ACC_XXXXXXXX).

    Returns:
        Formatted profile summary or error message.
    """
    try:
        if not validate_account_id(account_id):
            return f"Invalid account ID format: {account_id}. Expected ACC_XXXXXXXX."

        from src.data.database import get_engine

        engine = get_engine()

        query = """
        SELECT
            a.account_id, a.email, a.signup_date, a.country, a.age, a.gender,
            s.plan_type, s.start_date, s.end_date, s.status,
            s.cancel_reason, s.previous_plan, s.watchlist_size,
            s.app_rating, s.parental_control_enabled
        FROM accounts a
        LEFT JOIN subscriptions s ON a.account_id = s.account_id
        WHERE a.account_id = :account_id
        ORDER BY s.start_date DESC
        LIMIT 1
        """

        from sqlalchemy import text

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"account_id": account_id})

        if df.empty:
            return f"Account {account_id} not found."

        row = df.iloc[0]
        lines = [f"## Account Profile: {account_id}\n"]
        for col in df.columns:
            val = row[col]
            if pd.notna(val):
                lines.append(f"- **{col}**: {val}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"get_account_profile failed: {e}")
        return f"Error retrieving profile for {account_id}: {e}"


@tool
def get_account_support_history(account_id: str) -> str:
    """Get support ticket history for an account.

    Shows all support tickets with dates, categories, priorities,
    and resolution status. Useful for identifying support frustration.

    Args:
        account_id: Account ID (format: ACC_XXXXXXXX).

    Returns:
        Markdown table of support tickets or error message.
    """
    try:
        if not validate_account_id(account_id):
            return f"Invalid account ID format: {account_id}. Expected ACC_XXXXXXXX."

        from src.data.database import get_engine

        engine = get_engine()

        query = """
        SELECT ticket_id, created_at, category, priority, resolved_at
        FROM support_tickets
        WHERE account_id = :account_id
        ORDER BY created_at DESC
        """

        from sqlalchemy import text

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"account_id": account_id})

        if df.empty:
            return f"No support tickets found for {account_id}."

        return format_dataframe_as_markdown(df)

    except Exception as e:
        logger.error(f"get_account_support_history failed: {e}")
        return f"Error retrieving support history for {account_id}: {e}"


@tool
def get_account_payment_history(account_id: str) -> str:
    """Get payment transaction history for an account.

    Shows all payments with dates, amounts, methods, status, and
    failure reasons. Useful for identifying payment issues.

    Args:
        account_id: Account ID (format: ACC_XXXXXXXX).

    Returns:
        Markdown table of payments or error message.
    """
    try:
        if not validate_account_id(account_id):
            return f"Invalid account ID format: {account_id}. Expected ACC_XXXXXXXX."

        from src.data.database import get_engine

        engine = get_engine()

        query = """
        SELECT payment_id, payment_date, amount, currency,
               payment_method, status, failure_reason
        FROM payments
        WHERE account_id = :account_id
        ORDER BY payment_date DESC
        """

        from sqlalchemy import text

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"account_id": account_id})

        if df.empty:
            return f"No payment history found for {account_id}."

        return format_dataframe_as_markdown(df)

    except Exception as e:
        logger.error(f"get_account_payment_history failed: {e}")
        return f"Error retrieving payment history for {account_id}: {e}"


@tool
def get_account_viewing_summary(account_id: str) -> str:
    """Get streaming behavior summary for an account.

    Shows total watch hours, favorite genres, device usage breakdown,
    and last active date. Useful for identifying disengagement.

    Args:
        account_id: Account ID (format: ACC_XXXXXXXX).

    Returns:
        Formatted viewing summary or error message.
    """
    try:
        if not validate_account_id(account_id):
            return f"Invalid account ID format: {account_id}. Expected ACC_XXXXXXXX."

        from src.data.database import get_engine

        engine = get_engine()

        from sqlalchemy import text

        # Total watch hours and last active
        summary_query = """
        SELECT
            COUNT(*) as total_events,
            COALESCE(SUM(watch_duration_minutes) / 60.0, 0) as total_watch_hours,
            MAX(event_timestamp) as last_active,
            COUNT(DISTINCT device_type) as device_count
        FROM streaming_events
        WHERE account_id = :account_id
        """

        # Genre breakdown
        genre_query = """
        SELECT c.genre, COUNT(*) as watch_count,
               SUM(se.watch_duration_minutes) / 60.0 as hours
        FROM streaming_events se
        JOIN content_catalog c ON se.content_id = c.content_id
        WHERE se.account_id = :account_id
        GROUP BY c.genre
        ORDER BY hours DESC
        LIMIT 5
        """

        # Device breakdown
        device_query = """
        SELECT device_type, COUNT(*) as event_count
        FROM streaming_events
        WHERE account_id = :account_id
        GROUP BY device_type
        ORDER BY event_count DESC
        """

        with engine.connect() as conn:
            params = {"account_id": account_id}
            summary_df = pd.read_sql(text(summary_query), conn, params=params)
            genre_df = pd.read_sql(text(genre_query), conn, params=params)
            device_df = pd.read_sql(text(device_query), conn, params=params)

        if summary_df.iloc[0]["total_events"] == 0:
            return f"No streaming activity found for {account_id}."

        row = summary_df.iloc[0]
        lines = [f"## Viewing Summary: {account_id}\n"]
        lines.append(f"- **Total watch hours**: {row['total_watch_hours']:.1f}")
        lines.append(f"- **Total events**: {int(row['total_events'])}")
        lines.append(f"- **Last active**: {row['last_active']}")
        lines.append(f"- **Devices used**: {int(row['device_count'])}")

        if not genre_df.empty:
            lines.append("\n### Favorite Genres:")
            for _, g in genre_df.iterrows():
                lines.append(f"- {g['genre']}: {g['hours']:.1f} hours ({int(g['watch_count'])} views)")

        if not device_df.empty:
            lines.append("\n### Device Usage:")
            for _, d in device_df.iterrows():
                lines.append(f"- {d['device_type']}: {int(d['event_count'])} events")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"get_account_viewing_summary failed: {e}")
        return f"Error retrieving viewing summary for {account_id}: {e}"


@tool
def query_database_readonly(sql_query: str) -> str:
    """Execute a read-only SQL query against the Retain database.

    Only SELECT statements are allowed. INSERT, UPDATE, DELETE, DROP,
    ALTER, TRUNCATE, and other mutation statements are blocked.
    Results are limited to prevent context overflow.

    Args:
        sql_query: SQL SELECT query to execute.

    Returns:
        Markdown table of results, or error message.
    """
    try:
        # Validate: only allow SELECT statements
        normalized = sql_query.strip().upper()

        blocked_keywords = [
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
            "TRUNCATE", "CREATE", "GRANT", "REVOKE", "EXEC",
        ]

        for keyword in blocked_keywords:
            # Check for keyword at statement boundaries (start of statement or after semicolon)
            pattern = rf"(^|;\s*){keyword}\b"
            if re.search(pattern, normalized):
                return f"Error: {keyword} statements are not allowed. Only SELECT queries permitted."

        if not normalized.lstrip("(").startswith("SELECT") and not normalized.startswith("WITH"):
            return "Error: Only SELECT queries (including WITH/CTE) are allowed."

        # Add LIMIT if missing
        max_rows = _DEFAULT_CONFIG.max_sql_rows
        if "LIMIT" not in normalized:
            sql_query = sql_query.rstrip().rstrip(";") + f" LIMIT {max_rows}"

        from src.data.database import get_engine

        engine = get_engine()

        from sqlalchemy import text

        with engine.connect() as conn:
            df = pd.read_sql(text(sql_query), conn)

        if df.empty:
            return "Query returned no results."

        return truncate_for_context(format_dataframe_as_markdown(df))

    except Exception as e:
        logger.error(f"query_database_readonly failed: {e}")
        return f"SQL query error: {e}"


# =============================================================================
# Cohort Analysis Tools
# =============================================================================


@tool
def segment_high_risk_accounts(
    reference_date: str = "2025-12-01",
    min_cohort_size: int = 10,
) -> str:
    """Segment high-risk accounts into cohorts using k-means clustering.

    Identifies high-risk accounts, clusters them by feature similarity,
    and reports distinguishing characteristics of each cohort versus
    the overall population.

    Args:
        reference_date: Reference date for feature generation.
        min_cohort_size: Minimum accounts per cohort.

    Returns:
        JSON with cohort details or error message.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        from src.data.database import get_engine
        from src.features import FeatureConfig, FeaturePipeline
        from src.models.config import ScoringConfig
        from src.models.score import BatchScorer

        engine = get_engine()

        # Generate features
        feature_config = FeatureConfig(reference_date=reference_date)
        pipeline = FeaturePipeline(feature_config, engine)
        result = pipeline.run(include_target=False)

        if not result.success:
            return f"Feature pipeline failed: {result.errors}"

        features = result.features

        # Score accounts
        scorer = BatchScorer(engine, ScoringConfig())
        scoring_result = scorer.score(features)

        # Filter to high-risk accounts
        high_risk = scoring_result.predictions[
            scoring_result.predictions["risk_tier"] == "high"
        ]

        if len(high_risk) < min_cohort_size:
            return (
                f"Only {len(high_risk)} high-risk accounts found, "
                f"below minimum cohort size of {min_cohort_size}."
            )

        # Get features for high-risk accounts
        high_risk_ids = high_risk["account_id"].tolist()
        hr_features = features.loc[
            features.index.isin(high_risk_ids)
        ].copy()

        # Handle missing values for clustering
        hr_features = hr_features.fillna(0)

        # Determine k
        max_cohorts = _DEFAULT_CONFIG.max_cohorts
        k = min(max_cohorts, len(hr_features) // min_cohort_size)
        k = max(k, 2)  # At least 2 clusters

        # Standardize and cluster
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(hr_features)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)

        hr_features["cluster"] = clusters
        hr_features["churn_probability"] = high_risk.set_index("account_id")[
            "churn_probability"
        ]

        # Population statistics for comparison
        pop_means = features.fillna(0).mean()
        pop_stds = features.fillna(0).std()

        # Build cohort summaries
        cohorts = []
        for cluster_id in range(k):
            cluster_mask = hr_features["cluster"] == cluster_id
            cluster_data = hr_features[cluster_mask]

            cluster_means = cluster_data.drop(
                columns=["cluster", "churn_probability"]
            ).mean()

            # Z-scores vs population
            z_scores = (cluster_means - pop_means) / (pop_stds + 1e-10)
            top_features = z_scores.abs().nlargest(
                _DEFAULT_CONFIG.top_features_count
            )

            distinguishing = {}
            for feat in top_features.index:
                distinguishing[feat] = {
                    "cohort_mean": round(float(cluster_means[feat]), 4),
                    "population_mean": round(float(pop_means[feat]), 4),
                    "z_score": round(float(z_scores[feat]), 4),
                }

            sample_ids = cluster_data.index.tolist()[:5]

            cohorts.append({
                "cohort_id": int(cluster_id),
                "size": int(cluster_mask.sum()),
                "mean_churn_prob": round(
                    float(cluster_data["churn_probability"].mean()), 4
                ),
                "distinguishing_features": distinguishing,
                "sample_account_ids": sample_ids,
            })

        # Find highest-risk cohort
        highest_risk = max(cohorts, key=lambda c: c["mean_churn_prob"])

        result_dict = {
            "total_high_risk": len(high_risk),
            "n_cohorts": k,
            "cohorts": cohorts,
            "highest_risk_cohort_id": highest_risk["cohort_id"],
        }

        return json.dumps(safe_json_serialize(result_dict), indent=2)

    except Exception as e:
        logger.error(f"segment_high_risk_accounts failed: {e}")
        return f"Error segmenting accounts: {e}"


# =============================================================================
# Monitoring Tools
# =============================================================================


@tool
def get_model_health_status() -> str:
    """Get the current production model health status.

    Checks for data drift, prediction drift, and performance degradation.
    Returns health status (healthy/warning/critical), drifted features,
    and recommended actions.

    Returns:
        Formatted health report or error message.
    """
    try:
        from src.data.database import get_engine
        from src.features import FeatureConfig, FeaturePipeline
        from src.models.config import MonitoringConfig
        from src.models.monitoring import ModelMonitor

        engine = get_engine()
        config = MonitoringConfig()
        monitor = ModelMonitor(engine, config)

        # Generate current features
        feature_config = FeatureConfig(
            reference_date=_DEFAULT_CONFIG.reference_date
        )
        pipeline = FeaturePipeline(feature_config, engine)
        current_result = pipeline.run(include_target=False)

        if not current_result.success:
            return f"Feature pipeline failed: {current_result.errors}"

        # Generate reference features (30 days prior)
        ref_date_str = _DEFAULT_CONFIG.reference_date
        from datetime import datetime, timedelta

        ref_date = datetime.strptime(ref_date_str, "%Y-%m-%d")
        ref_date_30d = (ref_date - timedelta(days=30)).strftime("%Y-%m-%d")

        ref_config = FeatureConfig(reference_date=ref_date_30d)
        ref_pipeline = FeaturePipeline(ref_config, engine)
        ref_result = ref_pipeline.run(include_target=False)

        if not ref_result.success:
            return f"Reference feature pipeline failed: {ref_result.errors}"

        # Check drift
        report = monitor.check_drift(
            current_result.features, ref_result.features
        )

        return report.summary()

    except Exception as e:
        logger.error(f"get_model_health_status failed: {e}")
        return f"Error checking model health: {e}"


# =============================================================================
# Tool Registry
# =============================================================================

# All tools for easy access by agent nodes
ALL_TOOLS = [
    get_account_risk_scores,
    get_account_scoring_history,
    run_batch_scoring,
    get_account_features,
    explain_account_prediction,
    get_feature_importance_global,
    get_account_profile,
    get_account_support_history,
    get_account_payment_history,
    get_account_viewing_summary,
    query_database_readonly,
    segment_high_risk_accounts,
    get_model_health_status,
]

# Tool subsets for each agent role
DETECTION_TOOLS = [
    run_batch_scoring,
    get_account_risk_scores,
    segment_high_risk_accounts,
    query_database_readonly,
]

DIAGNOSIS_TOOLS = [
    explain_account_prediction,
    get_account_profile,
    get_account_support_history,
    get_account_payment_history,
    get_account_viewing_summary,
    get_account_features,
    query_database_readonly,
]

PRESCRIPTION_TOOLS = [
    query_database_readonly,
    get_account_profile,
    get_account_viewing_summary,
]


# =============================================================================
# Risk Transition Tools (for Early Warning Agent)
# =============================================================================


@tool
def get_risk_tier_transitions(
    current_date: str = "2025-12-01",
    previous_date: str = "2025-11-01",
) -> str:
    """Compare risk tiers between two scoring runs and identify accounts that changed tiers.

    Returns a summary of transitions: how many moved low→medium, medium→high,
    low→high, etc. Also returns the list of account IDs for each transition type.
    Critical for the Early Warning Agent to detect deteriorating accounts.

    Args:
        current_date: Reference date for the current scoring run (YYYY-MM-DD).
        previous_date: Reference date for the previous scoring run (YYYY-MM-DD).

    Returns:
        JSON with transition summary and account ID lists, or error message.
    """
    try:
        from src.data.database import get_engine
        from src.features import FeatureConfig, FeaturePipeline
        from src.models.config import ScoringConfig
        from src.models.score import BatchScorer

        engine = get_engine()
        scoring_config = ScoringConfig()

        def _score_for_date(ref_date: str) -> pd.DataFrame:
            feature_config = FeatureConfig(reference_date=ref_date)
            pipeline = FeaturePipeline(feature_config, engine)
            result = pipeline.run(include_target=False)
            if not result.success:
                raise RuntimeError(
                    f"Feature pipeline failed for {ref_date}: {result.errors}"
                )
            scorer = BatchScorer(engine, scoring_config)
            scoring_result = scorer.score(result.features)
            return scoring_result.predictions

        logger.info(
            f"Computing risk tier transitions: {previous_date} → {current_date}"
        )

        current_preds = _score_for_date(current_date)
        previous_preds = _score_for_date(previous_date)

        # Join on account_id
        merged = current_preds.merge(
            previous_preds,
            on="account_id",
            suffixes=("_current", "_previous"),
        )

        total_scored = len(merged)

        # Compute transitions
        transitions = []
        for _, row in merged.iterrows():
            prev_tier = row["risk_tier_previous"]
            curr_tier = row["risk_tier_current"]
            prev_prob = row["churn_probability_previous"]
            curr_prob = row["churn_probability_current"]

            tier_order = {"low": 0, "medium": 1, "high": 2}
            prev_rank = tier_order.get(prev_tier, 0)
            curr_rank = tier_order.get(curr_tier, 0)

            if curr_rank > prev_rank:
                direction = "escalated"
            elif curr_rank < prev_rank:
                direction = "improved"
            else:
                direction = "stable"

            transitions.append({
                "account_id": row["account_id"],
                "previous_tier": prev_tier,
                "current_tier": curr_tier,
                "previous_probability": round(float(prev_prob), 4),
                "current_probability": round(float(curr_prob), 4),
                "probability_delta": round(
                    float(curr_prob) - float(prev_prob), 4
                ),
                "direction": direction,
            })

        # Group by transition type
        groups: dict[str, list[dict]] = {}
        for t in transitions:
            key = f"{t['previous_tier']}→{t['current_tier']}"
            groups.setdefault(key, []).append(t)

        # Build summary
        summary = {
            "total_scored": total_scored,
            "current_date": current_date,
            "previous_date": previous_date,
            "transition_counts": {
                k: len(v) for k, v in groups.items()
            },
            "escalated_accounts": [
                t for t in transitions if t["direction"] == "escalated"
            ],
            "improved_accounts": [
                t for t in transitions if t["direction"] == "improved"
            ],
            "stable_count": sum(
                1 for t in transitions if t["direction"] == "stable"
            ),
        }

        return json.dumps(safe_json_serialize(summary), indent=2)

    except Exception as e:
        logger.error(f"get_risk_tier_transitions failed: {e}")
        return f"Error computing risk tier transitions: {e}"


@tool
def get_risk_velocity(
    account_ids: list[str] | None = None,
    lookback_scores: int = 3,
) -> str:
    """Calculate how fast accounts are moving toward or away from churn.

    For each account, computes the change in churn_probability across recent
    scoring runs. Returns accounts sorted by velocity (fastest-deteriorating
    first). Positive velocity = worsening, negative = improving.

    Args:
        account_ids: Optional list of account IDs. If None, uses high-risk accounts.
        lookback_scores: Number of recent scores to consider for velocity.

    Returns:
        JSON with velocity data sorted by fastest-deteriorating, or error message.
    """
    try:
        from src.data.database import get_engine
        from src.models.config import ScoringConfig
        from src.models.score import BatchScorer

        engine = get_engine()
        scorer = BatchScorer(engine, ScoringConfig())

        # If no IDs given, get high-risk accounts
        if not account_ids:
            high_risk_df = scorer.get_latest_scores(risk_tier="high")
            if high_risk_df.empty:
                return "No high-risk accounts found."
            account_ids = high_risk_df["account_id"].tolist()[:50]

        velocities = []
        for aid in account_ids:
            history_df = scorer.get_scoring_history(aid)
            if history_df.empty or len(history_df) < 2:
                continue

            # Take the most recent N scores
            recent = history_df.head(lookback_scores)
            probs = recent["churn_probability"].tolist()

            # Velocity = last - first (positive = worsening)
            velocity = float(probs[0]) - float(probs[-1])
            velocities.append({
                "account_id": aid,
                "current_probability": round(float(probs[0]), 4),
                "oldest_probability": round(float(probs[-1]), 4),
                "velocity": round(velocity, 4),
                "scores_used": len(recent),
                "direction": (
                    "worsening" if velocity > 0.01
                    else "improving" if velocity < -0.01
                    else "stable"
                ),
            })

        # Sort by velocity descending (fastest-deteriorating first)
        velocities.sort(key=lambda x: x["velocity"], reverse=True)

        result = {
            "total_accounts": len(velocities),
            "accounts": velocities,
            "avg_velocity": round(
                sum(v["velocity"] for v in velocities) / max(len(velocities), 1),
                4,
            ),
        }

        return json.dumps(safe_json_serialize(result), indent=2)

    except Exception as e:
        logger.error(f"get_risk_velocity failed: {e}")
        return f"Error computing risk velocity: {e}"


@tool
def get_new_high_risk_accounts(
    current_date: str = "2025-12-01",
    previous_date: str = "2025-11-01",
) -> str:
    """Identify accounts that were NOT high-risk in the previous run but ARE high-risk now.

    This is the core 'alert' set for the Early Warning Agent.
    Returns account IDs, current probability, previous probability, and
    the magnitude of change.

    Args:
        current_date: Reference date for the current scoring run (YYYY-MM-DD).
        previous_date: Reference date for the previous scoring run (YYYY-MM-DD).

    Returns:
        JSON with newly high-risk accounts and probability deltas, or error message.
    """
    try:
        from src.data.database import get_engine
        from src.features import FeatureConfig, FeaturePipeline
        from src.models.config import ScoringConfig
        from src.models.score import BatchScorer

        engine = get_engine()
        scoring_config = ScoringConfig()

        def _score_for_date(ref_date: str) -> pd.DataFrame:
            feature_config = FeatureConfig(reference_date=ref_date)
            pipeline = FeaturePipeline(feature_config, engine)
            result = pipeline.run(include_target=False)
            if not result.success:
                raise RuntimeError(
                    f"Feature pipeline failed for {ref_date}: {result.errors}"
                )
            scorer = BatchScorer(engine, scoring_config)
            scoring_result = scorer.score(result.features)
            return scoring_result.predictions

        logger.info(
            f"Finding new high-risk accounts: {previous_date} → {current_date}"
        )

        current_preds = _score_for_date(current_date)
        previous_preds = _score_for_date(previous_date)

        # Current high-risk
        current_high = set(
            current_preds[current_preds["risk_tier"] == "high"]["account_id"]
        )
        # Previous high-risk
        previous_high = set(
            previous_preds[previous_preds["risk_tier"] == "high"]["account_id"]
        )

        # Newly high-risk = in current_high but NOT in previous_high
        newly_high = current_high - previous_high

        if not newly_high:
            return json.dumps({
                "total_new_high_risk": 0,
                "accounts": [],
                "message": "No new high-risk accounts detected.",
            })

        # Build detail for each newly high-risk account
        prev_lookup = previous_preds.set_index("account_id")
        curr_lookup = current_preds.set_index("account_id")

        accounts = []
        for aid in newly_high:
            curr_prob = float(curr_lookup.loc[aid, "churn_probability"])
            prev_prob = (
                float(prev_lookup.loc[aid, "churn_probability"])
                if aid in prev_lookup.index
                else 0.0
            )
            prev_tier = (
                str(prev_lookup.loc[aid, "risk_tier"])
                if aid in prev_lookup.index
                else "unknown"
            )
            accounts.append({
                "account_id": aid,
                "current_probability": round(curr_prob, 4),
                "previous_probability": round(prev_prob, 4),
                "probability_delta": round(curr_prob - prev_prob, 4),
                "previous_tier": prev_tier,
                "current_tier": "high",
            })

        # Sort by delta descending
        accounts.sort(key=lambda x: x["probability_delta"], reverse=True)

        result = {
            "total_new_high_risk": len(accounts),
            "accounts": accounts,
        }

        return json.dumps(safe_json_serialize(result), indent=2)

    except Exception as e:
        logger.error(f"get_new_high_risk_accounts failed: {e}")
        return f"Error finding new high-risk accounts: {e}"


# =============================================================================
# KPI and Analytics Tools (for Analysis Agent)
# =============================================================================


@tool
def get_executive_kpis(reference_date: str = "2025-12-01") -> str:
    """Compute executive-level KPIs as of the reference date.

    Returns:
    - total_active_accounts: count of accounts with active subscriptions
    - total_churned_accounts: count cancelled/payment_failed in last 30 days
    - churn_rate_30d: churned / (active + churned) as percentage
    - avg_churn_probability: mean predicted churn prob across active accounts
    - high_risk_count: accounts with churn_probability >= 0.70
    - monthly_recurring_revenue: sum of active subscription payment amounts
    - avg_customer_lifetime_days: mean tenure of active accounts
    - retention_rate_30d: 1 - churn_rate_30d as percentage
    All values as of the reference date. Uses SQL queries against the database.

    Args:
        reference_date: Reference date for KPI computation (YYYY-MM-DD).

    Returns:
        JSON with all KPI values, or error message.
    """
    try:
        from datetime import datetime, timedelta

        from sqlalchemy import text

        from src.data.database import get_engine

        engine = get_engine()
        ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
        thirty_days_ago = (ref_date - timedelta(days=30)).strftime("%Y-%m-%d")

        with engine.connect() as conn:
            # Active accounts
            active_result = conn.execute(text(
                "SELECT COUNT(DISTINCT account_id) as cnt "
                "FROM subscriptions WHERE status = 'active'"
            ))
            total_active = active_result.scalar() or 0

            # Churned in last 30 days
            churned_result = conn.execute(text(
                "SELECT COUNT(DISTINCT account_id) as cnt "
                "FROM subscriptions "
                "WHERE status IN ('cancelled', 'payment_failed') "
                "AND end_date >= :start_date AND end_date <= :end_date"
            ), {"start_date": thirty_days_ago, "end_date": reference_date})
            total_churned = churned_result.scalar() or 0

            # Churn rate
            total_base = total_active + total_churned
            churn_rate = (
                round(total_churned / total_base * 100, 2)
                if total_base > 0 else 0.0
            )
            retention_rate = round(100 - churn_rate, 2)

            # MRR from recent payments
            mrr_result = conn.execute(text(
                "SELECT COALESCE(SUM(p.amount), 0) as mrr "
                "FROM payments p "
                "JOIN subscriptions s ON p.account_id = s.account_id "
                "WHERE s.status = 'active' AND p.status = 'success' "
                "AND p.payment_date >= :start_date "
                "AND p.payment_date <= :end_date"
            ), {"start_date": thirty_days_ago, "end_date": reference_date})
            mrr = round(float(mrr_result.scalar() or 0), 2)

            # Average customer lifetime
            lifetime_result = conn.execute(text(
                "SELECT AVG(:ref_date - a.signup_date) as avg_days "
                "FROM accounts a "
                "JOIN subscriptions s ON a.account_id = s.account_id "
                "WHERE s.status = 'active'"
            ), {"ref_date": reference_date})
            avg_lifetime = float(lifetime_result.scalar() or 0)

        # Churn probability from predictions table
        try:
            from src.models.config import ScoringConfig
            from src.models.score import BatchScorer

            scorer = BatchScorer(engine, ScoringConfig())
            latest_scores = scorer.get_latest_scores()
            if not latest_scores.empty:
                avg_prob = round(
                    float(latest_scores["churn_probability"].mean()), 4
                )
                high_risk_count = int(
                    (latest_scores["risk_tier"] == "high").sum()
                )
            else:
                avg_prob = 0.0
                high_risk_count = 0
        except Exception:
            avg_prob = 0.0
            high_risk_count = 0

        kpis = {
            "total_active_accounts": total_active,
            "total_churned_accounts": total_churned,
            "churn_rate_30d": churn_rate,
            "retention_rate_30d": retention_rate,
            "avg_churn_probability": avg_prob,
            "high_risk_count": high_risk_count,
            "monthly_recurring_revenue": mrr,
            "avg_customer_lifetime_days": round(avg_lifetime, 1),
            "reference_date": reference_date,
        }

        return json.dumps(safe_json_serialize(kpis), indent=2)

    except Exception as e:
        logger.error(f"get_executive_kpis failed: {e}")
        return f"Error computing executive KPIs: {e}"


@tool
def get_subscription_distribution(
    reference_date: str = "2025-12-01",
) -> str:
    """Get the breakdown of active subscriptions by plan type and status.

    Returns counts and percentages for each plan_type (Basic, Regular,
    Premium, Family) and status (active, cancelled, payment_failed).
    Useful for the Analysis Agent when narrating plan mix and health.

    Args:
        reference_date: Reference date (YYYY-MM-DD).

    Returns:
        JSON with plan type and status distributions, or error message.
    """
    try:
        from sqlalchemy import text

        from src.data.database import get_engine

        engine = get_engine()

        with engine.connect() as conn:
            # By plan type
            plan_df = pd.read_sql(text(
                "SELECT plan_type, COUNT(*) as count "
                "FROM subscriptions GROUP BY plan_type ORDER BY count DESC"
            ), conn)

            # By status
            status_df = pd.read_sql(text(
                "SELECT status, COUNT(*) as count "
                "FROM subscriptions GROUP BY status ORDER BY count DESC"
            ), conn)

            # Cross-tab: plan_type × status
            cross_df = pd.read_sql(text(
                "SELECT plan_type, status, COUNT(*) as count "
                "FROM subscriptions GROUP BY plan_type, status "
                "ORDER BY plan_type, status"
            ), conn)

        total = int(plan_df["count"].sum())

        result = {
            "total_subscriptions": total,
            "by_plan_type": {
                row["plan_type"]: {
                    "count": int(row["count"]),
                    "percentage": round(int(row["count"]) / max(total, 1) * 100, 1),
                }
                for _, row in plan_df.iterrows()
            },
            "by_status": {
                row["status"]: {
                    "count": int(row["count"]),
                    "percentage": round(int(row["count"]) / max(total, 1) * 100, 1),
                }
                for _, row in status_df.iterrows()
            },
            "cross_tab": [
                {
                    "plan_type": row["plan_type"],
                    "status": row["status"],
                    "count": int(row["count"]),
                }
                for _, row in cross_df.iterrows()
            ],
        }

        return json.dumps(safe_json_serialize(result), indent=2)

    except Exception as e:
        logger.error(f"get_subscription_distribution failed: {e}")
        return f"Error retrieving subscription distribution: {e}"


@tool
def get_engagement_trends(
    reference_date: str = "2025-12-01",
    lookback_days: int = 90,
) -> str:
    """Get streaming engagement trends over the lookback period.

    Returns weekly aggregates of: total_watch_hours, unique_active_accounts,
    avg_session_duration_minutes, sessions_per_account, top_genres_by_hours.
    Useful for the Analysis Agent when explaining engagement charts.

    Args:
        reference_date: End date for the trend period (YYYY-MM-DD).
        lookback_days: Number of days to look back from reference_date.

    Returns:
        JSON with weekly engagement data, or error message.
    """
    try:
        from datetime import datetime, timedelta

        from sqlalchemy import text

        from src.data.database import get_engine

        engine = get_engine()
        ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
        start_date = (ref_date - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )

        with engine.connect() as conn:
            # Weekly aggregates
            weekly_df = pd.read_sql(text(
                "SELECT "
                "  DATE_TRUNC('week', event_timestamp) as week, "
                "  COUNT(*) as total_sessions, "
                "  COUNT(DISTINCT account_id) as unique_accounts, "
                "  COALESCE(SUM(watch_duration_minutes) / 60.0, 0) "
                "    as total_watch_hours, "
                "  COALESCE(AVG(watch_duration_minutes), 0) "
                "    as avg_session_minutes "
                "FROM streaming_events "
                "WHERE event_timestamp >= :start AND event_timestamp <= :end "
                "GROUP BY DATE_TRUNC('week', event_timestamp) "
                "ORDER BY week"
            ), conn, params={"start": start_date, "end": reference_date})

            # Top genres
            genre_df = pd.read_sql(text(
                "SELECT c.genre, "
                "  SUM(se.watch_duration_minutes) / 60.0 as hours "
                "FROM streaming_events se "
                "JOIN content_catalog c ON se.content_id = c.content_id "
                "WHERE se.event_timestamp >= :start "
                "  AND se.event_timestamp <= :end "
                "GROUP BY c.genre ORDER BY hours DESC LIMIT 10"
            ), conn, params={"start": start_date, "end": reference_date})

        weeks = []
        for _, row in weekly_df.iterrows():
            week_str = str(row["week"])[:10] if row["week"] else "unknown"
            sessions_per_account = (
                round(int(row["total_sessions"]) / max(int(row["unique_accounts"]), 1), 2)
            )
            weeks.append({
                "week": week_str,
                "total_sessions": int(row["total_sessions"]),
                "unique_accounts": int(row["unique_accounts"]),
                "total_watch_hours": round(float(row["total_watch_hours"]), 1),
                "avg_session_minutes": round(
                    float(row["avg_session_minutes"]), 1
                ),
                "sessions_per_account": sessions_per_account,
            })

        # Compute trend direction
        if len(weeks) >= 2:
            first_half = weeks[: len(weeks) // 2]
            second_half = weeks[len(weeks) // 2 :]
            first_avg = sum(w["total_watch_hours"] for w in first_half) / max(
                len(first_half), 1
            )
            second_avg = sum(
                w["total_watch_hours"] for w in second_half
            ) / max(len(second_half), 1)
            trend = (
                "increasing" if second_avg > first_avg * 1.05
                else "decreasing" if second_avg < first_avg * 0.95
                else "stable"
            )
        else:
            trend = "insufficient_data"

        result = {
            "period": f"{start_date} to {reference_date}",
            "weekly_data": weeks,
            "top_genres": [
                {"genre": row["genre"], "hours": round(float(row["hours"]), 1)}
                for _, row in genre_df.iterrows()
            ],
            "engagement_trend": trend,
        }

        return json.dumps(safe_json_serialize(result), indent=2)

    except Exception as e:
        logger.error(f"get_engagement_trends failed: {e}")
        return f"Error retrieving engagement trends: {e}"


@tool
def get_support_health(
    reference_date: str = "2025-12-01",
    lookback_days: int = 90,
) -> str:
    """Get support ticket health metrics over the lookback period.

    Returns:
    - total_tickets, tickets_by_category (top 5), tickets_by_priority
    - avg_resolution_hours, median_resolution_hours
    - unresolved_count (tickets with no resolved_at)
    - weekly_ticket_volume trend
    Useful for the Analysis Agent when narrating support dashboard panels.

    Args:
        reference_date: End date for the trend period (YYYY-MM-DD).
        lookback_days: Number of days to look back from reference_date.

    Returns:
        JSON with support health metrics, or error message.
    """
    try:
        from datetime import datetime, timedelta

        from sqlalchemy import text

        from src.data.database import get_engine

        engine = get_engine()
        ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
        start_date = (ref_date - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )

        with engine.connect() as conn:
            # Overall metrics
            overall = conn.execute(text(
                "SELECT COUNT(*) as total, "
                "  SUM(CASE WHEN resolved_at IS NULL THEN 1 ELSE 0 END) "
                "    as unresolved "
                "FROM support_tickets "
                "WHERE created_at >= :start AND created_at <= :end"
            ), {"start": start_date, "end": reference_date}).fetchone()

            total_tickets = overall[0] or 0
            unresolved = overall[1] or 0

            # By category
            category_df = pd.read_sql(text(
                "SELECT category, COUNT(*) as count "
                "FROM support_tickets "
                "WHERE created_at >= :start AND created_at <= :end "
                "GROUP BY category ORDER BY count DESC LIMIT 5"
            ), conn, params={"start": start_date, "end": reference_date})

            # By priority
            priority_df = pd.read_sql(text(
                "SELECT priority, COUNT(*) as count "
                "FROM support_tickets "
                "WHERE created_at >= :start AND created_at <= :end "
                "GROUP BY priority ORDER BY count DESC"
            ), conn, params={"start": start_date, "end": reference_date})

            # Resolution time
            resolution = conn.execute(text(
                "SELECT "
                "  AVG(EXTRACT(EPOCH FROM (resolved_at - created_at)) / 3600) "
                "    as avg_hours, "
                "  PERCENTILE_CONT(0.5) WITHIN GROUP ("
                "    ORDER BY EXTRACT(EPOCH FROM (resolved_at - created_at)) "
                "    / 3600"
                "  ) as median_hours "
                "FROM support_tickets "
                "WHERE created_at >= :start AND created_at <= :end "
                "AND resolved_at IS NOT NULL"
            ), {"start": start_date, "end": reference_date}).fetchone()

            avg_hours = round(float(resolution[0] or 0), 1)
            median_hours = round(float(resolution[1] or 0), 1)

            # Weekly volume
            weekly_df = pd.read_sql(text(
                "SELECT DATE_TRUNC('week', created_at) as week, "
                "  COUNT(*) as count "
                "FROM support_tickets "
                "WHERE created_at >= :start AND created_at <= :end "
                "GROUP BY DATE_TRUNC('week', created_at) ORDER BY week"
            ), conn, params={"start": start_date, "end": reference_date})

        result = {
            "period": f"{start_date} to {reference_date}",
            "total_tickets": total_tickets,
            "unresolved_count": unresolved,
            "avg_resolution_hours": avg_hours,
            "median_resolution_hours": median_hours,
            "by_category": {
                row["category"]: int(row["count"])
                for _, row in category_df.iterrows()
            },
            "by_priority": {
                row["priority"]: int(row["count"])
                for _, row in priority_df.iterrows()
            },
            "weekly_volume": [
                {
                    "week": str(row["week"])[:10],
                    "count": int(row["count"]),
                }
                for _, row in weekly_df.iterrows()
            ],
        }

        return json.dumps(safe_json_serialize(result), indent=2)

    except Exception as e:
        logger.error(f"get_support_health failed: {e}")
        return f"Error retrieving support health: {e}"


@tool
def get_payment_health(
    reference_date: str = "2025-12-01",
    lookback_days: int = 90,
) -> str:
    """Get payment health metrics over the lookback period.

    Returns:
    - total_payments, successful_payments, failed_payments
    - failure_rate as percentage
    - total_revenue, avg_payment_amount
    - failures_by_reason (top reasons)
    - weekly_revenue trend
    Useful for the Analysis Agent when narrating financial health.

    Args:
        reference_date: End date for the trend period (YYYY-MM-DD).
        lookback_days: Number of days to look back from reference_date.

    Returns:
        JSON with payment health metrics, or error message.
    """
    try:
        from datetime import datetime, timedelta

        from sqlalchemy import text

        from src.data.database import get_engine

        engine = get_engine()
        ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
        start_date = (ref_date - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )

        with engine.connect() as conn:
            # Overall metrics
            overall = conn.execute(text(
                "SELECT "
                "  COUNT(*) as total, "
                "  SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) "
                "    as successful, "
                "  SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) "
                "    as failed, "
                "  COALESCE(SUM(CASE WHEN status = 'success' "
                "    THEN amount ELSE 0 END), 0) as total_revenue, "
                "  COALESCE(AVG(CASE WHEN status = 'success' "
                "    THEN amount END), 0) as avg_amount "
                "FROM payments "
                "WHERE payment_date >= :start AND payment_date <= :end"
            ), {"start": start_date, "end": reference_date}).fetchone()

            total = overall[0] or 0
            successful = overall[1] or 0
            failed = overall[2] or 0
            total_revenue = round(float(overall[3] or 0), 2)
            avg_amount = round(float(overall[4] or 0), 2)

            failure_rate = (
                round(failed / max(total, 1) * 100, 2)
            )

            # Failure reasons
            reason_df = pd.read_sql(text(
                "SELECT failure_reason, COUNT(*) as count "
                "FROM payments "
                "WHERE status = 'failed' "
                "AND payment_date >= :start AND payment_date <= :end "
                "AND failure_reason IS NOT NULL "
                "GROUP BY failure_reason ORDER BY count DESC LIMIT 5"
            ), conn, params={"start": start_date, "end": reference_date})

            # Weekly revenue
            weekly_df = pd.read_sql(text(
                "SELECT DATE_TRUNC('week', payment_date) as week, "
                "  SUM(CASE WHEN status = 'success' "
                "    THEN amount ELSE 0 END) as revenue "
                "FROM payments "
                "WHERE payment_date >= :start AND payment_date <= :end "
                "GROUP BY DATE_TRUNC('week', payment_date) ORDER BY week"
            ), conn, params={"start": start_date, "end": reference_date})

        result = {
            "period": f"{start_date} to {reference_date}",
            "total_payments": total,
            "successful_payments": successful,
            "failed_payments": failed,
            "failure_rate": failure_rate,
            "total_revenue": total_revenue,
            "avg_payment_amount": avg_amount,
            "failures_by_reason": {
                row["failure_reason"]: int(row["count"])
                for _, row in reason_df.iterrows()
            },
            "weekly_revenue": [
                {
                    "week": str(row["week"])[:10],
                    "revenue": round(float(row["revenue"]), 2),
                }
                for _, row in weekly_df.iterrows()
            ],
        }

        return json.dumps(safe_json_serialize(result), indent=2)

    except Exception as e:
        logger.error(f"get_payment_health failed: {e}")
        return f"Error retrieving payment health: {e}"


@tool
def get_churn_cohort_analysis(
    reference_date: str = "2025-12-01",
) -> str:
    """Analyze churn patterns by signup cohort (quarterly signup buckets).

    For each cohort: size, churn_count, churn_rate, avg_lifetime_days,
    avg_churn_probability. Reveals whether recent cohorts churn faster
    than older ones. Useful for the Analysis Agent when narrating
    retention curves.

    Args:
        reference_date: Reference date for analysis (YYYY-MM-DD).

    Returns:
        JSON with cohort analysis data, or error message.
    """
    try:
        from sqlalchemy import text

        from src.data.database import get_engine

        engine = get_engine()

        with engine.connect() as conn:
            cohort_df = pd.read_sql(text(
                "SELECT "
                "  DATE_TRUNC('quarter', a.signup_date) as cohort, "
                "  COUNT(DISTINCT a.account_id) as cohort_size, "
                "  SUM(CASE WHEN s.status IN ('cancelled', 'payment_failed') "
                "    THEN 1 ELSE 0 END) as churn_count, "
                "  AVG(:ref_date - a.signup_date) as avg_lifetime_days "
                "FROM accounts a "
                "LEFT JOIN subscriptions s ON a.account_id = s.account_id "
                "GROUP BY DATE_TRUNC('quarter', a.signup_date) "
                "ORDER BY cohort"
            ), conn, params={"ref_date": reference_date})

        cohorts = []
        for _, row in cohort_df.iterrows():
            size = int(row["cohort_size"])
            churned = int(row["churn_count"])
            churn_rate = round(churned / max(size, 1) * 100, 2)
            cohorts.append({
                "cohort": str(row["cohort"])[:10],
                "size": size,
                "churn_count": churned,
                "churn_rate": churn_rate,
                "avg_lifetime_days": round(float(row["avg_lifetime_days"] or 0), 1),
            })

        result = {
            "reference_date": reference_date,
            "total_cohorts": len(cohorts),
            "cohorts": cohorts,
        }

        return json.dumps(safe_json_serialize(result), indent=2)

    except Exception as e:
        logger.error(f"get_churn_cohort_analysis failed: {e}")
        return f"Error running cohort analysis: {e}"


# =============================================================================
# Early Warning & Analysis Tool Registries
# =============================================================================

EARLY_WARNING_TOOLS: list = [
    run_batch_scoring,
    get_account_risk_scores,
    get_risk_tier_transitions,
    get_risk_velocity,
    get_new_high_risk_accounts,
    explain_account_prediction,
    get_account_profile,
    get_account_support_history,
    get_account_payment_history,
    get_account_viewing_summary,
    segment_high_risk_accounts,
    query_database_readonly,
]

ANALYSIS_TOOLS: list = [
    get_executive_kpis,
    get_subscription_distribution,
    get_engagement_trends,
    get_support_health,
    get_payment_health,
    get_churn_cohort_analysis,
    get_account_risk_scores,
    get_feature_importance_global,
    get_model_health_status,
    query_database_readonly,
]
