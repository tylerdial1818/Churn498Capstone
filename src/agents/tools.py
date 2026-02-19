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
