"""
Configuration for the Retain agent system.

Centralizes all agent settings — LLM parameters, scoring thresholds,
cohort analysis limits, and safety guardrails.
"""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for the Retain agent system."""

    # LLM
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = 0.2
    max_tokens: int = 4096

    # Data
    reference_date: str = "2025-12-01"

    # Scoring thresholds (must match ScoringConfig defaults)
    high_risk_threshold: float = 0.70
    medium_risk_threshold: float = 0.40

    # Cohort analysis
    min_cohort_size: int = 10
    max_cohorts: int = 5
    top_features_count: int = 10

    # SHAP
    shap_sample_size: int = 100

    # Safety
    max_sql_rows: int = 1000
    read_only_sql: bool = True
