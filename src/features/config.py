"""
Feature engineering configuration.

This module defines feature specifications, time windows, and pipeline parameters.
Configuration-driven approach allows easy modification without code changes.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any


class FeatureType(Enum):
    """Categories of features for organization and documentation."""
    ENGAGEMENT = "engagement"
    BEHAVIORAL = "behavioral"
    FINANCIAL = "financial"
    SUPPORT = "support"
    DEMOGRAPHIC = "demographic"
    TEMPORAL = "temporal"


@dataclass(frozen=True)
class TimeWindow:
    """Defines a lookback window for feature aggregation."""
    name: str
    days: int
    
    @property
    def timedelta(self) -> timedelta:
        return timedelta(days=self.days)


@dataclass
class FeatureSpec:
    """Specification for a single feature."""
    name: str
    feature_type: FeatureType
    description: str
    source_tables: list[str]
    nullable: bool = False
    default_value: Any = None


# Standard time windows for rolling aggregations
TIME_WINDOWS = {
    "7d": TimeWindow("7d", 7),
    "30d": TimeWindow("30d", 30),
    "90d": TimeWindow("90d", 90),
    "lifetime": TimeWindow("lifetime", 9999),
}


@dataclass
class FeatureConfig:
    """
    Central configuration for the feature engineering pipeline.
    
    Modify these settings to adjust feature generation behavior
    without changing pipeline code.
    """
    # Reference date for feature calculation (None = use current date)
    reference_date: str | None = None
    
    # Time windows for rolling features
    rolling_windows: list[str] = field(
        default_factory=lambda: ["30d", "90d"]
    )
    
    # Minimum days of history required per account
    min_history_days: int = 14
    
    # Whether to include text-based features (requires NLP processing)
    include_text_features: bool = False
    
    # Output settings
    output_format: str = "parquet"  # parquet, csv, or database
    output_path: str = "data/processed/features"
    
    # Validation settings
    validate_output: bool = True
    fail_on_validation_error: bool = False
    
    # Processing settings
    chunk_size: int = 10000  # For batch processing
    n_jobs: int = -1  # Parallel jobs (-1 = all cores)


# Feature specifications registry
FEATURE_SPECS: dict[str, FeatureSpec] = {
    # Engagement features
    "avg_watch_hours_per_week": FeatureSpec(
        name="avg_watch_hours_per_week",
        feature_type=FeatureType.ENGAGEMENT,
        description="Average weekly watch time in hours",
        source_tables=["streaming_events"],
    ),
    "days_since_last_stream": FeatureSpec(
        name="days_since_last_stream",
        feature_type=FeatureType.ENGAGEMENT,
        description="Days between reference date and last streaming event",
        source_tables=["streaming_events"],
    ),
    "total_watch_sessions": FeatureSpec(
        name="total_watch_sessions",
        feature_type=FeatureType.ENGAGEMENT,
        description="Total number of streaming sessions",
        source_tables=["streaming_events"],
    ),
    "avg_session_duration_minutes": FeatureSpec(
        name="avg_session_duration_minutes",
        feature_type=FeatureType.ENGAGEMENT,
        description="Average duration per streaming session",
        source_tables=["streaming_events"],
    ),
    
    # Behavioral features
    "most_watched_genre": FeatureSpec(
        name="most_watched_genre",
        feature_type=FeatureType.BEHAVIORAL,
        description="Mode of genres watched (categorical)",
        source_tables=["streaming_events", "content_catalog"],
    ),
    "genre_diversity_score": FeatureSpec(
        name="genre_diversity_score",
        feature_type=FeatureType.BEHAVIORAL,
        description="Ratio of unique genres to total genres watched",
        source_tables=["streaming_events", "content_catalog"],
    ),
    "primary_device_type": FeatureSpec(
        name="primary_device_type",
        feature_type=FeatureType.BEHAVIORAL,
        description="Most frequently used device for streaming",
        source_tables=["streaming_events"],
    ),
    "device_diversity_count": FeatureSpec(
        name="device_diversity_count",
        feature_type=FeatureType.BEHAVIORAL,
        description="Number of unique devices used",
        source_tables=["streaming_events"],
    ),
    "avg_login_distance_km": FeatureSpec(
        name="avg_login_distance_km",
        feature_type=FeatureType.BEHAVIORAL,
        description="Average distance between login locations (account sharing signal)",
        source_tables=["streaming_events"],
        nullable=True,
    ),
    "binge_session_ratio": FeatureSpec(
        name="binge_session_ratio",
        feature_type=FeatureType.BEHAVIORAL,
        description="Ratio of sessions >2 hours to total sessions",
        source_tables=["streaming_events"],
    ),
    
    # Financial features
    "failed_payment_count": FeatureSpec(
        name="failed_payment_count",
        feature_type=FeatureType.FINANCIAL,
        description="Count of failed payments in window",
        source_tables=["payments"],
    ),
    "avg_monthly_spend": FeatureSpec(
        name="avg_monthly_spend",
        feature_type=FeatureType.FINANCIAL,
        description="Average monthly payment amount",
        source_tables=["payments"],
    ),
    "payment_method_changes": FeatureSpec(
        name="payment_method_changes",
        feature_type=FeatureType.FINANCIAL,
        description="Number of times payment method was changed",
        source_tables=["payments"],
    ),
    "days_since_last_payment": FeatureSpec(
        name="days_since_last_payment",
        feature_type=FeatureType.FINANCIAL,
        description="Days since last successful payment",
        source_tables=["payments"],
    ),
    
    # Support features
    "ticket_count": FeatureSpec(
        name="ticket_count",
        feature_type=FeatureType.SUPPORT,
        description="Number of support tickets in window",
        source_tables=["support_tickets"],
    ),
    "has_billing_complaint": FeatureSpec(
        name="has_billing_complaint",
        feature_type=FeatureType.SUPPORT,
        description="Whether account has billing-related tickets",
        source_tables=["support_tickets"],
    ),
    "avg_resolution_hours": FeatureSpec(
        name="avg_resolution_hours",
        feature_type=FeatureType.SUPPORT,
        description="Average time to resolve support tickets",
        source_tables=["support_tickets"],
        nullable=True,
    ),
    "open_ticket_count": FeatureSpec(
        name="open_ticket_count",
        feature_type=FeatureType.SUPPORT,
        description="Number of currently unresolved tickets",
        source_tables=["support_tickets"],
    ),
    
    # Demographic/Account features
    "account_tenure_days": FeatureSpec(
        name="account_tenure_days",
        feature_type=FeatureType.DEMOGRAPHIC,
        description="Days since account signup",
        source_tables=["accounts", "subscriptions"],
    ),
    "plan_tier_encoded": FeatureSpec(
        name="plan_tier_encoded",
        feature_type=FeatureType.DEMOGRAPHIC,
        description="Plan tier (1=Regular, 2=Premium, 3=Premium-Multi-Screen)",
        source_tables=["subscriptions"],
    ),
    "has_plan_upgrade": FeatureSpec(
        name="has_plan_upgrade",
        feature_type=FeatureType.DEMOGRAPHIC,
        description="Whether account has upgraded plans",
        source_tables=["subscriptions"],
    ),
    "has_plan_downgrade": FeatureSpec(
        name="has_plan_downgrade",
        feature_type=FeatureType.DEMOGRAPHIC,
        description="Whether account has downgraded plans",
        source_tables=["subscriptions"],
    ),
    "parental_control_enabled": FeatureSpec(
        name="parental_control_enabled",
        feature_type=FeatureType.DEMOGRAPHIC,
        description="Family account indicator",
        source_tables=["subscriptions"],
    ),
    "has_app_rating": FeatureSpec(
        name="has_app_rating",
        feature_type=FeatureType.DEMOGRAPHIC,
        description="Whether account has rated the app",
        source_tables=["subscriptions"],
    ),
    "app_rating_value": FeatureSpec(
        name="app_rating_value",
        feature_type=FeatureType.DEMOGRAPHIC,
        description="App rating (1-5, null if not rated)",
        source_tables=["subscriptions"],
        nullable=True,
    ),
    "watchlist_size": FeatureSpec(
        name="watchlist_size",
        feature_type=FeatureType.DEMOGRAPHIC,
        description="Number of items in watchlist",
        source_tables=["subscriptions"],
    ),
    "age": FeatureSpec(
        name="age",
        feature_type=FeatureType.DEMOGRAPHIC,
        description="Account holder age",
        source_tables=["accounts"],
    ),
    
    # Temporal features
    "signup_month": FeatureSpec(
        name="signup_month",
        feature_type=FeatureType.TEMPORAL,
        description="Month of signup (1-12)",
        source_tables=["accounts"],
    ),
    "signup_dayofweek": FeatureSpec(
        name="signup_dayofweek",
        feature_type=FeatureType.TEMPORAL,
        description="Day of week of signup (0=Monday)",
        source_tables=["accounts"],
    ),
    "is_weekend_viewer": FeatureSpec(
        name="is_weekend_viewer",
        feature_type=FeatureType.TEMPORAL,
        description="Whether >50% of viewing is on weekends",
        source_tables=["streaming_events"],
    ),
    "peak_viewing_hour": FeatureSpec(
        name="peak_viewing_hour",
        feature_type=FeatureType.TEMPORAL,
        description="Most common hour of day for viewing (0-23)",
        source_tables=["streaming_events"],
    ),
}


def get_features_by_type(feature_type: FeatureType) -> list[FeatureSpec]:
    """Get all feature specifications of a given type."""
    return [
        spec for spec in FEATURE_SPECS.values() 
        if spec.feature_type == feature_type
    ]


def get_required_tables(feature_names: list[str] | None = None) -> set[str]:
    """Get all source tables required for the specified features."""
    if feature_names is None:
        feature_names = list(FEATURE_SPECS.keys())
    
    tables = set()
    for name in feature_names:
        if name in FEATURE_SPECS:
            tables.update(FEATURE_SPECS[name].source_tables)
    return tables
