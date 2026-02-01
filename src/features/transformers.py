"""
Feature transformers for churn prediction.

Each transformer class handles a category of features, encapsulating the SQL
queries and transformation logic. This separation allows for:
- Independent testing of each feature category
- Easy addition of new features
- Clear documentation of feature derivation logic
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .config import TIME_WINDOWS, TimeWindow


@dataclass
class TransformerContext:
    """Context passed to all transformers."""
    engine: Engine
    reference_date: date
    time_windows: list[TimeWindow]
    account_ids: list[str] | None = None  # None = all accounts


class BaseTransformer(ABC):
    """Base class for all feature transformers."""
    
    name: str = "base"
    
    def __init__(self, context: TransformerContext):
        self.context = context
        self.engine = context.engine
        self.reference_date = context.reference_date
        self.time_windows = context.time_windows
    
    @abstractmethod
    def transform(self) -> pd.DataFrame:
        """
        Execute transformation and return features.
        
        Returns:
            DataFrame with account_id as index and feature columns.
        """
        pass
    
    def _execute_query(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
    
    def _get_account_filter(self) -> str:
        """Get SQL WHERE clause for account filtering."""
        if self.context.account_ids is None:
            return ""
        # For production, this would use a temp table for large lists
        ids = ", ".join(f"'{aid}'" for aid in self.context.account_ids)
        return f"AND account_id IN ({ids})"


class EngagementTransformer(BaseTransformer):
    """
    Transforms streaming_events into engagement features.
    
    Features:
    - avg_watch_hours_per_week
    - days_since_last_stream  
    - total_watch_sessions
    - avg_session_duration_minutes
    - binge_session_ratio
    """
    
    name = "engagement"
    
    def transform(self) -> pd.DataFrame:
        features_list = []
        
        for window in self.time_windows:
            window_features = self._compute_window_features(window)
            # Add window suffix to column names (except account_id)
            window_features = window_features.rename(
                columns={
                    col: f"{col}_{window.name}" 
                    for col in window_features.columns 
                    if col != "account_id"
                }
            )
            features_list.append(window_features)
        
        # Also compute lifetime/all-time features
        lifetime_features = self._compute_lifetime_features()
        features_list.append(lifetime_features)
        
        # Merge all feature sets on account_id
        result = features_list[0]
        for df in features_list[1:]:
            result = result.merge(df, on="account_id", how="outer")
        
        return result.set_index("account_id")
    
    def _compute_window_features(self, window: TimeWindow) -> pd.DataFrame:
        """Compute engagement features for a specific time window."""
        query = """
        WITH window_events AS (
            SELECT 
                account_id,
                event_timestamp,
                watch_duration_minutes,
                DATE(event_timestamp) as event_date
            FROM streaming_events
            WHERE event_timestamp >= :start_date
              AND event_timestamp < :end_date
              {account_filter}
        ),
        daily_stats AS (
            SELECT
                account_id,
                event_date,
                COUNT(*) as sessions,
                SUM(watch_duration_minutes) as total_minutes,
                MAX(watch_duration_minutes) as max_session_minutes
            FROM window_events
            GROUP BY account_id, event_date
        )
        SELECT
            account_id,
            SUM(total_minutes) / 60.0 / GREATEST(1, :window_weeks) as avg_watch_hours_per_week,
            SUM(sessions) as total_watch_sessions,
            AVG(total_minutes / NULLIF(sessions, 0)) as avg_session_duration_minutes,
            SUM(CASE WHEN max_session_minutes > 120 THEN 1 ELSE 0 END)::float / 
                NULLIF(COUNT(*), 0) as binge_session_ratio
        FROM daily_stats
        GROUP BY account_id
        """
        
        start_date = self.reference_date - window.timedelta
        params = {
            "start_date": start_date,
            "end_date": self.reference_date,
            "window_weeks": max(1, window.days // 7),
        }
        
        formatted_query = query.format(account_filter=self._get_account_filter())
        return self._execute_query(formatted_query, params)
    
    def _compute_lifetime_features(self) -> pd.DataFrame:
        """Compute features that don't depend on time windows."""
        query = """
        SELECT
            account_id,
            :reference_date - MAX(DATE(event_timestamp)) as days_since_last_stream
        FROM streaming_events
        WHERE event_timestamp < :reference_date
          {account_filter}
        GROUP BY account_id
        """
        
        params = {"reference_date": self.reference_date}
        formatted_query = query.format(account_filter=self._get_account_filter())
        return self._execute_query(formatted_query, params)


class BehavioralTransformer(BaseTransformer):
    """
    Transforms streaming behavior into behavioral features.
    
    Features:
    - most_watched_genre
    - genre_diversity_score
    - primary_device_type
    - device_diversity_count
    - avg_login_distance_km
    """
    
    name = "behavioral"
    
    def transform(self) -> pd.DataFrame:
        features_list = []
        
        # Genre features (lifetime - genre preferences are stable)
        genre_features = self._compute_genre_features()
        features_list.append(genre_features)
        
        # Device features (lifetime)
        device_features = self._compute_device_features()
        features_list.append(device_features)
        
        # Location features (potential account sharing)
        location_features = self._compute_location_features()
        features_list.append(location_features)
        
        # Merge all
        result = features_list[0]
        for df in features_list[1:]:
            result = result.merge(df, on="account_id", how="outer")
        
        return result.set_index("account_id")
    
    def _compute_genre_features(self) -> pd.DataFrame:
        """Compute genre-based behavioral features."""
        query = """
        WITH genre_watches AS (
            SELECT 
                se.account_id,
                cc.genre,
                COUNT(*) as watch_count
            FROM streaming_events se
            JOIN content_catalog cc ON se.content_id = cc.content_id
            WHERE se.event_timestamp < :reference_date
              {account_filter}
            GROUP BY se.account_id, cc.genre
        ),
        account_genres AS (
            SELECT
                account_id,
                COUNT(DISTINCT genre) as unique_genres,
                SUM(watch_count) as total_watches,
                MAX(watch_count) as max_genre_watches
            FROM genre_watches
            GROUP BY account_id
        ),
        top_genre AS (
            SELECT DISTINCT ON (account_id)
                account_id,
                genre as most_watched_genre
            FROM genre_watches
            ORDER BY account_id, watch_count DESC
        )
        SELECT
            ag.account_id,
            tg.most_watched_genre,
            ag.unique_genres::float / NULLIF(
                (SELECT COUNT(DISTINCT genre) FROM content_catalog), 0
            ) as genre_diversity_score
        FROM account_genres ag
        JOIN top_genre tg ON ag.account_id = tg.account_id
        """
        
        params = {"reference_date": self.reference_date}
        formatted_query = query.format(account_filter=self._get_account_filter())
        return self._execute_query(formatted_query, params)
    
    def _compute_device_features(self) -> pd.DataFrame:
        """Compute device usage features."""
        query = """
        WITH device_usage AS (
            SELECT 
                account_id,
                device_type,
                COUNT(*) as use_count
            FROM streaming_events
            WHERE event_timestamp < :reference_date
              {account_filter}
            GROUP BY account_id, device_type
        ),
        primary_device AS (
            SELECT DISTINCT ON (account_id)
                account_id,
                device_type as primary_device_type
            FROM device_usage
            ORDER BY account_id, use_count DESC
        )
        SELECT
            pd.account_id,
            pd.primary_device_type,
            (SELECT COUNT(DISTINCT device_type) 
             FROM device_usage du 
             WHERE du.account_id = pd.account_id) as device_diversity_count
        FROM primary_device pd
        """
        
        params = {"reference_date": self.reference_date}
        formatted_query = query.format(account_filter=self._get_account_filter())
        return self._execute_query(formatted_query, params)
    
    def _compute_location_features(self) -> pd.DataFrame:
        """
        Compute location-based features for account sharing detection.
        
        Uses Haversine formula for distance calculation.
        """
        query = """
        WITH location_pairs AS (
            SELECT 
                account_id,
                login_lat as lat1,
                login_long as long1,
                LEAD(login_lat) OVER (
                    PARTITION BY account_id ORDER BY event_timestamp
                ) as lat2,
                LEAD(login_long) OVER (
                    PARTITION BY account_id ORDER BY event_timestamp
                ) as long2
            FROM streaming_events
            WHERE event_timestamp < :reference_date
              AND login_lat IS NOT NULL
              AND login_long IS NOT NULL
              {account_filter}
        ),
        distances AS (
            SELECT
                account_id,
                -- Haversine formula (approximate km)
                6371 * 2 * ASIN(SQRT(
                    POWER(SIN(RADIANS(lat2 - lat1) / 2), 2) +
                    COS(RADIANS(lat1)) * COS(RADIANS(lat2)) *
                    POWER(SIN(RADIANS(long2 - long1) / 2), 2)
                )) as distance_km
            FROM location_pairs
            WHERE lat2 IS NOT NULL
        )
        SELECT
            account_id,
            AVG(distance_km) as avg_login_distance_km
        FROM distances
        GROUP BY account_id
        """
        
        params = {"reference_date": self.reference_date}
        formatted_query = query.format(account_filter=self._get_account_filter())
        return self._execute_query(formatted_query, params)


class FinancialTransformer(BaseTransformer):
    """
    Transforms payment data into financial features.
    
    Features:
    - failed_payment_count_{window}
    - avg_monthly_spend
    - payment_method_changes
    - days_since_last_payment
    """
    
    name = "financial"
    
    def transform(self) -> pd.DataFrame:
        features_list = []
        
        # Window-based payment failure counts
        for window in self.time_windows:
            failure_features = self._compute_failure_counts(window)
            failure_features = failure_features.rename(
                columns={
                    col: f"{col}_{window.name}" 
                    for col in failure_features.columns 
                    if col != "account_id"
                }
            )
            features_list.append(failure_features)
        
        # Lifetime financial features
        lifetime_features = self._compute_lifetime_features()
        features_list.append(lifetime_features)
        
        result = features_list[0]
        for df in features_list[1:]:
            result = result.merge(df, on="account_id", how="outer")
        
        return result.set_index("account_id")
    
    def _compute_failure_counts(self, window: TimeWindow) -> pd.DataFrame:
        """Compute payment failure counts for a time window."""
        query = """
        SELECT
            account_id,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_payment_count
        FROM payments
        WHERE payment_date >= :start_date
          AND payment_date < :end_date
          {account_filter}
        GROUP BY account_id
        """
        
        start_date = self.reference_date - window.timedelta
        params = {
            "start_date": start_date,
            "end_date": self.reference_date,
        }
        
        formatted_query = query.format(account_filter=self._get_account_filter())
        return self._execute_query(formatted_query, params)
    
    def _compute_lifetime_features(self) -> pd.DataFrame:
        """Compute lifetime financial features."""
        query = """
        WITH payment_stats AS (
            SELECT
                account_id,
                COUNT(DISTINCT payment_method) as unique_methods,
                AVG(CASE WHEN status = 'success' THEN amount END) as avg_payment,
                MAX(CASE WHEN status = 'success' THEN payment_date END) as last_payment_date,
                MIN(payment_date) as first_payment_date,
                COUNT(DISTINCT DATE_TRUNC('month', payment_date)) as months_with_payments
            FROM payments
            WHERE payment_date < :reference_date
              {account_filter}
            GROUP BY account_id
        )
        SELECT
            account_id,
            unique_methods - 1 as payment_method_changes,
            avg_payment as avg_monthly_spend,
            :reference_date - last_payment_date as days_since_last_payment
        FROM payment_stats
        """
        
        params = {"reference_date": self.reference_date}
        formatted_query = query.format(account_filter=self._get_account_filter())
        return self._execute_query(formatted_query, params)


class SupportTransformer(BaseTransformer):
    """
    Transforms support ticket data into support features.
    
    Features:
    - ticket_count_{window}
    - has_billing_complaint
    - avg_resolution_hours
    - open_ticket_count
    """
    
    name = "support"
    
    def transform(self) -> pd.DataFrame:
        features_list = []
        
        # Window-based ticket counts
        for window in self.time_windows:
            ticket_features = self._compute_ticket_counts(window)
            ticket_features = ticket_features.rename(
                columns={
                    col: f"{col}_{window.name}" 
                    for col in ticket_features.columns 
                    if col != "account_id"
                }
            )
            features_list.append(ticket_features)
        
        # Lifetime support features
        lifetime_features = self._compute_lifetime_features()
        features_list.append(lifetime_features)
        
        result = features_list[0]
        for df in features_list[1:]:
            result = result.merge(df, on="account_id", how="outer")
        
        return result.set_index("account_id")
    
    def _compute_ticket_counts(self, window: TimeWindow) -> pd.DataFrame:
        """Compute ticket counts for a time window."""
        query = """
        SELECT
            account_id,
            COUNT(*) as ticket_count
        FROM support_tickets
        WHERE created_at >= :start_date
          AND created_at < :end_date
          {account_filter}
        GROUP BY account_id
        """
        
        start_date = self.reference_date - window.timedelta
        params = {
            "start_date": start_date,
            "end_date": self.reference_date,
        }
        
        formatted_query = query.format(account_filter=self._get_account_filter())
        return self._execute_query(formatted_query, params)
    
    def _compute_lifetime_features(self) -> pd.DataFrame:
        """Compute lifetime support features."""
        query = """
        SELECT
            account_id,
            MAX(CASE WHEN category ILIKE '%billing%' THEN 1 ELSE 0 END) as has_billing_complaint,
            AVG(
                EXTRACT(EPOCH FROM (resolved_at - created_at)) / 3600
            ) as avg_resolution_hours,
            SUM(CASE WHEN resolved_at IS NULL THEN 1 ELSE 0 END) as open_ticket_count
        FROM support_tickets
        WHERE created_at < :reference_date
          {account_filter}
        GROUP BY account_id
        """
        
        params = {"reference_date": self.reference_date}
        formatted_query = query.format(account_filter=self._get_account_filter())
        return self._execute_query(formatted_query, params)


class DemographicTransformer(BaseTransformer):
    """
    Transforms account and subscription data into demographic features.
    
    Features:
    - account_tenure_days
    - plan_tier_encoded
    - has_plan_upgrade/downgrade
    - parental_control_enabled
    - has_app_rating, app_rating_value
    - watchlist_size
    - age
    - signup_month, signup_dayofweek
    """
    
    name = "demographic"
    
    PLAN_TIER_MAP = {
        "Regular": 1,
        "Premium": 2,
        "Premium-Multi-Screen": 3,
    }
    
    def transform(self) -> pd.DataFrame:
        query = """
        SELECT
            a.account_id,
            :reference_date - a.signup_date as account_tenure_days,
            s.plan_type,
            CASE 
                WHEN s.previous_plan IS NOT NULL 
                     AND s.plan_type IN ('Premium', 'Premium-Multi-Screen')
                     AND s.previous_plan = 'Regular'
                THEN 1 
                ELSE 0 
            END as has_plan_upgrade,
            CASE 
                WHEN s.previous_plan IS NOT NULL 
                     AND s.plan_type = 'Regular'
                     AND s.previous_plan IN ('Premium', 'Premium-Multi-Screen')
                THEN 1 
                ELSE 0 
            END as has_plan_downgrade,
            s.parental_control_enabled::int as parental_control_enabled,
            CASE WHEN s.app_rating IS NOT NULL THEN 1 ELSE 0 END as has_app_rating,
            s.app_rating as app_rating_value,
            COALESCE(s.watchlist_size, 0) as watchlist_size,
            a.age,
            EXTRACT(MONTH FROM a.signup_date)::int as signup_month,
            EXTRACT(DOW FROM a.signup_date)::int as signup_dayofweek
        FROM accounts a
        JOIN subscriptions s ON a.account_id = s.account_id
        WHERE 1=1
          {account_filter}
        """
        
        params = {"reference_date": self.reference_date}
        formatted_query = query.format(
            account_filter=self._get_account_filter().replace(
                "AND account_id", "AND a.account_id"
            )
        )
        
        df = self._execute_query(formatted_query, params)
        
        # Encode plan tier
        df["plan_tier_encoded"] = df["plan_type"].map(self.PLAN_TIER_MAP).fillna(0)
        df = df.drop(columns=["plan_type"])
        
        return df.set_index("account_id")


class TemporalTransformer(BaseTransformer):
    """
    Transforms viewing patterns into temporal features.
    
    Features:
    - is_weekend_viewer
    - peak_viewing_hour
    """
    
    name = "temporal"
    
    def transform(self) -> pd.DataFrame:
        query = """
        WITH viewing_times AS (
            SELECT 
                account_id,
                EXTRACT(DOW FROM event_timestamp) as day_of_week,
                EXTRACT(HOUR FROM event_timestamp) as hour_of_day,
                watch_duration_minutes
            FROM streaming_events
            WHERE event_timestamp < :reference_date
              {account_filter}
        ),
        weekend_stats AS (
            SELECT
                account_id,
                SUM(CASE WHEN day_of_week IN (0, 6) THEN watch_duration_minutes ELSE 0 END) as weekend_minutes,
                SUM(watch_duration_minutes) as total_minutes
            FROM viewing_times
            GROUP BY account_id
        ),
        hour_mode AS (
            SELECT DISTINCT ON (account_id)
                account_id,
                hour_of_day as peak_viewing_hour
            FROM (
                SELECT 
                    account_id, 
                    hour_of_day, 
                    COUNT(*) as hour_count
                FROM viewing_times
                GROUP BY account_id, hour_of_day
            ) hourly
            ORDER BY account_id, hour_count DESC
        )
        SELECT
            ws.account_id,
            CASE WHEN ws.weekend_minutes::float / NULLIF(ws.total_minutes, 0) > 0.5 
                 THEN 1 ELSE 0 END as is_weekend_viewer,
            hm.peak_viewing_hour
        FROM weekend_stats ws
        JOIN hour_mode hm ON ws.account_id = hm.account_id
        """
        
        params = {"reference_date": self.reference_date}
        formatted_query = query.format(account_filter=self._get_account_filter())
        
        return self._execute_query(formatted_query, params).set_index("account_id")


class TargetTransformer(BaseTransformer):
    """
    Creates the target variable for churn prediction.
    
    Target:
    - churned: 1 if status in ('cancelled', 'payment_failed'), else 0
    """
    
    name = "target"
    
    def transform(self) -> pd.DataFrame:
        query = """
        SELECT
            account_id,
            CASE 
                WHEN status IN ('cancelled', 'payment_failed') THEN 1 
                ELSE 0 
            END as churned,
            status as churn_status,
            cancel_reason
        FROM subscriptions
        WHERE 1=1
          {account_filter}
        """
        
        formatted_query = query.format(account_filter=self._get_account_filter())
        return self._execute_query(formatted_query).set_index("account_id")


# Registry of all transformers
TRANSFORMER_REGISTRY: dict[str, type[BaseTransformer]] = {
    "engagement": EngagementTransformer,
    "behavioral": BehavioralTransformer,
    "financial": FinancialTransformer,
    "support": SupportTransformer,
    "demographic": DemographicTransformer,
    "temporal": TemporalTransformer,
    "target": TargetTransformer,
}
