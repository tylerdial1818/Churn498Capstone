"""
Feature validation for quality assurance.

This module provides validation checks for generated features,
ensuring data quality before model training or inference.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of feature validation."""
    passed: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)


class FeatureValidator:
    """
    Validates feature quality and data integrity.
    
    Checks include:
    - Missing value thresholds
    - Value range validation
    - Feature correlation checks
    - Target leakage detection
    - Cardinality checks for categorical features
    """
    
    # Maximum allowed missing value ratio per column
    MAX_MISSING_RATIO = 0.3
    
    # Minimum required samples
    MIN_SAMPLES = 100
    
    # Suspicious correlation threshold (potential leakage)
    LEAKAGE_CORRELATION_THRESHOLD = 0.95
    
    def __init__(
        self,
        max_missing_ratio: float = MAX_MISSING_RATIO,
        min_samples: int = MIN_SAMPLES,
    ):
        self.max_missing_ratio = max_missing_ratio
        self.min_samples = min_samples
    
    def validate(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame | None = None,
    ) -> ValidationResult:
        """
        Run all validation checks on features.
        
        Args:
            features: Feature DataFrame to validate
            target: Optional target DataFrame for leakage checks
        
        Returns:
            ValidationResult with pass/fail status and details
        """
        issues: list[str] = []
        warnings: list[str] = []
        statistics: dict[str, Any] = {}
        
        # Basic statistics
        statistics["n_samples"] = len(features)
        statistics["n_features"] = len(features.columns)
        
        # Check minimum samples
        if len(features) < self.min_samples:
            issues.append(
                f"Insufficient samples: {len(features)} < {self.min_samples}"
            )
        
        # Check missing values
        missing_issues, missing_stats = self._check_missing_values(features)
        issues.extend(missing_issues)
        statistics["missing_values"] = missing_stats
        
        # Check value ranges
        range_warnings = self._check_value_ranges(features)
        warnings.extend(range_warnings)
        
        # Check for infinite values
        inf_issues = self._check_infinite_values(features)
        issues.extend(inf_issues)
        
        # Check for constant features
        constant_warnings = self._check_constant_features(features)
        warnings.extend(constant_warnings)
        
        # Check for high cardinality categoricals
        cardinality_warnings = self._check_cardinality(features)
        warnings.extend(cardinality_warnings)
        
        # Check for target leakage if target provided
        if target is not None and "churned" in target.columns:
            leakage_issues = self._check_target_leakage(features, target["churned"])
            issues.extend(leakage_issues)
            
            # Target distribution
            statistics["target_distribution"] = target["churned"].value_counts().to_dict()
        
        # Compute feature statistics summary
        statistics["feature_summary"] = self._compute_feature_summary(features)
        
        passed = len(issues) == 0
        
        return ValidationResult(
            passed=passed,
            issues=issues,
            warnings=warnings,
            statistics=statistics,
        )
    
    def _check_missing_values(
        self, 
        df: pd.DataFrame,
    ) -> tuple[list[str], dict[str, float]]:
        """Check for excessive missing values."""
        issues = []
        missing_ratios = {}
        
        for col in df.columns:
            ratio = df[col].isna().mean()
            missing_ratios[col] = round(ratio, 4)
            
            if ratio > self.max_missing_ratio:
                issues.append(
                    f"High missing ratio in '{col}': {ratio:.1%} > {self.max_missing_ratio:.0%}"
                )
        
        return issues, missing_ratios
    
    def _check_value_ranges(self, df: pd.DataFrame) -> list[str]:
        """Check for values outside expected ranges."""
        warnings = []
        
        # Define expected ranges for known features
        range_checks = {
            "age": (0, 120),
            "account_tenure_days": (0, 10000),
            "avg_watch_hours_per_week": (0, 168),  # Max hours in a week
            "genre_diversity_score": (0, 1),
            "binge_session_ratio": (0, 1),
            "plan_tier_encoded": (1, 3),
            "app_rating_value": (1, 5),
            "signup_month": (1, 12),
            "signup_dayofweek": (0, 6),
            "peak_viewing_hour": (0, 23),
        }
        
        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                
                if pd.notna(col_min) and col_min < min_val:
                    warnings.append(
                        f"'{col}' has values below expected minimum: {col_min} < {min_val}"
                    )
                if pd.notna(col_max) and col_max > max_val:
                    warnings.append(
                        f"'{col}' has values above expected maximum: {col_max} > {max_val}"
                    )
        
        return warnings
    
    def _check_infinite_values(self, df: pd.DataFrame) -> list[str]:
        """Check for infinite values in numeric columns."""
        issues = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                issues.append(f"'{col}' contains {inf_count} infinite values")
        
        return issues
    
    def _check_constant_features(self, df: pd.DataFrame) -> list[str]:
        """Check for features with zero variance."""
        warnings = []
        
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 1:
                warnings.append(f"'{col}' is constant (zero variance)")
        
        return warnings
    
    def _check_cardinality(self, df: pd.DataFrame) -> list[str]:
        """Check for high cardinality in categorical features."""
        warnings = []
        max_cardinality = 50
        
        # Identify likely categorical columns
        object_cols = df.select_dtypes(include=["object", "category"]).columns
        
        for col in object_cols:
            cardinality = df[col].nunique()
            if cardinality > max_cardinality:
                warnings.append(
                    f"High cardinality in '{col}': {cardinality} unique values"
                )
        
        return warnings
    
    def _check_target_leakage(
        self, 
        features: pd.DataFrame, 
        target: pd.Series,
    ) -> list[str]:
        """Check for suspiciously high correlation with target."""
        issues = []
        
        # Align indices
        common_idx = features.index.intersection(target.index)
        features_aligned = features.loc[common_idx]
        target_aligned = target.loc[common_idx]
        
        numeric_cols = features_aligned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                corr = features_aligned[col].corr(target_aligned)
                if pd.notna(corr) and abs(corr) > self.LEAKAGE_CORRELATION_THRESHOLD:
                    issues.append(
                        f"Potential target leakage in '{col}': correlation = {corr:.3f}"
                    )
            except Exception:
                pass  # Skip columns that can't compute correlation
        
        return issues
    
    def _compute_feature_summary(self, df: pd.DataFrame) -> dict[str, dict]:
        """Compute summary statistics for each feature."""
        summary = {}
        
        for col in df.columns:
            col_summary = {
                "dtype": str(df[col].dtype),
                "missing_pct": round(df[col].isna().mean() * 100, 2),
                "unique_count": int(df[col].nunique()),
            }
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                col_summary.update({
                    "mean": round(float(df[col].mean()), 4) if pd.notna(df[col].mean()) else None,
                    "std": round(float(df[col].std()), 4) if pd.notna(df[col].std()) else None,
                    "min": round(float(df[col].min()), 4) if pd.notna(df[col].min()) else None,
                    "max": round(float(df[col].max()), 4) if pd.notna(df[col].max()) else None,
                })
            elif pd.api.types.is_bool_dtype(df[col]):
                col_summary.update({
                    "true_count": int(df[col].sum()),
                    "true_ratio": round(float(df[col].mean()), 4) if pd.notna(df[col].mean()) else None,
                })
            
            summary[col] = col_summary
        
        return summary


def validate_feature_schema(
    df: pd.DataFrame,
    expected_columns: list[str],
    strict: bool = False,
) -> ValidationResult:
    """
    Validate that DataFrame has expected columns.
    
    Args:
        df: DataFrame to validate
        expected_columns: List of required column names
        strict: If True, extra columns are also flagged
    
    Returns:
        ValidationResult
    """
    issues = []
    warnings = []
    
    # Check for missing columns
    missing = set(expected_columns) - set(df.columns)
    if missing:
        issues.append(f"Missing expected columns: {sorted(missing)}")
    
    # Check for extra columns (warning only unless strict)
    extra = set(df.columns) - set(expected_columns)
    if extra:
        msg = f"Unexpected columns present: {sorted(extra)}"
        if strict:
            issues.append(msg)
        else:
            warnings.append(msg)
    
    return ValidationResult(
        passed=len(issues) == 0,
        issues=issues,
        warnings=warnings,
        statistics={
            "expected_columns": len(expected_columns),
            "actual_columns": len(df.columns),
            "missing_columns": len(missing),
            "extra_columns": len(extra),
        },
    )
