# Retain Data Dictionary

## Overview

This dataset simulates a streaming service's operational data for customer churn prediction.

- **Date Range:** January 2024 - December 2025 (24 months)
- **Accounts:** ~60,000
- **Seed:** 42 (reproducible)

## Tables

### accounts
Customer account information.

| Column | Type | Description |
|--------|------|-------------|
| account_id | VARCHAR(12) | Primary key (ACC_XXXXXXXX) |
| email | VARCHAR | Unique email address |
| signup_date | DATE | Account creation date |
| country | VARCHAR(2) | ISO country code (US only) |
| age | INTEGER | Age at signup (18-75) |
| gender | VARCHAR | Male / Female / Other / null |

### subscriptions
Current subscription state for each account.

| Column | Type | Description |
|--------|------|-------------|
| subscription_id | VARCHAR(14) | Primary key |
| account_id | VARCHAR(12) | Foreign key → accounts |
| plan_type | VARCHAR | Regular / Premium / Premium-Multi-Screen |
| start_date | DATE | Subscription start (same as signup) |
| end_date | DATE | Null if active; otherwise churn date |
| status | VARCHAR | active / cancelled / payment_failed |
| cancel_reason | VARCHAR | Null unless cancelled |
| previous_plan | VARCHAR | Previous plan if changed, else null |
| parental_control_enabled | BOOLEAN | Family account indicator |
| watchlist_size | INTEGER | Items in watchlist (40% have 0) |
| app_rating | INTEGER | 1-5 stars (15-20% of accounts) |
| app_review_text | TEXT | Review text (4-6% of accounts) |

### content_catalog
Available content library.

| Column | Type | Description |
|--------|------|-------------|
| content_id | VARCHAR(10) | Primary key |
| title | VARCHAR | Content title |
| genre | VARCHAR | Drama, Comedy, Action, etc. |
| content_type | VARCHAR | movie / series |
| release_year | INTEGER | Year released |
| duration_minutes | INTEGER | Runtime |

### payments
Monthly payment transactions.

| Column | Type | Description |
|--------|------|-------------|
| payment_id | VARCHAR(14) | Primary key |
| account_id | VARCHAR(12) | Foreign key → accounts |
| payment_date | DATE | Transaction date |
| amount | DECIMAL | Charge amount |
| currency | VARCHAR(3) | USD |
| payment_method | VARCHAR | credit_card / debit_card / paypal / bank_transfer |
| status | VARCHAR | success / failed |
| failure_reason | VARCHAR | Null if success |

### support_tickets
Customer service interactions.

| Column | Type | Description |
|--------|------|-------------|
| ticket_id | VARCHAR(12) | Primary key |
| account_id | VARCHAR(12) | Foreign key → accounts |
| created_at | TIMESTAMP | Ticket creation time |
| resolved_at | TIMESTAMP | Resolution time (null if open) |
| category | VARCHAR | 20 categories (see below) |
| priority | VARCHAR | low / medium / high |

**Support Categories:**
Cancellation Request, Account Access, Buffering Issues, Video Quality, Error Code, 
Payment Failure, Login Problems, Feature Request, Audio Sync, Black Screen, 
Billing Dispute, Parental Controls, App Crash, Streaming Lag, Content Not Available, 
Subtitle Issues, Device Compatibility, Playback Error, Profile Issues, Download Failed

### streaming_events
Playback event logs.

| Column | Type | Description |
|--------|------|-------------|
| event_id | VARCHAR(16) | Primary key |
| account_id | VARCHAR(12) | Foreign key → accounts |
| event_timestamp | TIMESTAMP | Playback start time |
| content_id | VARCHAR(10) | Foreign key → content_catalog |
| device_type | VARCHAR | mobile / tablet / desktop / streaming_stick / smart_tv |
| watch_duration_minutes | DECIMAL | Actual time watched |
| login_lat | DECIMAL | Latitude (US metros) |
| login_long | DECIMAL | Longitude (US metros) |

## Churn Definitions

1. **Explicit Cancellation:** `status = 'cancelled'`
2. **Payment Failure:** `status = 'payment_failed'` (2 consecutive failures)

## Key Relationships

```
accounts (1) ─── (1) subscriptions
accounts (1) ─── (N) payments
accounts (1) ─── (N) support_tickets
accounts (1) ─── (N) streaming_events
content_catalog (1) ─── (N) streaming_events
```

## Injected Trends

| Factor | Churn Correlation |
|--------|-------------------|
| Low engagement (<2 hrs/week) | Strong positive |
| Payment failures | Strong positive |
| Billing complaints | Moderate positive |
| Short tenure (<90 days) | Moderate positive |
| Regular plan tier | Weak positive |
| Parental controls enabled | Moderate negative |
| High app rating (4-5) | Moderate negative |

## Seasonal Patterns

Monthly viewing weights (applied to streaming events):
- Winter peak: December (1.35), January (1.30)
- Summer trough: July (0.80), June (0.85)
- Weekend + Friday boost: 15-30% higher viewing
