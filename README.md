# Retain: Customer Lifecycle Management Platform

A customer churn prediction and prevention platform for streaming services. This project demonstrates end-to-end ML engineering including data pipelines, model development, and automated interventions.

## 🎯 Project Overview

Retain classifies streaming service accounts likely to cancel and triggers automated email interventions. The platform showcases:

- **Data Engineering**: ETL pipelines, data validation, feature engineering
- **ML Engineering**: Churn prediction models, training pipelines, model registry
- **MLOps**: Scheduled retraining, model monitoring, prediction serving
- **Application**: Dashboard with risk scores and intervention triggers

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/retain.git
cd retain
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 3. Set Up PostgreSQL Database

```bash
# Create the database
createdb retain_dev

# Or via psql
psql -c "CREATE DATABASE retain_dev;"
```

### 4. Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your database credentials
# DATABASE_URL=postgresql://username:password@localhost:5432/retain_dev
```

### 5. Initialize the Database

```bash
# Run schema creation
make db-init

# Load sample data
make db-load
```

### 6. Verify Setup

```bash
# Run tests
make test

# Check database connection
make db-check
```

## 📁 Project Structure

```
retain/
├── src/                    # Source code
│   ├── data/              # Data loading and processing
│   ├── features/          # Feature engineering
│   ├── models/            # ML models
│   └── app/               # Web application
├── sql/                   # SQL scripts
├── data/                  # Data files (gitignored)
├── notebooks/             # Jupyter notebooks
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## 🗄️ Database Schema

| Table | Description |
|-------|-------------|
| `accounts` | Customer account information |
| `subscriptions` | Subscription status and plan details |
| `content_catalog` | Available streaming content |
| `payments` | Payment transaction history |
| `support_tickets` | Customer service interactions |
| `streaming_events` | Playback event logs |

See [Data Dictionary](docs/data_dictionary.md) for complete field descriptions.

## 🔧 Development Commands

```bash
make help          # Show all available commands
make db-init       # Initialize database schema
make db-load       # Load data into database
make db-reset      # Drop and recreate database
make test          # Run test suite
make lint          # Run linters
make format        # Format code
```

## 🧪 Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src

# Specific test file
pytest tests/test_data.py
```

## 📊 Data Generation

To regenerate the synthetic dataset:

```bash
python -m src.data.generate --seed 42 --accounts 60000
```

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and add tests
3. Run `make lint` and `make test`
4. Commit with descriptive messages
5. Push and create a Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.
