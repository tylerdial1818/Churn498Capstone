# Development Setup Guide

This guide walks through setting up the Retain project for local development.

## Prerequisites

Before starting, ensure you have installed:

- **Python 3.11+**: [Download Python](https://www.python.org/downloads/)
- **PostgreSQL 14+**: [Download PostgreSQL](https://www.postgresql.org/download/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **VS Code** (recommended): [Download VS Code](https://code.visualstudio.com/)

### Verify Prerequisites

```bash
python --version    # Should be 3.11+
psql --version      # Should be 14+
git --version       # Any recent version
```

---

## Step 1: Clone the Repository

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/retain.git
cd retain

# If starting fresh (no existing repo), initialize git
git init
git add .
git commit -m "Initial commit: project structure"
```

---

## Step 2: Set Up Python Environment

### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# macOS/Linux:
source .venv/bin/activate

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Windows (CMD):
.venv\Scripts\activate.bat
```

### Option B: Using conda

```bash
conda create -n retain python=3.11
conda activate retain
```

### Install Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

---

## Step 3: Set Up PostgreSQL

### Install PostgreSQL

**macOS (Homebrew):**
```bash
brew install postgresql@14
brew services start postgresql@14
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**Windows:**
- Download installer from [postgresql.org](https://www.postgresql.org/download/windows/)
- Run installer and note the password you set for the `postgres` user

### Create the Database

```bash
# Connect to PostgreSQL
psql -U postgres

# In psql, create the database
CREATE DATABASE retain_dev;

# Create a user (optional, but recommended)
CREATE USER retain_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE retain_dev TO retain_user;

# Exit psql
\q
```

---

## Step 4: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env
```

Edit `.env` with your database credentials:

```bash
# Option 1: Full connection string
DATABASE_URL=postgresql://retain_user:your_secure_password@localhost:5432/retain_dev

# Option 2: Individual components (if not using DATABASE_URL)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=retain_dev
DB_USER=retain_user
DB_PASSWORD=your_secure_password
```

---

## Step 5: Initialize the Database

```bash
# Create tables
make db-init

# Load the CSV data
# First, copy your CSV files to data/raw/
cp path/to/retain_*.csv data/raw/

# Then load them into the database
make db-load

# Verify everything worked
make db-check
```

---

## Step 6: VS Code Setup

### Install Recommended Extensions

1. Open the project in VS Code: `code .`
2. VS Code will prompt you to install recommended extensions
3. Click "Install All"

Or install manually:
- **Python** (ms-python.python)
- **Pylance** (ms-python.vscode-pylance)
- **Ruff** (charliermarsh.ruff)
- **SQLTools** (mtxr.sqltools)
- **SQLTools PostgreSQL** (mtxr.sqltools-driver-pg)

### Configure SQLTools (Database Explorer)

1. Open Command Palette (Cmd/Ctrl + Shift + P)
2. Type "SQLTools: Add New Connection"
3. Select "PostgreSQL"
4. Fill in connection details:
   - **Connection Name**: Retain Dev
   - **Server**: localhost
   - **Port**: 5432
   - **Database**: retain_dev
   - **Username**: retain_user
   - **Password**: your_secure_password
5. Click "Test Connection" then "Save"

Now you can browse tables directly in VS Code!

### Select Python Interpreter

1. Open Command Palette (Cmd/Ctrl + Shift + P)
2. Type "Python: Select Interpreter"
3. Choose the `.venv` interpreter

---

## Step 7: Verify Setup

Run the test suite to verify everything is working:

```bash
# Run all tests
make test

# Check database connection
make db-check
```

You should see:
```
Checking database connection...
✓ Connection successful!
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Table              ┃      Rows ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ accounts           │    60,000 │
│ subscriptions      │    60,000 │
│ content_catalog    │     2,000 │
│ payments           │   605,502 │
│ support_tickets    │    59,870 │
│ streaming_events   │ 1,898,463 │
├────────────────────┼───────────┤
│ Total              │ 2,685,835 │
└────────────────────┴───────────┘
```

---

## GitHub Setup

### Create New Repository

1. Go to [github.com/new](https://github.com/new)
2. Name: `retain`
3. Description: "Customer lifecycle management platform for streaming services"
4. Visibility: Private (or Public)
5. **Don't** initialize with README (we already have one)
6. Click "Create repository"

### Push to GitHub

```bash
# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/retain.git

# Push initial commit
git branch -M main
git push -u origin main
```

### Set Up Branch Protection (Optional but Recommended)

1. Go to Repository → Settings → Branches
2. Click "Add branch protection rule"
3. Branch name pattern: `main`
4. Enable:
   - ✅ Require a pull request before merging
   - ✅ Require status checks to pass (after adding CI)

---

## Common Issues

### "psql: command not found"

PostgreSQL isn't in your PATH. 

**macOS**: `export PATH="/usr/local/opt/postgresql@14/bin:$PATH"`

**Windows**: Add `C:\Program Files\PostgreSQL\14\bin` to your PATH

### "FATAL: password authentication failed"

Check your `.env` file has the correct password. Try connecting directly:
```bash
psql -U postgres -d retain_dev
```

### "relation does not exist"

Tables haven't been created. Run:
```bash
make db-init
```

### Import errors when running Python

Ensure your virtual environment is activated and package is installed:
```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Next Steps

Once setup is complete, you can:

1. **Explore the data** in Jupyter:
   ```bash
   jupyter lab notebooks/
   ```

2. **Run the test suite**:
   ```bash
   make test
   ```

3. **Start building features** in `src/features/`

4. **Check code quality**:
   ```bash
   make lint
   make format
   ```
