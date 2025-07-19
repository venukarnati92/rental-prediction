# Rental-Prediction

A comprehensive machine learning pipeline for rental price prediction using AWS services, Prefect orchestration, and modern DevOps practices.

## 🚀 Features

- **ML Pipeline**: Automated data processing, feature engineering, and model training
- **AWS Infrastructure**: Serverless Lambda functions, Kinesis streams, RDS database
- **Monitoring**: Data drift detection with Evidently and Grafana dashboards
- **Testing**: Comprehensive unit tests with pytest
- **CI/CD**: Pre-commit hooks for code quality and automated testing

## 📋 Prerequisites

- Python 3.8+
- AWS CLI configured
- Terraform
- Docker (for local development)

## 🛠️ Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd rental-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Setup Pre-commit Hooks (Recommended)
```bash
# Run the setup script
./scripts/setup-hooks.sh

# Or manually install pre-commit
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
```

## 🧪 Testing

### Run All Tests
```bash
make test
```

### Run Tests with Verbose Output
```bash
make test-verbose
```

### Run Tests with Coverage
```bash
make test-coverage
```

### Run Specific Test File
```bash
make test-specific FILE=test_orchestration.py
```

## 🔧 Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. The hooks run automatically on every commit and include:

### Pre-commit Checks:
- **Python Syntax**: Validates Python syntax
- **Code Formatting**: Black formatter with 88-character line length
- **Import Sorting**: isort for consistent import organization
- **Linting**: flake8 for style and error checking
- **Type Checking**: mypy for static type analysis
- **Security**: bandit for security vulnerability scanning
- **Unit Tests**: Runs all unit tests before commit

### Commit Message Validation:
- **Non-empty**: Commit messages cannot be empty
- **Minimum Length**: At least 10 characters
- **Format**: Suggests conventional commit format
- **Style**: Warns about common bad practices

### Manual Hook Execution:
```bash
# Run all hooks on all files
pre-commit run --all-files

# Run hooks on staged files only
pre-commit run

# Update hook versions
pre-commit autoupdate
```

## 🏗️ Infrastructure

### Deploy Infrastructure
```bash
# Create Terraform state bucket (first time only)
make create-tfstate-bucket

# Deploy infrastructure
make infra-apply

# Deploy application
make app-apply
```

### Destroy Infrastructure
```bash
# Destroy application first
make app-destroy

# Then destroy infrastructure
make infra-destroy
```

## 📊 Monitoring

### Start Monitoring Stack
```bash
# Start Grafana and monitoring services
docker-compose -f docker/docker-compose.yml up -d
```

### Access Dashboards
- **Grafana**: http://localhost:3000 (admin/admin)
- **Data Drift Dashboard**: Available in Grafana

## 🏃‍♂️ Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code
   - Add tests
   - Update documentation

3. **Test Your Changes**
   ```bash
   make test
   pre-commit run --all-files
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📁 Project Structure

```
rental-prediction/
├── src/                    # Source code
│   ├── lambda_service/     # AWS Lambda functions
│   └── prefect/           # Prefect orchestration
├── tests/                 # Test files
│   └── unit/             # Unit tests
├── terraform/             # Infrastructure as Code
│   ├── infra/            # Infrastructure resources
│   └── app/              # Application resources
├── docker/               # Docker configurations
├── scripts/              # Utility scripts
└── docs/                 # Documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 
