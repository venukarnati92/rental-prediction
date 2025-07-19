# Rental-Prediction

A comprehensive machine learning pipeline for rental price prediction using AWS services, Prefect orchestration, and modern DevOps practices.

## 🚀 Features

- **ML Pipeline**: Automated data processing, feature engineering, and model training
- **AWS Infrastructure**: Serverless Lambda functions, Kinesis streams, RDS database
- **Monitoring**: Data drift detection with Evidently and Grafana dashboards
- **Testing**: Comprehensive unit tests with pytest
- **CI/CD**: Pre-commit hooks for code quality and automated testing
- **Code Quality**: Automated linting, formatting, and type checking

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

### Unit Tests

Run unit tests with:
```bash
make test                    # Run all unit tests
make test-verbose           # Run with verbose output
make test-coverage          # Run with coverage report
make test-specific FILE=test_file.py  # Run specific test file
```

### Integration Tests

Integration tests verify end-to-end functionality:

```bash
make test-integration       # Run all integration tests
make test-integration-real  # Run real data integration tests
make test-integration-specific FILE=test_file.py  # Run specific integration test
make test-all              # Run all tests (unit + integration)
```

### Available Test Files

#### Unit Tests
- `tests/unit/test_orchestration.py` - Tests for Prefect orchestration pipeline
- `tests/unit/test_lambda_function.py` - Tests for AWS Lambda handler
- `tests/unit/test_model.py` - Tests for ML model service and utilities

#### Integration Tests
- `tests/integration/test_pipeline_integration.py` - Complete pipeline integration tests
- `tests/integration/test_aws_integration.py` - AWS service integration tests (mocked)
- `tests/integration/test_monitoring_integration.py` - Monitoring and observability tests
- `tests/integration/test_real_data_integration.py` - Real data integration tests
- `tests/integration/test_real_aws_integration.py` - Real AWS service tests (when credentials available)

### Integration Test Types

1. **Mocked Integration Tests**: Use mocked services for fast, reliable testing
2. **Real Data Integration Tests**: Use realistic data generation with real ML models
3. **Real AWS Integration Tests**: Use actual AWS services when credentials are available
4. **Monitoring Integration Tests**: Test Evidently, Grafana, and alerting systems

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
│   │   ├── lambda_function.py
│   │   ├── model.py
│   │   └── requirements.txt
│   └── prefect/           # Prefect orchestration
│       ├── orchestration.py
│       └── setup.sh
├── tests/                 # Test files
│   ├── unit/             # Unit tests
│   │   ├── test_orchestration.py
│   │   ├── test_lambda_function.py
│   │   └── test_model.py
│   └── integration/      # Integration tests
│       ├── test_pipeline_integration.py
│       ├── test_aws_integration.py
│       ├── test_monitoring_integration.py
│       ├── test_real_data_integration.py
│       └── test_real_aws_integration.py
├── terraform/             # Infrastructure as Code
│   ├── infra/            # Infrastructure resources
│   ├── app/              # Application resources
│   └── modules/          # Reusable Terraform modules
├── docker/               # Docker configurations
│   ├── docker-compose.yml
│   ├── config/           # Grafana configurations
│   └── dashboards/       # Monitoring dashboards
├── scripts/              # Utility scripts
│   └── setup-hooks.sh   # Pre-commit setup script
├── .pre-commit-config.yaml  # Pre-commit configuration
├── .flake8              # Flake8 linting configuration
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
└── Makefile             # Build and deployment commands
```

## 🛠️ Available Make Commands

```bash
# Testing
make test                 # Run all unit tests
make test-verbose         # Run tests with verbose output
make test-coverage        # Run tests with coverage report
make test-specific FILE=test_file.py  # Run specific test file
make test-integration     # Run all integration tests
make test-integration-real # Run real data integration tests
make test-all             # Run all tests (unit + integration)

# Infrastructure
make infra-plan          # Plan infrastructure changes
make infra-apply         # Apply infrastructure changes
make infra-destroy       # Destroy infrastructure
make app-plan            # Plan application changes
make app-apply           # Apply application changes
make app-destroy         # Destroy application

# Combined operations
make all-init            # Initialize everything
make all-destroy         # Destroy everything

# Prefect
make prefect-setup       # Setup Prefect with EC2
make generate-ssh-key    # Generate SSH key for EC2
make ssh-tunnel          # Start SSH tunnel for PostgreSQL
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Quality Standards
- All code must pass pre-commit hooks
- Unit tests must be written for new functionality
- Follow PEP 8 style guidelines
- Use conventional commit messages

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 
