# 🏠 Rental Price Prediction - ML Pipeline

A comprehensive **Machine Learning Pipeline** for **rental price prediction** built on **AWS Cloud** with **Infrastructure as Code (IaC)**, **MLflow experiment tracking**, **Prefect orchestration**, and **real-time monitoring**. This project demonstrates MLOps solution for predicting rental prices using real estate data.

## 🎯 Problem Statement

The rental market is highly dynamic with prices varying significantly based on location, property features, and market conditions. This project solves the challenge of:

- **Accurate Price Prediction**: Predicting rental prices based on property characteristics (bedrooms, bathrooms, square footage, location)
- **Model Monitoring**: Detecting data drift and model performance degradation
- **Scalable Infrastructure**: Deploying ML models in a cloud-native, serverless architecture
- **Reproducible ML**: Tracking experiments, model versions, and deployment history

## 🏗️ AWS Cloud Infrastructure & IaC

This project is built entirely on **AWS Cloud** using **Terraform** for Infrastructure as Code (IaC) to ensure reproducible, version-controlled infrastructure deployment.

## 🏗️ Project Structure

```
rental-prediction/
├── src/                          # Source code
│   ├── lambda_service/           # AWS Lambda functions
│   │   ├── lambda_function.py    # Prediction service
│   │   ├── model.py             # Model loading & inference
│   │   ├── Dockerfile           # Container configuration
│   │   └── requirements.txt     # Lambda dependencies
│   └── prefect/                 # Prefect orchestration
│       ├── orchestration.py     # Main ML pipeline
│       └── setup.sh            # EC2 setup script
├── terraform/                   # Infrastructure as Code
│   ├── infra/                  # Core infrastructure
│   │   ├── main.tf            # VPC, RDS, EC2, S3
│   │   └── variables.tf       # Infrastructure variables
│   ├── app/                   # Application infrastructure
│   │   ├── main.tf           # Lambda, Kinesis, ECR
│   │   └── variables.tf      # Application variables
│   └── modules/              # Reusable Terraform modules
│       ├── ec2/             # EC2 instance module
│       ├── lambda/          # Lambda function module
│       ├── rds/            # RDS database module
│       ├── vpc/            # VPC networking module
│       └── kinesis/        # Kinesis streams module
├── tests/                    # Comprehensive testing
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── docker/                 # Container configurations
│   ├── docker-compose.yml  # Monitoring stack
│   ├── config/            # Grafana configurations
│   └── dashboards/        # Monitoring dashboards
├── scripts/               # Utility scripts
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
└── Makefile              # Build and deployment commands
```

## 🚀 Quick Start Guide

### 📋 Prerequisites

- **Python 3.8+**
- **AWS CLI** configured with appropriate permissions
- **Terraform** installed
- **Docker** for local development
- **Git** for version control

### 🛠️ Installation & Setup

#### 1. **Clone Repository**
```bash
git clone https://github.com/venukarnati92/rental-prediction.git
cd rental-prediction
```

#### 2. **Install Dependencies**
```bash
#virtual env setup is recommended
# Production dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

#### 3. **Setup Pre-commit Hooks**
```bash
# Automated code quality checks
./scripts/setup-hooks.sh
```

#### 4. **Configure AWS**

This project uses the acg AWS profile for authentication and deployment. You need to configure your AWS credentials and profile before running the infrastructure deployment.

Create or update your AWS credentials file at `~/.aws/credentials`:

```ini
[acg]
aws_access_key_id     = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
```

Create or update your AWS config file at `~/.aws/config`:

```ini
[profile acg]
region = us-east-1
output = json
```

**Environment Variables (Alternative)**

If you prefer using environment variables instead of the profile, you can set:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

**Note**: The project automatically uses the `acg` profile. The `acg` profile takes precedence over environment variables.

### 🏗️ Infrastructure Deployment

#### 1. **Generate SSH key for EC2 access**
```bash
#Generate SSH key for EC2 access
make generate-ssh-key
```

#### 2. **Create Terraform state bucket(first time only)**
```bash
# Create Terraform state bucket 
make create-tfstate-bucket
```

#### 3. **Deploy Core Infrastructure**
```bash
# Deploy core infrastructure (VPC, RDS, EC2, S3)
make infra-apply
```

### 📊 These are the service infrastructure deploy on EC2

- **MLflow UI**: `http://<EC2-PUBLIC-IP>:5000`
- **Prefect UI**: `http://<EC2-PUBLIC-IP>:4200`
- **Grafana**: `http://<EC2-PUBLIC-IP>:3000` (admin/admin)
- **Adminer**: `http://<EC2-PUBLIC-IP>:8080` (database management)
- **PostgreSQL**: `<EC2-PUBLIC-IP>:5432` (local database)


#### 4. **Access EC2 instance**
```bash
ssh -i my-key.pem ec2-user@<EC2-PUBLIC-IP>
```

### 🏗️ Prefect Server Orchestration

#### 1. **Setup MLflow & Prefect**
```bash
# Setup services on EC2
make prefect-setup
```

#### 2. **Execute Deployment**
```
Naviate to browser http://<<EC2-PUBLIC-IP>>:4200/deployments and execute the deployment. 
This will build the model, upload model to S3 using mlflow and Publish evidently metrics
```

### 📊 Prefect Capabilities

- **Task Retries**: Automatic retry with exponential backoff
- **Monitoring**: Real-time workflow monitoring and alerting
- **Scheduling**: Automated pipeline scheduling and triggering
- **Error Handling**: Robust error handling and recovery
- **Artifacts**: Rich artifact storage and visualization

### Deploy Application Service

#### 1. **Deploy App Infrastructure**
```bash
# Deploy Lambda, Kinesis, ECR
make app-apply
```

#### 2. **AWS CLI to update Lambda function**
```bash
# Get the model location from s3 example s3://<bucket_name>/1/models/m-d07fd6cf5ed6418bbbfc3668f5c95042/artifacts/
make lambda-deploy MODEL_LOCATION=<s3_path>
```

#### 3. **Add records to Kinesis Stream**
```bash
make kinesis-put-record
```

### 📊 View Results

After executing the steps above:

1. **Check Lambda Logs**: Navigate to AWS Console > CloudWatch > Log groups
2. **Find Log Group**: Look for `/aws/lambda/lambda_function_rental_prediction_mlops-zoomcamp`
3. **View Predictions**: You should see rental price predictions in the logs

### 🔍 Expected Output

The Lambda function will process the test data and output something like:
```
Prediction event: {
  'model': 'rental_price_prediction_model',
  'version': '1',
  'prediction': {'price': 1850.75}
}
```

### 📊 Infrastructure Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS Cloud                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   VPC       │  │   Security  │  │   S3        │         │
│  │   (Custom)  │  │   Groups    │  │   Bucket    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    EC2 Instance                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
│  │  │   MLflow    │  │   Prefect   │  │   Grafana   │     │ │
│  │  │   Server    │  │   Server    │  │   (Port     │     │ │
│  │  │   (Port     │  │   (Port     │  │   3000)     │     │ │
│  │  │   5000)     │  │   4200)     │  │             │     │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
│  │  │ PostgreSQL  │  │   Adminer   │  │   Docker    │     │ │
│  │  │   (Port     │  │   (Port     │  │   Compose   │     │ │
│  │  │   5432)     │  │   8080)     │  │   Stack     │     │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   RDS       │  │   ECR       │  │   Lambda    │         │
│  │   PostgreSQL│  │   Registry  │  │   Functions │         │
│  │   (Backend) │  │             │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Kinesis   │  │   CloudWatch│  │   IAM       │         │
│  │   Streams   │  │   Logs      │  │   Roles     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 🏢 Infrastructure Components

#### **Core Infrastructure** (`terraform/infra/`)
- **VPC & Networking**: Custom VPC with public/private subnets across multiple AZs
- **Security Groups**: Fine-grained access control for EC2, RDS, and Lambda services
- **S3 Bucket**: Artifact storage for models, data, and MLflow artifacts
- **RDS PostgreSQL**: Managed database for MLflow metadata and experiment tracking
- **EC2 Instance**: Multi-service host with automated deployment via user script
  - **MLflow Server** (Port 5000): Experiment tracking and model registry
  - **Prefect Server** (Port 4200): Workflow orchestration
  - **Grafana** (Port 3000): Monitoring dashboards
  - **PostgreSQL** (Port 5432): Local metrics database
  - **Adminer** (Port 8080): Database management interface

#### **Application Infrastructure** (`terraform/app/`)
- **ECR Repository**: Container registry for Dockerized ML models
- **Lambda Functions**: Serverless prediction service with auto-scaling
- **Kinesis Streams**: Real-time data streaming for input/output processing
- **IAM Roles & Policies**: Secure access management for all services

### 🚀 Infrastructure Teardown

```bash
make app-destroy
make infra-destroy
```

## ⚡ Prefect Server Orchestration

**Prefect** provides robust workflow orchestration for the ML pipeline with:

### 🎯 Prefect Services on EC2

- **Prefect Server**: Workflow orchestration and scheduling
- **Prefect Agent**: Task execution and monitoring
- **PostgreSQL Backend**: Workflow state and metadata storage


### 📊 Prefect Capabilities

- **Task Retries**: Automatic retry with exponential backoff
- **Monitoring**: Real-time workflow monitoring and alerting
- **Scheduling**: Automated pipeline scheduling and triggering
- **Error Handling**: Robust error handling and recovery
- **Artifacts**: Rich artifact storage and visualization

## 🔬 MLflow Experiment Tracking & Model Registry

**MLflow** is used for comprehensive experiment tracking and model registry, hosted on EC2 with S3 backend storage.

### 🎯 Services Deployed on EC2

The EC2 instance hosts multiple services deployed via Terraform:

#### **MLflow Services**
- **MLflow Tracking Server** (Port 5000): Experiment tracking and metric logging
- **MLflow Model Registry**: Model versioning and lifecycle management
- **S3 Backend**: Artifact storage for models and experiment data
- **RDS PostgreSQL Backend**: Metadata storage for experiments and runs

#### **Prefect Services**
- **Prefect Server** (Port 4200): Workflow orchestration and scheduling
- **Prefect Worker**: Task execution and monitoring
- **PostgreSQL Backend**: Workflow state and metadata storage

#### **Monitoring Services (Docker Compose)**
- **Grafana** (Port 3000): Visualization and monitoring dashboards
- **PostgreSQL** (Port 5432): Local database for metrics storage
- **Adminer** (Port 8080): Database management interface

### 📈 Experiment Tracking Features

```python
# MLflow integration in orchestration
mlflow.set_tracking_uri(f"http://{host_name}:5000")
mlflow.set_experiment("rental-prediction")

with mlflow.start_run():
    mlflow.set_tag("model", "xgboost")
    mlflow.log_params(xgb_params)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(pipeline, artifact_path="model")
```

### 🔍 Model Registry Capabilities

- **Model Versioning**: Track model versions with metadata
- **Model Lineage**: Link models to specific experiments and data
- **Model Deployment**: Manage model deployment stages (Staging, Production)
- **Artifact Storage**: Store model artifacts in S3 with versioning
- **Performance Tracking**: Monitor model performance over time

## 🐳 Containerized Model Deployment

The ML model is **containerized** using **Docker** and deployed to **AWS ECR** for scalable, reproducible deployments.

### 🚀 Deployment Pipeline

1. **Model Training**: MLflow tracks model training and artifacts
2. **Container Build**: Docker image built with trained model
3. **ECR Push**: Image pushed to AWS ECR registry
4. **Lambda Deployment**: Serverless function updated with new model
5. **Traffic Routing**: Kinesis streams route traffic to new model


## 📊 Comprehensive Model Monitoring

**Real-time monitoring** with **conditional workflows** and **automated alerts** when metrics thresholds are violated.

### 🔍 Monitoring Stack

#### **Evidently AI** - Data Drift Detection
- **Column Drift**: Monitor individual feature drift
- **Dataset Drift**: Overall dataset distribution changes
- **Missing Values**: Track data quality degradation
- **Quantile Monitoring**: Price distribution changes

#### **Grafana** - Visualization & Dashboards
- **Real-time Metrics**: Live model performance metrics
- **Data Drift Alerts**: Visual alerts for drift detection
- **Performance Trends**: Historical performance tracking
- **Custom Dashboards**: Tailored monitoring views

#### **PostgreSQL** - Metrics Storage
- **Structured Storage**: Organized metrics storage
- **Historical Data**: Long-term performance tracking
- **Query Capabilities**: Complex metric analysis

### 🚨 Conditional Workflows

When monitoring thresholds are violated, the system triggers:

1. **Alert Generation**: Immediate notification via CloudWatch
2. **Model Retraining**: Automatic retraining pipeline initiation
3. **Debug Dashboard**: Enhanced monitoring dashboard activation
4. **Model Rollback**: Fallback to previous stable model version
5. **Performance Analysis**: Detailed performance investigation

### 📈 Monitoring Metrics

```python
# Key monitoring metrics
- Prediction Drift: Model prediction distribution changes
- Data Drift: Input feature distribution changes
- Missing Values: Data quality degradation
- Price Quantiles: Rental price distribution changes
- Model Performance: RMSE, MAE tracking over time
```

### 🧪 Testing

#### **Unit Tests**
```bash
make test                    # Run all unit tests
make test-verbose           # Run with verbose output
make test-coverage          # Run with coverage report
```

#### **Integration Tests**
```bash
make test-integration       # Run all integration tests
make test-integration-real  # Run real data integration tests
make test-all              # Run all tests
```

### ⚠️ Important Notes

- **Costs**: Running this infrastructure will incur AWS charges. Remember to destroy resources when done.
- **Cleanup**: Use `make app-destroy` and `make infra-destroy` to clean up resources.

## 🛠️ Available Commands

```bash
make help
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'feat: add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Code Quality Standards**
- All code must pass pre-commit hooks
- Unit tests required for new functionality
- Follow PEP 8 style guidelines
- Use conventional commit messages
- Integration tests for new features

---

**Built with ❤️ using AWS, Terraform, MLflow, Prefect, and modern MLOps practices** 
