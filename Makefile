# Makefile for managing Terraform infrastructure and application deployments

# Default target when 'make' is run
.DEFAULT_GOAL := help

export AWS_PROFILE=acg

# ==============================================================================
# HELP
# ==============================================================================
.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  create-tfstate-bucket - Create S3 bucket for Terraform state (run first if not already created)"
	@echo "  infra-plan            - Plan the infrastructure changes"
	@echo "  infra-apply           - Apply the infrastructure changes"
	@echo "  infra-destroy         - Destroy the infrastructure"
	@echo ""
	@echo "  app-plan              - Plan the application deployment changes"
	@echo "  app-apply             - Apply the application deployment changes"
	@echo "  app-destroy           - Destroy the application deployment"
	@echo ""
	@echo "  all-init              - Initialize S3 bucket, infra, and app (in order)"
	@echo "  all-destroy           - Destroy both app and infra (in correct order)"
	@echo ""
	@echo "  prefect-setup         - Run Prefect setup script with EC2 DNS from infra output"
	@echo "  generate-ssh-key      - Generate RSA SSH key pair for EC2"
	@echo "  prefect-server        - Start SSH tunnel for PostgreSQL to EC2 instance"
	@echo ""
	@echo "Testing:"
	@echo "  test                  - Run all unit tests"
	@echo "  test-verbose          - Run all unit tests with verbose output"
	@echo "  test-coverage         - Run unit tests with coverage report"
	@echo "  test-specific         - Run specific test file (usage: make test-specific FILE=test_file.py)"
	@echo "  test-integration      - Run all integration tests"
	@echo "  test-integration-real - Run real data integration tests"
	@echo "  test-integration-specific - Run specific integration test file"
	@echo "  test-all              - Run all tests (unit + integration)"

# ==============================================================================
# TESTING TARGETS
# ==============================================================================
.PHONY: test
test:
	@echo "--> Running all unit tests..."
	@PYTHONPATH=src pytest tests/unit/ -q

.PHONY: test-verbose
test-verbose:
	@echo "--> Running all unit tests with verbose output..."
	@PYTHONPATH=src pytest tests/unit/ -v

.PHONY: test-coverage
test-coverage:
	@echo "--> Running unit tests with coverage report..."
	@PYTHONPATH=src pytest tests/unit/ --cov=src --cov-report=term-missing --cov-report=html

.PHONY: test-specific
test-specific:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: Please specify a test file. Usage: make test-specific FILE=test_file.py"; \
		exit 1; \
	fi
	@echo "--> Running specific test file: $(FILE)..."
	@PYTHONPATH=src pytest tests/unit/$(FILE) -v

.PHONY: test-integration
test-integration:
	@echo "--> Running all integration tests..."
	@PYTHONPATH=src pytest tests/integration/ -v

.PHONY: test-integration-real
test-integration-real:
	@echo "--> Running real data integration tests..."
	@PYTHONPATH=src pytest tests/integration/test_real_data_integration.py tests/integration/test_real_aws_integration.py -v

.PHONY: test-integration-specific
test-integration-specific:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: Please specify a test file. Usage: make test-integration-specific FILE=test_file.py"; \
		exit 1; \
	fi
	@echo "--> Running specific integration test file: $(FILE)..."
	@PYTHONPATH=src pytest tests/integration/$(FILE) -v

.PHONY: test-all
test-all:
	@echo "--> Running all tests (unit + integration)..."
	@PYTHONPATH=src pytest tests/ -v

.PHONY: prefect-setup
prefect-setup: ssh-tunnel
	@echo "--> Getting EC2 public IP from Terraform output..."
	@host_name=$$(terraform -chdir=terraform/infra output -raw ec2_dns_name) && \
	echo "--> Running Prefect setup script with host: $$host_name" && \
	bash src/prefect/setup.sh $$host_name

# ==============================================================================
.PHONY: ssh-tunnel
ssh-tunnel:
	@echo "--> Starting SSH tunnel for PostgreSQL on port 5432 to EC2..."
	@ip_address=$$(terraform -chdir=terraform/infra output -raw ec2_public_ip) && \
	ssh -i terraform/modules/ec2/my-key.pem -L 5432:localhost:5432 ec2-user@$$ip_address

.PHONY: generate-ssh-key
generate-ssh-key:
	@echo "--> Generating RSA SSH key pair..."
	@ssh-keygen -t rsa -b 2048 -f terraform/modules/ec2/my-key.pem -N ''

# INFRASTRUCTURE TARGETS
# ==============================================================================
.PHONY: infra-init
infra-init:
	@echo "--> Initializing infrastructure directory..."
	@terraform -chdir=terraform/infra init

.PHONY: infra-plan
infra-plan: infra-init
	@echo "--> Planning infrastructure changes..."
	@terraform -chdir=terraform/infra plan

.PHONY: infra-apply
infra-apply: infra-init
	@echo "--> Applying infrastructure changes..."
	@terraform -chdir=terraform/infra apply -auto-approve

.PHONY: infra-destroy
infra-destroy: infra-init
	@echo "--> Destroying infrastructure..."
	@terraform -chdir=terraform/infra destroy -auto-approve


# ==============================================================================
# APPLICATION TARGETS
# ==============================================================================
.PHONY: app-init
app-init:
	@echo "--> Initializing application directory..."
	@terraform -chdir=terraform/app init

.PHONY: app-plan
app-plan: app-init
	@echo "--> Planning application changes..."
	@terraform -chdir=terraform/app plan

.PHONY: app-apply
app-apply: app-init
	@echo "--> Applying application changes..."
	@terraform -chdir=terraform/app apply -auto-approve

.PHONY: app-destroy
app-destroy: app-init
	@echo "--> Destroying application..."
	@terraform -chdir=terraform/app destroy -auto-approve


# ==============================================================================
# COMBINED TARGETS
# ==============================================================================
.PHONY: all-init
all-init: create-tfstate-bucket infra-init app-init

.PHONY: all-destroy
all-destroy: app-destroy infra-destroy

