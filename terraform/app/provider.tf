terraform {
  required_version = ">= 1.0.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "mlops-zoomcamp-tfstate-2025"
    key    = "mlops-zoomcamp/app/terraform.tfstate"
    region = "us-east-1"
    # Note: Use AWS_PROFILE=acg terraform <command> for backend operations
  }
}

# Provider configuration
provider "aws" {
  region  = var.aws_region
  profile = "acg"
}

data "terraform_remote_state" "infra" {
  backend = "s3"
  config = {
    bucket = "mlops-zoomcamp-tfstate-2025"
    key    = "mlops-zoomcamp/infra/terraform.tfstate"
    region = "us-east-1"
  }
}