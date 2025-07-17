variable "aws_region" {
  type = string
  default = "us-east-1"
  description = "AWS region for the backend S3 bucket"
}

variable "project_id" {
  type = string
  description = "Project ID "
  default = "mlops-zoomcamp"
}

variable "rds_username" {
  type = string
  description = "RDS username"
  default = "mlflow"
}

variable "rds_password" {
  type = string
  description = "RDS password"
  default = "mlops-zoomcamp"
}

variable "rds_db_name" {
  type = string
  description = "RDS database name"
  default = "mlflow_db"
}

variable "rds_instance_class" {
  type = string
  description = "RDS instance class"
  default = "db.t3.micro"
}
  
variable "rds_engine" {
  type = string
  description = "RDS engine"
  default = "postgres"
}

variable "rds_engine_version" {
  type = string
  default = "15.3"
  description = "RDS engine version"
}

variable "allocated_storage" {
  type = number
  description = "The allocated storage in gigabytes"
  default = 20
}