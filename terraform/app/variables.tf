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

variable "source_stream_name" {
  type = string
  description = "Source stream name"
  default = "source_stream"
}

variable "output_stream_name" {
  type = string
  description = "Output stream name"
  default = "output_stream"
}

variable "ecr_repo_name" {
  type = string
  description = "ECR repository name"
  default = "rental-prediction"
}

variable "lambda_function_local_path" {
  type = string
  description = "Local path to lambda function / python file"
  default = "../../src/webservice/lambda_function.py"
}

variable "docker_image_local_path" {
  type = string
  description = "Local path to Dockerfile"
  default = "../../src/webservice/Dockerfile"
}

variable "ecr_image_tag" {
  type = string
  description = "ECR image tag"
  default = "v1.0.1"
}

variable "lambda_function_name" {
  type = string
  description = "Lambda function name"
  default = "lambda_function_rental_prediction"
}
  