# ./modules/vpc/variables.tf

variable "project_id" {
  description = "A unique identifier for the project, used for naming resources."
  type        = string
}

variable "aws_region" {
  description = "The AWS region to create resources in."
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr_block" {
  description = "The main CIDR block for the VPC."
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "A list of availability zones to create subnets in."
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

variable "public_subnet_cidr_blocks" {
  description = "A list of CIDR blocks for the public subnets."
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidr_blocks" {
  description = "A list of CIDR blocks for the private subnets."
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24"]
}
