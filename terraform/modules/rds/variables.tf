

variable "project_id" {
    type = string
    description = "Project ID"
}

variable "rds_username" {
    type = string
    description = "RDS username"
}

variable "rds_password" {
    type = string
    description = "RDS password"
}

variable "rds_db_name" {
    type = string
    description = "RDS database name"
}

variable "rds_instance_class" {
    type = string
    description = "RDS instance class"
}

variable "rds_engine" {
    type = string
    description = "RDS engine"
}


variable "allocated_storage" {
  type = number
  description = "The allocated storage in gigabytes"
}


variable "vpc_id" {
  type = string
  description = "VPC ID for RDS and its security group"
}


# Add this new variable
variable "private_subnet_ids" {
  description = "A list of private subnet IDs for the DB subnet group."
  type        = list(string)
}

variable "aws_ec2_security_group_id" {
  description = "The ID of the EC2 security group that needs access to the RDS instance."
  type        = string
}



    