variable "vpc_id" {
    type = string
    description = "VPC ID"
}

variable "ec2_instance_type" {
    type = string
    description = "EC2 instance type"
    # default = "t2.micro"
    default = "t3.medium"
}

variable "ec2_ami" {
    type = string
    description = "EC2 AMI"
    default = "ami-0c55b159cbfafe1f0"
}

variable "project_id" {
    type = string
    description = "Project ID"
}

variable "ec2_sg_id" {
    type = string
    description = "EC2 security group ID"
}

variable "rds_endpoint" {
    type = string
    description = "RDS endpoint"
}

variable "s3_bucket_name" {
    type = string
    description = "S3 bucket name"
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

variable "subnet_id" {
    type = string
    description = "Subnet ID"
}