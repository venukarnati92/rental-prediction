module "s3" {
    source = "../modules/s3"
    bucket_name = "${var.project_id}-bucket-2025"
}

module "vpc" {
  source = "../modules/vpc"
  project_id = var.project_id
}

module "ec2_sg" {
  source    = "../modules/ec2_sg"
  project_id = var.project_id
  vpc_id    = module.vpc.vpc_id
}

module "rds" {
    source = "../modules/rds"
    aws_ec2_security_group_id = module.ec2_sg.ec2_sg_id
    private_subnet_ids = module.vpc.private_subnet_ids
    project_id = var.project_id
    rds_username = var.rds_username
    rds_password = var.rds_password
    rds_db_name = var.rds_db_name
    rds_instance_class = var.rds_instance_class
    rds_engine = var.rds_engine
    allocated_storage = var.allocated_storage
    vpc_id = module.vpc.vpc_id
}

module "ec2" {
    source = "../modules/ec2"
    subnet_id = module.vpc.public_subnet_ids[0]
    vpc_id = module.vpc.vpc_id
    project_id = var.project_id
    ec2_sg_id = module.ec2_sg.ec2_sg_id
    rds_endpoint = module.rds.db_endpoint
    s3_bucket_name = module.s3.bucket_name
    rds_username = var.rds_username
    rds_password = var.rds_password
    rds_db_name = var.rds_db_name
}