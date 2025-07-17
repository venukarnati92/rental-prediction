output "s3_bucket_name" {
    value = module.s3.bucket_name
}

#ec2 public ip
output "ec2_public_ip" {
    value = module.ec2.ec2_instance_public_ip
}

#ec2 dns name
output "ec2_dns_name" {
    value = module.ec2.ec2_dns_name
}

#rds endpoint
output "rds_endpoint" {
    value = module.rds.db_endpoint
}