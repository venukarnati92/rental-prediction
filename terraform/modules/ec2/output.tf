output "ec2_instance_id" {
    value = aws_instance.ec2.id
}

output "ec2_instance_public_ip" {
    value = aws_instance.ec2.public_ip
}

output "ec2_dns_name" {
    value = aws_instance.ec2.public_dns
}