# This resource uploads your public key to AWS
resource "aws_key_pair" "ec2_key" {
  key_name   = "${var.project_id}-ec2-key"
  public_key = file("${path.module}/my-key.pem.pub") # Assumes the key is in the same directory
}

# Data source to dynamically find the latest Amazon Linux 2 AMI
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    # This filter specifically finds Amazon Linux 2023 AMIs
    values = ["al2023-ami-*-kernel-6.1-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Create an IAM Role that the EC2 instance can assume
resource "aws_iam_role" "mlflow_server" {
  name = "${var.project_id}-mlflow-ec2-role"

  # Trust policy allowing EC2 to assume this role
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Action    = "sts:AssumeRole",
        Effect    = "Allow",
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# Create a custom policy to allow access to the S3 artifact bucket
resource "aws_iam_policy" "s3_artifact_access" {
  name        = "${var.project_id}-mlflow-s3-policy"
  description = "Allows MLflow server to read/write artifacts from its S3 bucket."

  policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ],
        Effect   = "Allow",
        Resource = [
          "arn:aws:s3:::${var.s3_bucket_name}",
          "arn:aws:s3:::${var.s3_bucket_name}/*"
        ]
      }
    ]
  })
}

# Attach the custom S3 policy to the role
resource "aws_iam_role_policy_attachment" "s3_artifact_access" {
  role       = aws_iam_role.mlflow_server.name
  policy_arn = aws_iam_policy.s3_artifact_access.arn
}

# Create an instance profile to pass the role to the EC2 instance
resource "aws_iam_instance_profile" "mlflow_server" {
  name = "${var.project_id}-mlflow-instance-profile"
  role = aws_iam_role.mlflow_server.name
}

#Launch ec2 instance
resource "aws_instance" "ec2" {
  instance_type   = var.ec2_instance_type
  ami             = data.aws_ami.amazon_linux_2023.id
  security_groups = [var.ec2_sg_id]
  vpc_security_group_ids = [var.ec2_sg_id]
  subnet_id = var.subnet_id
  iam_instance_profile = aws_iam_instance_profile.mlflow_server.name
  associate_public_ip_address = true
  key_name = aws_key_pair.ec2_key.key_name
  root_block_device {
    volume_size = 40
  }
  tags = {
    Name = "${var.project_id}-ec2"
  }
  user_data = <<EOF
#!/bin/bash
# --- Base System Setup ---
echo "--> Updating system and installing base packages..."
sudo dnf update -y
sudo dnf install -y python3-pip git

# --- MLflow & Prefect Python Packages ---
echo "--> Installing Python packages for MLflow and Prefect..."
sudo python3 -m pip install --ignore-installed mlflow boto3 psycopg2-binary prefect

# --- Install Docker and Docker Compose ---
echo "--> Installing Docker..."
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

echo "--> Installing Docker Compose..."
sudo curl -SL https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# --- Clone Configuration Repository ---
echo "--> Cloning configuration repository from GitHub..."
git clone https://github.com/venukarnati92/rental-prediction.git /home/ec2-user/app

# Set correct ownership for the cloned repository
sudo chown -R ec2-user:ec2-user /home/ec2-user/app

# --- Start All Services ---
echo "--> Starting MLflow, Prefect, and Docker Compose services..."

# Start MLflow server
nohup mlflow server -h 0.0.0.0 -p 5000 \
--backend-store-uri postgresql://${var.rds_username}:${var.rds_password}@${var.rds_endpoint}/${var.rds_db_name} \
--default-artifact-root s3://${var.s3_bucket_name} &

# Start Prefect server and worker
# Loop until a public IP is successfully retrieved
while [ -z "$PUBLIC_IP" ]; do
  echo "Attempting to fetch public IP..."
  PUBLIC_IP=$(curl -s http://ifconfig.me)
  if [ -z "$PUBLIC_IP" ]; then
    echo "Public IP not available yet. Retrying in 5 seconds..."
    sleep 5
  fi
done
echo "Successfully fetched public IP: $PUBLIC_IP"

export PREFECT_API_URL="http://$PUBLIC_IP:4200/api"
nohup prefect server start --host 0.0.0.0 --port 4200 &
sleep 10
# Use the modern 'prefect worker start' command
nohup prefect worker start -p 'default' &

# Start Docker Compose services from the correct subdirectory
sudo -u ec2-user -H sh -c "cd /home/ec2-user/app/docker && docker-compose up -d"
EOF  
}




