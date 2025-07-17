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
# Log everything to a file for easier debugging
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

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
sudo usermod -aG docker ec2-user # Allow ec2-user to run docker commands

echo "--> Installing Docker Compose..."
sudo curl -SL https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# --- Create Directory and Config Files for Docker Compose ---
echo "--> Creating configuration directories and files for monitoring stack..."
mkdir -p /home/ec2-user/monitoring/config
mkdir -p /home/ec2-user/monitoring/dashboards

# Create Grafana datasource config
cat <<'EOT' > /home/ec2-user/monitoring/config/grafana_datasources.yaml
apiVersion: 1
datasources:
  - name: PostgreSQL
    type: postgres
    url: db:5432
    user: postgres
    password: example
    database: rentalprediction # Or your specific DB name
    isDefault: true
    editable: true
EOT

# Create Grafana dashboard config
cat <<'EOT' > /home/ec2-user/monitoring/config/grafana_dashboards.yaml
apiVersion: 1
providers:
- name: 'default'
  orgId: 1
  folder: ''
  type: file
  disableDeletion: false
  editable: true
  options:
    path: /opt/grafana/dashboards
EOT

# --- Create the docker-compose.yml file ---
echo "--> Creating docker-compose.yml..."
cat <<'EOT' > /home/ec2-user/monitoring/docker-compose.yml
version: '3.7'

volumes: 
  grafana_data: {}

networks:
  front-tier:
  back-tier:

services:
  db:
    image: postgres
    restart: always
    environment:
      POSTGRES_PASSWORD: example
    ports:
      - "5432:5432"
    networks:
      - back-tier

  adminer:
    image: adminer
    restart: always
    ports:
      - "8080:8080"
    networks:
      - back-tier
      - front-tier  

  grafana:
    image: grafana/grafana-enterprise
    user: "472"
    ports:
      - "3000:3000"
    volumes:
      - ./config/grafana_datasources.yaml:/etc/grafana/provisioning/datasources/datasource.yaml:ro
      - ./config/grafana_dashboards.yaml:/etc/grafana/provisioning/dashboards/dashboards.yaml:ro
      - ./dashboards:/opt/grafana/dashboards
      - grafana_data:/var/lib/grafana
    networks:
      - back-tier
      - front-tier
    restart: always
EOT

# Set correct ownership for the created files
sudo chown -R ec2-user:ec2-user /home/ec2-user/monitoring

# --- Start All Services ---
echo "--> Starting MLflow, Prefect, and Docker Compose services..."

# Start MLflow server
nohup mlflow server -h 0.0.0.0 -p 5000 \
--backend-store-uri postgresql://${var.rds_username}:${var.rds_password}@${var.rds_endpoint}/${var.rds_db_name} \
--default-artifact-root s3://${var.s3_bucket_name} &

# Start Prefect server and agent
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
nohup prefect agent start -p 'default' &

# Start Docker Compose services in detached mode as the ec2-user
sudo -u ec2-user -H sh -c "cd /home/ec2-user/monitoring && docker-compose up -d"
EOF
}




