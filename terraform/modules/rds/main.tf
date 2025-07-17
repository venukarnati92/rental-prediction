#Create a new PostgreSQL database
resource "aws_security_group" "rds" {
  name        = "${var.project_id}-rds-sg"
  description = "Allow PostgreSQL from EC2 SG"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.aws_ec2_security_group_id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1" # "-1" means all protocols
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = {
    Name = "${var.project_id}-rds-sg"
  }
}

resource "aws_db_subnet_group" "db" {
  name       = "${var.project_id}-db-subnet-group"
  subnet_ids = var.private_subnet_ids # Use the passed-in subnet IDs

  tags = {
    Name = "${var.project_id}-db-subnet-group"
  }
}

resource "aws_db_instance" "db" {
  allocated_storage       = var.allocated_storage
  instance_class          = var.rds_instance_class
  engine                  = var.rds_engine
  username                = var.rds_username
  password                = var.rds_password
  db_name                 = var.rds_db_name
  db_subnet_group_name    = aws_db_subnet_group.db.name
  vpc_security_group_ids  = [aws_security_group.rds.id]
  skip_final_snapshot = true
}







    


