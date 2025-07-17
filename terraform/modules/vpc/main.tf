
# ------------------------------------------------------------------------------
# VPC (Virtual Private Cloud)
# ------------------------------------------------------------------------------
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr_block
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.project_id}-vpc"
  }
}

# ------------------------------------------------------------------------------
# SUBNETS
# ------------------------------------------------------------------------------
# Create public subnets in different availability zones
resource "aws_subnet" "public" {
  count                   = length(var.public_subnet_cidr_blocks)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = element(var.public_subnet_cidr_blocks, count.index)
  availability_zone       = element(var.availability_zones, count.index)
  map_public_ip_on_launch = true # Instances in public subnets get a public IP

  tags = {
    Name = "${var.project_id}-public-subnet-${count.index + 1}"
  }
}

# Create private subnets in different availability zones
resource "aws_subnet" "private" {
  count             = length(var.private_subnet_cidr_blocks)
  vpc_id            = aws_vpc.main.id
  cidr_block        = element(var.private_subnet_cidr_blocks, count.index)
  availability_zone = element(var.availability_zones, count.index)

  tags = {
    Name = "${var.project_id}-private-subnet-${count.index + 1}"
  }
}

# ------------------------------------------------------------------------------
# INTERNET GATEWAY - For Public Subnets
# ------------------------------------------------------------------------------
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_id}-igw"
  }
}

# ------------------------------------------------------------------------------
# NAT GATEWAY - For Private Subnets
# ------------------------------------------------------------------------------
# An Elastic IP is required for the NAT Gateway
resource "aws_eip" "nat" {
  domain = "vpc"
  depends_on = [aws_internet_gateway.main]
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  # Place the NAT Gateway in the first public subnet
  subnet_id     = aws_subnet.public[0].id

  tags = {
    Name = "${var.project_id}-nat-gw"
  }

  # The NAT Gateway depends on the Internet Gateway being available
  depends_on = [aws_internet_gateway.main]
}

# ------------------------------------------------------------------------------
# ROUTE TABLES
# ------------------------------------------------------------------------------
# Route table for the public subnets
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  # Route for internet-bound traffic via the Internet Gateway
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.project_id}-public-rt"
  }
}

# Associate the public route table with the public subnets
resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = element(aws_subnet.public, count.index).id
  route_table_id = aws_route_table.public.id
}

# Route table for the private subnets
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  # Route for internet-bound traffic via the NAT Gateway
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }

  tags = {
    Name = "${var.project_id}-private-rt"
  }
}

# Associate the private route table with the private subnets
resource "aws_route_table_association" "private" {
  count          = length(aws_subnet.private)
  subnet_id      = element(aws_subnet.private, count.index).id
  route_table_id = aws_route_table.private.id
}
