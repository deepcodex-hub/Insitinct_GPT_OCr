# provider.tf
provider "aws" {
  region = var.region
}

# vpc.tf
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = { Name = "ag-ocr-vpc" }
}

# s3.tf
resource "aws_s3_bucket" "artifacts" {
  bucket = "ag-ocr-artifacts-${var.environment}"
}

resource "aws_s3_bucket_public_access_block" "artifacts_block" {
  bucket = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# rds.tf
resource "aws_db_instance" "postgres" {
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "14"
  instance_class       = "db.t3.micro"
  name                 = "agocrdb"
  username             = "admin"
  password             = var.db_password
  skip_final_snapshot  = true
  vpc_security_group_ids = [aws_security_group.db_sg.id]
}

# eks.tf - Simplified
resource "aws_eks_cluster" "main" {
  name     = "ag-ocr-cluster"
  role_arn = aws_iam_role.eks_role.arn
  vpc_config {
    subnet_ids = aws_subnet.private.*.id
  }
}

# 1. EKS Cluster
resource "aws_eks_cluster" "agm_cluster" {
  name     = "agm-production-cluster"
  role_arn = aws_iam_role.eks_role.arn
  vpc_config {
    subnet_ids = ["subnet-private1", "subnet-private2"] # Private subnet for security
  }
}

# 2. Managed Node Group for GPU workloads (Triton)
resource "aws_eks_node_group" "gpu_nodes" {
  cluster_name    = aws_eks_cluster.agm_cluster.name
  node_group_name = "agm-gpu-nodes"
  node_role_arn   = aws_iam_role.node_role.arn
  subnet_ids      = ["subnet-private1", "subnet-private2"]

  scaling_config {
    desired_size = 2
    max_size     = 10
    min_size     = 2
  }
  instance_types = ["g4dn.xlarge"] # Using T4 GPUs for inference
  ami_type       = "AL2_x86_64_GPU"
}

# 3. Dedicated S3 Bucket for Artifacts (presigned URLs)
resource "aws_s3_bucket" "agm_artifacts" {
  bucket = "agm-inference-artifacts-prod"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "agm_sse" {
  bucket = aws_s3_bucket.agm_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# 4. Secrets Manager
resource "aws_secretsmanager_secret" "agm_api_keys" {
  name = "agm/prod/api_keys"
}

# 5. RDS setup for metadata/MLFlow backend
resource "aws_db_instance" "mlflow_db" {
  allocated_storage    = 20
  storage_type         = "gp2"
  engine               = "postgres"
  engine_version       = "13.4"
  instance_class       = "db.t3.micro"
  name                 = "mlflow_production"
  username             = "mlflow_admin"
  password             = "REPLACE_VIA_SECRET_MANAGER"
  publicly_accessible  = false
  skip_final_snapshot  = true
}

# (IAM Roles omitted for brevity in skeleton)
