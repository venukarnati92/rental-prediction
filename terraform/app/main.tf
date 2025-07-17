data "aws_caller_identity" "current_identity" {}

locals {
  account_id = data.aws_caller_identity.current_identity.account_id
}


module "source_kinesis_stream" {
  source = "../modules/kinesis"
  retention_period = 48
  shard_count = 2
  stream_name = "${var.source_stream_name}-${var.project_id}"
  tags = var.project_id
}


module "output_kinesis_stream" {
  source = "../modules/kinesis"
  retention_period = 48
  shard_count = 2
  stream_name = "${var.output_stream_name}-${var.project_id}"
  tags = var.project_id
}

# image registry
module "ecr_image" {
   source = "../modules/ecr"
   ecr_repo_name = "${var.ecr_repo_name}_${var.project_id}"
   account_id = local.account_id
   lambda_function_local_path = var.lambda_function_local_path
   docker_image_local_path = var.docker_image_local_path
   ecr_image_tag = var.ecr_image_tag
}

module "lambda_function" {
  source = "../modules/lambda"
  image_uri = module.ecr_image.image_uri
  lambda_function_name = "${var.lambda_function_name}_${var.project_id}"
  # bucket_name = data.terraform_remote_state.infra.outputs.s3_bucket_name
  bucket_name = "mlops-zoomcamp-bucket-2025"
  output_stream_arn = module.output_kinesis_stream.stream_arn
  source_stream_arn = module.source_kinesis_stream.stream_arn
  output_stream_name = module.output_kinesis_stream.stream_name
  source_stream_name = module.source_kinesis_stream.stream_name
}