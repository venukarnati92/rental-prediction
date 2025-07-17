resource "aws_ecr_repository" "repo" {
  name                 = var.ecr_repo_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = false
  }

  force_delete = true
}

# In practice, the Image build-and-push step is handled separately by the CI/CD pipeline and not the IaC script.
# But because the lambda config would fail without an existing Image URI in ECR,
# we can also upload any base image to bootstrap the lambda config, unrelated to your Inference logic
resource "null_resource" "ecr_image" {
  triggers = {
    python_file = md5(file(var.lambda_function_local_path))
    docker_file = md5(file(var.docker_image_local_path))
    timestamp = timestamp()
  }

  provisioner "local-exec" {
    command = <<EOF
      # Step 1: Log in to the AWS ECR registry.
      aws ecr get-login-password --region ${var.region} | docker login --username AWS --password-stdin ${var.account_id}.dkr.ecr.${var.region}.amazonaws.com

      # Step 2: Build for the correct platform and push in a single command.
      # The --provenance=false flag is CRITICAL to generate a Lambda-compatible manifest.
      docker buildx build --platform linux/amd64 --provenance=false -t ${aws_ecr_repository.repo.repository_url}:${var.ecr_image_tag} -f ${var.docker_image_local_path} $(dirname ${var.docker_image_local_path}) --push
    EOF
  }
}


// Wait for the image to be uploaded, before lambda config runs
data aws_ecr_image lambda_image {
  depends_on = [
    null_resource.ecr_image
  ]
 repository_name = var.ecr_repo_name
 image_tag       = var.ecr_image_tag
}

output "image_uri" {
  value     = "${aws_ecr_repository.repo.repository_url}:${data.aws_ecr_image.lambda_image.image_tag}"
}
