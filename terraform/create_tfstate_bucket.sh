#!/bin/bash

# --- Configuration ---
# REQUIRED: A unique name for the S3 bucket.
# S3 bucket names must be globally unique across all of AWS.
BUCKET_NAME="mlops-zoomcamp-tfstate-2025"

# REQUIRED: The AWS region to create the bucket in.
AWS_REGION="us-east-1"


# --- Script Logic ---
# Use `aws s3api head-bucket` to check if the bucket exists.
# We redirect stdout and stderr to /dev/null to keep the output clean.
aws s3api head-bucket --bucket "${BUCKET_NAME}" >/dev/null 2>&1

# Check the exit code of the last command ($?). 0 means success (bucket exists).
if [ $? -eq 0 ]; then
  echo "Bucket '${BUCKET_NAME}' already exists. No action taken."
else
  echo "Bucket '${BUCKET_NAME}' not found. Creating it now in region ${AWS_REGION}..."

  # Create the S3 bucket with a location constraint for regions other than us-east-1
  if [ "$AWS_REGION" == "us-east-1" ]; then
    aws s3api create-bucket \
      --bucket "${BUCKET_NAME}" \
      --region "${AWS_REGION}"
  else
    aws s3api create-bucket \
      --bucket "${BUCKET_NAME}" \
      --region "${AWS_REGION}" \
      --create-bucket-configuration LocationConstraint=${AWS_REGION}
  fi

  # Enable versioning on the S3 bucket to keep state history (best practice)
  echo "Enabling versioning for bucket: ${BUCKET_NAME}"
  aws s3api put-bucket-versioning \
    --bucket "${BUCKET_NAME}" \
    --versioning-configuration Status=Enabled

  echo "Bucket '${BUCKET_NAME}' created and versioning enabled."
fi

echo "--------------------------------------------------"
echo "Terraform backend S3 bucket: ${BUCKET_NAME}"
echo "--------------------------------------------------"