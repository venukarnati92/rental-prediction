export PREDICTIONS_STREAM_NAME="output_stream-mlops-zoomcamp"
export LAMBDA_FUNCTION="lambda_function_rental_prediction_mlops-zoomcamp"
export RUN_ID="fd0368334e064785aaa2c42c2be133a3"
export MODEL_BUCKET="mlops-zoomcamp-bucket-2025"
export MLFLOW_EXPERIMENT_ID="1"
export MODEL_LOCATION="s3://mlops-zoomcamp-bucket-2025/1/models/m-2a11d280a71d4630a129147209f6e0b3/artifacts"
export MODEL_VERSION="1"

variables="{RUN_ID=${RUN_ID}, MODEL_BUCKET=${MODEL_BUCKET}, MLFLOW_EXPERIMENT_ID=${MLFLOW_EXPERIMENT_ID}, MODEL_LOCATION=${MODEL_LOCATION}, MODEL_VERSION=${MODEL_VERSION}}"

aws lambda update-function-configuration --function-name ${LAMBDA_FUNCTION} --environment "Variables=${variables}"