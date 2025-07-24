export PREDICTIONS_STREAM_NAME="output_stream-mlops-zoomcamp"
export LAMBDA_FUNCTION="lambda_function_rental_prediction_mlops-zoomcamp"
export RUN_ID="d07fd6cf5ed6418bbbfc3668f5c95042" #not taking this route we will use model location
export MODEL_BUCKET="mlops-zoomcamp-bucket-2025"
export MLFLOW_EXPERIMENT_ID="1"
export MODEL_VERSION="1"
export MODEL_LOCATION=$1

variables="{RUN_ID=${RUN_ID}, MODEL_BUCKET=${MODEL_BUCKET}, MLFLOW_EXPERIMENT_ID=${MLFLOW_EXPERIMENT_ID}, MODEL_LOCATION=${MODEL_LOCATION}, MODEL_VERSION=${MODEL_VERSION}}"

aws lambda update-function-configuration --function-name ${LAMBDA_FUNCTION} --environment "Variables=${variables}"