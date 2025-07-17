import os
import json
import base64

import boto3
import mlflow


def get_model_location(run_id):
    model_location = os.getenv('MODEL_LOCATION')

    if model_location is not None:
        return model_location

    model_bucket = os.getenv('MODEL_BUCKET', 'mlops-zoomcamp-bucket-2025')
    experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID', '1')
    model_location = f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/'
    return model_location


def load_model(run_id):
    model_path = get_model_location(run_id)
    # model_path = "s3://mlops-zoomcamp-bucket-2025/1/models/m-205ae18cbfae4070a9cce59339d449c0/artifacts"
    model = mlflow.pyfunc.load_model(model_path)
    return model


def base64_decode(encoded_data):
    decoded_data = base64.b64decode(encoded_data).decode('utf-8')
    event_data = json.loads(decoded_data)
    return event_data


class ModelService:
    def __init__(self, model, model_version=None, callbacks=None):
        self.model = model
        self.model_version = model_version
        self.callbacks = callbacks or []

    def predict(self, features):
        pred = self.model.predict(features)
        return float(pred[0])

    def lambda_handler(self, event):
        predictions_events = []

        for record in event['Records']:
            encoded_data = record['kinesis']['data']
            event_data = base64_decode(encoded_data)
            
            prediction = self.predict(event_data)

            prediction_event = {
                'model': 'rental_price_prediction_model',
                'version': self.model_version,
                'prediction': {'price': prediction},
            }
            print(f"Prediction event: {prediction_event}")

            for callback in self.callbacks:
                callback(prediction_event)

            predictions_events.append(prediction_event)

        return {'predictions': predictions_events}


class KinesisCallback:
    def __init__(self, kinesis_client, prediction_stream_name):
        self.kinesis_client = kinesis_client
        self.prediction_stream_name = prediction_stream_name

    def put_record(self, prediction_event):
        try:
            price = prediction_event['prediction']['price']
            stream_name = self.prediction_stream_name
            print(f"Attempting to put record to stream: {stream_name}")
            
            if not self.kinesis_client:
                print("Error: Kinesis client is not initialized")
                return

            response = self.kinesis_client.put_record(
                StreamName=stream_name,
                Data=json.dumps(prediction_event),
                PartitionKey=str(price),
            )
            print(f"Successfully put record to Kinesis. SequenceNumber: {response.get('SequenceNumber')}, ShardId: {response.get('ShardId')}")
            return response
            
        except self.kinesis_client.exceptions.ResourceNotFoundException as e:
            print(f"Error: Stream {stream_name} not found. Please check if the stream exists and is in ACTIVE state.")
            print(f"Error details: {str(e)}")
            raise
        except self.kinesis_client.exceptions.AccessDeniedException as e:
            print("Error: Permission denied when calling PutRecord. Check IAM permissions.")
            print(f"Required permission: kinesis:PutRecord on resource: {stream_name}")
            print(f"Error details: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error when putting record to Kinesis: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            raise


def create_kinesis_client():
    endpoint_url = os.getenv('KINESIS_ENDPOINT_URL')

    if endpoint_url is None:
        return boto3.client('kinesis')

    try:
        return boto3.client('kinesis', endpoint_url=endpoint_url)
    except Exception as e:
        print(f"Failed to create kinesis client: {e}")
        return None


def init(prediction_stream_name: str, run_id: str, test_run: bool):
    print("Loading model from S3...")
    model = load_model(run_id)
    print("Model loaded successfully.")

    callbacks = []

    if not test_run:
        print("Initializing kinesis client...")
        kinesis_client = create_kinesis_client()
        kinesis_callback = KinesisCallback(kinesis_client, prediction_stream_name)
        callbacks.append(kinesis_callback.put_record)
        print("Kinesis client initialized successfully.")

    print("Initializing model service...")
    model_service = ModelService(model=model, model_version=run_id, callbacks=callbacks)
    print("Model service initialized successfully.")

    return model_service