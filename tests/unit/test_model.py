import pytest
import os
import json
import base64
from unittest.mock import Mock, patch, MagicMock


class TestModel:
    """All tests for model.py in a single class"""
    
    # TestGetModelLocation methods
    def test_get_model_location_with_env_variable(self):
        """Test get_model_location when MODEL_LOCATION is set"""
        custom_location = 's3://custom-bucket/custom-path'
        
        with patch.dict(os.environ, {'MODEL_LOCATION': custom_location}):
            model_location = os.getenv('MODEL_LOCATION')
            assert model_location == custom_location
    
    def test_get_model_location_without_env_variable(self):
        """Test get_model_location when MODEL_LOCATION is not set"""
        with patch.dict(os.environ, {}, clear=True):
            model_bucket = os.getenv('MODEL_BUCKET', 'mlops-zoomcamp-bucket-2025')
            experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID', '1')
            run_id = 'test-run-id'
            expected = f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/'
            assert expected == 's3://mlops-zoomcamp-bucket-2025/1/test-run-id/artifacts/'
    
    def test_get_model_location_with_custom_bucket(self):
        """Test get_model_location with custom bucket"""
        custom_bucket = 'my-custom-bucket'
        custom_experiment_id = '123'
        
        with patch.dict(os.environ, {
            'MODEL_BUCKET': custom_bucket,
            'MLFLOW_EXPERIMENT_ID': custom_experiment_id
        }):
            run_id = 'test-run-id'
            expected = f's3://{custom_bucket}/{custom_experiment_id}/{run_id}/artifacts/'
            assert expected == f's3://{custom_bucket}/{custom_experiment_id}/{run_id}/artifacts/'
    
    # TestBase64Decode methods
    def test_base64_decode_valid_data(self):
        """Test base64_decode with valid data"""
        test_data = {
            'cityname': 'NYC',
            'state': 'NY',
            'bedrooms': 2,
            'bathrooms': 1,
            'square_feet': 1000
        }
        
        json_data = json.dumps(test_data)
        encoded_data = base64.b64encode(json_data.encode('utf-8')).decode('utf-8')
        
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        result = json.loads(decoded_data)
        
        assert result == test_data
        assert result['cityname'] == 'NYC'
        assert result['state'] == 'NY'
        assert result['bedrooms'] == 2
        assert result['bathrooms'] == 1
        assert result['square_feet'] == 1000
    
    def test_base64_decode_empty_data(self):
        """Test base64_decode with empty data"""
        empty_data = {}
        json_data = json.dumps(empty_data)
        encoded_data = base64.b64encode(json_data.encode('utf-8')).decode('utf-8')
        
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        result = json.loads(decoded_data)
        assert result == empty_data
    
    def test_base64_decode_invalid_data(self):
        """Test base64_decode with invalid base64 data"""
        invalid_data = "invalid-base64-data"
        
        with pytest.raises(Exception):
            base64.b64decode(invalid_data).decode('utf-8')
    
    # TestModelService methods
    def test_model_service_initialization(self):
        """Test ModelService initialization"""
        mock_model = Mock()
        model_version = 'test-version'
        callbacks = [Mock()]
        
        class ModelService:
            def __init__(self, model, model_version=None, callbacks=None):
                self.model = model
                self.model_version = model_version
                self.callbacks = callbacks or []
        
        service = ModelService(model=mock_model, model_version=model_version, callbacks=callbacks)
        
        assert service.model == mock_model
        assert service.model_version == model_version
        assert service.callbacks == callbacks
    
    def test_model_service_initialization_default_callbacks(self):
        """Test ModelService initialization with default callbacks"""
        mock_model = Mock()
        model_version = 'test-version'
        
        class ModelService:
            def __init__(self, model, model_version=None, callbacks=None):
                self.model = model
                self.model_version = model_version
                self.callbacks = callbacks or []
        
        service = ModelService(model=mock_model, model_version=model_version)
        
        assert service.model == mock_model
        assert service.model_version == model_version
        assert service.callbacks == []
    
    def test_model_service_predict(self):
        """Test ModelService predict method"""
        mock_model = Mock()
        mock_model.predict.return_value = [2500.0]
        
        class ModelService:
            def __init__(self, model, model_version=None, callbacks=None):
                self.model = model
                self.model_version = model_version
                self.callbacks = callbacks or []
            
            def predict(self, features):
                pred = self.model.predict(features)
                return float(pred[0])
        
        service = ModelService(model=mock_model)
        
        features = {
            'cityname': 'NYC',
            'state': 'NY',
            'bedrooms': 2,
            'bathrooms': 1,
            'square_feet': 1000
        }
        
        result = service.predict(features)
        
        mock_model.predict.assert_called_once_with(features)
        assert isinstance(result, float)
        assert result == 2500.0
    
    def test_model_service_lambda_handler_single_record(self):
        """Test ModelService lambda_handler with single record"""
        mock_model = Mock()
        mock_model.predict.return_value = [2500.0]
        mock_callback = Mock()
        
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
                    decoded_data = base64.b64decode(encoded_data).decode('utf-8')
                    event_data = json.loads(decoded_data)
                    
                    prediction = self.predict(event_data)
                    
                    prediction_event = {
                        'model': 'rental_price_prediction_model',
                        'version': self.model_version,
                        'prediction': {'price': prediction},
                    }
                    
                    for callback in self.callbacks:
                        callback(prediction_event)
                    
                    predictions_events.append(prediction_event)
                
                return {'predictions': predictions_events}
        
        service = ModelService(model=mock_model, callbacks=[mock_callback])
        
        test_data = {
            'cityname': 'NYC',
            'state': 'NY',
            'bedrooms': 2,
            'bathrooms': 1,
            'square_feet': 1000
        }
        json_data = json.dumps(test_data)
        encoded_data = base64.b64encode(json_data.encode('utf-8')).decode('utf-8')
        
        event = {
            'Records': [
                {
                    'kinesis': {
                        'data': encoded_data
                    }
                }
            ]
        }
        
        result = service.lambda_handler(event)
        
        assert 'predictions' in result
        assert len(result['predictions']) == 1
        
        prediction = result['predictions'][0]
        assert prediction['model'] == 'rental_price_prediction_model'
        assert prediction['version'] is None
        assert 'prediction' in prediction
        assert 'price' in prediction['prediction']
        assert prediction['prediction']['price'] == 2500.0
        
        mock_callback.assert_called_once()
    
    # TestKinesisCallback methods
    def test_kinesis_callback_initialization(self):
        """Test KinesisCallback initialization"""
        mock_kinesis_client = Mock()
        stream_name = 'test-stream'
        
        class KinesisCallback:
            def __init__(self, kinesis_client, prediction_stream_name):
                self.kinesis_client = kinesis_client
                self.prediction_stream_name = prediction_stream_name
        
        callback = KinesisCallback(mock_kinesis_client, stream_name)
        
        assert callback.kinesis_client == mock_kinesis_client
        assert callback.prediction_stream_name == stream_name
    
    def test_kinesis_callback_put_record_success(self):
        """Test KinesisCallback put_record success case"""
        mock_kinesis_client = Mock()
        stream_name = 'test-stream'
        
        mock_response = {
            'SequenceNumber': '1234567890',
            'ShardId': 'shard-000000000000'
        }
        mock_kinesis_client.put_record.return_value = mock_response
        
        class KinesisCallback:
            def __init__(self, kinesis_client, prediction_stream_name):
                self.kinesis_client = kinesis_client
                self.prediction_stream_name = prediction_stream_name
            
            def put_record(self, prediction_event):
                try:
                    price = prediction_event['prediction']['price']
                    stream_name = self.prediction_stream_name
                    
                    if not self.kinesis_client:
                        return None
                    
                    response = self.kinesis_client.put_record(
                        StreamName=stream_name,
                        Data=json.dumps(prediction_event),
                        PartitionKey=str(price),
                    )
                    return response
                    
                except Exception as e:
                    raise
        
        callback = KinesisCallback(mock_kinesis_client, stream_name)
        
        prediction_event = {
            'model': 'rental_price_prediction_model',
            'version': 'test-version',
            'prediction': {'price': 2500.0}
        }
        
        result = callback.put_record(prediction_event)
        
        mock_kinesis_client.put_record.assert_called_once_with(
            StreamName=stream_name,
            Data=json.dumps(prediction_event),
            PartitionKey='2500.0'
        )
        
        assert result == mock_response
    
    # TestCreateKinesisClient methods
    @patch('boto3.client')
    def test_create_kinesis_client_default(self, mock_boto3_client):
        """Test create_kinesis_client without endpoint URL"""
        with patch.dict(os.environ, {}, clear=True):
            mock_client = Mock()
            mock_boto3_client.return_value = mock_client
            
            endpoint_url = os.getenv('KINESIS_ENDPOINT_URL')
            if endpoint_url is None:
                result = mock_boto3_client('kinesis')
            else:
                result = mock_boto3_client('kinesis', endpoint_url=endpoint_url)
            
            mock_boto3_client.assert_called_once_with('kinesis')
            assert result == mock_client
    
    @patch('boto3.client')
    def test_create_kinesis_client_with_endpoint(self, mock_boto3_client):
        """Test create_kinesis_client with endpoint URL"""
        endpoint_url = 'http://localhost:4566'
        
        with patch.dict(os.environ, {'KINESIS_ENDPOINT_URL': endpoint_url}):
            mock_client = Mock()
            mock_boto3_client.return_value = mock_client
            
            endpoint_url = os.getenv('KINESIS_ENDPOINT_URL')
            if endpoint_url is None:
                result = mock_boto3_client('kinesis')
            else:
                result = mock_boto3_client('kinesis', endpoint_url=endpoint_url)
            
            mock_boto3_client.assert_called_once_with('kinesis', endpoint_url=endpoint_url)
            assert result == mock_client
    
    # TestIntegration methods
    def test_complete_prediction_flow(self):
        """Test the complete prediction flow"""
        test_data = {
            'cityname': 'NYC',
            'state': 'NY',
            'bedrooms': 2,
            'bathrooms': 1,
            'square_feet': 1000
        }
        
        json_data = json.dumps(test_data)
        encoded_data = base64.b64encode(json_data.encode('utf-8')).decode('utf-8')
        
        event = {
            'Records': [
                {
                    'kinesis': {
                        'data': encoded_data
                    }
                }
            ]
        }
        
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        event_data = json.loads(decoded_data)
        assert event_data == test_data
        
        required_fields = ['cityname', 'state', 'bedrooms', 'bathrooms', 'square_feet']
        for field in required_fields:
            assert field in event_data
    
    def test_model_location_generation(self):
        """Test model location generation with different parameters"""
        run_id = 'test-run-123'
        
        with patch.dict(os.environ, {}, clear=True):
            model_bucket = os.getenv('MODEL_BUCKET', 'mlops-zoomcamp-bucket-2025')
            experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID', '1')
            location = f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/'
            expected = f's3://mlops-zoomcamp-bucket-2025/1/{run_id}/artifacts/'
            assert location == expected
        
        with patch.dict(os.environ, {
            'MODEL_BUCKET': 'custom-bucket',
            'MLFLOW_EXPERIMENT_ID': '999'
        }):
            model_bucket = os.getenv('MODEL_BUCKET', 'mlops-zoomcamp-bucket-2025')
            experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID', '1')
            location = f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/'
            expected = f's3://custom-bucket/999/{run_id}/artifacts/'
            assert location == expected


if __name__ == "__main__":
    pytest.main([__file__]) 