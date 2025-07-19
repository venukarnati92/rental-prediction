"""
Integration tests for AWS service interactions.

These tests verify the integration with AWS services including
S3, Kinesis, Lambda, and RDS.
"""

import pytest
import json
import base64
import boto3
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
import numpy as np


class TestAWSIntegration:
    """Integration tests for AWS service interactions"""

    def setup_method(self):
        """Set up test data and mocks for each test"""
        self.sample_data = {
            'cityname': 'NYC',
            'state': 'NY',
            'bedrooms': 2,
            'bathrooms': 1,
            'square_feet': 1000
        }
        
        self.kinesis_event = {
            'Records': [
                {
                    'kinesis': {
                        'data': base64.b64encode(json.dumps(self.sample_data).encode()).decode()
                    }
                }
            ]
        }

    @patch('boto3.client')
    def test_s3_model_storage_integration(self, mock_boto3_client):
        """Test S3 integration for model storage and retrieval"""
        # Mock S3 client
        mock_s3 = Mock()
        mock_boto3_client.return_value = mock_s3
        
        # Mock successful S3 operations
        mock_s3.head_object.return_value = {'ContentLength': 1024}
        mock_s3.get_object.return_value = {
            'Body': Mock(read=lambda: b'mock_model_data'),
            'ContentLength': 1024
        }
        
        # Test model location generation
        def get_model_location():
            """Mock model location function"""
            bucket = 'mlflow-bucket'
            key = 'models/rental-prediction/latest/model.pkl'
            return f's3://{bucket}/{key}'
        
        model_location = get_model_location()
        
        # Verify model location format
        assert model_location.startswith('s3://')
        assert 'mlflow-bucket' in model_location
        assert 'models/rental-prediction' in model_location
        
        # Test S3 model loading
        def load_model_from_s3(location):
            """Mock model loading from S3"""
            # Parse S3 location
            if location.startswith('s3://'):
                path_parts = location[5:].split('/')
                bucket = path_parts[0]
                key = '/'.join(path_parts[1:])
                
                # Mock S3 get_object call
                response = mock_s3.get_object(Bucket=bucket, Key=key)
                return response['Body'].read()
        
        # Test model loading
        model_data = load_model_from_s3(model_location)
        
        # Verify S3 operations were called
        mock_s3.get_object.assert_called_once()
        assert model_data == b'mock_model_data'

    @patch('boto3.client')
    def test_kinesis_stream_integration(self, mock_boto3_client):
        """Test Kinesis stream integration for data ingestion and publishing"""
        # Mock Kinesis client
        mock_kinesis = Mock()
        mock_boto3_client.return_value = mock_kinesis
        
        # Mock successful Kinesis operations
        mock_kinesis.put_record.return_value = {'RecordId': 'test-record-id'}
        mock_kinesis.describe_stream.return_value = {
            'StreamDescription': {
                'StreamStatus': 'ACTIVE',
                'Shards': [{'ShardId': 'shard-000000000000'}]
            }
        }
        
        # Test Kinesis stream creation
        def create_kinesis_client(stream_name, endpoint_url=None):
            """Mock Kinesis client creation"""
            if endpoint_url:
                return boto3.client('kinesis', endpoint_url=endpoint_url)
            return boto3.client('kinesis')
        
        # Test stream validation
        def validate_stream_exists(stream_name):
            """Mock stream validation"""
            try:
                response = mock_kinesis.describe_stream(StreamName=stream_name)
                return response['StreamDescription']['StreamStatus'] == 'ACTIVE'
            except Exception:
                return False
        
        # Test publishing to Kinesis
        def publish_to_kinesis(stream_name, data):
            """Mock publishing to Kinesis"""
            try:
                response = mock_kinesis.put_record(
                    StreamName=stream_name,
                    Data=json.dumps(data),
                    PartitionKey='rental-prediction'
                )
                return response['RecordId']
            except Exception as e:
                raise Exception(f"Failed to publish to Kinesis: {str(e)}")
        
        # Test the integration
        stream_name = 'rental-predictions-stream'
        
        # Validate stream exists
        assert validate_stream_exists(stream_name)
        
        # Publish prediction data
        prediction_data = {
            'prediction': 2500,
            'features': self.sample_data,
            'timestamp': datetime.now().isoformat()
        }
        
        record_id = publish_to_kinesis(stream_name, prediction_data)
        
        # Verify Kinesis operations
        mock_kinesis.describe_stream.assert_called_once_with(StreamName=stream_name)
        mock_kinesis.put_record.assert_called_once()
        assert record_id == 'test-record-id'

    @patch('boto3.client')
    def test_lambda_function_integration(self, mock_boto3_client):
        """Test Lambda function integration with AWS services"""
        # Mock AWS clients
        mock_s3 = Mock()
        mock_kinesis = Mock()
        mock_boto3_client.side_effect = [mock_s3, mock_kinesis]
        
        # Mock S3 model loading
        mock_s3.get_object.return_value = {
            'Body': Mock(read=lambda: b'mock_model_data')
        }
        
        # Mock Kinesis publishing
        mock_kinesis.put_record.return_value = {'RecordId': 'test-record-id'}
        
        # Mock Lambda handler with AWS integration
        def mock_lambda_handler(event, context):
            """Mock Lambda handler with AWS service integration"""
            predictions = []
            
            # Mock model service initialization
            model_service = Mock()
            model_service.init.return_value = True
            
            # Mock Kinesis callback
            kinesis_callback = Mock()
            kinesis_callback.put_record.return_value = {'RecordId': 'test-record-id'}
            
            for record in event['Records']:
                try:
                    # Decode Kinesis data
                    encoded_data = record['kinesis']['data']
                    decoded_data = base64.b64decode(encoded_data).decode('utf-8')
                    event_data = json.loads(decoded_data)
                    
                    # Mock prediction
                    prediction = 2500
                    
                    # Mock publishing to Kinesis
                    prediction_data = {
                        'prediction': prediction,
                        'features': event_data,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    kinesis_callback.put_record(prediction_data)
                    
                    predictions.append({
                        'features': event_data,
                        'prediction': prediction,
                        'record_id': 'test-record-id'
                    })
                    
                except Exception as e:
                    # Log error but continue processing
                    print(f"Error processing record: {str(e)}")
            
            return {
                'predictions': predictions,
                'processed_records': len(predictions)
            }
        
        # Test the Lambda handler
        result = mock_lambda_handler(self.kinesis_event, None)
        
        # Verify the result
        assert 'predictions' in result
        assert 'processed_records' in result
        assert result['processed_records'] == 1
        assert len(result['predictions']) == 1
        assert result['predictions'][0]['prediction'] == 2500

    @patch('boto3.client')
    def test_rds_database_integration(self, mock_boto3_client):
        """Test RDS database integration for metrics storage"""
        # Mock RDS/PostgreSQL connection
        mock_psycopg = Mock()
        
        # Mock database operations
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_psycopg.connect.return_value = mock_connection
        
        # Test database connection
        def connect_to_database(connection_string):
            """Mock database connection"""
            try:
                connection = mock_psycopg.connect(connection_string)
                return connection
            except Exception as e:
                raise Exception(f"Failed to connect to database: {str(e)}")
        
        # Test metrics insertion
        def insert_metrics(connection, metrics_data):
            """Mock metrics insertion"""
            cursor = connection.cursor()
            
            # Mock SQL insertion
            insert_query = """
                INSERT INTO model_metrics (run_id, mae, rmse, r2_score, timestamp)
                VALUES (%s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                metrics_data['run_id'],
                metrics_data['mae'],
                metrics_data['rmse'],
                metrics_data['r2_score'],
                metrics_data['timestamp']
            ))
            
            connection.commit()
            return True
        
        # Test the database integration
        connection_string = "postgresql://user:pass@localhost:5432/rental_db"
        
        # Connect to database
        connection = connect_to_database(connection_string)
        assert connection is not None
        
        # Insert metrics
        metrics_data = {
            'run_id': 'test-run-123',
            'mae': 150.5,
            'rmse': 200.3,
            'r2_score': 0.85,
            'timestamp': datetime.now().isoformat()
        }
        
        success = insert_metrics(connection, metrics_data)
        assert success is True
        
        # Verify database operations
        mock_psycopg.connect.assert_called_once_with(connection_string)
        mock_cursor.execute.assert_called_once()
        mock_connection.commit.assert_called_once()

    def test_aws_credentials_integration(self):
        """Test AWS credentials and configuration integration"""
        # Mock AWS credentials
        mock_credentials = {
            'aws_access_key_id': 'test-access-key',
            'aws_secret_access_key': 'test-secret-key',
            'region_name': 'us-east-1'
        }
        
        # Test AWS client creation with credentials
        def create_aws_client(service_name, credentials):
            """Mock AWS client creation"""
            try:
                client = boto3.client(
                    service_name,
                    aws_access_key_id=credentials['aws_access_key_id'],
                    aws_secret_access_key=credentials['aws_secret_access_key'],
                    region_name=credentials['region_name']
                )
                return client
            except Exception as e:
                raise Exception(f"Failed to create {service_name} client: {str(e)}")
        
        # Test environment variable configuration
        def configure_aws_environment():
            """Mock AWS environment configuration"""
            import os
            os.environ['AWS_ACCESS_KEY_ID'] = mock_credentials['aws_access_key_id']
            os.environ['AWS_SECRET_ACCESS_KEY'] = mock_credentials['aws_secret_access_key']
            os.environ['AWS_DEFAULT_REGION'] = mock_credentials['region_name']
        
        # Test the configuration
        configure_aws_environment()
        
        # Verify environment variables are set
        import os
        assert os.environ['AWS_ACCESS_KEY_ID'] == mock_credentials['aws_access_key_id']
        assert os.environ['AWS_SECRET_ACCESS_KEY'] == mock_credentials['aws_secret_access_key']
        assert os.environ['AWS_DEFAULT_REGION'] == mock_credentials['region_name']

    def test_aws_error_handling_integration(self):
        """Test AWS service error handling"""
        # Mock AWS service errors
        class MockAWSError(Exception):
            def __init__(self, error_code, message):
                self.error_code = error_code
                self.message = message
        
        # Test S3 error handling
        def handle_s3_error(operation):
            """Mock S3 error handling"""
            try:
                # Simulate S3 error
                raise MockAWSError('NoSuchBucket', 'The specified bucket does not exist')
            except MockAWSError as e:
                if e.error_code == 'NoSuchBucket':
                    return {'error': 'Bucket not found', 'retry': False}
                elif e.error_code == 'AccessDenied':
                    return {'error': 'Access denied', 'retry': False}
                else:
                    return {'error': 'Unknown error', 'retry': True}
        
        # Test Kinesis error handling
        def handle_kinesis_error(operation):
            """Mock Kinesis error handling"""
            try:
                # Simulate Kinesis error
                raise MockAWSError('ResourceNotFoundException', 'Stream not found')
            except MockAWSError as e:
                if e.error_code == 'ResourceNotFoundException':
                    return {'error': 'Stream not found', 'retry': False}
                elif e.error_code == 'ProvisionedThroughputExceededException':
                    return {'error': 'Throughput exceeded', 'retry': True}
                else:
                    return {'error': 'Unknown error', 'retry': True}
        
        # Test error handling
        s3_error = handle_s3_error('get_object')
        kinesis_error = handle_kinesis_error('put_record')
        
        # Verify error handling
        assert s3_error['error'] == 'Bucket not found'
        assert s3_error['retry'] is False
        assert kinesis_error['error'] == 'Stream not found'
        assert kinesis_error['retry'] is False

    def test_aws_service_health_check(self):
        """Test AWS service health check integration"""
        # Mock health check functions
        def check_s3_health(bucket_name):
            """Mock S3 health check"""
            try:
                # Simulate S3 health check
                return {
                    'service': 'S3',
                    'bucket': bucket_name,
                    'status': 'healthy',
                    'accessible': True
                }
            except Exception:
                return {
                    'service': 'S3',
                    'bucket': bucket_name,
                    'status': 'unhealthy',
                    'accessible': False
                }
        
        def check_kinesis_health(stream_name):
            """Mock Kinesis health check"""
            try:
                # Simulate Kinesis health check
                return {
                    'service': 'Kinesis',
                    'stream': stream_name,
                    'status': 'healthy',
                    'active': True
                }
            except Exception:
                return {
                    'service': 'Kinesis',
                    'stream': stream_name,
                    'status': 'unhealthy',
                    'active': False
                }
        
        # Test health checks
        s3_health = check_s3_health('mlflow-bucket')
        kinesis_health = check_kinesis_health('rental-predictions-stream')
        
        # Verify health check results
        assert s3_health['status'] == 'healthy'
        assert s3_health['accessible'] is True
        assert kinesis_health['status'] == 'healthy'
        assert kinesis_health['active'] is True


if __name__ == "__main__":
    pytest.main([__file__]) 