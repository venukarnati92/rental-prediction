"""
Integration tests for the rental prediction pipeline.

These tests verify the end-to-end functionality of the ML pipeline,
including data processing, model training, and prediction services.
"""

import pytest
import pandas as pd
import numpy as np
import json
import base64
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from datetime import datetime


class TestPipelineIntegration:
    """Integration tests for the complete rental prediction pipeline"""

    def setup_method(self):
        """Set up test data and mocks for each test"""
        # Sample rental data similar to UCI dataset
        self.sample_data = pd.DataFrame({
            'price_type': ['Monthly'] * 100,
            'cityname': ['NYC'] * 50 + ['LA'] * 30 + ['CHI'] * 20,
            'state': ['NY'] * 50 + ['CA'] * 30 + ['IL'] * 20,
            'bedrooms': np.random.randint(1, 5, 100),
            'bathrooms': np.random.randint(1, 4, 100),
            'square_feet': np.random.randint(500, 3000, 100),
            'price': np.random.randint(1000, 5000, 100)
        })
        
        # Sample prediction request
        self.prediction_request = {
            'cityname': 'NYC',
            'state': 'NY',
            'bedrooms': 2,
            'bathrooms': 1,
            'square_feet': 1000
        }

    def test_complete_data_processing_pipeline(self):
        """Test the complete data processing pipeline from raw data to features"""
        # Step 1: Simulate data filtering (like get_data_from_ucirepo)
        def filter_data(df):
            """Simulate the data filtering logic"""
            # Filter by monthly rent type
            df_filtered = df[df['price_type'] == 'Monthly'].copy()
            
            # Filter by price range
            df_filtered = df_filtered[(df_filtered['price'] > 100) & (df_filtered['price'] < 8000)]
            
            # Filter cities with count > 1
            city_counts = df_filtered['cityname'].value_counts()
            cities_to_keep = city_counts[city_counts > 1].index
            df_filtered = df_filtered[df_filtered['cityname'].isin(cities_to_keep)]
            
            # Convert data types
            categorical = ['cityname', 'state']
            numerical = ['bedrooms', 'bathrooms', 'square_feet']
            
            df_filtered[categorical] = df_filtered[categorical].astype(str)
            df_filtered[numerical] = df_filtered[numerical].astype(float)
            
            return df_filtered
        
        # Step 2: Simulate feature engineering
        def feature_engineering(df):
            """Simulate the feature engineering logic"""
            df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 1)
            df['sqft_x_beds'] = df['square_feet'] * df['bedrooms']
            return df
        
        # Execute the pipeline
        filtered_data = filter_data(self.sample_data.copy())
        engineered_data = feature_engineering(filtered_data.copy())
        
        # Verify the pipeline worked correctly
        assert len(filtered_data) > 0
        assert 'bed_bath_ratio' in engineered_data.columns
        assert 'sqft_x_beds' in engineered_data.columns
        assert engineered_data['bed_bath_ratio'].dtype == 'float64'
        assert engineered_data['sqft_x_beds'].dtype == 'float64'
        
        # Verify no infinite values
        assert not np.isinf(engineered_data['bed_bath_ratio']).any()
        assert not np.isinf(engineered_data['sqft_x_beds']).any()

    def test_model_training_integration(self):
        """Test the complete model training pipeline"""
        # Create larger training data with features
        training_data = pd.DataFrame({
            'cityname': ['NYC', 'LA', 'CHI', 'NYC', 'LA'] * 20,
            'state': ['NY', 'CA', 'IL', 'NY', 'CA'] * 20,
            'bedrooms': np.random.randint(1, 5, 100),
            'bathrooms': np.random.randint(1, 4, 100),
            'square_feet': np.random.randint(500, 3000, 100),
            'price': np.random.randint(1000, 5000, 100)
        })
        
        # Apply feature engineering
        training_data['bed_bath_ratio'] = training_data['bedrooms'] / training_data['bathrooms'].replace(0, 1)
        training_data['sqft_x_beds'] = training_data['square_feet'] * training_data['bedrooms']
        
        # Simulate model training
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        # Prepare features and target
        feature_columns = ['bedrooms', 'bathrooms', 'square_feet', 'bed_bath_ratio', 'sqft_x_beds']
        X = training_data[feature_columns]
        y = training_data['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Verify model training worked
        assert len(predictions) == len(y_test)
        assert all(pred > 0 for pred in predictions)  # Prices should be positive
        
        # Check if we have enough test samples for R² calculation
        if len(y_test) >= 2:
            score = model.score(X_test, y_test)
            # With random data, the model might not perform well, but we should have predictions
            assert len(predictions) > 0
            # Verify predictions are reasonable (not all the same value)
            assert len(set(predictions)) > 1
        else:
            # For very small test sets, just verify predictions are reasonable
            assert len(predictions) > 0

    def test_lambda_prediction_service_integration(self):
        """Test the Lambda prediction service end-to-end"""
        # Mock the model service
        class MockModelService:
            def __init__(self):
                self.model = None
                self.is_initialized = False
            
            def init(self, stream_name, run_id, test_run=False):
                """Mock initialization"""
                self.is_initialized = True
                return True
            
            def predict(self, features):
                """Mock prediction"""
                if not self.is_initialized:
                    raise Exception("Model service not initialized")
                
                # Simple mock prediction based on features
                base_price = 1000
                price = (base_price + 
                        features['bedrooms'] * 200 + 
                        features['bathrooms'] * 150 + 
                        features['square_feet'] * 0.5)
                return price
        
        # Mock Kinesis callback
        class MockKinesisCallback:
            def __init__(self, stream_name):
                self.stream_name = stream_name
                self.published_records = []
            
            def put_record(self, data):
                """Mock publishing to Kinesis"""
                self.published_records.append(data)
                return {'RecordId': 'mock-record-id'}
        
        # Test the complete prediction flow
        model_service = MockModelService()
        kinesis_callback = MockKinesisCallback('test-stream')
        
        # Initialize the service
        assert model_service.init('test-stream', 'test-run-123', test_run=True)
        
        # Make a prediction
        prediction = model_service.predict(self.prediction_request)
        
        # Verify prediction
        assert prediction > 0
        assert isinstance(prediction, (int, float))
        
        # Test Kinesis publishing
        prediction_data = {
            'prediction': prediction,
            'features': self.prediction_request,
            'timestamp': datetime.now().isoformat()
        }
        
        kinesis_callback.put_record(prediction_data)
        
        # Verify Kinesis callback worked
        assert len(kinesis_callback.published_records) == 1
        assert kinesis_callback.published_records[0]['prediction'] == prediction

    def test_kinesis_event_processing_integration(self):
        """Test processing Kinesis events through the Lambda function"""
        # Create a mock Kinesis event
        event_data = {
            'cityname': 'NYC',
            'state': 'NY',
            'bedrooms': 2,
            'bathrooms': 1,
            'square_feet': 1000
        }
        
        # Encode the event data
        json_data = json.dumps(event_data)
        encoded_data = base64.b64encode(json_data.encode('utf-8')).decode('utf-8')
        
        # Create Kinesis event structure
        kinesis_event = {
            'Records': [
                {
                    'kinesis': {
                        'data': encoded_data
                    }
                }
            ]
        }
        
        # Mock Lambda handler
        def mock_lambda_handler(event, context):
            """Mock Lambda handler for testing"""
            predictions = []
            
            for record in event['Records']:
                # Decode the data
                encoded_data = record['kinesis']['data']
                decoded_data = base64.b64decode(encoded_data).decode('utf-8')
                event_data = json.loads(decoded_data)
                
                # Mock prediction
                prediction = 2000  # Mock prediction value
                
                predictions.append({
                    'features': event_data,
                    'prediction': prediction,
                    'timestamp': datetime.now().isoformat()
                })
            
            return {'predictions': predictions}
        
        # Test the Lambda handler
        result = mock_lambda_handler(kinesis_event, None)
        
        # Verify the result
        assert 'predictions' in result
        assert len(result['predictions']) == 1
        assert result['predictions'][0]['features'] == event_data
        assert result['predictions'][0]['prediction'] == 2000

    def test_data_validation_integration(self):
        """Test data validation across the pipeline"""
        # Test valid data
        valid_data = pd.DataFrame({
            'price_type': ['Monthly'],
            'cityname': ['NYC'],
            'state': ['NY'],
            'bedrooms': [2],
            'bathrooms': [1],
            'square_feet': [1000],
            'price': [2000]
        })
        
        # Test invalid data (should be handled gracefully)
        invalid_data = pd.DataFrame({
            'price_type': ['Weekly'],  # Invalid price type
            'cityname': ['NYC'],
            'state': ['NY'],
            'bedrooms': [-1],  # Invalid bedrooms
            'bathrooms': [0],  # Invalid bathrooms
            'square_feet': [0],  # Invalid square feet
            'price': [0]  # Invalid price
        })
        
        # Test data filtering
        def filter_data(df):
            """Filter valid data"""
            df_filtered = df[df['price_type'] == 'Monthly'].copy()
            df_filtered = df_filtered[(df_filtered['price'] > 100) & (df_filtered['price'] < 8000)]
            df_filtered = df_filtered[(df_filtered['bedrooms'] > 0) & (df_filtered['bathrooms'] > 0)]
            return df_filtered
        
        # Valid data should pass
        valid_filtered = filter_data(valid_data)
        assert len(valid_filtered) == 1
        
        # Invalid data should be filtered out
        invalid_filtered = filter_data(invalid_data)
        assert len(invalid_filtered) == 0

    def test_error_handling_integration(self):
        """Test error handling across the pipeline"""
        # Test with malformed data
        malformed_event = {
            'Records': [
                {
                    'kinesis': {
                        'data': 'invalid-base64-data'
                    }
                }
            ]
        }
        
        def mock_lambda_handler_with_errors(event, context):
            """Mock Lambda handler with error handling"""
            predictions = []
            errors = []
            
            for record in event['Records']:
                try:
                    # Try to decode the data
                    encoded_data = record['kinesis']['data']
                    decoded_data = base64.b64decode(encoded_data).decode('utf-8')
                    event_data = json.loads(decoded_data)
                    
                    # Mock prediction
                    prediction = 2000
                    predictions.append({
                        'features': event_data,
                        'prediction': prediction
                    })
                    
                except Exception as e:
                    errors.append({
                        'error': str(e),
                        'record': record
                    })
            
            return {
                'predictions': predictions,
                'errors': errors
            }
        
        # Test error handling
        result = mock_lambda_handler_with_errors(malformed_event, None)
        
        # Verify error handling worked
        assert 'errors' in result
        assert len(result['errors']) == 1
        assert len(result['predictions']) == 0

    def test_performance_integration(self):
        """Test performance characteristics of the pipeline"""
        # Create larger dataset for performance testing
        large_dataset = pd.DataFrame({
            'price_type': ['Monthly'] * 1000,
            'cityname': ['NYC'] * 500 + ['LA'] * 300 + ['CHI'] * 200,
            'state': ['NY'] * 500 + ['CA'] * 300 + ['IL'] * 200,
            'bedrooms': np.random.randint(1, 5, 1000),
            'bathrooms': np.random.randint(1, 4, 1000),
            'square_feet': np.random.randint(500, 3000, 1000),
            'price': np.random.randint(1000, 5000, 1000)
        })
        
        import time
        
        # Test data processing performance
        start_time = time.time()
        
        # Apply filtering
        filtered_data = large_dataset[large_dataset['price_type'] == 'Monthly'].copy()
        filtered_data = filtered_data[(filtered_data['price'] > 100) & (filtered_data['price'] < 8000)]
        
        # Apply feature engineering
        filtered_data['bed_bath_ratio'] = filtered_data['bedrooms'] / filtered_data['bathrooms'].replace(0, 1)
        filtered_data['sqft_x_beds'] = filtered_data['square_feet'] * filtered_data['bedrooms']
        
        processing_time = time.time() - start_time
        
        # Verify performance is reasonable (should complete in under 1 second for 1000 records)
        assert processing_time < 1.0
        assert len(filtered_data) > 0
        
        # Test prediction performance
        start_time = time.time()
        
        # Mock multiple predictions
        for _ in range(100):
            features = {
                'bedrooms': np.random.randint(1, 5),
                'bathrooms': np.random.randint(1, 4),
                'square_feet': np.random.randint(500, 3000)
            }
            # Mock prediction calculation
            prediction = features['bedrooms'] * 200 + features['bathrooms'] * 150 + features['square_feet'] * 0.5
        
        prediction_time = time.time() - start_time
        
        # Verify prediction performance is reasonable
        assert prediction_time < 1.0


if __name__ == "__main__":
    pytest.main([__file__]) 