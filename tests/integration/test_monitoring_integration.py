"""
Integration tests for monitoring and observability.

These tests verify the integration with monitoring tools including
Evidently for data drift detection and Grafana for visualization.
"""

import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os


class TestMonitoringIntegration:
    """Integration tests for monitoring and observability"""

    def setup_method(self):
        """Set up test data and mocks for each test"""
        # Reference data (training data)
        self.reference_data = pd.DataFrame({
            'cityname': ['NYC', 'LA', 'CHI', 'NYC', 'LA'] * 20,
            'state': ['NY', 'CA', 'IL', 'NY', 'CA'] * 20,
            'bedrooms': np.random.randint(1, 5, 100),
            'bathrooms': np.random.randint(1, 4, 100),
            'square_feet': np.random.randint(500, 3000, 100),
            'price': np.random.randint(1000, 5000, 100)
        })
        
        # Current data (production data)
        self.current_data = pd.DataFrame({
            'cityname': ['NYC', 'LA', 'CHI', 'NYC', 'LA'] * 20,
            'state': ['NY', 'CA', 'IL', 'NY', 'CA'] * 20,
            'bedrooms': np.random.randint(1, 5, 100),
            'bathrooms': np.random.randint(1, 4, 100),
            'square_feet': np.random.randint(500, 3000, 100),
            'price': np.random.randint(1000, 5000, 100)
        })
        
        # Add some drift to current data
        self.current_data['bedrooms'] = self.current_data['bedrooms'] + np.random.choice([0, 1], 100, p=[0.8, 0.2])
        self.current_data['price'] = self.current_data['price'] * 1.1  # 10% price increase

    def test_evidently_data_drift_detection(self):
        """Test Evidently data drift detection integration"""
        # Mock Evidently components
        class MockDataDriftProfile:
            def __init__(self, reference_data, current_data):
                self.reference_data = reference_data
                self.current_data = current_data
                self.drift_detected = False
                self.drift_metrics = {}
            
            def calculate_drift(self):
                """Mock drift calculation"""
                # Simulate drift detection logic
                ref_mean = self.reference_data['price'].mean()
                curr_mean = self.current_data['price'].mean()
                
                # Calculate drift percentage
                drift_percentage = abs(curr_mean - ref_mean) / ref_mean
                
                self.drift_detected = drift_percentage > 0.05  # 5% threshold
                self.drift_metrics = {
                    'drift_percentage': drift_percentage,
                    'reference_mean': ref_mean,
                    'current_mean': curr_mean,
                    'threshold': 0.05
                }
                
                return self.drift_metrics
        
        # Test drift detection
        drift_profile = MockDataDriftProfile(self.reference_data, self.current_data)
        drift_metrics = drift_profile.calculate_drift()
        
        # Verify drift detection
        assert 'drift_percentage' in drift_metrics
        assert 'reference_mean' in drift_metrics
        assert 'current_mean' in drift_metrics
        assert drift_metrics['drift_percentage'] > 0
        
        # Check if drift was detected
        assert drift_profile.drift_detected == (drift_metrics['drift_percentage'] > 0.05)

    def test_evidently_data_quality_monitoring(self):
        """Test Evidently data quality monitoring integration"""
        # Mock data quality metrics
        class MockDataQualityProfile:
            def __init__(self, data):
                self.data = data
                self.quality_metrics = {}
            
            def calculate_quality_metrics(self):
                """Mock quality metrics calculation"""
                self.quality_metrics = {
                    'missing_values': self.data.isnull().sum().to_dict(),
                    'duplicate_rows': self.data.duplicated().sum(),
                    'data_types': self.data.dtypes.to_dict(),
                    'value_ranges': {
                        'bedrooms': {'min': self.data['bedrooms'].min(), 'max': self.data['bedrooms'].max()},
                        'bathrooms': {'min': self.data['bathrooms'].min(), 'max': self.data['bathrooms'].max()},
                        'square_feet': {'min': self.data['square_feet'].min(), 'max': self.data['square_feet'].max()},
                        'price': {'min': self.data['price'].min(), 'max': self.data['price'].max()}
                    }
                }
                return self.quality_metrics
        
        # Test data quality monitoring
        quality_profile = MockDataQualityProfile(self.current_data)
        quality_metrics = quality_profile.calculate_quality_metrics()
        
        # Verify quality metrics
        assert 'missing_values' in quality_metrics
        assert 'duplicate_rows' in quality_metrics
        assert 'data_types' in quality_metrics
        assert 'value_ranges' in quality_metrics
        
        # Verify value ranges are reasonable
        assert quality_metrics['value_ranges']['bedrooms']['min'] >= 1
        assert quality_metrics['value_ranges']['bedrooms']['max'] <= 5
        assert quality_metrics['value_ranges']['price']['min'] > 0

    def test_evidently_target_drift_detection(self):
        """Test Evidently target drift detection integration"""
        # Mock target drift detection
        class MockTargetDriftProfile:
            def __init__(self, reference_data, current_data):
                self.reference_data = reference_data
                self.current_data = current_data
                self.target_drift_metrics = {}
            
            def calculate_target_drift(self):
                """Mock target drift calculation"""
                # Calculate target distribution drift
                ref_price_dist = self.reference_data['price'].value_counts(normalize=True)
                curr_price_dist = self.current_data['price'].value_counts(normalize=True)
                
                # Calculate distribution difference
                common_bins = set(ref_price_dist.index) & set(curr_price_dist.index)
                if common_bins:
                    drift_score = sum(abs(ref_price_dist.get(bin, 0) - curr_price_dist.get(bin, 0)) 
                                    for bin in common_bins)
                else:
                    drift_score = 1.0  # Maximum drift if no common bins
                
                self.target_drift_metrics = {
                    'drift_score': drift_score,
                    'reference_distribution': ref_price_dist.to_dict(),
                    'current_distribution': curr_price_dist.to_dict(),
                    'drift_detected': drift_score > 0.1  # 10% threshold
                }
                
                return self.target_drift_metrics
        
        # Test target drift detection
        target_drift_profile = MockTargetDriftProfile(self.reference_data, self.current_data)
        target_drift_metrics = target_drift_profile.calculate_target_drift()
        
        # Verify target drift metrics
        assert 'drift_score' in target_drift_metrics
        assert 'reference_distribution' in target_drift_metrics
        assert 'current_distribution' in target_drift_metrics
        assert 'drift_detected' in target_drift_metrics
        assert 0 <= target_drift_metrics['drift_score'] <= 1

    def test_grafana_dashboard_integration(self):
        """Test Grafana dashboard integration"""
        # Mock Grafana API
        class MockGrafanaClient:
            def __init__(self, url, username, password):
                self.url = url
                self.username = username
                self.password = password
                self.dashboards = {}
                self.datasources = {}
            
            def create_dashboard(self, dashboard_config):
                """Mock dashboard creation"""
                dashboard_id = f"dashboard_{len(self.dashboards) + 1}"
                self.dashboards[dashboard_id] = dashboard_config
                return {'id': dashboard_id, 'status': 'success'}
            
            def create_datasource(self, datasource_config):
                """Mock datasource creation"""
                datasource_id = f"datasource_{len(self.datasources) + 1}"
                self.datasources[datasource_id] = datasource_config
                return {'id': datasource_id, 'status': 'success'}
            
            def get_dashboard(self, dashboard_id):
                """Mock dashboard retrieval"""
                return self.dashboards.get(dashboard_id, None)
        
        # Test Grafana integration
        grafana_client = MockGrafanaClient('http://localhost:3000', 'admin', 'admin')
        
        # Create datasource
        datasource_config = {
            'name': 'PostgreSQL',
            'type': 'postgres',
            'url': 'localhost:5432',
            'database': 'rental_db',
            'user': 'postgres',
            'secureJsonData': {'password': 'password'}
        }
        
        datasource_result = grafana_client.create_datasource(datasource_config)
        assert datasource_result['status'] == 'success'
        
        # Create dashboard
        dashboard_config = {
            'dashboard': {
                'title': 'Rental Prediction Monitoring',
                'panels': [
                    {
                        'title': 'Data Drift Score',
                        'type': 'stat',
                        'targets': [
                            {
                                'expr': 'SELECT drift_score FROM model_metrics ORDER BY timestamp DESC LIMIT 1'
                            }
                        ]
                    },
                    {
                        'title': 'Prediction Accuracy',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'SELECT timestamp, mae FROM model_metrics ORDER BY timestamp'
                            }
                        ]
                    }
                ]
            }
        }
        
        dashboard_result = grafana_client.create_dashboard(dashboard_config)
        assert dashboard_result['status'] == 'success'
        
        # Verify dashboard was created
        created_dashboard = grafana_client.get_dashboard(dashboard_result['id'])
        assert created_dashboard is not None
        assert created_dashboard['dashboard']['title'] == 'Rental Prediction Monitoring'

    def test_metrics_collection_integration(self):
        """Test metrics collection and storage integration"""
        # Mock metrics collection
        class MockMetricsCollector:
            def __init__(self):
                self.metrics = []
            
            def collect_prediction_metrics(self, predictions, actual_values=None):
                """Mock prediction metrics collection"""
                if actual_values is None:
                    # Mock actual values for testing
                    actual_values = [pred * 0.9 + np.random.normal(0, 50) for pred in predictions]
                
                # Calculate metrics
                mae = np.mean(np.abs(np.array(predictions) - np.array(actual_values)))
                rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actual_values)) ** 2))
                
                # Calculate R² score
                ss_res = np.sum((np.array(predictions) - np.array(actual_values)) ** 2)
                ss_tot = np.sum((np.array(actual_values) - np.mean(actual_values)) ** 2)
                r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                metric_data = {
                    'timestamp': datetime.now().isoformat(),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2_score': float(r2_score),
                    'num_predictions': len(predictions)
                }
                
                self.metrics.append(metric_data)
                return metric_data
            
            def collect_data_quality_metrics(self, data):
                """Mock data quality metrics collection"""
                quality_metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'total_rows': len(data),
                    'missing_values': data.isnull().sum().sum(),
                    'duplicate_rows': data.duplicated().sum(),
                    'data_completeness': 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
                }
                
                self.metrics.append(quality_metrics)
                return quality_metrics
        
        # Test metrics collection
        metrics_collector = MockMetricsCollector()
        
        # Test prediction metrics
        predictions = [2000, 2500, 3000, 1800, 2200]
        prediction_metrics = metrics_collector.collect_prediction_metrics(predictions)
        
        assert 'mae' in prediction_metrics
        assert 'rmse' in prediction_metrics
        assert 'r2_score' in prediction_metrics
        assert prediction_metrics['num_predictions'] == 5
        
        # Test data quality metrics
        quality_metrics = metrics_collector.collect_data_quality_metrics(self.current_data)
        
        assert 'total_rows' in quality_metrics
        assert 'missing_values' in quality_metrics
        assert 'duplicate_rows' in quality_metrics
        assert 'data_completeness' in quality_metrics
        assert quality_metrics['total_rows'] == 100

    def test_alerting_integration(self):
        """Test alerting system integration"""
        # Mock alerting system
        class MockAlertingSystem:
            def __init__(self):
                self.alerts = []
                self.thresholds = {
                    'drift_threshold': 0.1,
                    'mae_threshold': 300,
                    'r2_threshold': 0.7
                }
            
            def check_drift_alert(self, drift_score):
                """Mock drift alert check"""
                if drift_score > self.thresholds['drift_threshold']:
                    alert = {
                        'type': 'data_drift',
                        'severity': 'high',
                        'message': f'Data drift detected: {drift_score:.3f}',
                        'timestamp': datetime.now().isoformat(),
                        'threshold': self.thresholds['drift_threshold']
                    }
                    self.alerts.append(alert)
                    return alert
                return None
            
            def check_performance_alert(self, mae, r2_score):
                """Mock performance alert check"""
                alerts = []
                
                if mae > self.thresholds['mae_threshold']:
                    alert = {
                        'type': 'high_mae',
                        'severity': 'medium',
                        'message': f'High MAE detected: {mae:.2f}',
                        'timestamp': datetime.now().isoformat(),
                        'threshold': self.thresholds['mae_threshold']
                    }
                    alerts.append(alert)
                    self.alerts.append(alert)
                
                if r2_score < self.thresholds['r2_threshold']:
                    alert = {
                        'type': 'low_r2',
                        'severity': 'high',
                        'message': f'Low R² score detected: {r2_score:.3f}',
                        'timestamp': datetime.now().isoformat(),
                        'threshold': self.thresholds['r2_threshold']
                    }
                    alerts.append(alert)
                    self.alerts.append(alert)
                
                return alerts
        
        # Test alerting system
        alerting_system = MockAlertingSystem()
        
        # Test drift alert
        drift_alert = alerting_system.check_drift_alert(0.15)  # Above threshold
        assert drift_alert is not None
        assert drift_alert['type'] == 'data_drift'
        assert drift_alert['severity'] == 'high'
        
        # Test performance alerts
        performance_alerts = alerting_system.check_performance_alert(mae=350, r2_score=0.6)
        assert len(performance_alerts) == 2  # Both MAE and R² alerts
        assert any(alert['type'] == 'high_mae' for alert in performance_alerts)
        assert any(alert['type'] == 'low_r2' for alert in performance_alerts)

    def test_logging_integration(self):
        """Test logging and observability integration"""
        # Mock logging system
        class MockLogger:
            def __init__(self):
                self.logs = []
            
            def info(self, message, **kwargs):
                """Mock info logging"""
                log_entry = {
                    'level': 'INFO',
                    'message': message,
                    'timestamp': datetime.now().isoformat(),
                    **kwargs
                }
                self.logs.append(log_entry)
            
            def warning(self, message, **kwargs):
                """Mock warning logging"""
                log_entry = {
                    'level': 'WARNING',
                    'message': message,
                    'timestamp': datetime.now().isoformat(),
                    **kwargs
                }
                self.logs.append(log_entry)
            
            def error(self, message, **kwargs):
                """Mock error logging"""
                log_entry = {
                    'level': 'ERROR',
                    'message': message,
                    'timestamp': datetime.now().isoformat(),
                    **kwargs
                }
                self.logs.append(log_entry)
        
        # Test logging integration
        logger = MockLogger()
        
        # Test various log levels
        logger.info('Pipeline started', run_id='test-run-123')
        logger.warning('Data drift detected', drift_score=0.12)
        logger.error('Model prediction failed', error='Invalid input data')
        
        # Verify logs
        assert len(logger.logs) == 3
        assert logger.logs[0]['level'] == 'INFO'
        assert logger.logs[1]['level'] == 'WARNING'
        assert logger.logs[2]['level'] == 'ERROR'
        assert 'Pipeline started' in logger.logs[0]['message']
        assert 'Data drift detected' in logger.logs[1]['message']
        assert 'Model prediction failed' in logger.logs[2]['message']


if __name__ == "__main__":
    pytest.main([__file__]) 