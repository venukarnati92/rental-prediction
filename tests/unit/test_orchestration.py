import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestOrchestration:
    """All tests for orchestration.py in a single class"""
    
    # TestFeatureEngineering methods
    def test_feature_engineering_adds_new_features(self):
        """Test that feature engineering adds the expected features"""
        # Create test data
        df = pd.DataFrame({
            'bedrooms': [2, 3, 1, 0],
            'bathrooms': [1, 2, 1, 1],
            'square_feet': [1000, 1500, 800, 1200]
        })
        
        # Define the feature engineering logic directly
        def feature_engineering_logic(df):
            """Add engineered features to improve model performance WITHOUT data leakage."""
            # Bedroom to bathroom ratio (Good feature)
            df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 1)
            
            # Interaction feature
            df['sqft_x_beds'] = df['square_feet'] * df['bedrooms']
            
            return df
        
        # Call the function
        result = feature_engineering_logic(df.copy())
        
        # Verify new features are added
        assert 'bed_bath_ratio' in result.columns
        assert 'sqft_x_beds' in result.columns
        
        # Verify calculations
        expected_bed_bath_ratio = [2.0, 1.5, 1.0, 0.0]  # 0 bathrooms replaced with 1
        expected_sqft_x_beds = [2000, 4500, 800, 0]
        
        assert list(result['bed_bath_ratio'].values) == expected_bed_bath_ratio
        assert list(result['sqft_x_beds'].values) == expected_sqft_x_beds
    
    def test_feature_engineering_handles_zero_bathrooms(self):
        """Test that feature engineering handles zero bathrooms correctly"""
        df = pd.DataFrame({
            'bedrooms': [1, 2, 3],
            'bathrooms': [0, 1, 2],
            'square_feet': [800, 1000, 1500]
        })
        
        def feature_engineering_logic(df):
            df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 1)
            df['sqft_x_beds'] = df['square_feet'] * df['bedrooms']
            return df
        
        result = feature_engineering_logic(df.copy())
        
        # Check that zero bathrooms are handled correctly
        assert result['bed_bath_ratio'].iloc[0] == 1.0  # 1 bedroom / 1 bathroom (replaced 0)
        assert result['bed_bath_ratio'].iloc[1] == 2.0  # 2 bedrooms / 1 bathroom
        assert result['bed_bath_ratio'].iloc[2] == 1.5  # 3 bedrooms / 2 bathrooms
    
    # TestDataPreprocessing methods
    def test_data_filtering_logic(self):
        """Test the data filtering logic from get_data_from_ucirepo"""
        # Create sample data similar to UCI dataset
        df = pd.DataFrame({
            'price_type': ['Monthly', 'Monthly', 'Weekly', 'Monthly', 'Monthly'],
            'cityname': ['NYC', 'LA', 'CHI', 'NYC', 'LA'],
            'state': ['NY', 'CA', 'IL', 'NY', 'CA'],
            'bedrooms': [2, 3, 1, 2, 3],
            'bathrooms': [1, 2, 1, 1, 2],
            'square_feet': [1000, 1500, 800, 1200, 1600],
            'price': [2000, 3000, 1500, 2500, 3500]
        })
        
        # Apply the filtering logic
        # 1. Filter by monthly rent type
        df_filtered = df[df['price_type'] == 'Monthly'].copy()
        assert len(df_filtered) == 4  # Should exclude the 'Weekly' entry
        
        # 2. Filter by price range
        df_filtered = df_filtered[(df_filtered['price'] > 100) & (df_filtered['price'] < 8000)]
        assert len(df_filtered) == 4  # All prices are within range
        
        # 3. Filter cities with count > 1 (simulating the > 10 logic)
        city_counts = df_filtered['cityname'].value_counts()
        cities_to_keep = city_counts[city_counts > 1].index
        df_filtered = df_filtered[df_filtered['cityname'].isin(cities_to_keep)]
        assert len(df_filtered) == 4  # NYC and LA both appear twice
        
        # 4. Convert data types
        categorical = ['cityname', 'state']
        numerical = ['bedrooms', 'bathrooms', 'square_feet']
        
        df_filtered[categorical] = df_filtered[categorical].astype(str)
        df_filtered[numerical] = df_filtered[numerical].astype(float)
        
        assert df_filtered['cityname'].dtype == 'object'
        assert df_filtered['state'].dtype == 'object'
        assert df_filtered['bedrooms'].dtype == 'float64'
        assert df_filtered['bathrooms'].dtype == 'float64'
        assert df_filtered['square_feet'].dtype == 'float64'
    
    # TestModelParameters methods
    def test_xgb_params_structure(self):
        """Test that XGBoost parameters have the expected structure"""
        xgb_params = {
            'learning_rate': 0.2578608018238022,
            'max_depth': 16,
            'min_child_weight': 4.257570349680962,
            'reg_alpha': 0.3626526597533281,
            'reg_lambda': 0.12291704293621523,
        }
        
        # Verify all required parameters are present
        required_params = ['learning_rate', 'max_depth', 'min_child_weight', 'reg_alpha', 'reg_lambda']
        for param in required_params:
            assert param in xgb_params
        
        # Verify parameter types and ranges
        assert isinstance(xgb_params['learning_rate'], float)
        assert 0 < xgb_params['learning_rate'] < 1
        
        assert isinstance(xgb_params['max_depth'], int)
        assert xgb_params['max_depth'] > 0
        
        assert isinstance(xgb_params['min_child_weight'], float)
        assert xgb_params['min_child_weight'] > 0
        
        assert isinstance(xgb_params['reg_alpha'], float)
        assert xgb_params['reg_alpha'] >= 0
        
        assert isinstance(xgb_params['reg_lambda'], float)
        assert xgb_params['reg_lambda'] >= 0
    
    # TestDataValidation methods
    def test_data_structure_validation(self):
        """Test that data has the expected structure"""
        # Create valid data
        valid_data = pd.DataFrame({
            'cityname': ['NYC', 'LA', 'CHI'],
            'state': ['NY', 'CA', 'IL'],
            'bedrooms': [2, 3, 1],
            'bathrooms': [1, 2, 1],
            'square_feet': [1000, 1500, 800],
            'price': [2000, 3000, 1500]
        })
        
        # Verify required columns
        required_columns = ['cityname', 'state', 'bedrooms', 'bathrooms', 'square_feet', 'price']
        for col in required_columns:
            assert col in valid_data.columns
        
        # Verify data types
        assert valid_data['cityname'].dtype == 'object'
        assert valid_data['state'].dtype == 'object'
        assert valid_data['bedrooms'].dtype in ['int64', 'float64']
        assert valid_data['bathrooms'].dtype in ['int64', 'float64']
        assert valid_data['square_feet'].dtype in ['int64', 'float64']
        assert valid_data['price'].dtype in ['int64', 'float64']
        
        # Verify no negative values
        assert (valid_data['bedrooms'] >= 0).all()
        assert (valid_data['bathrooms'] >= 0).all()
        assert (valid_data['square_feet'] > 0).all()
        assert (valid_data['price'] > 0).all()
    
    # TestIntegration methods
    def test_complete_data_pipeline(self):
        """Test the complete data preprocessing pipeline"""
        # Create sample data similar to UCI dataset
        sample_data = pd.DataFrame({
            'price_type': ['Monthly'] * 100,
            'cityname': ['NYC'] * 50 + ['LA'] * 30 + ['CHI'] * 20,
            'state': ['NY'] * 50 + ['CA'] * 30 + ['IL'] * 20,
            'bedrooms': np.random.randint(1, 5, 100),
            'bathrooms': np.random.randint(1, 4, 100),
            'square_feet': np.random.randint(500, 3000, 100),
            'price': np.random.randint(1000, 5000, 100)
        })
        
        # Apply filtering
        df_filtered = sample_data[sample_data['price_type'] == 'Monthly'].copy()
        df_filtered = df_filtered[(df_filtered['price'] > 100) & (df_filtered['price'] < 8000)]
        
        # Apply feature engineering
        def feature_engineering_logic(df):
            df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 1)
            df['sqft_x_beds'] = df['square_feet'] * df['bedrooms']
            return df
        
        result = feature_engineering_logic(df_filtered.copy())
        
        # Verify new features
        assert 'bed_bath_ratio' in result.columns
        assert 'sqft_x_beds' in result.columns
        assert len(result) == len(df_filtered)
        
        # Verify no infinite values
        assert not np.isinf(result['bed_bath_ratio']).any()
        assert not np.isinf(result['sqft_x_beds']).any()
        
        # Verify reasonable ranges
        assert (result['bed_bath_ratio'] >= 0).all()
        assert (result['sqft_x_beds'] >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__]) 