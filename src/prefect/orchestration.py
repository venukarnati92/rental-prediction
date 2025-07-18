from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline, Pipeline
import xgboost as xgb
import mlflow
import os
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from datetime import date
import sys
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, ColumnQuantileMetric
from tqdm import tqdm
import logging
import psycopg
import datetime, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


@task
def prep_db(host_name):
    create_table_statement = """
    drop table if exists metrics;
    create table metrics(
        timestamp timestamp,
        prediction_drift float,
        num_drifted_columns integer,
        share_missing_values float,
        price_quantile_25 float,
        price_quantile_50 float,
        price_quantile_75 float
    )
    """
    with psycopg.connect(f"host={host_name} port=5432 user=postgres password=example", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='rentalprediction'")
        if len(res.fetchall()) == 0:
            # Use a lowercase name without hyphens for simplicity and compatibility.
            conn.execute("create database rentalprediction;")
            
    # Now, connect to the 'rentalprediction' database to create the table.
    with psycopg.connect(f"host={host_name} port=5432 dbname=rentalprediction user=postgres password=example") as conn:
        conn.execute(create_table_statement)


@task(retries=3, retry_delay_seconds=60)
def get_data_from_ucirepo():
    """
    Fetches the "Apartment for Rent Classified" dataset (ID 555) from the UCI repository
    and performs basic preprocessing and return dataframe
    """
    # Fetch the dataset from the UCI repository
    apartment_data = fetch_ucirepo(id=555)

    df = apartment_data.data.features
    #filter df by monthly rent type
    df = df[df['price_type'] == 'Monthly'].copy()

    categorical = ['cityname', 'state']
    numerical = ['bedrooms', 'bathrooms', 'square_feet']

    df[categorical] = df[categorical].astype(str)
    df[numerical] = df[numerical].astype(float)

    # Fill missing values in numerical columns with the median
    df[numerical] = df[numerical].fillna(df[numerical].median())

    # Fill missing values in categorical columns with a placeholder (e.g., 'Missing')
    df[categorical] = df[categorical].fillna('Missing')

    # Apply this right after loading the data
    df = df[(df['price'] > 100) & (df['price'] < 8000)]

    #filter cities which has count more than 10 times
    city_counts = df['cityname'].value_counts()
    cities_to_keep = city_counts[city_counts > 10].index
    df = df[df['cityname'].isin(cities_to_keep)]

    return df

@task(retries=3, retry_delay_seconds=60)
def feature_engineering(df):
    """Add engineered features to improve model performance WITHOUT data leakage."""
    # Bedroom to bathroom ratio (Good feature)
    df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 1)

    # Interaction feature
    df['sqft_x_beds'] = df['square_feet'] * df['bedrooms']
    
    return df

@task(retries=3, retry_delay_seconds=60)
def store_data_in_prefect_artifact(rmse):
    rmse_value = float(rmse)  # Ensure rmse is a float
    markdown__rmse_report = f"""# RMSE Report
        ## Summary

        Rental Price Prediction 

        ## RMSE XGBoost Model

        | Region    | RMSE |
        |:----------|-------:|
        | {date.today()} | {rmse_value:.2f} |
        """
    create_markdown_artifact(
            key="rental-prediction-report", markdown=markdown__rmse_report
        )
    
@task(log_prints=True)
def train_model(train_data, val_data, xgb_params, host_name):    
    """
    Train a model using the provided training and validation data.
    """
    categorical = ['cityname', 'state']
    numerical = ['bedrooms', 'bathrooms', 'square_feet'] #, 'bed_bath_ratio', 'sqft_x_beds'

    # Prepare data
    train_dicts = train_data[categorical+numerical].to_dict(orient='records')
    val_dicts = val_data[categorical+numerical].to_dict(orient='records')

    target = 'price'
    y_train = train_data[target].values
    y_val = val_data[target].values

    # Use the provided host_name parameter for MLflow tracking URI
    mlflow.set_tracking_uri(f"http://{host_name}:5000")
    mlflow.set_experiment("rental-prediction")

    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(xgb_params) 

        # Try multiple models and select the best one
        models = {
            'XGBoost': xgb.XGBRegressor(**xgb_params)
        }
        
        model_data = {}
        for name, model in models.items():
            pipeline = make_pipeline(DictVectorizer(), model)
            pipeline.fit(train_dicts, y_train)
            y_pred = pipeline.predict(val_dicts)
            #store prediction in val data and train data
            x_pred = pipeline.predict(train_dicts)
            val_data['prediction'] = y_pred
            train_data['prediction'] = x_pred
            if 'price_display' in val_data.columns:
                val_data['price_display'] = val_data['price_display'].astype(str)
            val_data.to_parquet('data/reference.parquet')
            
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            model_data[name] = f"{rmse:.2f}"
            store_data_in_prefect_artifact(rmse)
            model_data['model_uri'] = mlflow.get_artifact_uri()
            model_data['run_id'] = mlflow.active_run().info.run_id
            print(f"Model {name} is stored in {model_data['model_uri']} with run_id {model_data['run_id']}")

            return model_data, pipeline

@task(cache_policy=None)
def calculate_metrics_postgresql(host_name, df, model):
    #get the unique states from df
    states = df['state'].unique()
    reference_data = pd.read_parquet('data/reference.parquet')
    num_features = ['bedrooms', 'bathrooms', 'square_feet']
    cat_features = ['cityname', 'state']
    target = 'price'
    column_mapping = ColumnMapping(
        prediction='prediction',
        numerical_features=num_features,
        categorical_features=cat_features,
        target=target
    )
    report = Report(metrics = [
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnQuantileMetric(column_name=target, quantile=0.25),
        ColumnQuantileMetric(column_name=target, quantile=0.5),
        ColumnQuantileMetric(column_name=target, quantile=0.75)
    ])
    
    with psycopg.connect(f"host={host_name} port=5432 dbname=rentalprediction user=postgres password=example", autocommit=True) as conn:       
        counter = 0
        for state in states:
            current_data = df[df['state'] == state]

            features_for_prediction = current_data[num_features + cat_features].to_dict(orient='records')
            current_data['prediction'] = model.predict(features_for_prediction)

            report.run(reference_data = reference_data, current_data = current_data,
                column_mapping=column_mapping)

            result = report.as_dict()
            
            prediction_drift = result['metrics'][0]['result']['drift_score']
            num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
            share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
            price_quantile_25 = result['metrics'][3]['result']['current']['value']
            price_quantile_50 = result['metrics'][4]['result']['current']['value']
            price_quantile_75 = result['metrics'][5]['result']['current']['value']

            with conn.cursor() as curr:
                curr.execute(
                    "insert into metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values, price_quantile_25, price_quantile_50, price_quantile_75) values (%s, %s, %s, %s, %s, %s, %s)",
                    (datetime.datetime.now(), prediction_drift, num_drifted_columns, share_missing_values, price_quantile_25, price_quantile_50, price_quantile_75)
                )
                counter += 1
            if counter >15:
                break
            #sleep for 10 seconds
            time.sleep(1)

@flow(log_prints=True)
def run(host_name):
    db_host_name = "localhost"
    df = get_data_from_ucirepo()

    #divide df into train and val with 80/20 split
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

    # Feature engineering
    # train_data = feature_engineering(train_data)
    # val_data = feature_engineering(val_data)

    # XGBoost parameters
    xgb_params = {
        'learning_rate': 0.2578608018238022,
        'max_depth': 16,
        'min_child_weight': 4.257570349680962,
        'reg_alpha': 0.3626526597533281,
        'reg_lambda': 0.12291704293621523,
    }

    model_data, pipeline = train_model(train_data, val_data, xgb_params, host_name)
    store_data_in_prefect_artifact(model_data['XGBoost'])

    prep_db(db_host_name)
    calculate_metrics_postgresql(db_host_name, df, pipeline)

if __name__ == "__main__":
    # read host name from arguments
    host_name = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    

    run.serve(
        name="rental-prediction-deployment",
        cron="0 8 * * *",
        tags=["ml", "prediction", "daily-run"],
        description="Scheduled daily run of the rental price prediction model.",
        version="1.0",
        parameters={"host_name": host_name}
    )

