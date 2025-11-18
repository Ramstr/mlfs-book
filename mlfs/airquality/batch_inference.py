"""
Daily Batch Inference for Air Quality
Generates predictions for a specific sensor and creates forecast visualizations.
"""

import datetime
import json
import sys
import os
from pathlib import Path
import pandas as pd
import hopsworks
from xgboost import XGBRegressor
from mlfs.airquality import util
from rich import print


def run_batch_inference(sensor_config: dict, root_dir: str = None) -> bool:
    """
    Run batch inference for one sensor.

    Args:
        sensor_config: Dictionary containing sensor configuration:
            - street: Street identifier
            - city: City name
            - country: Country name
        root_dir: Root directory of the project (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract sensor configuration
        street = sensor_config['street']
        city = sensor_config.get('city', 'stuttgart')
        country = sensor_config.get('country', 'germany')

        print(f"\n{'='*60}")
        print(f"Batch Inference: {sensor_config.get('display_name', street)}")
        print(f"Street: {street}")
        print(f"{'='*60}\n")

        today = datetime.datetime.now() - datetime.timedelta(0)

        # Connect to Hopsworks
        print("Connecting to Hopsworks...")
        project = hopsworks.login(engine="python")
        fs = project.get_feature_store()
        mr = project.get_model_registry()

        # Load the model
        model_name = f"air_quality_xgboost_model_{street.replace('-', '_')}"
        print(f"Loading model: {model_name}...")

        retrieved_model = mr.get_model(name=model_name, version=1)
        saved_model_dir = retrieved_model.download()

        retrieved_xgboost_model = XGBRegressor()
        retrieved_xgboost_model.load_model(saved_model_dir + "/model.json")

        # Get feature groups
        print("Loading feature groups...")
        air_quality_fg = fs.get_feature_group(name='air_quality', version=3)
        weather_fg = fs.get_feature_group(name='weather', version=2)

        # Get weather forecast data
        print("Fetching weather forecast data...")
        batch_data = weather_fg.filter(weather_fg.date >= today).read()

        if len(batch_data) == 0:
            print("No weather forecast data available")
            return False

        # Get lagged features
        print("Adding lagged PM2.5 features...")
        lag_features = util.get_pm25_lagged_features(
            air_quality_fg, today, country, city, street)

        # Prepare batch data with lagged features
        batch_data = batch_data.sort_values('date').reset_index(drop=True)
        batch_data['pm25_lag1'] = None
        batch_data['pm25_lag2'] = None
        batch_data['pm25_lag3'] = None

        # Set initial lagged values
        batch_data.loc[0, 'pm25_lag1'] = lag_features['pm25_lag1']
        batch_data.loc[0, 'pm25_lag2'] = lag_features['pm25_lag2']
        batch_data.loc[0, 'pm25_lag3'] = lag_features['pm25_lag3']

        batch_data['pm25_lag1'] = batch_data['pm25_lag1'].astype('float32')
        batch_data['pm25_lag2'] = batch_data['pm25_lag2'].astype('float32')
        batch_data['pm25_lag3'] = batch_data['pm25_lag3'].astype('float32')

        # Add day of week feature
        batch_data['day_of_week'] = batch_data['date'].dt.dayofweek
        batch_data['day_of_week'] = batch_data['day_of_week'].astype('int32')

        # Make predictions iteratively
        print(f"Making predictions for {len(batch_data)} days...")
        predictions = []

        for i in range(len(batch_data)):
            # Get current row features
            current_features = batch_data.iloc[[i]][[
                'pm25_lag1', 'pm25_lag2', 'pm25_lag3', 'day_of_week',
                'temperature_2m_mean', 'precipitation_sum',
                'wind_speed_10m_max', 'wind_direction_10m_dominant',
                'relative_humidity_2m_mean', 'surface_pressure_mean'
            ]].copy()

            # Make prediction
            pred = retrieved_xgboost_model.predict(current_features)[0]
            predictions.append(pred)

            # Update lagged features for next prediction
            if i + 1 < len(batch_data):
                batch_data.loc[i + 1, 'pm25_lag1'] = pred
                batch_data.loc[i + 1,
                               'pm25_lag2'] = batch_data.loc[i, 'pm25_lag1']
                batch_data.loc[i + 1,
                               'pm25_lag3'] = batch_data.loc[i, 'pm25_lag2']

        batch_data['predicted_pm25'] = predictions

        # Prepare monitoring data
        print("Preparing monitoring data...")
        batch_data_monitor = batch_data.copy()
        batch_data_monitor['street'] = street
        batch_data_monitor['city'] = city
        batch_data_monitor['country'] = country
        batch_data_monitor['days_before_forecast_day'] = range(
            1, len(batch_data_monitor) + 1)
        batch_data_monitor = batch_data_monitor.sort_values(by=['date'])

        # Generate forecast plot
        if root_dir is None:
            root_dir = Path(__file__).parent.parent.parent
        else:
            root_dir = Path(root_dir)

        print("Generating forecast plot...")
        pred_file_path = root_dir / "docs" / "air-quality" / \
            "assets" / "img" / f"pm25_forecast_{street}.png"
        pred_file_path.parent.mkdir(parents=True, exist_ok=True)

        plt = util.plot_air_quality_forecast(
            city, street, batch_data_monitor, str(pred_file_path))

        # Save predictions to monitoring feature group
        print("Saving predictions to feature group...")
        monitor_fg = fs.get_or_create_feature_group(
            name='aq_predictions',
            description='Air Quality prediction monitoring',
            version=3,
            primary_key=['city', 'street', 'date', 'days_before_forecast_day'],
            event_time="date"
        )
        monitor_fg.insert(batch_data_monitor, wait=True)

        # Generate hindcast plot
        print("Generating hindcast plot...")
        monitoring_df = monitor_fg.filter(
            monitor_fg.days_before_forecast_day == 1).read()
        air_quality_df = air_quality_fg.read()

        outcome_df = air_quality_df[['date', 'pm25']]
        preds_df = monitoring_df[['date', 'predicted_pm25']]

        hindcast_df = pd.merge(preds_df, outcome_df, on="date")
        hindcast_df = hindcast_df.sort_values(by=['date'])

        if len(hindcast_df) == 0:
            print("No hindcast data available yet (this is normal initially)")
            hindcast_df = util.backfill_predictions_for_monitoring(
                weather_fg, air_quality_df, monitor_fg, retrieved_xgboost_model
            )

        hindcast_file_path = root_dir / "docs" / "air-quality" / \
            "assets" / "img" / f"pm25_hindcast_1day_{street}.png"
        plt = util.plot_air_quality_forecast(
            city, street, hindcast_df, str(hindcast_file_path), hindcast=True)

        # Upload to Hopsworks
        print("Uploading plots to Hopsworks...")
        dataset_api = project.get_dataset_api()
        str_today = today.strftime("%Y-%m-%d")

        if not dataset_api.exists("Resources/airquality"):
            dataset_api.mkdir("Resources/airquality")

        dataset_api.upload(str(
            pred_file_path), f"Resources/airquality/{city}_{street}_{str_today}", overwrite=True)
        dataset_api.upload(str(hindcast_file_path),
                           f"Resources/airquality/{city}_{street}_{str_today}", overwrite=True)

        print(f"✓ Batch inference completed successfully for {street}")
        print(f"  - Forecast plot: {pred_file_path}")
        print(f"  - Hindcast plot: {hindcast_file_path}")

        return True

    except Exception as e:
        print(
            f"✗ Batch inference failed for {sensor_config.get('street', 'unknown')}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point when run as a script"""
    if len(sys.argv) < 2:
        print("Usage: python batch_inference.py <sensor_street>")
        print("Example: python batch_inference.py arnulf-klett-platz")
        sys.exit(1)

    sensor_street = sys.argv[1]

    # Load sensor configuration
    root_dir = Path(__file__).parent.parent.parent
    sensors_file = root_dir / "sensors.json"

    with open(sensors_file, 'r') as f:
        config = json.load(f)

    # Find the sensor
    sensor_config = None
    for sensor in config['sensors']:
        if sensor['street'] == sensor_street:
            sensor_config = sensor
            sensor_config['city'] = config['city']
            sensor_config['country'] = config['country']
            break

    if not sensor_config:
        print(f"Error: Sensor '{sensor_street}' not found in sensors.json")
        sys.exit(1)

    # Run the inference
    success = run_batch_inference(sensor_config, str(root_dir))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
