"""
Daily Feature Pipeline for Air Quality
Fetches daily air quality (PM2.5) and weather forecast data for a specific sensor.
"""

import datetime
import json
import sys
from pathlib import Path
import pandas as pd
import hopsworks
from mlfs.airquality import util
from rich import print


def run_feature_pipeline(sensor_config: dict, api_key: str, root_dir: str = None) -> bool:
    """
    Run the daily feature pipeline for one sensor.

    Args:
        sensor_config: Dictionary containing sensor configuration:
            - street: Street identifier
            - url: AQICN API URL
            - latitude/longitude: GPS coordinates
        api_key: AQICN API key
        root_dir: Root directory of the project (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract sensor configuration
        street = sensor_config['street']
        aqicn_url = sensor_config['url']
        latitude = sensor_config['latitude']
        longitude = sensor_config['longitude']
        city = sensor_config.get('city', 'stuttgart')
        country = sensor_config.get('country', 'germany')

        print(f"\n{'='*60}")
        print(f"Feature Pipeline: {sensor_config.get('display_name', street)}")
        print(f"Street: {street}")
        print(f"{'='*60}\n")

        today = datetime.date.today()

        # Connect to Hopsworks
        print("Connecting to Hopsworks...")
        project = hopsworks.login(engine="python")
        fs = project.get_feature_store()

        # Get feature groups
        print("Loading feature groups...")
        air_quality_fg = fs.get_feature_group(name='air_quality', version=3)
        weather_fg = fs.get_feature_group(name='weather', version=2)

        # Fetch today's air quality data
        print(f"Fetching air quality data for {today}...")
        aq_today_df = util.get_pm25(
            aqicn_url, country, city, street, today, api_key)

        # Add lagged features
        print("Adding lagged PM2.5 features...")
        lag_features = util.get_pm25_lagged_features(
            air_quality_fg, today, country, city, street)
        for lag_name, lag_value in lag_features.items():
            aq_today_df[lag_name] = lag_value

        # Add day of week feature
        aq_today_df['day_of_week'] = today.weekday()
        aq_today_df['day_of_week'] = aq_today_df['day_of_week'].astype('int32')

        print(f"Air quality data shape: {aq_today_df.shape}")

        # Fetch weather forecast
        print("Fetching weather forecast...")
        hourly_df = util.get_hourly_weather_forecast(city, latitude, longitude)
        hourly_df = hourly_df.set_index('date')

        # Get daily weather at noon
        daily_df = hourly_df.between_time('11:59', '12:01')
        daily_df = daily_df.reset_index()
        daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df['city'] = city

        print(f"Weather forecast shape: {daily_df.shape}")

        # Insert data into feature groups
        print("Inserting air quality data...")
        air_quality_fg.insert(aq_today_df)

        print("Inserting weather data...")
        weather_fg.insert(daily_df, wait=True)

        print(f"✓ Feature pipeline completed successfully for {street}")
        return True

    except Exception as e:
        print(
            f"✗ Feature pipeline failed for {sensor_config.get('street', 'unknown')}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point when run as a script"""
    if len(sys.argv) < 2:
        print("Usage: python feature_pipeline.py <sensor_street>")
        print("Example: python feature_pipeline.py arnulf-klett-platz")
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

    # Get API key from Hopsworks secrets
    project = hopsworks.login(engine="python")
    secrets = hopsworks.get_secrets_api()
    api_key = secrets.get_secret("AQICN_API_KEY").value

    # Run the pipeline
    success = run_feature_pipeline(sensor_config, api_key, str(root_dir))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
