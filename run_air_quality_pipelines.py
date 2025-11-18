#!/usr/bin/env python3
"""
Orchestration script to run air quality feature pipeline and batch inference
for multiple sensors in Stuttgart.

This script:
1. Loads sensor configurations from sensors.json
2. For each sensor, runs the feature pipeline
3. For each sensor, runs the batch inference
4. Uses pure Python functions instead of notebooks
"""

from mlfs.airquality.batch_inference import run_batch_inference
from mlfs.airquality.feature_pipeline import run_feature_pipeline
import sys
import json
from pathlib import Path
from datetime import datetime
import hopsworks
from rich import print

# Get the root directory
root_dir = Path(__file__).parent.absolute()
sensors_file = root_dir / "sensors.json"

# Import our pipeline functions
sys.path.insert(0, str(root_dir))


def load_sensors():
    """Load sensor configurations from sensors.json"""
    with open(sensors_file, 'r') as f:
        config = json.load(f)
    return config


def run_feature_pipeline_for_sensor(sensor_config: dict, api_key: str):
    """Run the feature pipeline for a specific sensor"""
    sensor_street = sensor_config['street']

    print(f"\n{'='*60}")
    print(f"Running feature pipeline for sensor: {sensor_street}")
    print(f"{'='*60}\n")

    try:
        success = run_feature_pipeline(sensor_config, api_key, str(root_dir))
        if success:
            print(
                f"✓ Feature pipeline completed successfully for {sensor_street}")
        return success
    except Exception as e:
        print(f"✗ Feature pipeline failed for {sensor_street}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_batch_inference_for_sensor(sensor_config: dict):
    """Run the batch inference for a specific sensor"""
    sensor_street = sensor_config['street']

    print(f"\n{'='*60}")
    print(f"Running batch inference for sensor: {sensor_street}")
    print(f"{'='*60}\n")

    try:
        success = run_batch_inference(sensor_config, str(root_dir))
        if success:
            print(
                f"✓ Batch inference completed successfully for {sensor_street}")
        return success
    except Exception as e:
        print(f"✗ Batch inference failed for {sensor_street}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main orchestration function"""
    print(f"\n{'='*60}")
    print(f"[BLUE] Air Quality Pipeline Orchestration")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Load sensor configurations
    config = load_sensors()
    sensors = config['sensors']

    # Add city and country to each sensor config
    for sensor in sensors:
        sensor['city'] = config['city']
        sensor['country'] = config['country']

    print(f" Found {len(sensors)} sensors to process:")
    for sensor in sensors:
        print(f"  - {sensor['display_name']} ({sensor['street']})")

    # Get API key from Hopsworks
    print("\nConnecting to Hopsworks to get API key...")
    try:
        project = hopsworks.login(engine="python")
        secrets = hopsworks.get_secrets_api()
        api_key = secrets.get_secret("AQICN_API_KEY").value
        print("✓ API key retrieved")
    except Exception as e:
        print(f"✗ Failed to get API key from Hopsworks: {str(e)}")
        sys.exit(1)

    # Track results
    feature_pipeline_results = []
    batch_inference_results = []

    # Run feature pipeline for all sensors
    print(f"\n[bold magenta]{'='*60}")
    print("[bold magenta] PHASE 1: Running Feature Pipelines")
    print(f"[bold magenta]{'='*60}")

    for sensor in sensors:
        success = run_feature_pipeline_for_sensor(sensor, api_key)
        feature_pipeline_results.append({
            'sensor': sensor['street'],
            'success': success
        })

    # Run batch inference for all sensors
    print(f"\n [bold magenta]{'='*60}")
    print("[bold magenta] PHASE 2: Running Batch Inference")
    print(f"[bold magenta]{'='*60}")

    for sensor in sensors:
        success = run_batch_inference_for_sensor(sensor)
        batch_inference_results.append({
            'sensor': sensor['street'],
            'success': success
        })

    # Print summary
    print(f"\n [bold green]{'='*60}")
    print("[bold green]EXECUTION SUMMARY")
    print(f"[bold green]{'='*60}\n")

    print("Feature Pipeline Results:")
    for result in feature_pipeline_results:
        status = "✓ Success" if result['success'] else "✗ Failed"
        print(f"  {result['sensor']}: {status}")

    print("\nBatch Inference Results:")
    for result in batch_inference_results:
        status = "✓ Success" if result['success'] else "✗ Failed"
        print(f"  {result['sensor']}: {status}")

    # Check if all succeeded
    all_success = all(r['success']
                      for r in feature_pipeline_results + batch_inference_results)

    print(f"\n{'='*60}")
    if all_success:
        print("All pipelines completed successfully!")
        print(f"{'='*60}\n")
        sys.exit(0)
    else:
        print("Some pipelines failed. Check the logs above for details.")
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
