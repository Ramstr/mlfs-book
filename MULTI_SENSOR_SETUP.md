# Air Quality Multi-Sensor Setup

This codebase has been parametrized to support multiple air quality sensors in Stuttgart.

## Configuration

Sensors are configured in `sensors.json` at the root of the repository. Each sensor has:

- `street`: Unique identifier for the sensor location
- `feed_id`: AQICN feed ID
- `url`: AQICN API URL
- `display_name`: Human-readable name
- `csv_name`: Historical data CSV filename
- `latitude`/`longitude`: GPS coordinates

## Architecture

The system uses **Python scripts** for daily automation (cleaner and easier than notebooks):

- **Notebooks 1 & 3**: Interactive notebooks for initial setup and training (run manually)
- **Python Scripts**: Production-ready scripts for daily operations (automated)

### Python Scripts

#### `mlfs/airquality/feature_pipeline.py`

Fetches daily air quality and weather data for a sensor.

```bash
python -m mlfs.airquality.feature_pipeline arnulf-klett-platz
```

#### `mlfs/airquality/batch_inference.py`

Generates predictions and visualizations for a sensor.

```bash
python -m mlfs.airquality.batch_inference arnulf-klett-platz
```

### Notebooks (for manual operations)

#### 1. Feature Backfill (Notebook 1)

**Usage**: Run once per sensor to initialize historical data.

Change the `SENSOR_INDEX` variable at the top:

```python
SENSOR_INDEX = 0  # 0 for arnulf-klett-platz, 1 for am-neckartor
```

#### 3. Training Pipeline (Notebook 3)

**Usage**: Train a model for a specific sensor (run when you want to retrain).

Change the `SENSOR_STREET` parameter:

```python
SENSOR_STREET = "arnulf-klett-platz"  # or "am-neckartor"
```

Each sensor gets its own model: `air_quality_xgboost_model_{street}`

## Automated Daily Pipeline

The script `run_air_quality_pipelines.py` orchestrates both pipelines for all sensors:

```bash
python run_air_quality_pipelines.py
```

This script:

1. Reads all sensors from `sensors.json`
2. Runs the feature pipeline for each sensor
3. Runs batch inference for each sensor
4. Reports success/failure for each step

**Benefits of Python scripts over notebooks:**

- Simpler error handling and debugging
- No notebook overhead or papermill complexity
- Cleaner code and better version control
- Native function calls with clear parameters
- Easier to test and maintain

## GitHub Actions

The workflow `.github/workflows/air-quality-daily.yml` runs daily and executes the orchestration script for all sensors.

## Adding New Sensors

1. Add sensor configuration to `sensors.json`
2. Run notebook 1 with the new sensor's index to backfill historical data
3. Run notebook 3 to train a model for the new sensor
4. The daily pipeline will automatically include the new sensor

## Feature Groups

All sensors share the same feature groups:

- `air_quality` (version 3): PM2.5 measurements with primary key `[country, city, street]`
- `weather` (version 2): Weather data with primary key `[city]`

## Models

Each sensor has its own model in Hopsworks:

- `air_quality_xgboost_model_arnulf_klett_platz`
- `air_quality_xgboost_model_am_neckartor`

## Secrets

Each sensor has its own location secret in Hopsworks:

- `SENSOR_LOCATION_JSON_ARNULF_KLETT_PLATZ`
- `SENSOR_LOCATION_JSON_AM_NECKARTOR`

These are created automatically when running notebook 1 for each sensor.
