# Air Quality Pipeline Workflow

## What Changed?

Instead of running individual Jupyter notebooks for each sensor and each step, we now have a single script that handles notebook 2 and 4 for every sensor.

## New Features for the Model

The model now uses additional features to improve air quality predictions:

**Weather Features:**

- **Temperature** (`temperature_2m_mean`): Daily average temperature affects pollution dispersion
- **Precipitation** (`precipitation_sum`): Rain helps clear pollutants from the air
- **Wind speed** (`wind_speed_10m_max`): Stronger winds disperse pollution more effectively
- **Wind direction** (`wind_direction_10m_dominant`): Shows where pollution is coming from
- **Humidity** (`relative_humidity_2m_mean`): Affects how pollutants behave in the atmosphere
- **Surface pressure** (`surface_pressure_mean`): High/low pressure systems influence air quality

**Temporal Features:**

- **Day of week** (`day_of_week`): Captures weekly patterns (e.g., weekday traffic vs. weekend)

**Lagged Features (Past PM2.5 values):**

- **Yesterday's PM2.5** (`pm25_lag1`)
- **2 days ago** (`pm25_lag2`)
- **3 days ago** (`pm25_lag3`)

These features help the model understand both the environmental conditions that affect air quality and the recent pollution trends.

## The New Way

Now, github Actions will just run one command:

```bash
python run_air_quality_pipelines.py
```

This works in the command prompt as well. This script will:

- Load all sensor configurations from `sensors.json`
- Run the feature pipeline for each sensor
- Run the batch inference for each sensor
- Give a summary of what succeeded and what failed

## How It Works

### 1. Sensor Configuration

All sensors are defined in `sensors.json`. Each sensor has:

- Street location
- Feed ID from the air quality API
- Display name
- Coordinates
- CSV filename for historical data

### 2. Two-Phase Execution

**Phase 1: Feature Pipeline**

- Fetches current air quality data
- Gets weather data
- Creates lagged features (past 3 days)
- Saves everything to the feature store

**Phase 2: Batch Inference**

- Loads the trained model
- Makes predictions for the next 7-10 days
- Generates forecast plots
- Saves predictions back to Hopsworks

### 3. Clear Output

The script shows you exactly what's happening:

- Which sensor is being processed
- Whether each step succeeded or failed
- A final summary of all results

## Example Output

```
==========================================
Air Quality Pipeline Orchestration
Started at: 2024-11-18 14:30:00
==========================================

Found 5 sensors to process:
  - Arnulf-Klett-Platz (arnulf-klett-platz)
  - Stuttgart Am Neckartor (am-neckartor)
  - Bad Cannstatt (bad-cannstatt)
  - Ludwigsburg (ludwigsburg)
  - Bernhausen (bernhausen)

==========================================
PHASE 1: Running Feature Pipelines
==========================================

✓ Feature pipeline completed successfully for arnulf-klett-platz
✓ Feature pipeline completed successfully for am-neckartor
...

==========================================
PHASE 2: Running Batch Inference
==========================================

✓ Batch inference completed successfully for arnulf-klett-platz
✓ Batch inference completed successfully for am-neckartor
...

==========================================
EXECUTION SUMMARY
==========================================

All pipelines completed successfully!
```

## Adding New Sensors

Want to monitor a new location? Just add it to `sensors.json`:

```json
{
  "street": "your-street-name",
  "feed_id": "12345",
  "url": "https://api.waqi.info/feed/@12345",
  "display_name": "Your Location",
  "csv_name": "your-location-air-quality.csv",
  "latitude": "48.xxx",
  "longitude": "9.xxx"
}
```

Then run the script again. No code changes needed!

## What You Need

Before running the script, make sure:

1. You're logged into Hopsworks
2. Your `AQICN_API_KEY` is stored in Hopsworks secrets
3. You have the historical CSV files in the `data/` folder
4. Your air quality model is trained and saved in Hopsworks

## Benefits

- **Less manual work**: One command instead of many notebooks
- **Consistent**: Same code runs for all sensors
- **Scalable**: Easy to add more sensors
- **Error handling**: Clear messages if something goes wrong
- **Trackable**: See results for all sensors at once

## Next Steps

After running the script, check the generated forecast plots in your repository. They'll show the predicted PM2.5 levels for each sensor location.
