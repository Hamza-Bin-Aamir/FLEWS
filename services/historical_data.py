"""
Historical Data Manager for FLEWS
Stores and manages historical weather and flood data for model retraining
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

# Data storage path
DATA_DIR = Path(__file__).parent / "data"
HISTORICAL_DATA_FILE = DATA_DIR / "historical_flood_data.json"

def ensure_data_dir():
    """Create data directory if it doesn't exist"""
    DATA_DIR.mkdir(exist_ok=True)
    if not HISTORICAL_DATA_FILE.exists():
        with open(HISTORICAL_DATA_FILE, 'w') as f:
            json.dump({"records": [], "metadata": {"created": datetime.now().isoformat()}}, f)

def load_historical_data() -> Dict:
    """Load historical data from JSON file"""
    ensure_data_dir()
    try:
        with open(HISTORICAL_DATA_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"records": [], "metadata": {"created": datetime.now().isoformat()}}

def save_historical_data(data: Dict):
    """Save historical data to JSON file"""
    ensure_data_dir()
    with open(HISTORICAL_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def add_weather_record(
    lat: float,
    lon: float,
    location_name: str,
    weather_data: Dict,
    flood_status: str,
    risk_score: float
):
    """
    Add a new weather/flood record to historical data
    
    Args:
        lat: Latitude
        lon: Longitude
        location_name: Name of location
        weather_data: Weather data from API (rainfall, humidity, temp, etc.)
        flood_status: Current flood status (Safe/At Risk/Flooded)
        risk_score: Calculated risk score (0-100)
    """
    data = load_historical_data()
    
    record = {
        "timestamp": datetime.now().isoformat(),
        "location": {
            "lat": lat,
            "lon": lon,
            "name": location_name
        },
        "weather": {
            "rainfall_1h": weather_data.get("rainfall_1h", 0),
            "rainfall_3h": weather_data.get("rainfall_3h", 0),
            "humidity": weather_data.get("humidity", 0),
            "temperature": weather_data.get("temperature", 0),
            "pressure": weather_data.get("pressure", 1013),
            "wind_speed": weather_data.get("wind_speed", 0),
            "cloud_cover": weather_data.get("cloud_cover", 0)
        },
        "flood_status": flood_status,
        "risk_score": risk_score
    }
    
    data["records"].append(record)
    data["metadata"]["last_updated"] = datetime.now().isoformat()
    data["metadata"]["total_records"] = len(data["records"])
    
    save_historical_data(data)
    return record

def get_records_since(days: int = 30) -> List[Dict]:
    """Get records from the last N days"""
    data = load_historical_data()
    cutoff = datetime.now() - timedelta(days=days)
    
    recent_records = []
    for record in data["records"]:
        record_time = datetime.fromisoformat(record["timestamp"])
        if record_time >= cutoff:
            recent_records.append(record)
    
    return recent_records

def get_records_for_training() -> List[Dict]:
    """Get all records formatted for ML training"""
    data = load_historical_data()
    return data["records"]

def get_data_stats() -> Dict:
    """Get statistics about the historical data"""
    data = load_historical_data()
    records = data["records"]
    
    if not records:
        return {
            "total_records": 0,
            "date_range": None,
            "locations": [],
            "status_distribution": {}
        }
    
    # Calculate stats
    locations = set()
    status_counts = {"Safe": 0, "At Risk": 0, "Flooded": 0}
    
    for record in records:
        locations.add(record["location"]["name"])
        status = record.get("flood_status", "Safe")
        if status in status_counts:
            status_counts[status] += 1
    
    timestamps = [datetime.fromisoformat(r["timestamp"]) for r in records]
    
    return {
        "total_records": len(records),
        "date_range": {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat()
        },
        "locations": list(locations),
        "status_distribution": status_counts,
        "metadata": data.get("metadata", {})
    }

def cleanup_old_records(days_to_keep: int = 365):
    """Remove records older than specified days"""
    data = load_historical_data()
    cutoff = datetime.now() - timedelta(days=days_to_keep)
    
    original_count = len(data["records"])
    data["records"] = [
        r for r in data["records"]
        if datetime.fromisoformat(r["timestamp"]) >= cutoff
    ]
    
    removed = original_count - len(data["records"])
    if removed > 0:
        data["metadata"]["last_cleanup"] = datetime.now().isoformat()
        data["metadata"]["records_removed"] = removed
        save_historical_data(data)
    
    return removed

def convert_to_training_format(records: List[Dict]) -> tuple:
    """
    Convert historical records to ML training format
    
    Returns:
        features: List of feature arrays
        labels: List of labels (0=Safe, 1=At Risk, 2=Flooded)
    """
    features = []
    labels = []
    
    label_map = {"Safe": 0, "At Risk": 1, "Flooded": 2}
    
    for record in records:
        weather = record["weather"]
        
        # Extract features (matching ml_prediction.py format)
        feature = [
            weather.get("rainfall_1h", 0),
            weather.get("rainfall_3h", 0),
            weather.get("rainfall_1h", 0) * 8,  # Estimate 24h from 3h
            weather.get("humidity", 50),
            weather.get("temperature", 25),
            weather.get("pressure", 1013),
            weather.get("wind_speed", 0),
            weather.get("cloud_cover", 50),
            weather.get("rainfall_1h", 0) * 4,  # Forecast estimate
            0.5,  # river_proximity (default)
            0.5,  # elevation (default)
            weather.get("humidity", 50) / 100,  # soil_saturation estimate
            0,    # previous_flood (default)
            1 if datetime.now().month in [7, 8, 9] else 0  # monsoon_season
        ]
        
        features.append(feature)
        labels.append(label_map.get(record.get("flood_status", "Safe"), 0))
    
    return features, labels