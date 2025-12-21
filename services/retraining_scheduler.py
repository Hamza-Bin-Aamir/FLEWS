"""
Retraining Scheduler for FLEWS
Handles periodic data collection and model retraining
"""

import asyncio
from datetime import datetime
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# Import local modules
from historical_data import (
    add_weather_record,
    get_records_for_training,
    convert_to_training_format,
    cleanup_old_records,
    get_data_stats
)
from weather_api import WeatherAPIClient
from flood_prediction import flood_predictor
from ml_prediction import train_models, get_model_info

# Cities to collect data from
COLLECTION_CITIES = [
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479},
    {"name": "Lahore", "lat": 31.5497, "lon": 74.3436},
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011},
    {"name": "Peshawar", "lat": 34.0151, "lon": 71.5249},
    {"name": "Multan", "lat": 30.1575, "lon": 71.5249},
    {"name": "Rawalpindi", "lat": 33.5651, "lon": 73.0169},
    {"name": "Faisalabad", "lat": 31.4504, "lon": 73.1350},
    {"name": "Quetta", "lat": 30.1798, "lon": 66.9750}
]

# Scheduler instance
scheduler: Optional[AsyncIOScheduler] = None
training_history = []

async def collect_weather_data():
    """Collect weather data from all monitored cities"""
    print(f"[{datetime.now()}] Starting data collection...")
    
    weather_client = WeatherAPIClient()
    collected = 0
    
    for city in COLLECTION_CITIES:
        try:
            # Get current weather
            weather = await weather_client.get_current_weather(city["lat"], city["lon"])
            
            if weather:
                # Calculate flood risk
                prediction = await flood_predictor.predict_flood_risk(city["lat"], city["lon"])
                
                # Prepare weather data
                weather_data = {
                    "rainfall_1h": weather.get("rain", {}).get("1h", 0),
                    "rainfall_3h": weather.get("rain", {}).get("3h", 0),
                    "humidity": weather.get("main", {}).get("humidity", 0),
                    "temperature": weather.get("main", {}).get("temp", 0),
                    "pressure": weather.get("main", {}).get("pressure", 1013),
                    "wind_speed": weather.get("wind", {}).get("speed", 0),
                    "cloud_cover": weather.get("clouds", {}).get("all", 0)
                }
                
                # Add record to historical data
                add_weather_record(
                    lat=city["lat"],
                    lon=city["lon"],
                    location_name=city["name"],
                    weather_data=weather_data,
                    flood_status=prediction.get("risk_level", "Safe"),
                    risk_score=prediction.get("risk_score", 0)
                )
                collected += 1
                
        except Exception as e:
            print(f"Error collecting data for {city['name']}: {e}")
    
    print(f"[{datetime.now()}] Data collection complete. Collected {collected} records.")
    return collected

async def retrain_models():
    """Retrain ML models with historical data"""
    print(f"[{datetime.now()}] Starting model retraining...")
    
    try:
        # Get historical records
        records = get_records_for_training()
        
        if len(records) < 100:
            print(f"Not enough data for retraining ({len(records)} records). Need at least 100.")
            return None
        
        # Convert to training format
        features, labels = convert_to_training_format(records)
        
        # Retrain models
        results = train_models()
        
        # Log training history
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "records_used": len(records),
            "results": results
        }
        training_history.append(history_entry)
        
        # Cleanup old data (keep 1 year)
        removed = cleanup_old_records(365)
        if removed > 0:
            print(f"Cleaned up {removed} old records")
        
        print(f"[{datetime.now()}] Model retraining complete. Results: {results}")
        return results
        
    except Exception as e:
        print(f"Error during retraining: {e}")
        return None

def start_scheduler():
    """Start the background scheduler"""
    global scheduler
    
    if scheduler is not None:
        return scheduler
    
    scheduler = AsyncIOScheduler()
    
    # Data collection every 6 hours
    scheduler.add_job(
        collect_weather_data,
        trigger=IntervalTrigger(hours=6),
        id="data_collection",
        name="Collect Weather Data",
        replace_existing=True
    )
    
    # Model retraining weekly (Sunday at 2 AM)
    scheduler.add_job(
        retrain_models,
        trigger=CronTrigger(day_of_week="sun", hour=2, minute=0),
        id="model_retraining",
        name="Retrain ML Models",
        replace_existing=True
    )
    
    scheduler.start()
    print(f"[{datetime.now()}] Scheduler started with jobs: data_collection (6h), model_retraining (weekly)")
    
    return scheduler

def stop_scheduler():
    """Stop the scheduler"""
    global scheduler
    if scheduler:
        scheduler.shutdown()
        scheduler = None
        print("Scheduler stopped")

def get_scheduler_status():
    """Get current scheduler status"""
    if scheduler is None:
        return {"running": False, "jobs": []}
    
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None
        })
    
    return {
        "running": scheduler.running,
        "jobs": jobs
    }

def get_training_history():
    """Get model training history"""
    return training_history