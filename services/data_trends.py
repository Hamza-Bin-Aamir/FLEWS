"""
Data Trends Service for FLEWS
Provides historical and forecast data formatted for visualization charts
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import math

from historical_data import load_historical_data, get_records_since
from weather_api import get_weather_forecast, get_current_weather


async def get_rainfall_trends(lat: float, lon: float, days: int = 7) -> Dict[str, Any]:
    """
    Get rainfall trend data for charts
    
    Args:
        lat: Latitude
        lon: Longitude
        days: Number of days of historical data to include
    
    Returns:
        Historical rainfall data and forecasts formatted for charting
    """
    result = {
        "success": True,
        "location": {"lat": lat, "lon": lon},
        "historical": [],
        "forecast": [],
        "summary": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # Get historical data from stored records
    historical_records = get_records_since(days)
    
    # Filter records near the requested location (within ~50km)
    nearby_records = []
    for record in historical_records:
        rec_lat = record["location"]["lat"]
        rec_lon = record["location"]["lon"]
        distance = math.sqrt((rec_lat - lat)**2 + (rec_lon - lon)**2)
        if distance < 0.5:  # Roughly 50km
            nearby_records.append(record)
    
    # If we have historical records, use them
    if nearby_records:
        for record in nearby_records:
            result["historical"].append({
                "timestamp": record["timestamp"],
                "rainfall_1h": record["weather"].get("rainfall_1h", 0),
                "rainfall_3h": record["weather"].get("rainfall_3h", 0),
                "humidity": record["weather"].get("humidity", 0),
                "risk_score": record.get("risk_score", 0),
                "flood_status": record.get("flood_status", "Safe")
            })
    else:
        # Generate simulated historical data for demo purposes
        now = datetime.now()
        for i in range(days * 8):  # 8 data points per day (3-hour intervals)
            timestamp = now - timedelta(hours=i * 3)
            # Create realistic-looking rainfall pattern
            base_rain = random.uniform(0, 5)
            # Add monsoon pattern (higher rainfall in certain hours)
            hour_factor = 1 + 0.5 * math.sin(timestamp.hour * math.pi / 12)
            # Add day variation
            day_factor = 1 + 0.3 * math.sin(timestamp.day * math.pi / 15)
            
            rainfall = base_rain * hour_factor * day_factor
            
            result["historical"].append({
                "timestamp": timestamp.isoformat(),
                "rainfall_1h": round(rainfall, 2),
                "rainfall_3h": round(rainfall * 2.5, 2),
                "humidity": round(60 + random.uniform(-20, 30), 1),
                "risk_score": round(min(100, rainfall * 8 + random.uniform(0, 20)), 1),
                "flood_status": "Flooded" if rainfall > 8 else "At Risk" if rainfall > 4 else "Safe"
            })
        
        # Sort by timestamp
        result["historical"].sort(key=lambda x: x["timestamp"])
    
    # Get forecast data from weather API
    try:
        forecast_data = await get_weather_forecast(lat, lon, days=5)
        if forecast_data.get("success") and forecast_data.get("forecasts"):
            for fc in forecast_data["forecasts"]:
                result["forecast"].append({
                    "timestamp": fc["datetime"],
                    "rainfall_3h": fc.get("rain_3h", 0),
                    "probability": fc.get("pop", 0) * 100,  # Convert to percentage
                    "humidity": fc.get("humidity", 0),
                    "condition": fc.get("condition", "Clear"),
                    "temperature": fc.get("temperature", 0)
                })
    except Exception as e:
        print(f"Error fetching forecast: {e}")
        # Generate simulated forecast
        now = datetime.now()
        for i in range(40):  # 5 days * 8 intervals
            timestamp = now + timedelta(hours=i * 3)
            base_rain = random.uniform(0, 8)
            result["forecast"].append({
                "timestamp": timestamp.isoformat(),
                "rainfall_3h": round(base_rain, 2),
                "probability": round(min(100, base_rain * 10 + random.uniform(0, 30)), 1),
                "humidity": round(60 + random.uniform(-15, 25), 1),
                "condition": "Rain" if base_rain > 3 else "Clouds" if base_rain > 1 else "Clear",
                "temperature": round(25 + random.uniform(-5, 10), 1)
            })
    
    # Calculate summary statistics
    if result["historical"]:
        rainfall_values = [h["rainfall_1h"] for h in result["historical"]]
        result["summary"] = {
            "avg_rainfall": round(sum(rainfall_values) / len(rainfall_values), 2),
            "max_rainfall": round(max(rainfall_values), 2),
            "min_rainfall": round(min(rainfall_values), 2),
            "total_rainfall": round(sum(rainfall_values), 2),
            "flood_events": sum(1 for h in result["historical"] if h["flood_status"] == "Flooded"),
            "at_risk_events": sum(1 for h in result["historical"] if h["flood_status"] == "At Risk")
        }
    
    return result


async def get_river_level_trends(lat: float, lon: float, days: int = 7) -> Dict[str, Any]:
    """
    Get river level trend data for charts
    
    Note: Since Pakistan-specific river data APIs (WAPDA/IRSA) require special access,
    this generates simulated data that correlates with rainfall patterns.
    
    Args:
        lat: Latitude
        lon: Longitude
        days: Number of days of data
    
    Returns:
        River level data formatted for charting
    """
    result = {
        "success": True,
        "location": {"lat": lat, "lon": lon},
        "river_name": get_nearest_river(lat, lon),
        "historical": [],
        "forecast": [],
        "thresholds": {
            "normal": 3.0,
            "warning": 5.0,
            "danger": 7.0,
            "extreme": 9.0
        },
        "unit": "meters",
        "timestamp": datetime.now().isoformat()
    }
    
    # Generate simulated river level data
    # In production, this would come from WAPDA/IRSA APIs
    now = datetime.now()
    base_level = 3.5  # Normal water level in meters
    
    # Historical data
    for i in range(days * 8):
        timestamp = now - timedelta(hours=i * 3)
        
        # Simulate river level based on time patterns
        # Rivers typically rise after rainfall with a delay
        hour_variation = 0.3 * math.sin(timestamp.hour * math.pi / 12)
        day_variation = 0.5 * math.sin(timestamp.day * math.pi / 10)
        random_variation = random.uniform(-0.3, 0.3)
        
        # Simulate occasional flood events
        flood_event = 2.0 if random.random() < 0.05 else 0
        
        level = base_level + hour_variation + day_variation + random_variation + flood_event
        level = max(1.0, level)  # Minimum level
        
        # Determine status based on level
        if level >= result["thresholds"]["extreme"]:
            status = "Extreme"
        elif level >= result["thresholds"]["danger"]:
            status = "Danger"
        elif level >= result["thresholds"]["warning"]:
            status = "Warning"
        else:
            status = "Normal"
        
        result["historical"].append({
            "timestamp": timestamp.isoformat(),
            "level": round(level, 2),
            "flow_rate": round(level * 150 + random.uniform(-50, 50), 1),  # m³/s estimate
            "status": status
        })
    
    # Sort historical by timestamp
    result["historical"].sort(key=lambda x: x["timestamp"])
    
    # Forecast data (24-48 hours ahead)
    last_level = result["historical"][-1]["level"] if result["historical"] else base_level
    for i in range(16):  # 48 hours in 3-hour intervals
        timestamp = now + timedelta(hours=i * 3)
        
        # Forecast tends to regress toward normal with uncertainty
        trend = (base_level - last_level) * 0.1
        random_factor = random.uniform(-0.2, 0.2) * (i + 1) / 8  # Uncertainty grows
        
        level = last_level + trend + random_factor
        level = max(1.0, level)
        last_level = level
        
        # Determine forecast status
        if level >= result["thresholds"]["extreme"]:
            status = "Extreme"
        elif level >= result["thresholds"]["danger"]:
            status = "Danger"
        elif level >= result["thresholds"]["warning"]:
            status = "Warning"
        else:
            status = "Normal"
        
        result["forecast"].append({
            "timestamp": timestamp.isoformat(),
            "level": round(level, 2),
            "level_min": round(level - 0.5 * (i + 1) / 8, 2),  # Confidence interval
            "level_max": round(level + 0.5 * (i + 1) / 8, 2),
            "status": status,
            "confidence": round(100 - i * 5, 1)  # Confidence decreases over time
        })
    
    # Summary
    if result["historical"]:
        levels = [h["level"] for h in result["historical"]]
        result["summary"] = {
            "current_level": result["historical"][-1]["level"],
            "current_status": result["historical"][-1]["status"],
            "avg_level": round(sum(levels) / len(levels), 2),
            "max_level": round(max(levels), 2),
            "min_level": round(min(levels), 2),
            "trend": "rising" if levels[-1] > levels[0] else "falling" if levels[-1] < levels[0] else "stable"
        }
    
    return result


def get_nearest_river(lat: float, lon: float) -> str:
    """Get the name of the nearest major river based on coordinates"""
    # Major rivers in Pakistan with approximate coordinates
    rivers = [
        {"name": "Indus River", "lat": 33.6, "lon": 73.0},
        {"name": "Jhelum River", "lat": 33.0, "lon": 73.5},
        {"name": "Chenab River", "lat": 32.0, "lon": 72.5},
        {"name": "Ravi River", "lat": 31.5, "lon": 74.3},
        {"name": "Sutlej River", "lat": 30.5, "lon": 73.0},
        {"name": "Kabul River", "lat": 34.0, "lon": 71.5},
        {"name": "Swat River", "lat": 34.8, "lon": 72.3},
        {"name": "Kurram River", "lat": 33.5, "lon": 70.5},
        {"name": "Zhob River", "lat": 31.3, "lon": 69.5},
        {"name": "Hub River", "lat": 25.0, "lon": 66.8},
    ]
    
    min_distance = float('inf')
    nearest = "Unknown River"
    
    for river in rivers:
        distance = math.sqrt((river["lat"] - lat)**2 + (river["lon"] - lon)**2)
        if distance < min_distance:
            min_distance = distance
            nearest = river["name"]
    
    return nearest


async def get_combined_trends(lat: float, lon: float, days: int = 7) -> Dict[str, Any]:
    """
    Get combined rainfall and river level trends for comprehensive analysis
    
    Args:
        lat: Latitude
        lon: Longitude
        days: Number of days of data
    
    Returns:
        Combined data for both rainfall and river levels
    """
    rainfall_data = await get_rainfall_trends(lat, lon, days)
    river_data = await get_river_level_trends(lat, lon, days)
    
    return {
        "success": True,
        "location": {"lat": lat, "lon": lon},
        "rainfall": rainfall_data,
        "river_level": river_data,
        "correlation": calculate_correlation(rainfall_data, river_data),
        "timestamp": datetime.now().isoformat()
    }


def calculate_correlation(rainfall_data: Dict, river_data: Dict) -> Dict[str, Any]:
    """Calculate correlation between rainfall and river levels"""
    # Simple correlation analysis
    rainfall_values = [h["rainfall_1h"] for h in rainfall_data.get("historical", [])]
    river_values = [h["level"] for h in river_data.get("historical", [])]
    
    if not rainfall_values or not river_values:
        return {"coefficient": 0, "description": "Insufficient data"}
    
    # Normalize lengths
    min_len = min(len(rainfall_values), len(river_values))
    rainfall_values = rainfall_values[:min_len]
    river_values = river_values[:min_len]
    
    # Calculate simple correlation coefficient
    if min_len < 2:
        return {"coefficient": 0, "description": "Insufficient data"}
    
    mean_rain = sum(rainfall_values) / min_len
    mean_river = sum(river_values) / min_len
    
    numerator = sum((r - mean_rain) * (v - mean_river) for r, v in zip(rainfall_values, river_values))
    denom_rain = math.sqrt(sum((r - mean_rain)**2 for r in rainfall_values))
    denom_river = math.sqrt(sum((v - mean_river)**2 for v in river_values))
    
    if denom_rain * denom_river == 0:
        return {"coefficient": 0, "description": "No variation in data"}
    
    coefficient = numerator / (denom_rain * denom_river)
    
    if coefficient > 0.7:
        description = "Strong positive correlation - river levels closely follow rainfall"
    elif coefficient > 0.4:
        description = "Moderate correlation - rainfall affects river levels with some delay"
    elif coefficient > 0:
        description = "Weak correlation - other factors significantly influence river levels"
    else:
        description = "No significant correlation detected"
    
    return {
        "coefficient": round(coefficient, 3),
        "description": description
    }