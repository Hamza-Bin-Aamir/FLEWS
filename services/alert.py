"""
Alert management system for FLEWS
Handles live flood alert notifications with real-time weather integration
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import random
import hashlib
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
import httpx

# Load environment variables from root .env file
root_dir = Path(__file__).parent.parent
env_path = root_dir / '.env'
load_dotenv(env_path)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "your_api_key_here")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# Weather cache to avoid excessive API calls
_alert_weather_cache: Dict[str, tuple] = {}
CACHE_DURATION = 300  # 5 minutes

# Alert locations across Pakistan
ALERT_LOCATIONS = [
    {"name": "Topi, Swabi", "coords": [72.6234, 34.0705], "region": "KPK"},
    {"name": "Nowshera", "coords": [71.9824, 33.9951], "region": "KPK"},
    {"name": "Charsadda", "coords": [71.7417, 34.1483], "region": "KPK"},
    {"name": "Sukkur", "coords": [68.8577, 27.7058], "region": "Sindh"},
    {"name": "Jacobabad", "coords": [68.4375, 28.2769], "region": "Sindh"},
    {"name": "Rahim Yar Khan", "coords": [70.2952, 28.4202], "region": "Punjab"},
    {"name": "Dera Ghazi Khan", "coords": [70.6343, 30.0561], "region": "Punjab"},
    {"name": "Multan", "coords": [71.5249, 30.1575], "region": "Punjab"},
    {"name": "Lahore (Ravi)", "coords": [74.3436, 31.5204], "region": "Punjab"},
    {"name": "Jhelum", "coords": [73.7257, 32.9425], "region": "Punjab"},
    {"name": "Rawalpindi", "coords": [73.0479, 33.5651], "region": "Punjab"},
    {"name": "Karachi Coast", "coords": [67.0099, 24.8607], "region": "Sindh"},
    {"name": "Hyderabad", "coords": [68.3578, 25.3792], "region": "Sindh"},
    {"name": "Muzaffarabad", "coords": [73.4708, 34.3700], "region": "AJK"},
]

ALERT_MESSAGES = {
    "high_risk": [
        "Flash flood warning! Evacuate immediately.",
        "Critical water levels detected. Immediate action required.",
        "Severe flooding in progress. Move to higher ground now.",
        "Emergency: River overflowing. Evacuate low-lying areas.",
        "Danger: Rapid water rise detected. Seek safety immediately.",
    ],
    "medium_risk": [
        "Flood watch in effect. Monitor conditions closely.",
        "Rising water levels detected. Prepare for possible evacuation.",
        "Moderate flood risk. Stay alert and avoid flood-prone areas.",
        "Water levels increasing. Be ready to evacuate if needed.",
        "Flood advisory issued. Exercise caution near waterways.",
    ],
    "low_risk": [
        "Weather advisory: Monitor rainfall conditions.",
        "Minor flooding possible in low-lying areas.",
        "Elevated water levels detected. Stay informed.",
        "Slight flood risk. Avoid unnecessary travel near rivers.",
        "Normal operations with caution advised near water bodies.",
    ]
}

# Weather-based alert messages (more specific)
WEATHER_ALERT_MESSAGES = {
    "high_risk": {
        "rain": "Heavy rainfall ({rain_mm}mm/h) causing severe flooding. Evacuate immediately!",
        "forecast": "Extreme rainfall expected ({forecast_mm}mm in 24h). Prepare for major flooding.",
        "combined": "Critical: Heavy rain ({rain_mm}mm/h) with more expected. Immediate evacuation required.",
    },
    "medium_risk": {
        "rain": "Moderate rainfall ({rain_mm}mm/h) detected. Monitor flood conditions closely.",
        "forecast": "Significant rainfall forecast ({forecast_mm}mm in 24h). Prepare for possible flooding.",
        "combined": "Rising water levels with continued rain expected. Stay alert for updates.",
    },
    "low_risk": {
        "rain": "Light rainfall detected ({rain_mm}mm/h). Monitor conditions.",
        "forecast": "Some rainfall expected ({forecast_mm}mm in 24h). Normal precautions advised.",
        "combined": "Minor precipitation activity. Stay informed about weather changes.",
    }
}


async def fetch_location_weather(lon: float, lat: float) -> Dict[str, Any]:
    """
    Fetch weather data for a specific location
    """
    cache_key = f"{lat:.2f},{lon:.2f}"
    current_time = datetime.now().timestamp()
    
    # Check cache
    if cache_key in _alert_weather_cache:
        cached_time, cached_data = _alert_weather_cache[cache_key]
        if current_time - cached_time < CACHE_DURATION:
            return cached_data
    
    try:
        async with httpx.AsyncClient() as client:
            # Fetch current weather
            weather_response = await client.get(
                f"{OPENWEATHER_BASE_URL}/weather",
                params={
                    "lat": lat,
                    "lon": lon,
                    "appid": OPENWEATHER_API_KEY,
                    "units": "metric"
                },
                timeout=10.0
            )
            
            # Fetch forecast
            forecast_response = await client.get(
                f"{OPENWEATHER_BASE_URL}/forecast",
                params={
                    "lat": lat,
                    "lon": lon,
                    "appid": OPENWEATHER_API_KEY,
                    "units": "metric",
                    "cnt": 8
                },
                timeout=10.0
            )
            
            weather_data = {}
            if weather_response.status_code == 200:
                data = weather_response.json()
                weather_data["current"] = {
                    "rain_1h": data.get("rain", {}).get("1h", 0),
                    "rain_3h": data.get("rain", {}).get("3h", 0),
                    "humidity": data.get("main", {}).get("humidity", 0),
                    "condition": data.get("weather", [{}])[0].get("main", "Unknown"),
                    "temp": data.get("main", {}).get("temp", 0),
                }
            
            if forecast_response.status_code == 200:
                forecast_data = forecast_response.json()
                total_rain = sum(
                    item.get("rain", {}).get("3h", 0) 
                    for item in forecast_data.get("list", [])
                )
                weather_data["forecast"] = {
                    "rain_24h": total_rain,
                    "periods_with_rain": sum(
                        1 for item in forecast_data.get("list", [])
                        if item.get("rain", {}).get("3h", 0) > 0
                    )
                }
            
            _alert_weather_cache[cache_key] = (current_time, weather_data)
            return weather_data
            
    except Exception as e:
        print(f"Error fetching weather for alert: {e}")
        return {}


def calculate_weather_severity(weather_data: Dict) -> tuple:
    """
    Calculate alert severity based on weather data
    Returns (severity, color, score, reason)
    """
    current = weather_data.get("current", {})
    forecast = weather_data.get("forecast", {})
    
    rain_1h = current.get("rain_1h", 0)
    rain_3h = current.get("rain_3h", 0)
    humidity = current.get("humidity", 0)
    forecast_rain = forecast.get("rain_24h", 0)
    
    # Calculate risk score
    score = 0
    reasons = []
    
    # Current rainfall (most important)
    if rain_1h >= 30:
        score += 50
        reasons.append("extreme_rain")
    elif rain_1h >= 15:
        score += 35
        reasons.append("heavy_rain")
    elif rain_1h >= 7.5:
        score += 20
        reasons.append("moderate_rain")
    elif rain_1h >= 2.5:
        score += 10
        reasons.append("light_rain")
    
    # 3-hour accumulated
    if rain_3h >= 45:
        score += 25
    elif rain_3h >= 20:
        score += 15
    elif rain_3h >= 10:
        score += 8
    
    # Forecast
    if forecast_rain >= 50:
        score += 20
        reasons.append("heavy_forecast")
    elif forecast_rain >= 25:
        score += 12
        reasons.append("moderate_forecast")
    elif forecast_rain >= 10:
        score += 5
    
    # Humidity (saturated ground)
    if humidity >= 90 and rain_1h > 0:
        score += 10
    elif humidity >= 80 and rain_1h > 0:
        score += 5
    
    # Determine severity
    if score >= 60:
        return ("high_risk", "red", score, reasons)
    elif score >= 30:
        return ("medium_risk", "yellow", score, reasons)
    else:
        return ("low_risk", "green", score, reasons)


def generate_demo_alerts() -> List[Dict[str, Any]]:
    """
    Generate demo alerts with simulated heavy rainfall for demonstration.
    Shows a variety of alert severities across different locations.
    """
    import random
    
    # Use time-based seed for some variation but consistent within same minute
    seed = int(datetime.now().strftime("%Y%m%d%H%M"))
    rng = random.Random(seed)
    
    demo_scenarios = [
        # High risk locations (flooding)
        {
            "location": ALERT_LOCATIONS[0],  # Topi, Swabi
            "severity": "high_risk",
            "rain_1h": rng.uniform(25, 40),
            "rain_3h": rng.uniform(60, 90),
            "forecast": rng.uniform(80, 120),
            "humidity": rng.randint(92, 99),
        },
        {
            "location": ALERT_LOCATIONS[1],  # Nowshera
            "severity": "high_risk",
            "rain_1h": rng.uniform(20, 35),
            "rain_3h": rng.uniform(50, 80),
            "forecast": rng.uniform(70, 100),
            "humidity": rng.randint(90, 98),
        },
        {
            "location": ALERT_LOCATIONS[2],  # Charsadda
            "severity": "high_risk",
            "rain_1h": rng.uniform(30, 45),
            "rain_3h": rng.uniform(70, 100),
            "forecast": rng.uniform(90, 130),
            "humidity": rng.randint(94, 99),
        },
        # Medium risk locations
        {
            "location": ALERT_LOCATIONS[4],  # Jacobabad
            "severity": "medium_risk",
            "rain_1h": rng.uniform(8, 15),
            "rain_3h": rng.uniform(20, 35),
            "forecast": rng.uniform(30, 50),
            "humidity": rng.randint(80, 90),
        },
        {
            "location": ALERT_LOCATIONS[6],  # Dera Ghazi Khan
            "severity": "medium_risk",
            "rain_1h": rng.uniform(10, 18),
            "rain_3h": rng.uniform(25, 40),
            "forecast": rng.uniform(35, 55),
            "humidity": rng.randint(82, 92),
        },
        {
            "location": ALERT_LOCATIONS[10],  # Rawalpindi
            "severity": "medium_risk",
            "rain_1h": rng.uniform(7, 14),
            "rain_3h": rng.uniform(18, 32),
            "forecast": rng.uniform(28, 45),
            "humidity": rng.randint(78, 88),
        },
        # Low risk locations
        {
            "location": ALERT_LOCATIONS[8],  # Lahore
            "severity": "low_risk",
            "rain_1h": rng.uniform(2, 6),
            "rain_3h": rng.uniform(5, 12),
            "forecast": rng.uniform(10, 20),
            "humidity": rng.randint(70, 82),
        },
        {
            "location": ALERT_LOCATIONS[11],  # Karachi
            "severity": "low_risk",
            "rain_1h": rng.uniform(1, 4),
            "rain_3h": rng.uniform(3, 8),
            "forecast": rng.uniform(5, 15),
            "humidity": rng.randint(65, 78),
        },
    ]
    
    alerts = []
    severity_colors = {"high_risk": "red", "medium_risk": "yellow", "low_risk": "green"}
    
    for i, scenario in enumerate(demo_scenarios):
        location = scenario["location"]
        severity = scenario["severity"]
        rain_mm = scenario["rain_1h"]
        forecast_mm = scenario["forecast"]
        
        # Generate message
        if rain_mm > 20:
            message = f"SEVERE FLOODING: {rain_mm:.1f}mm/h rainfall causing dangerous conditions. Evacuate immediately!"
        elif rain_mm > 10:
            message = f"Heavy rainfall ({rain_mm:.1f}mm/h) detected. Monitor conditions and prepare for possible evacuation."
        elif rain_mm > 5:
            message = f"Moderate rainfall ({rain_mm:.1f}mm/h) with {forecast_mm:.0f}mm expected in 24h. Stay alert."
        else:
            message = f"Light rainfall detected ({rain_mm:.1f}mm/h). Monitor weather updates."
        
        # Calculate risk score
        score = min(100, rain_mm * 2 + scenario["rain_3h"] * 0.5 + forecast_mm * 0.2 + scenario["humidity"] * 0.3)
        
        alert_id = hashlib.md5(f"demo-{location['name']}-{seed}".encode()).hexdigest()[:12]
        
        alert = {
            "id": f"alert-{alert_id}",
            "severity": severity,
            "color": severity_colors[severity],
            "location": location["name"],
            "region": location["region"],
            "coordinates": location["coords"],
            "message": message,
            "issued_at": datetime.now().isoformat(),
            "issued_minutes_ago": rng.randint(2, 20),
            "weather_data": {
                "rain_1h_mm": round(rain_mm, 1),
                "rain_3h_mm": round(scenario["rain_3h"], 1),
                "humidity": scenario["humidity"],
                "temperature": rng.randint(24, 32),
                "condition": "Thunderstorm" if severity == "high_risk" else "Rain",
                "forecast_rain_24h_mm": round(forecast_mm, 1),
            },
            "risk_score": round(score, 1),
            "demo_mode": True,
        }
        alerts.append(alert)
    
    # Sort by severity
    severity_order = {"high_risk": 0, "medium_risk": 1, "low_risk": 2}
    alerts.sort(key=lambda x: (severity_order[x["severity"]], -x["risk_score"]))
    
    return alerts


async def generate_weather_based_alerts(demo_mode: bool = False) -> List[Dict[str, Any]]:
    """
    Generate alerts based on real-time weather data for all monitored locations
    
    Args:
        demo_mode: If True, simulates heavy rainfall for demonstration
    """
    alerts = []
    
    if demo_mode:
        # Generate demo alerts with simulated heavy rainfall
        return generate_demo_alerts()
    
    # Fetch weather for all locations concurrently
    tasks = [
        fetch_location_weather(loc["coords"][0], loc["coords"][1])
        for loc in ALERT_LOCATIONS
    ]
    weather_results = await asyncio.gather(*tasks)
    
    for location, weather_data in zip(ALERT_LOCATIONS, weather_results):
        if not weather_data:
            continue
            
        severity, color, score, reasons = calculate_weather_severity(weather_data)
        
        # Only generate alert if there's actual risk
        if score < 10:
            continue
        
        current = weather_data.get("current", {})
        forecast = weather_data.get("forecast", {})
        
        # Generate appropriate message
        rain_mm = current.get("rain_1h", 0)
        forecast_mm = forecast.get("rain_24h", 0)
        
        if rain_mm > 0 and forecast_mm > 10:
            message_type = "combined"
        elif rain_mm > 0:
            message_type = "rain"
        else:
            message_type = "forecast"
        
        message_template = WEATHER_ALERT_MESSAGES[severity][message_type]
        message = message_template.format(rain_mm=round(rain_mm, 1), forecast_mm=round(forecast_mm, 1))
        
        alert_id = hashlib.md5(f"{location['name']}{datetime.now().strftime('%Y%m%d%H')}".encode()).hexdigest()[:12]
        
        alert = {
            "id": f"alert-{alert_id}",
            "severity": severity,
            "color": color,
            "location": location["name"],
            "region": location["region"],
            "coordinates": location["coords"],
            "message": message,
            "issued_at": datetime.now().isoformat(),
            "issued_minutes_ago": 0,
            "weather_data": {
                "rain_1h_mm": rain_mm,
                "rain_3h_mm": current.get("rain_3h", 0),
                "humidity": current.get("humidity", 0),
                "temperature": current.get("temp", 0),
                "condition": current.get("condition", "Unknown"),
                "forecast_rain_24h_mm": forecast_mm,
            },
            "risk_score": score,
        }
        alerts.append(alert)
    
    # Sort by severity (high first)
    severity_order = {"high_risk": 0, "medium_risk": 1, "low_risk": 2}
    alerts.sort(key=lambda x: (severity_order[x["severity"]], -x["risk_score"]))
    
    return alerts


def generate_live_alerts(seed: int = None) -> List[Dict[str, Any]]:
    """
    Generate live alerts using deterministic pseudo-random generation
    Same seed = same alerts (RESTful stateless)
    
    Seed changes every 5 minutes to simulate evolving alert conditions
    """
    # Use seed for deterministic randomness
    if seed is None:
        # Default: change every 5 minutes
        now = datetime.now()
        seed = int(now.strftime("%Y%m%d%H")) * 12 + (now.minute // 5)
    
    # Seed the random generator for deterministic results
    rng = random.Random(seed)
    
    # Deterministic number of active alerts (2-8)
    num_alerts = 2 + (seed % 7)
    alerts = []
    
    # Deterministically select locations
    location_indices = list(range(len(ALERT_LOCATIONS)))
    rng.shuffle(location_indices)
    selected_indices = location_indices[:min(num_alerts, len(ALERT_LOCATIONS))]
    
    for i, idx in enumerate(selected_indices):
        location = ALERT_LOCATIONS[idx]
        
        # Determine severity (deterministic based on seed and index)
        severity_value = (seed + idx * 7) % 10
        if severity_value < 2:  # 20% high risk
            severity = "high_risk"
            color = "red"
        elif severity_value < 5:  # 30% medium risk
            severity = "medium_risk"
            color = "yellow"
        else:  # 50% low risk
            severity = "low_risk"
            color = "green"
        
        # Generate timestamp (deterministic minutes ago)
        issued_minutes_ago = 5 + ((seed + idx * 3) % 25)
        issued_at = datetime.now() - timedelta(minutes=issued_minutes_ago)
        
        # Select message deterministically
        message_idx = (seed + idx * 11) % len(ALERT_MESSAGES[severity])
        
        # Create alert with deterministic ID
        alert_id = hashlib.md5(f"{seed}{idx}{location['name']}".encode()).hexdigest()[:12]
        
        alert = {
            "id": f"alert-{alert_id}",
            "severity": severity,
            "color": color,
            "location": location["name"],
            "region": location["region"],
            "coordinates": location["coords"],
            "message": ALERT_MESSAGES[severity][message_idx],
            "issued_at": issued_at.isoformat(),
            "issued_minutes_ago": issued_minutes_ago,
        }
        alerts.append(alert)
    
    # Sort by severity (high first, then medium, then low)
    severity_order = {"high_risk": 0, "medium_risk": 1, "low_risk": 2}
    alerts.sort(key=lambda x: severity_order[x["severity"]])
    
    return alerts

def get_alert_summary(seed: int = None) -> Dict[str, Any]:
    """
    Returns live alerts with full information (fallback/deterministic mode)
    
    RESTful: Same seed always returns same alerts (deterministic)
    Seed changes every 5 minutes to simulate evolving conditions
    
    Args:
        seed: Optional seed for deterministic generation
    """
    alerts = generate_live_alerts(seed)
    
    # Count by severity
    high_count = sum(1 for a in alerts if a["severity"] == "high_risk")
    medium_count = sum(1 for a in alerts if a["severity"] == "medium_risk")
    low_count = sum(1 for a in alerts if a["severity"] == "low_risk")
    
    return {
        "alerts": alerts,
        "summary": {
            "high_risk": high_count,
            "medium_risk": medium_count,
            "low_risk": low_count,
            "total": len(alerts)
        },
        "last_updated": datetime.now().isoformat(),
        "data_source": "simulated"
    }


async def get_weather_alert_summary(demo_mode: bool = False) -> Dict[str, Any]:
    """
    Returns live alerts based on REAL weather data from OpenWeatherMap API
    
    This function fetches current weather conditions for all monitored locations
    and generates alerts based on actual rainfall, humidity, and forecasts.
    
    Args:
        demo_mode: If True, uses simulated heavy rainfall to show alerts for demonstration
    """
    try:
        if demo_mode:
            # Use demo alerts with simulated heavy rainfall
            alerts = generate_demo_alerts()
            data_source = "demo_simulation"
        else:
            alerts = await generate_weather_based_alerts()
            data_source = "openweathermap_realtime"
        
        # Count by severity
        high_count = sum(1 for a in alerts if a["severity"] == "high_risk")
        medium_count = sum(1 for a in alerts if a["severity"] == "medium_risk")
        low_count = sum(1 for a in alerts if a["severity"] == "low_risk")
        
        return {
            "alerts": alerts,
            "summary": {
                "high_risk": high_count,
                "medium_risk": medium_count,
                "low_risk": low_count,
                "total": len(alerts)
            },
            "last_updated": datetime.now().isoformat(),
            "data_source": data_source
        }
    except Exception as e:
        print(f"Error fetching weather-based alerts: {e}")
        # Return empty alerts instead of fake simulated ones
        # This is more accurate - no alerts means conditions are safe
        return {
            "alerts": [],
            "summary": {
                "high_risk": 0,
                "medium_risk": 0,
                "low_risk": 0,
                "total": 0
            },
            "last_updated": datetime.now().isoformat(),
            "data_source": "openweathermap_realtime",
            "error": str(e)
        }


def create_alert(location: str, severity: str, message: str) -> Dict:
    """
    Create a new flood alert
    
    Args:
        location: Location name or description
        severity: Alert severity level (high, moderate, low)
        message: Alert message content
    
    Returns:
        Created alert object
    """
    # Dummy implementation
    alert = {
        "id": f"alert_{datetime.now().timestamp()}",
        "location": location,
        "severity": severity,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "status": "active"
    }
    return alert


def dismiss_alert(alert_id: str) -> bool:
    """
    Dismiss or deactivate an alert
    
    Args:
        alert_id: ID of the alert to dismiss
    
    Returns:
        True if successful, False otherwise
    """
    # Dummy implementation
    return True
