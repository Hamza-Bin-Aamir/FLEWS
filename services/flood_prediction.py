"""
Flood Prediction Model for FLEWS
Combines real-time weather data with geographical flood risk factors
and machine learning models to generate accurate, current flood predictions
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import httpx
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import math
import hashlib
import asyncio

# Load environment variables from root .env file
root_dir = Path(__file__).parent.parent
env_path = root_dir / '.env'
load_dotenv(env_path)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "your_api_key_here")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# Cache for weather data (avoid hitting API too frequently)
_weather_cache: Dict[str, Tuple[datetime, Dict]] = {}
CACHE_DURATION = timedelta(minutes=10)

# Try to import ML module
try:
    from ml_prediction import predict_flood_risk_ml, get_model_info, FEATURE_NAMES
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML prediction module not available - using rule-based predictions only")


class WeatherBasedFloodPredictor:
    """
    Prediction model that uses real-time weather data to assess flood risk
    
    Factors considered:
    - Current rainfall intensity
    - Accumulated rainfall (1h, 3h)
    - Humidity levels
    - Recent weather history
    - Forecasted precipitation
    - Geographical flood-prone areas
    """
    
    # Risk weights for different weather factors
    WEIGHTS = {
        "rainfall_1h": 0.35,      # Current hourly rainfall is most important
        "rainfall_3h": 0.20,      # 3-hour accumulated rainfall
        "humidity": 0.10,         # High humidity indicates saturated ground
        "forecast_rain": 0.20,    # Predicted rainfall
        "geographical": 0.15,     # Historical flood-prone areas
    }
    
    # Thresholds for rainfall (mm)
    RAINFALL_THRESHOLDS = {
        "light": 2.5,      # Light rain
        "moderate": 7.5,   # Moderate rain
        "heavy": 15.0,     # Heavy rain
        "extreme": 30.0,   # Extreme/torrential rain
    }
    
    def __init__(self):
        self.api_key = OPENWEATHER_API_KEY
        
    async def fetch_weather_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch current weather data from OpenWeatherMap API with caching
        """
        cache_key = f"{lat:.2f},{lon:.2f}"
        
        # Check cache
        if cache_key in _weather_cache:
            cached_time, cached_data = _weather_cache[cache_key]
            if datetime.now() - cached_time < CACHE_DURATION:
                return cached_data
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{OPENWEATHER_BASE_URL}/weather",
                    params={
                        "lat": lat,
                        "lon": lon,
                        "appid": self.api_key,
                        "units": "metric"
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    _weather_cache[cache_key] = (datetime.now(), data)
                    return data
                else:
                    return {}
        except Exception as e:
            print(f"Weather API error: {e}")
            return {}
    
    async def fetch_forecast_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch weather forecast data from OpenWeatherMap API
        """
        cache_key = f"forecast_{lat:.2f},{lon:.2f}"
        
        # Check cache
        if cache_key in _weather_cache:
            cached_time, cached_data = _weather_cache[cache_key]
            if datetime.now() - cached_time < CACHE_DURATION:
                return cached_data
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{OPENWEATHER_BASE_URL}/forecast",
                    params={
                        "lat": lat,
                        "lon": lon,
                        "appid": self.api_key,
                        "units": "metric",
                        "cnt": 8  # Next 24 hours (3-hour intervals)
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    _weather_cache[cache_key] = (datetime.now(), data)
                    return data
                else:
                    return {}
        except Exception as e:
            print(f"Forecast API error: {e}")
            return {}
    
    def calculate_rainfall_score(self, weather_data: Dict) -> Tuple[float, Dict]:
        """
        Calculate flood risk score based on rainfall data
        Returns score (0-100) and details
        """
        rain = weather_data.get("rain", {})
        rain_1h = rain.get("1h", 0)
        rain_3h = rain.get("3h", 0)
        
        # Calculate scores for each rainfall metric
        score_1h = 0
        if rain_1h >= self.RAINFALL_THRESHOLDS["extreme"]:
            score_1h = 100
        elif rain_1h >= self.RAINFALL_THRESHOLDS["heavy"]:
            score_1h = 75
        elif rain_1h >= self.RAINFALL_THRESHOLDS["moderate"]:
            score_1h = 50
        elif rain_1h >= self.RAINFALL_THRESHOLDS["light"]:
            score_1h = 25
        
        score_3h = 0
        if rain_3h >= self.RAINFALL_THRESHOLDS["extreme"] * 3:
            score_3h = 100
        elif rain_3h >= self.RAINFALL_THRESHOLDS["heavy"] * 3:
            score_3h = 75
        elif rain_3h >= self.RAINFALL_THRESHOLDS["moderate"] * 3:
            score_3h = 50
        elif rain_3h >= self.RAINFALL_THRESHOLDS["light"] * 3:
            score_3h = 25
        
        return {
            "score_1h": score_1h,
            "score_3h": score_3h,
            "rain_1h_mm": rain_1h,
            "rain_3h_mm": rain_3h,
        }
    
    def calculate_humidity_score(self, weather_data: Dict) -> Tuple[float, Dict]:
        """
        High humidity indicates saturated ground = higher flood risk
        """
        main = weather_data.get("main", {})
        humidity = main.get("humidity", 50)
        
        # Score based on humidity (higher humidity = higher risk when combined with rain)
        if humidity >= 95:
            score = 100
        elif humidity >= 85:
            score = 70
        elif humidity >= 75:
            score = 40
        elif humidity >= 65:
            score = 20
        else:
            score = 0
        
        return {
            "score": score,
            "humidity": humidity,
        }
    
    def calculate_forecast_score(self, forecast_data: Dict) -> Tuple[float, Dict]:
        """
        Calculate risk based on forecasted rainfall in next 24 hours
        """
        forecasts = forecast_data.get("list", [])
        
        total_rain = 0
        max_rain_3h = 0
        rainy_periods = 0
        
        for item in forecasts:
            rain = item.get("rain", {}).get("3h", 0)
            total_rain += rain
            max_rain_3h = max(max_rain_3h, rain)
            if rain > 0:
                rainy_periods += 1
        
        # Score based on forecast
        score = 0
        if total_rain >= 100:
            score = 100
        elif total_rain >= 50:
            score = 75
        elif total_rain >= 25:
            score = 50
        elif total_rain >= 10:
            score = 25
        
        return {
            "score": score,
            "total_rain_24h": total_rain,
            "max_rain_3h": max_rain_3h,
            "rainy_periods": rainy_periods,
        }
    
    async def predict_flood_risk(
        self, 
        lat: float, 
        lon: float,
        geographical_risk: float = 0.0  # Base risk from flood_risk.py
    ) -> Dict[str, Any]:
        """
        Main prediction function - combines all weather factors
        
        Args:
            lat, lon: Location coordinates
            geographical_risk: Base risk score from geographical analysis (0-100)
            
        Returns:
            Complete flood risk assessment
        """
        # Fetch weather data concurrently
        weather_data, forecast_data = await asyncio.gather(
            self.fetch_weather_data(lat, lon),
            self.fetch_forecast_data(lat, lon)
        )
        
        # Check if we got valid data
        has_weather = bool(weather_data)
        has_forecast = bool(forecast_data)
        
        # Calculate individual scores
        rainfall_scores = self.calculate_rainfall_score(weather_data) if has_weather else {"score_1h": 0, "score_3h": 0, "rain_1h_mm": 0, "rain_3h_mm": 0}
        humidity_data = self.calculate_humidity_score(weather_data) if has_weather else {"score": 0, "humidity": 0}
        forecast_scores = self.calculate_forecast_score(forecast_data) if has_forecast else {"score": 0, "total_rain_24h": 0, "max_rain_3h": 0, "rainy_periods": 0}
        
        # Calculate weighted total score
        total_score = (
            rainfall_scores["score_1h"] * self.WEIGHTS["rainfall_1h"] +
            rainfall_scores["score_3h"] * self.WEIGHTS["rainfall_3h"] +
            humidity_data["score"] * self.WEIGHTS["humidity"] +
            forecast_scores["score"] * self.WEIGHTS["forecast_rain"] +
            geographical_risk * self.WEIGHTS["geographical"]
        )
        
        # Determine risk level
        if total_score >= 70:
            risk_level = "Flooded"
            severity = "high_risk"
        elif total_score >= 40:
            risk_level = "At Risk"
            severity = "medium_risk"
        else:
            risk_level = "Safe"
            severity = "low_risk"
        
        # Get weather condition description
        condition = "Unknown"
        description = ""
        if has_weather and weather_data.get("weather"):
            condition = weather_data["weather"][0].get("main", "Unknown")
            description = weather_data["weather"][0].get("description", "")
        
        return {
            "success": True,
            "location": {"lat": lat, "lon": lon},
            "risk_level": risk_level,
            "severity": severity,
            "total_score": round(total_score, 1),
            "weather_available": has_weather,
            "forecast_available": has_forecast,
            "current_conditions": {
                "condition": condition,
                "description": description,
                "temperature": weather_data.get("main", {}).get("temp") if has_weather else None,
                "humidity": humidity_data["humidity"],
                "rain_1h_mm": rainfall_scores["rain_1h_mm"],
                "rain_3h_mm": rainfall_scores["rain_3h_mm"],
            },
            "forecast": {
                "total_rain_24h_mm": forecast_scores["total_rain_24h"],
                "max_rain_3h_mm": forecast_scores["max_rain_3h"],
                "rainy_periods_24h": forecast_scores["rainy_periods"],
            },
            "risk_factors": {
                "rainfall_current": rainfall_scores["score_1h"],
                "rainfall_accumulated": rainfall_scores["score_3h"],
                "humidity": humidity_data["score"],
                "forecast": forecast_scores["score"],
                "geographical": geographical_risk,
            },
            "timestamp": datetime.now().isoformat(),
        }
    
    def _prepare_ml_features(
        self,
        weather_data: Dict,
        forecast_data: Dict,
        geographical_risk: float,
        lat: float,
        lon: float
    ) -> Dict[str, float]:
        """
        Prepare features for ML model prediction from weather data
        """
        rain = weather_data.get("rain", {})
        main = weather_data.get("main", {})
        wind = weather_data.get("wind", {})
        clouds = weather_data.get("clouds", {})
        
        # Calculate accumulated rainfall from forecast
        forecasts = forecast_data.get("list", [])
        rainfall_24h = sum(f.get("rain", {}).get("3h", 0) for f in forecasts)
        forecast_rain_24h = rainfall_24h
        
        # Estimate soil saturation based on recent rainfall
        soil_saturation = min(1.0, (rain.get("3h", 0) / 50.0) + 0.2)
        
        # River proximity estimation (simplified - would use GIS data in production)
        # For Pakistan flood zones, assume closer to rivers
        river_proximity = max(0.2, 1.0 - (geographical_risk / 100.0))
        
        # Elevation estimation (simplified - would use DEM data in production)
        elevation = max(0.3, 1.0 - (geographical_risk / 100.0) * 0.7)
        
        # Check if monsoon season (June-September in Pakistan)
        month = datetime.now().month
        monsoon_season = 1 if 6 <= month <= 9 else 0
        
        # Previous flood indicator (simplified)
        previous_flood = 1 if geographical_risk > 70 else 0
        
        return {
            "rainfall_1h": rain.get("1h", 0),
            "rainfall_3h": rain.get("3h", 0),
            "rainfall_24h": rainfall_24h,
            "humidity": main.get("humidity", 50),
            "temperature": main.get("temp", 25),
            "pressure": main.get("pressure", 1013),
            "wind_speed": wind.get("speed", 0),
            "cloud_cover": clouds.get("all", 0),
            "forecast_rain_24h": forecast_rain_24h,
            "river_proximity": river_proximity,
            "elevation": elevation,
            "soil_saturation": soil_saturation,
            "previous_flood": previous_flood,
            "monsoon_season": monsoon_season,
        }
    
    async def predict_with_ml(
        self,
        lat: float,
        lon: float,
        geographical_risk: float = 0.0,
        model_type: str = "ensemble"
    ) -> Dict[str, Any]:
        """
        Predict flood risk using machine learning models
        
        Args:
            lat, lon: Location coordinates
            geographical_risk: Base risk from geographical analysis
            model_type: "ensemble", "random_forest", "gradient_boosting", or "lstm"
            
        Returns:
            ML-enhanced flood risk assessment
        """
        if not ML_AVAILABLE:
            return {
                "success": False,
                "error": "ML module not available",
                "fallback": await self.predict_flood_risk(lat, lon, geographical_risk)
            }
        
        # Fetch weather data
        weather_data, forecast_data = await asyncio.gather(
            self.fetch_weather_data(lat, lon),
            self.fetch_forecast_data(lat, lon)
        )
        
        has_weather = bool(weather_data)
        has_forecast = bool(forecast_data)
        
        if not has_weather:
            return {
                "success": False,
                "error": "No weather data available",
                "fallback": await self.predict_flood_risk(lat, lon, geographical_risk)
            }
        
        # Prepare features for ML model
        features = self._prepare_ml_features(
            weather_data, forecast_data, geographical_risk, lat, lon
        )
        
        # Build weather sequence for LSTM (if available)
        weather_sequence = None
        if forecast_data and forecast_data.get("list"):
            weather_sequence = []
            for item in forecast_data["list"]:
                weather_sequence.append({
                    "rainfall_1h": item.get("rain", {}).get("3h", 0) / 3,  # Estimate hourly
                    "humidity": item.get("main", {}).get("humidity", 50),
                    "temperature": item.get("main", {}).get("temp", 25),
                    "pressure": item.get("main", {}).get("pressure", 1013),
                    "wind_speed": item.get("wind", {}).get("speed", 0),
                    "cloud_cover": item.get("clouds", {}).get("all", 0),
                })
        
        # Get ML prediction
        ml_result = predict_flood_risk_ml(features, weather_sequence, model_type)
        
        # Get condition description
        condition = "Unknown"
        description = ""
        if has_weather and weather_data.get("weather"):
            condition = weather_data["weather"][0].get("main", "Unknown")
            description = weather_data["weather"][0].get("description", "")
        
        return {
            "success": True,
            "prediction_method": "machine_learning",
            "model_type": model_type,
            "location": {"lat": lat, "lon": lon},
            "risk_level": ml_result["risk_level"],
            "risk_index": ml_result["risk_index"],
            "confidence": ml_result["confidence"],
            "probabilities": ml_result["probabilities"],
            "current_conditions": {
                "condition": condition,
                "description": description,
                "temperature": features["temperature"],
                "humidity": features["humidity"],
                "rain_1h_mm": features["rainfall_1h"],
                "rain_3h_mm": features["rainfall_3h"],
            },
            "ml_features": features,
            "individual_predictions": ml_result.get("individual_predictions", {}),
            "models_used": ml_result.get("models_used", [model_type]),
            "feature_importance": ml_result.get("feature_importance", {}),
            "timestamp": datetime.now().isoformat(),
        }


# Singleton instance
flood_predictor = WeatherBasedFloodPredictor()


def _generate_demo_flood_risk(lat: float, lon: float, geographical_risk: float = 0.0) -> Dict[str, Any]:
    """
    Generate demo flood risk data simulating heavy rainfall conditions.
    Used for demonstration when there's no actual rainfall.
    """
    import random
    
    # Use location for deterministic "randomness" so same location = same demo data
    loc_seed = int(hashlib.md5(f"{lat:.4f},{lon:.4f}".encode()).hexdigest(), 16) % 1000
    random.seed(loc_seed)
    
    # Simulate various rainfall scenarios based on location
    scenario = loc_seed % 3  # 0=heavy, 1=moderate, 2=extreme
    
    if scenario == 0:  # Heavy rain
        rain_1h = random.uniform(15, 25)
        rain_3h = random.uniform(40, 60)
        humidity = random.randint(90, 98)
        forecast_rain = random.uniform(50, 80)
        condition = "Rain"
        description = "heavy intensity rain"
        risk_level = "At Risk"
        severity = "medium_risk"
        total_score = random.uniform(50, 69)
    elif scenario == 1:  # Moderate rain
        rain_1h = random.uniform(5, 12)
        rain_3h = random.uniform(15, 30)
        humidity = random.randint(80, 90)
        forecast_rain = random.uniform(20, 40)
        condition = "Rain"
        description = "moderate rain"
        risk_level = "At Risk"
        severity = "medium_risk"
        total_score = random.uniform(40, 55)
    else:  # Extreme/flood conditions
        rain_1h = random.uniform(30, 50)
        rain_3h = random.uniform(80, 120)
        humidity = random.randint(95, 100)
        forecast_rain = random.uniform(100, 150)
        condition = "Thunderstorm"
        description = "thunderstorm with heavy rain"
        risk_level = "Flooded"
        severity = "high_risk"
        total_score = random.uniform(75, 95)
    
    # Reset random seed
    random.seed()
    
    return {
        "success": True,
        "demo_mode": True,
        "location": {"lat": lat, "lon": lon},
        "risk_level": risk_level,
        "severity": severity,
        "total_score": round(total_score, 1),
        "weather_available": True,
        "forecast_available": True,
        "current_conditions": {
            "condition": condition,
            "description": description,
            "temperature": round(random.uniform(22, 32), 1),
            "humidity": humidity,
            "rain_1h_mm": round(rain_1h, 1),
            "rain_3h_mm": round(rain_3h, 1),
        },
        "forecast": {
            "total_rain_24h_mm": round(forecast_rain, 1),
            "max_rain_3h_mm": round(rain_3h * 0.6, 1),
            "rainy_periods_24h": random.randint(5, 8),
        },
        "risk_factors": {
            "rainfall_current": 75 if scenario == 2 else 50,
            "rainfall_accumulated": 75 if scenario == 2 else 50,
            "humidity": 70 if humidity > 90 else 40,
            "forecast": 75 if forecast_rain > 50 else 50,
            "geographical": geographical_risk,
        },
        "timestamp": datetime.now().isoformat(),
    }


async def get_weather_based_flood_risk(lat: float, lon: float, geographical_risk: float = 0.0, demo_mode: bool = False) -> Dict[str, Any]:
    """
    Get flood risk prediction based on real-time weather data
    
    Args:
        lat, lon: Location coordinates
        geographical_risk: Base risk from geographical analysis
        demo_mode: If True, returns simulated heavy rain data for demonstration
    """
    if demo_mode:
        return _generate_demo_flood_risk(lat, lon, geographical_risk)
    
    return await flood_predictor.predict_flood_risk(lat, lon, geographical_risk)


async def get_weather_for_location(lat: float, lon: float) -> Dict[str, Any]:
    """
    Get simplified weather data for a location
    """
    data = await flood_predictor.fetch_weather_data(lat, lon)
    
    if not data:
        return {"success": False, "error": "Could not fetch weather data"}
    
    return {
        "success": True,
        "location": {"lat": lat, "lon": lon},
        "temperature": data.get("main", {}).get("temp"),
        "humidity": data.get("main", {}).get("humidity"),
        "condition": data.get("weather", [{}])[0].get("main", "Unknown"),
        "description": data.get("weather", [{}])[0].get("description", ""),
        "rain_1h": data.get("rain", {}).get("1h", 0),
        "rain_3h": data.get("rain", {}).get("3h", 0),
        "wind_speed": data.get("wind", {}).get("speed", 0),
        "clouds": data.get("clouds", {}).get("all", 0),
        "timestamp": datetime.now().isoformat(),
    }


async def get_bulk_weather_predictions(locations: List[Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Get weather-based flood predictions for multiple locations concurrently
    
    Args:
        locations: List of {"lat": float, "lon": float, "geographical_risk": float} dicts
    """
    tasks = [
        get_weather_based_flood_risk(
            loc["lat"], 
            loc["lon"], 
            loc.get("geographical_risk", 0.0)
        )
        for loc in locations
    ]
    return await asyncio.gather(*tasks)


async def get_ml_flood_prediction(
    lat: float,
    lon: float,
    geographical_risk: float = 0.0,
    model_type: str = "ensemble",
    demo_mode: bool = False
) -> Dict[str, Any]:
    """
    Get ML-based flood risk prediction
    
    Args:
        lat, lon: Location coordinates
        geographical_risk: Base risk from geographical analysis
        model_type: "ensemble", "random_forest", "gradient_boosting", or "lstm"
        demo_mode: If True, generates demo features for ML prediction
        
    Returns:
        ML prediction with confidence scores and feature importance
    """
    if demo_mode:
        # Generate demo data and use it for ML prediction
        demo_data = _generate_demo_flood_risk(lat, lon, geographical_risk)
        
        if ML_AVAILABLE:
            # Create features from demo data
            import random
            loc_seed = int(hashlib.md5(f"{lat:.4f},{lon:.4f}".encode()).hexdigest(), 16) % 1000
            random.seed(loc_seed)
            
            features = {
                "rainfall_1h": demo_data["current_conditions"]["rain_1h_mm"],
                "rainfall_3h": demo_data["current_conditions"]["rain_3h_mm"],
                "rainfall_24h": demo_data["forecast"]["total_rain_24h_mm"],
                "humidity": demo_data["current_conditions"]["humidity"],
                "temperature": demo_data["current_conditions"]["temperature"],
                "pressure": random.uniform(1000, 1020),
                "wind_speed": random.uniform(5, 15),
                "cloud_cover": random.uniform(70, 100),
                "forecast_rain_24h": demo_data["forecast"]["total_rain_24h_mm"],
                "river_proximity": random.uniform(0.1, 0.5),
                "elevation": random.uniform(0.2, 0.6),
                "soil_saturation": random.uniform(0.5, 0.9),
                "previous_flood": 1 if random.random() > 0.6 else 0,
                "monsoon_season": 1 if datetime.now().month in [6, 7, 8, 9] else 0,
            }
            random.seed()
            
            ml_result = predict_flood_risk_ml(features, None, model_type)
            
            return {
                "success": True,
                "demo_mode": True,
                "prediction_method": "machine_learning",
                "model_type": model_type,
                "location": {"lat": lat, "lon": lon},
                "risk_level": ml_result["risk_level"],
                "risk_index": ml_result["risk_index"],
                "confidence": ml_result["confidence"],
                "probabilities": ml_result["probabilities"],
                "current_conditions": demo_data["current_conditions"],
                "ml_features": features,
                "individual_predictions": ml_result.get("individual_predictions", {}),
                "models_used": ml_result.get("models_used", [model_type]),
                "feature_importance": ml_result.get("feature_importance", {}),
                "timestamp": datetime.now().isoformat(),
            }
        else:
            demo_data["note"] = "ML not available, returning rule-based demo prediction"
            return demo_data
    
    return await flood_predictor.predict_with_ml(lat, lon, geographical_risk, model_type)


def get_ml_model_info() -> Dict[str, Any]:
    """Get information about available ML models"""
    if ML_AVAILABLE:
        return {
            "ml_available": True,
            **get_model_info()
        }
    return {
        "ml_available": False,
        "available_models": ["rule_based"],
        "note": "Install scikit-learn and tensorflow for ML features"
    }
