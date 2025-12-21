"""
FastAPI server for FLEWS (Flood Early Warning System)
Provides REST API endpoints for flood risk data and alerts
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn

# Import service modules
from alert import get_alert_summary, get_weather_alert_summary
from flood_risk import get_flood_risk_areas, get_weather_enhanced_tiles
from chatbot import get_chat_response
from flood_prediction import (
    get_weather_based_flood_risk, 
    get_weather_for_location,
    get_ml_flood_prediction,
    get_ml_model_info
)
from weather_api import (
    get_current_weather,
    get_weather_forecast,
    get_precipitation_data,
    get_city_weather,
    get_regional_weather,
    get_flood_risk_weather,
    get_water_level
)
from historical_data import add_weather_record, get_data_stats, get_records_for_training
from retraining_scheduler import (
    start_scheduler, 
    stop_scheduler, 
    get_scheduler_status,
    get_training_history,
    collect_weather_data,
    retrain_models
)
from data_trends import get_rainfall_trends, get_river_level_trends, get_combined_trends

# Try to import ML training function
try:
    from ml_prediction import train_models as ml_train_models
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="FLEWS API",
    description="Flood Early Warning System API",
    version="1.0.0"
)

# Configure CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    context: Optional[Dict] = None


@app.get("/")
def root():
    """Root endpoint - API information"""
    return {
        "name": "FLEWS API",
        "version": "1.0.0",
        "description": "Flood Early Warning System - Pakistan-wide flood monitoring",
        "endpoints": {
            "alerts": "/api/alerts",
            "flood_risk": "/api/flood-risk",
            "chat": "/api/chat",
            "health": "/api/health",
            "ml": {
                "predict": "/api/ml/predict",
                "models": "/api/ml/models",
                "train": "/api/ml/train"
            },
            "weather": {
                "current": "/api/weather/current",
                "forecast": "/api/weather/forecast",
                "precipitation": "/api/weather/precipitation",
                "flood_risk": "/api/weather/flood-risk",
                "city": "/api/weather/city/{city_name}",
                "region": "/api/weather/region/{region}",
                "hydrology": "/api/weather/hydrology"
            }
        }
    }


@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "FLEWS API"
    }


# ============================================================================
# ALERTS ENDPOINTS
# ============================================================================

@app.get("/api/alerts")
async def get_alerts(seed: int = None, use_weather: bool = True, demo: bool = False):
    """
    Get live alert notifications based on real-time weather data
    
    Query parameters:
        seed: Optional seed for deterministic generation (fallback mode)
        use_weather: If True (default), uses real OpenWeatherMap data
                     If False, uses simulated deterministic alerts
        demo: If True, generates demo alerts with simulated heavy rainfall
              for demonstration purposes (shows alerts even when no rain)
    """
    try:
        if use_weather:
            # Use real-time weather data for alerts (with optional demo mode)
            return await get_weather_alert_summary(demo_mode=demo)
        else:
            # Fallback to simulated alerts
            return get_alert_summary(seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alerts/simulated")
def get_simulated_alerts(seed: int = None):
    """
    Get simulated alert notifications (for testing/demo purposes)
    
    RESTful: Same seed parameter returns same alerts (stateless, deterministic)
    Omit seed to use current 5-minute window (alerts change every 5 minutes)
    """
    try:
        return get_alert_summary(seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FLOOD RISK ENDPOINTS
# ============================================================================

@app.get("/api/flood-risk")
async def get_flood_risk(min_lat: float = None, max_lat: float = None, 
                   min_lon: float = None, max_lon: float = None,
                   seed: int = None, use_weather: bool = True, demo: bool = False):
    """
    Get flood status grid for visible region
    
    RESTful: Same parameters return same tiles (stateless, deterministic)
    Omit seed to use current date (tiles change daily)
    
    Query parameters:
        min_lat, max_lat, min_lon, max_lon: Bounding box of visible map region
        seed: Optional seed for deterministic generation (defaults to current date)
        use_weather: If True (default), enhances tiles with real-time weather data
                     If False, uses only geographical factors
        demo: If True, simulates heavy rainfall for demonstration
    
    Returns tiles with status: Safe (green), At Risk (yellow), Flooded (red)
    Weather-enhanced tiles include influence from real-time precipitation data.
    """
    try:
        if use_weather or demo:
            # Get weather-enhanced tiles
            result = await get_weather_enhanced_tiles(min_lat, max_lat, min_lon, max_lon, seed, demo_mode=demo)
            return result
        else:
            # Fallback to geographical-only tiles
            tiles = get_flood_risk_areas(min_lat, max_lat, min_lon, max_lon, seed)
            return {"tiles": tiles, "data_source": "geographical"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/flood-risk/simple")
def get_flood_risk_simple(min_lat: float = None, max_lat: float = None, 
                         min_lon: float = None, max_lon: float = None,
                         seed: int = None):
    """
    Get flood status grid using geographical factors only (no weather API calls)
    
    Faster than /api/flood-risk but doesn't include real-time weather influence.
    Use for large region views or when weather API is unavailable.
    """
    try:
        tiles = get_flood_risk_areas(min_lat, max_lat, min_lon, max_lon, seed)
        return {"tiles": tiles, "data_source": "geographical"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/flood-prediction")
async def get_flood_prediction(lat: float, lon: float, demo: bool = False):
    """
    Get weather-based flood risk prediction for a specific location
    
    Uses real-time weather data from OpenWeatherMap API to calculate
    flood risk based on:
    - Current rainfall intensity (1h and 3h accumulated)
    - Humidity levels (saturated ground = higher risk)
    - Weather forecast (predicted rainfall in next 24h)
    - Geographical flood-prone factors
    
    Query parameters:
        lat: Latitude of the location
        lon: Longitude of the location
        demo: If True, simulates heavy rainfall for demonstration
    
    Returns:
        Comprehensive flood risk assessment with weather data
    """
    try:
        result = await get_weather_based_flood_risk(lat, lon, demo_mode=demo)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/location-weather")
async def get_location_weather(lat: float, lon: float):
    """
    Get current weather data for a specific location
    
    Query parameters:
        lat: Latitude
        lon: Longitude
    
    Returns:
        Current weather conditions including temperature, humidity, rainfall
    """
    try:
        result = await get_weather_for_location(lat, lon)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CHATBOT ENDPOINTS
# ============================================================================

@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Process chatbot messages and return responses
    
    Supports:
    - Simple status questions (US17)
    - Dynamic recommendations based on context (US16)
    - Emergency guidance
    - Evacuation information
    
    Request body:
        message: User's message
        context: Optional context with location and flood_data
    
    Returns:
        response: Chatbot's text response
        recommendations: List of actionable recommendations
    """
    try:
        result = get_chat_response(request.message, request.context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WEATHER & HYDROLOGICAL DATA ENDPOINTS
# ============================================================================

@app.get("/api/weather/current")
async def weather_current(lat: float, lon: float):
    """
    Get current weather data for a specific location
    
    Query parameters:
        lat: Latitude of the location
        lon: Longitude of the location
    
    Returns:
        Current weather conditions including temperature, humidity,
        precipitation, wind, and more from OpenWeatherMap API
    """
    try:
        result = await get_current_weather(lat, lon)
        if not result.get("success"):
            raise HTTPException(status_code=503, detail=result.get("message", "Weather service unavailable"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/weather/forecast")
async def weather_forecast(lat: float, lon: float, days: int = 5):
    """
    Get weather forecast for a specific location
    
    Query parameters:
        lat: Latitude of the location
        lon: Longitude of the location
        days: Number of days to forecast (1-5, default: 5)
    
    Returns:
        Weather forecast with 3-hour intervals including temperature,
        precipitation probability, and conditions
    """
    try:
        days = min(max(days, 1), 5)  # Clamp to 1-5 days
        result = await get_weather_forecast(lat, lon, days)
        if not result.get("success"):
            raise HTTPException(status_code=503, detail=result.get("message", "Forecast service unavailable"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/weather/precipitation")
async def weather_precipitation(lat: float, lon: float):
    """
    Get precipitation-specific data for flood risk analysis
    
    Query parameters:
        lat: Latitude of the location
        lon: Longitude of the location
    
    Returns:
        Current and forecasted precipitation data with risk assessment
    """
    try:
        result = await get_precipitation_data(lat, lon)
        if not result.get("success"):
            raise HTTPException(status_code=503, detail=result.get("message", "Precipitation data unavailable"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/weather/flood-risk")
async def weather_flood_risk(lat: float, lon: float):
    """
    Get comprehensive weather data for flood risk assessment
    
    Combines current weather, forecast, and precipitation analysis
    to provide a flood risk score based on weather conditions.
    
    Query parameters:
        lat: Latitude of the location
        lon: Longitude of the location
    
    Returns:
        Comprehensive weather data with flood risk assessment including:
        - Current weather conditions
        - 5-day forecast
        - Precipitation analysis
        - Risk level (low/medium/high) and contributing factors
    """
    try:
        result = await get_flood_risk_weather(lat, lon)
        if not result.get("success"):
            raise HTTPException(status_code=503, detail=result.get("message", "Flood risk weather data unavailable"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/weather/city/{city_name}")
async def weather_city(city_name: str):
    """
    Get weather data for a major Pakistani city
    
    Path parameters:
        city_name: Name of the city (e.g., 'lahore', 'karachi', 'islamabad')
    
    Supported cities: karachi, lahore, islamabad, rawalpindi, peshawar,
    quetta, multan, faisalabad, hyderabad, sukkur, nowshera, swat, muzaffarabad
    
    Returns:
        Current weather conditions for the specified city
    """
    try:
        result = await get_city_weather(city_name)
        if not result.get("success"):
            if "Unknown city" in result.get("error", ""):
                raise HTTPException(
                    status_code=404, 
                    detail={
                        "message": result.get("error"),
                        "available_cities": result.get("available_cities", [])
                    }
                )
            raise HTTPException(status_code=503, detail=result.get("message", "City weather unavailable"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/weather/region/{region}")
async def weather_region(region: str = "all"):
    """
    Get weather data for multiple cities in a Pakistan region
    
    Path parameters:
        region: Region name or 'all' for entire country
                Options: punjab, sindh, kpk, balochistan, capital, ajk, all
    
    Returns:
        Weather data for all major cities in the specified region
    """
    try:
        result = await get_regional_weather(region)
        if not result.get("success"):
            if "Unknown region" in result.get("error", ""):
                raise HTTPException(
                    status_code=404,
                    detail={
                        "message": result.get("error"),
                        "available_regions": result.get("available_regions", [])
                    }
                )
            raise HTTPException(status_code=503, detail=result.get("message", "Regional weather unavailable"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/weather/hydrology")
async def weather_hydrology(site_code: str = None, 
                            min_lon: float = None, min_lat: float = None,
                            max_lon: float = None, max_lat: float = None):
    """
    Get hydrological data (water levels, streamflow) from USGS Water Services
    
    Note: USGS data covers primarily US locations. For Pakistan-specific
    hydrological data, consider integrating with WAPDA or IRSA APIs.
    
    Query parameters:
        site_code: USGS site code (optional)
        min_lon, min_lat, max_lon, max_lat: Bounding box for site search (optional)
    
    Returns:
        Water level and streamflow data for monitoring sites
    """
    try:
        bbox = None
        if all([min_lon, min_lat, max_lon, max_lat]):
            bbox = (min_lon, min_lat, max_lon, max_lat)
        
        result = await get_water_level(site_code=site_code, bbox=bbox)
        if not result.get("success"):
            raise HTTPException(status_code=503, detail=result.get("message", "Hydrology data unavailable"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MACHINE LEARNING ENDPOINTS
# ============================================================================

@app.get("/api/ml/models")
def get_ml_models():
    """
    Get information about available machine learning models
    
    Returns:
        List of available models (ensemble, random_forest, gradient_boosting, lstm)
        Model capabilities and feature requirements
    """
    try:
        return get_ml_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/predict")
async def ml_predict(
    lat: float, 
    lon: float, 
    model: str = "ensemble",
    demo: bool = False
):
    """
    Get ML-based flood risk prediction for a specific location
    
    Uses trained machine learning models (Random Forest, Gradient Boosting, LSTM)
    combined in an ensemble for high-accuracy predictions.
    
    Query parameters:
        lat: Latitude of the location
        lon: Longitude of the location
        model: Model type - "ensemble" (default), "random_forest", "gradient_boosting", or "lstm"
        demo: If True, simulates heavy rainfall conditions for demonstration
    
    Returns:
        ML prediction with:
        - risk_level: Safe, At Risk, or Flooded
        - confidence: Model confidence score (0-1)
        - probabilities: Probability distribution across risk levels
        - feature_importance: Which features contributed most to prediction
        - individual_predictions: Predictions from each model in ensemble
    """
    try:
        valid_models = ["ensemble", "random_forest", "gradient_boosting", "lstm"]
        if model not in valid_models:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model type. Choose from: {valid_models}"
            )
        
        result = await get_ml_flood_prediction(lat, lon, model_type=model, demo_mode=demo)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/train")
async def train_ml_models():
    """
    Train all machine learning models
    
    Triggers training of Random Forest, Gradient Boosting, and LSTM models
    using synthetic training data (in production, use historical flood data).
    
    Note: Training may take several minutes. Models are saved to disk
    and automatically loaded on subsequent predictions.
    
    Returns:
        Training results with accuracy scores for each model
    """
    try:
        if not ML_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="ML module not available. Install scikit-learn and tensorflow."
            )
        
        # Run training in background would be better for production
        # For now, run synchronously
        accuracies = ml_train_models()
        
        return {
            "success": True,
            "message": "Model training completed",
            "accuracies": accuracies
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/compare")
async def compare_predictions(lat: float, lon: float, demo: bool = False):
    """
    Compare predictions from rule-based and ML models
    
    Useful for evaluating model performance and understanding
    the difference between approaches.
    
    Query parameters:
        lat: Latitude
        lon: Longitude
        demo: If True, simulates heavy rainfall
    
    Returns:
        Side-by-side comparison of rule-based and ML predictions
    """
    try:
        # Get rule-based prediction
        rule_based = await get_weather_based_flood_risk(lat, lon, demo_mode=demo)
        
        # Get ML prediction
        ml_prediction = await get_ml_flood_prediction(lat, lon, model_type="ensemble", demo_mode=demo)
        
        return {
            "location": {"lat": lat, "lon": lon},
            "demo_mode": demo,
            "rule_based_prediction": {
                "risk_level": rule_based.get("risk_level"),
                "total_score": rule_based.get("total_score"),
                "method": "weighted_rules"
            },
            "ml_prediction": {
                "risk_level": ml_prediction.get("risk_level"),
                "confidence": ml_prediction.get("confidence"),
                "probabilities": ml_prediction.get("probabilities"),
                "models_used": ml_prediction.get("models_used", []),
                "method": "machine_learning"
            },
            "agreement": rule_based.get("risk_level") == ml_prediction.get("risk_level"),
            "timestamp": rule_based.get("timestamp")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/scheduler/status")
async def scheduler_status():
    """Get scheduler status"""
    return get_scheduler_status()


@app.get("/api/ml/training-history")
async def training_history():
    """Get model training history"""
    return {
        "history": get_training_history(),
        "total_trainings": len(get_training_history())
    }


@app.get("/api/ml/historical-data/stats")
async def historical_data_stats():
    """Get statistics about collected historical data"""
    return get_data_stats()


@app.post("/api/ml/historical-data/collect")
async def trigger_data_collection():
    """Manually trigger data collection"""
    collected = await collect_weather_data()
    return {"message": f"Collected {collected} records", "collected": collected}


@app.post("/api/ml/retrain")
async def trigger_retraining():
    """Manually trigger model retraining"""
    results = await retrain_models()
    if results:
        return {"message": "Retraining complete", "results": results}
    else:
        return {"message": "Retraining failed or insufficient data", "results": None}


# ============================================================================
# DATA TRENDS ENDPOINTS (Charts & Visualizations)
# ============================================================================

@app.get("/api/trends/rainfall")
async def trends_rainfall(lat: float, lon: float, days: int = 7):
    """
    Get rainfall trend data for visualization charts
    
    Query parameters:
        lat: Latitude of the location
        lon: Longitude of the location
        days: Number of days of historical data (default: 7)
    
    Returns:
        Historical rainfall data and forecasts formatted for charting
    """
    try:
        days = min(max(days, 1), 30)  # Clamp to 1-30 days
        result = await get_rainfall_trends(lat, lon, days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trends/river-level")
async def trends_river_level(lat: float, lon: float, days: int = 7):
    """
    Get river level trend data for visualization charts
    
    Query parameters:
        lat: Latitude of the location
        lon: Longitude of the location
        days: Number of days of historical data (default: 7)
    
    Returns:
        River level history and forecasts with threshold indicators
    """
    try:
        days = min(max(days, 1), 30)  # Clamp to 1-30 days
        result = await get_river_level_trends(lat, lon, days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trends/combined")
async def trends_combined(lat: float, lon: float, days: int = 7):
    """
    Get combined rainfall and river level trends with correlation analysis
    
    Query parameters:
        lat: Latitude of the location
        lon: Longitude of the location
        days: Number of days of historical data (default: 7)
    
    Returns:
        Combined data for both parameters with correlation analysis
    """
    try:
        days = min(max(days, 1), 30)  # Clamp to 1-30 days
        result = await get_combined_trends(lat, lon, days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SERVER STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Start the retraining scheduler on server startup"""
    start_scheduler()
    print("FLEWS server started with retraining scheduler")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the scheduler on shutdown"""
    stop_scheduler()

if __name__ == "__main__":
    print("Starting FLEWS API Server...")
    print("API Documentation available at: http://localhost:8000/docs")
    print("Frontend should run on: http://localhost:5173")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )

