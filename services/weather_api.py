"""
Weather and Hydrological Data API Integration for FLEWS
Fetches real-time weather data from OpenWeatherMap and hydrological data from water services APIs
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import httpx
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from functools import lru_cache
import asyncio

# Load environment variables from root .env file (FLEWS/.env)
root_dir = Path(__file__).parent.parent  # Go up from services/ to FLEWS/
env_path = root_dir / '.env'
load_dotenv(env_path)

# API Configuration
# Set these as environment variables for security
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "your_api_key_here")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# USGS Water Services API (free, no key required) - Alternative to HydroServer
USGS_WATER_BASE_URL = "https://waterservices.usgs.gov/nwis"

# Pakistan Meteorological Department - Alternative weather source
PMD_BASE_URL = "https://www.pmd.gov.pk"  # Note: May require web scraping

# Cache settings (in seconds)
CACHE_DURATION = 300  # 5 minutes


class WeatherAPIClient:
    """Client for fetching weather data from OpenWeatherMap API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENWEATHER_API_KEY
        self.base_url = OPENWEATHER_BASE_URL
        
    async def get_current_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch current weather data for a specific location
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            
        Returns:
            Dictionary containing weather data including:
            - temperature, humidity, pressure
            - weather conditions (rain, clouds, etc.)
            - wind speed and direction
        """
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric"  # Use Celsius
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/weather",
                    params=params,
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                
                return {
                    "success": True,
                    "location": {
                        "lat": lat,
                        "lon": lon,
                        "name": data.get("name", "Unknown"),
                        "country": data.get("sys", {}).get("country", "")
                    },
                    "weather": {
                        "condition": data.get("weather", [{}])[0].get("main", "Unknown"),
                        "description": data.get("weather", [{}])[0].get("description", ""),
                        "icon": data.get("weather", [{}])[0].get("icon", ""),
                        "temperature": data.get("main", {}).get("temp"),
                        "feels_like": data.get("main", {}).get("feels_like"),
                        "humidity": data.get("main", {}).get("humidity"),
                        "pressure": data.get("main", {}).get("pressure"),
                        "visibility": data.get("visibility"),
                        "clouds": data.get("clouds", {}).get("all", 0),
                        "wind": {
                            "speed": data.get("wind", {}).get("speed"),
                            "direction": data.get("wind", {}).get("deg"),
                            "gust": data.get("wind", {}).get("gust")
                        },
                        "rain": data.get("rain", {}),
                        "snow": data.get("snow", {})
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "OpenWeatherMap"
                }
                
            except httpx.HTTPStatusError as e:
                return {
                    "success": False,
                    "error": f"HTTP error: {e.response.status_code}",
                    "message": "Failed to fetch weather data"
                }
            except httpx.RequestError as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Network error while fetching weather data"
                }
    
    async def get_weather_forecast(self, lat: float, lon: float, days: int = 5) -> Dict[str, Any]:
        """
        Fetch weather forecast for the next several days
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            days: Number of days to forecast (max 5 for free tier)
            
        Returns:
            Dictionary containing forecast data
        """
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
            "cnt": min(days * 8, 40)  # API returns 3-hour intervals, 8 per day
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/forecast",
                    params=params,
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                
                # Process forecast data
                forecasts = []
                for item in data.get("list", []):
                    forecasts.append({
                        "datetime": item.get("dt_txt"),
                        "timestamp": item.get("dt"),
                        "temperature": item.get("main", {}).get("temp"),
                        "feels_like": item.get("main", {}).get("feels_like"),
                        "humidity": item.get("main", {}).get("humidity"),
                        "pressure": item.get("main", {}).get("pressure"),
                        "condition": item.get("weather", [{}])[0].get("main"),
                        "description": item.get("weather", [{}])[0].get("description"),
                        "icon": item.get("weather", [{}])[0].get("icon"),
                        "clouds": item.get("clouds", {}).get("all", 0),
                        "wind_speed": item.get("wind", {}).get("speed"),
                        "rain_3h": item.get("rain", {}).get("3h", 0),
                        "snow_3h": item.get("snow", {}).get("3h", 0),
                        "pop": item.get("pop", 0)  # Probability of precipitation
                    })
                
                return {
                    "success": True,
                    "location": {
                        "lat": lat,
                        "lon": lon,
                        "name": data.get("city", {}).get("name", "Unknown"),
                        "country": data.get("city", {}).get("country", "")
                    },
                    "forecasts": forecasts,
                    "total_forecasts": len(forecasts),
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "OpenWeatherMap"
                }
                
            except httpx.HTTPStatusError as e:
                return {
                    "success": False,
                    "error": f"HTTP error: {e.response.status_code}",
                    "message": "Failed to fetch forecast data"
                }
            except httpx.RequestError as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Network error while fetching forecast data"
                }

    async def get_precipitation_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get precipitation-specific data useful for flood prediction
        Combines current and forecast data to assess flood risk from rainfall
        """
        current = await self.get_current_weather(lat, lon)
        forecast = await self.get_weather_forecast(lat, lon, days=3)
        
        if not current.get("success") or not forecast.get("success"):
            return {
                "success": False,
                "error": "Failed to fetch precipitation data"
            }
        
        # Calculate precipitation totals
        total_rain_forecast = sum(
            f.get("rain_3h", 0) for f in forecast.get("forecasts", [])
        )
        
        # Assess precipitation intensity
        current_rain = current.get("weather", {}).get("rain", {})
        rain_1h = current_rain.get("1h", 0)
        rain_3h = current_rain.get("3h", 0)
        
        # Determine precipitation risk level
        risk_level = "low"
        if total_rain_forecast > 50 or rain_1h > 10:
            risk_level = "high"
        elif total_rain_forecast > 20 or rain_1h > 5:
            risk_level = "medium"
        
        return {
            "success": True,
            "location": current.get("location"),
            "current_precipitation": {
                "rain_1h_mm": rain_1h,
                "rain_3h_mm": rain_3h,
                "humidity": current.get("weather", {}).get("humidity"),
                "clouds": current.get("weather", {}).get("clouds")
            },
            "forecast_precipitation": {
                "total_rain_mm": total_rain_forecast,
                "max_probability": max(
                    (f.get("pop", 0) for f in forecast.get("forecasts", [])),
                    default=0
                ) * 100,
                "rainy_periods": sum(
                    1 for f in forecast.get("forecasts", []) 
                    if f.get("rain_3h", 0) > 0
                )
            },
            "risk_assessment": {
                "precipitation_risk": risk_level,
                "flood_contribution": "high" if risk_level == "high" else "moderate" if risk_level == "medium" else "low"
            },
            "timestamp": datetime.utcnow().isoformat()
        }


class HydrologicalAPIClient:
    """
    Client for fetching hydrological data from water services APIs
    Uses USGS Water Services as the primary source (free, no key required)
    Note: For Pakistan-specific data, consider integrating with:
    - Pakistan Water and Power Development Authority (WAPDA)
    - Indus River System Authority (IRSA)
    """
    
    def __init__(self):
        self.base_url = USGS_WATER_BASE_URL
        
    async def get_water_level(
        self, 
        site_code: str = None,
        lat: float = None, 
        lon: float = None,
        bbox: tuple = None
    ) -> Dict[str, Any]:
        """
        Fetch water level data from USGS Water Services
        
        Args:
            site_code: USGS site code (e.g., '01646500')
            lat, lon: Location coordinates (for nearest site search)
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            Dictionary containing water level/streamflow data
        """
        params = {
            "format": "json",
            "parameterCd": "00065,00060",  # Gage height and Discharge
            "siteStatus": "active",
            "period": "P1D"  # Past 1 day
        }
        
        if site_code:
            params["sites"] = site_code
        elif bbox:
            params["bBox"] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/iv/",
                    params=params,
                    timeout=15.0
                )
                response.raise_for_status()
                data = response.json()
                
                # Parse USGS response format
                time_series = data.get("value", {}).get("timeSeries", [])
                sites = []
                
                for series in time_series:
                    site_info = series.get("sourceInfo", {})
                    variable = series.get("variable", {})
                    values = series.get("values", [{}])[0].get("value", [])
                    
                    if values:
                        latest = values[-1]
                        sites.append({
                            "site_code": site_info.get("siteCode", [{}])[0].get("value"),
                            "site_name": site_info.get("siteName"),
                            "location": {
                                "lat": site_info.get("geoLocation", {}).get("geogLocation", {}).get("latitude"),
                                "lon": site_info.get("geoLocation", {}).get("geogLocation", {}).get("longitude")
                            },
                            "variable": {
                                "code": variable.get("variableCode", [{}])[0].get("value"),
                                "name": variable.get("variableName"),
                                "unit": variable.get("unit", {}).get("unitCode")
                            },
                            "latest_reading": {
                                "value": latest.get("value"),
                                "datetime": latest.get("dateTime"),
                                "qualifiers": latest.get("qualifiers", [])
                            }
                        })
                
                return {
                    "success": True,
                    "sites": sites,
                    "total_sites": len(sites),
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "USGS Water Services"
                }
                
            except httpx.HTTPStatusError as e:
                return {
                    "success": False,
                    "error": f"HTTP error: {e.response.status_code}",
                    "message": "Failed to fetch hydrological data"
                }
            except httpx.RequestError as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Network error while fetching hydrological data"
                }

    async def get_flood_stage_data(self, site_code: str) -> Dict[str, Any]:
        """
        Get flood stage information for a specific monitoring site
        Includes action, flood, and major flood stages
        """
        params = {
            "format": "json",
            "sites": site_code,
            "parameterCd": "00065",  # Gage height
            "siteStatus": "active"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                # Get current water level
                response = await client.get(
                    f"{self.base_url}/iv/",
                    params={**params, "period": "P1D"},
                    timeout=15.0
                )
                response.raise_for_status()
                current_data = response.json()
                
                # Parse current level
                time_series = current_data.get("value", {}).get("timeSeries", [])
                current_level = None
                site_name = None
                
                for series in time_series:
                    values = series.get("values", [{}])[0].get("value", [])
                    if values:
                        current_level = float(values[-1].get("value", 0))
                        site_name = series.get("sourceInfo", {}).get("siteName")
                        break
                
                # Note: Flood stages would typically come from NWS or local authorities
                # This is a simplified example
                return {
                    "success": True,
                    "site_code": site_code,
                    "site_name": site_name,
                    "current_level_ft": current_level,
                    "flood_stages": {
                        "note": "Flood stages vary by location - contact local authorities for accurate data"
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "USGS Water Services"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to fetch flood stage data"
                }


class PakistanWeatherService:
    """
    Specialized service for Pakistan-specific weather and flood data
    Combines multiple data sources for comprehensive coverage
    """
    
    # Major cities in Pakistan with coordinates
    MAJOR_CITIES = {
        "karachi": {"lat": 24.8607, "lon": 67.0011},
        "lahore": {"lat": 31.5497, "lon": 74.3436},
        "islamabad": {"lat": 33.6844, "lon": 73.0479},
        "rawalpindi": {"lat": 33.5651, "lon": 73.0169},
        "peshawar": {"lat": 34.0151, "lon": 71.5249},
        "quetta": {"lat": 30.1798, "lon": 66.9750},
        "multan": {"lat": 30.1575, "lon": 71.5249},
        "faisalabad": {"lat": 31.4504, "lon": 73.1350},
        "hyderabad": {"lat": 25.3960, "lon": 68.3578},
        "sukkur": {"lat": 27.7052, "lon": 68.8574},
        "nowshera": {"lat": 34.0153, "lon": 71.9747},
        "swat": {"lat": 35.2227, "lon": 72.4258},
        "muzaffarabad": {"lat": 34.3700, "lon": 73.4700}
    }
    
    def __init__(self, openweather_api_key: str = None):
        self.weather_client = WeatherAPIClient(openweather_api_key)
        self.hydro_client = HydrologicalAPIClient()
    
    async def get_city_weather(self, city_name: str) -> Dict[str, Any]:
        """Get weather data for a major Pakistani city"""
        city = city_name.lower()
        if city not in self.MAJOR_CITIES:
            return {
                "success": False,
                "error": f"Unknown city: {city_name}",
                "available_cities": list(self.MAJOR_CITIES.keys())
            }
        
        coords = self.MAJOR_CITIES[city]
        return await self.weather_client.get_current_weather(coords["lat"], coords["lon"])
    
    async def get_regional_weather(self, region: str = "all") -> Dict[str, Any]:
        """
        Get weather data for multiple cities in a region
        
        Args:
            region: 'punjab', 'sindh', 'kpk', 'balochistan', or 'all'
        """
        regions = {
            "punjab": ["lahore", "multan", "faisalabad", "rawalpindi"],
            "sindh": ["karachi", "hyderabad", "sukkur"],
            "kpk": ["peshawar", "nowshera", "swat"],
            "balochistan": ["quetta"],
            "capital": ["islamabad"],
            "ajk": ["muzaffarabad"]
        }
        
        if region == "all":
            cities = list(self.MAJOR_CITIES.keys())
        elif region.lower() in regions:
            cities = regions[region.lower()]
        else:
            return {
                "success": False,
                "error": f"Unknown region: {region}",
                "available_regions": list(regions.keys()) + ["all"]
            }
        
        # Fetch weather for all cities concurrently
        tasks = [self.get_city_weather(city) for city in cities]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        weather_data = []
        for city, result in zip(cities, results):
            if isinstance(result, Exception):
                weather_data.append({
                    "city": city,
                    "success": False,
                    "error": str(result)
                })
            else:
                weather_data.append({
                    "city": city,
                    **result
                })
        
        return {
            "success": True,
            "region": region,
            "cities": weather_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_flood_risk_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get comprehensive weather data relevant to flood risk assessment
        Combines current weather, forecast, and precipitation analysis
        """
        # Fetch all data concurrently
        current, forecast, precipitation = await asyncio.gather(
            self.weather_client.get_current_weather(lat, lon),
            self.weather_client.get_weather_forecast(lat, lon, days=5),
            self.weather_client.get_precipitation_data(lat, lon)
        )
        
        # Calculate flood risk factors from weather
        risk_factors = []
        risk_score = 0
        
        if current.get("success"):
            weather = current.get("weather", {})
            
            # Heavy rain indicator
            rain = weather.get("rain", {})
            if rain.get("1h", 0) > 10:
                risk_factors.append("Heavy rainfall in last hour")
                risk_score += 30
            elif rain.get("1h", 0) > 5:
                risk_factors.append("Moderate rainfall in last hour")
                risk_score += 15
            
            # High humidity with clouds
            if weather.get("humidity", 0) > 80 and weather.get("clouds", 0) > 70:
                risk_factors.append("High humidity with heavy cloud cover")
                risk_score += 10
        
        if forecast.get("success"):
            forecasts = forecast.get("forecasts", [])
            
            # Check for rain in forecast
            rain_periods = sum(1 for f in forecasts if f.get("rain_3h", 0) > 0)
            if rain_periods > 10:
                risk_factors.append(f"Rain expected in {rain_periods} forecast periods")
                risk_score += 20
            
            # High precipitation probability
            max_pop = max((f.get("pop", 0) for f in forecasts), default=0)
            if max_pop > 0.8:
                risk_factors.append("High probability of precipitation (>80%)")
                risk_score += 15
        
        # Determine overall risk level
        if risk_score >= 50:
            overall_risk = "high"
        elif risk_score >= 25:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        return {
            "success": True,
            "location": {"lat": lat, "lon": lon},
            "current_weather": current if current.get("success") else None,
            "forecast": forecast if forecast.get("success") else None,
            "precipitation": precipitation if precipitation.get("success") else None,
            "flood_risk_assessment": {
                "risk_level": overall_risk,
                "risk_score": risk_score,
                "risk_factors": risk_factors
            },
            "timestamp": datetime.utcnow().isoformat()
        }


# Singleton instances for use in server
weather_client = WeatherAPIClient()
hydro_client = HydrologicalAPIClient()
pakistan_service = PakistanWeatherService()


# Helper functions for server integration
async def get_current_weather(lat: float, lon: float) -> Dict[str, Any]:
    """Get current weather for coordinates"""
    return await weather_client.get_current_weather(lat, lon)


async def get_weather_forecast(lat: float, lon: float, days: int = 5) -> Dict[str, Any]:
    """Get weather forecast for coordinates"""
    return await weather_client.get_weather_forecast(lat, lon, days)


async def get_precipitation_data(lat: float, lon: float) -> Dict[str, Any]:
    """Get precipitation data for coordinates"""
    return await weather_client.get_precipitation_data(lat, lon)


async def get_city_weather(city_name: str) -> Dict[str, Any]:
    """Get weather for a Pakistani city"""
    return await pakistan_service.get_city_weather(city_name)


async def get_regional_weather(region: str = "all") -> Dict[str, Any]:
    """Get weather for a region in Pakistan"""
    return await pakistan_service.get_regional_weather(region)


async def get_flood_risk_weather(lat: float, lon: float) -> Dict[str, Any]:
    """Get comprehensive flood risk weather data"""
    return await pakistan_service.get_flood_risk_weather(lat, lon)


async def get_water_level(site_code: str = None, bbox: tuple = None) -> Dict[str, Any]:
    """Get water level data from hydrological services"""
    return await hydro_client.get_water_level(site_code=site_code, bbox=bbox)
