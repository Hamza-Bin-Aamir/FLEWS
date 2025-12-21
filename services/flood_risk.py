"""
Flood risk assessment module for FLEWS
Generates a grid of flood status tiles covering Pakistan
Integrates with real-time weather data for accurate predictions
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import random
import math
import hashlib
import asyncio

# Pakistan bounding box (approximate)
PAKISTAN_BOUNDS = {
    "min_lat": 23.5,  # Southern tip
    "max_lat": 37.0,  # Northern tip (Kashmir)
    "min_lon": 60.5,  # Western border
    "max_lon": 77.5,  # Eastern border
}

# Tile size in degrees (approximately 1km = 0.009 degrees at equator)
# Dynamic tile sizing based on zoom level:
# - Minimum: 0.005 degrees (~0.5km) for zoomed-in views
# - Maximum: 0.1 degrees (~11km) for country-wide views
MIN_TILE_SIZE = 0.005  # ~500m (most detailed)
MAX_TILE_SIZE = 0.1    # ~11km (least detailed)

def calculate_optimal_tile_size(min_lat: float, max_lat: float, 
                                min_lon: float, max_lon: float) -> float:
    """
    Calculate optimal tile size based on the requested region
    Larger regions get larger tiles for performance
    Smaller regions get smaller tiles for detail
    
    Args:
        min_lat, max_lat, min_lon, max_lon: Bounding box of requested region
        
    Returns:
        Tile size in degrees
    """
    # Calculate region dimensions
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon
    region_area = lat_span * lon_span
    
    # Determine tile size based on region size
    # Small region (zoomed in): use small tiles
    # Large region (zoomed out): use large tiles
    
    if region_area < 0.1:  # Very small region (< ~10km x 10km)
        return MIN_TILE_SIZE  # 500m tiles
    elif region_area < 1.0:  # Small region (< ~100km x 100km)
        return 0.01  # ~1km tiles
    elif region_area < 10.0:  # Medium region (< ~1000km x 1000km)
        return 0.02  # ~2km tiles
    elif region_area < 50.0:  # Large region
        return 0.05  # ~5km tiles
    else:  # Very large region (country-wide view)
        return MAX_TILE_SIZE  # ~11km tiles

# Flood epicenters - major river systems and flood-prone areas in Pakistan
FLOOD_EPICENTERS = [
    # Indus River - Southern Punjab/Sindh
    (67.0, 26.5, 0.8),  # Near Sukkur
    (68.5, 27.5, 0.6),  # Jacobabad area
    (68.0, 28.5, 0.7),  # Rahim Yar Khan area
    
    # Indus River - Central
    (72.0, 32.0, 0.9),  # Dera Ghazi Khan
    (71.5, 33.0, 0.7),  # Mianwali area
    
    # Indus River - Northern
    (72.6, 34.1, 1.0),  # Topi/Swabi (your location)
    (71.9, 34.0, 0.8),  # Nowshera
    (71.5, 34.2, 0.7),  # Charsadda
    
    # Jhelum River
    (73.7, 32.9, 0.6),  # Jhelum city area
    
    # Chenab River
    (72.3, 30.2, 0.7),  # Multan area
    
    # Ravi River
    (74.3, 31.5, 0.5),  # Lahore area
    
    # Monsoon-affected areas
    (73.0, 33.6, 0.6),  # Islamabad/Rawalpindi
    (74.9, 34.1, 0.5),  # AJK region
    
    # Coastal areas (occasional flooding)
    (66.9, 24.9, 0.4),  # Karachi coast
]

def calculate_flood_risk(lat: float, lon: float, seed: int = 0) -> Tuple[str, float]:
    """
    Calculate flood risk based on distance from flood epicenters
    Uses seed for deterministic randomness (same location = same result)
    
    Returns:
        Tuple of (status, risk_score) where:
        - status: 'Flooded', 'At Risk', or 'Safe'
        - risk_score: Numeric score 0-100 for smoothing purposes
    """
    min_distance = float('inf')
    max_intensity = 0.0
    closest_epicenter = None
    
    for epi_lon, epi_lat, intensity in FLOOD_EPICENTERS:
        # Calculate distance (approximate, using euclidean distance)
        distance = math.sqrt((lat - epi_lat)**2 + (lon - epi_lon)**2)
        
        if distance < min_distance:
            min_distance = distance
            max_intensity = intensity
            closest_epicenter = (epi_lon, epi_lat)
    
    # Adjust thresholds based on intensity
    flooded_radius = 0.15 * max_intensity  # Core flood zone
    at_risk_radius = 0.4 * max_intensity   # Buffer zone
    
    # Calculate risk score (0-100) based on distance
    if min_distance < flooded_radius:
        # Inside flood zone: score 70-100
        risk_score = 100 - (min_distance / flooded_radius) * 30
        return ("Flooded", risk_score)
    elif min_distance < at_risk_radius:
        # In buffer zone: score 40-70
        normalized_dist = (min_distance - flooded_radius) / (at_risk_radius - flooded_radius)
        risk_score = 70 - normalized_dist * 30
        return ("At Risk", risk_score)
    else:
        # Outside risk zones: score 0-40 (with some randomness for isolated incidents)
        location_hash = int(hashlib.md5(f"{lat:.6f}{lon:.6f}{seed}".encode()).hexdigest(), 16) % 1000
        rand_value = location_hash / 1000.0
        
        # Base score decreases with distance
        base_score = max(0, 40 - (min_distance - at_risk_radius) * 20)
        
        if rand_value < 0.02:
            return ("At Risk", 45)
        elif rand_value < 0.03:
            return ("Flooded", 75)
        return ("Safe", base_score)


def smooth_tiles(tiles: List[Dict[str, Any]], tile_size: float) -> List[Dict[str, Any]]:
    """
    Apply spatial smoothing to tiles for cleaner visualization.
    
    Rules:
    1. Flooded tiles should be surrounded by At Risk (not directly Safe)
    2. Safe tiles surrounded by At Risk/Flooded should become At Risk
    3. Creates gradual transitions between zones
    
    Args:
        tiles: List of tile dictionaries with coordinates and status
        tile_size: Size of tiles in degrees
        
    Returns:
        Smoothed tiles list
    """
    # Build a spatial index for quick neighbor lookup
    tile_map = {}
    for tile in tiles:
        # Use bottom-left corner as key (rounded to tile grid)
        coords = tile["coordinates"][0]
        key = (round(coords[0] / tile_size) * tile_size, 
               round(coords[1] / tile_size) * tile_size)
        tile_map[key] = tile
    
    # Status values for comparison
    status_value = {"Safe": 0, "At Risk": 1, "Flooded": 2}
    value_status = {0: "Safe", 1: "At Risk", 2: "Flooded"}
    
    # First pass: collect neighbor information
    for tile in tiles:
        coords = tile["coordinates"][0]
        lon, lat = coords[0], coords[1]
        
        # Find neighbors (8-directional)
        neighbor_statuses = []
        for dlat in [-tile_size, 0, tile_size]:
            for dlon in [-tile_size, 0, tile_size]:
                if dlat == 0 and dlon == 0:
                    continue
                neighbor_key = (round((lon + dlon) / tile_size) * tile_size,
                               round((lat + dlat) / tile_size) * tile_size)
                if neighbor_key in tile_map:
                    neighbor_statuses.append(status_value[tile_map[neighbor_key]["status"]])
        
        tile["_neighbor_statuses"] = neighbor_statuses
        tile["_original_status"] = tile["status"]
    
    # Second pass: apply smoothing rules
    changes_made = True
    iterations = 0
    max_iterations = 3  # Limit iterations to prevent infinite loops
    
    while changes_made and iterations < max_iterations:
        changes_made = False
        iterations += 1
        
        for tile in tiles:
            neighbor_statuses = tile.get("_neighbor_statuses", [])
            if not neighbor_statuses:
                continue
            
            current_value = status_value[tile["status"]]
            avg_neighbor = sum(neighbor_statuses) / len(neighbor_statuses)
            max_neighbor = max(neighbor_statuses) if neighbor_statuses else 0
            min_neighbor = min(neighbor_statuses) if neighbor_statuses else 0
            
            new_status = tile["status"]
            
            # Rule 1: Safe tile surrounded mostly by At Risk/Flooded -> At Risk
            if tile["status"] == "Safe":
                flooded_neighbors = sum(1 for s in neighbor_statuses if s == 2)
                at_risk_neighbors = sum(1 for s in neighbor_statuses if s == 1)
                
                if flooded_neighbors >= 2 or (flooded_neighbors >= 1 and at_risk_neighbors >= 2):
                    new_status = "At Risk"
                elif at_risk_neighbors >= 4:
                    new_status = "At Risk"
            
            # Rule 2: Flooded tile with mostly Safe neighbors -> At Risk
            # (isolated flood makes no sense)
            elif tile["status"] == "Flooded":
                safe_neighbors = sum(1 for s in neighbor_statuses if s == 0)
                if safe_neighbors >= 5:  # Mostly surrounded by Safe
                    new_status = "At Risk"
            
            # Rule 3: At Risk tile completely surrounded by Safe -> Safe
            # (isolated yellow doesn't make sense)
            elif tile["status"] == "At Risk":
                safe_neighbors = sum(1 for s in neighbor_statuses if s == 0)
                if safe_neighbors == len(neighbor_statuses) and len(neighbor_statuses) >= 6:
                    new_status = "Safe"
            
            if new_status != tile["status"]:
                tile["status"] = new_status
                tile["smoothed"] = True
                changes_made = True
        
        # Update neighbor statuses for next iteration
        for tile in tiles:
            coords = tile["coordinates"][0]
            lon, lat = coords[0], coords[1]
            
            neighbor_statuses = []
            for dlat in [-tile_size, 0, tile_size]:
                for dlon in [-tile_size, 0, tile_size]:
                    if dlat == 0 and dlon == 0:
                        continue
                    neighbor_key = (round((lon + dlon) / tile_size) * tile_size,
                                   round((lat + dlat) / tile_size) * tile_size)
                    if neighbor_key in tile_map:
                        neighbor_statuses.append(status_value[tile_map[neighbor_key]["status"]])
            
            tile["_neighbor_statuses"] = neighbor_statuses
    
    # Clean up temporary fields
    for tile in tiles:
        tile.pop("_neighbor_statuses", None)
        tile.pop("_original_status", None)
    
    return tiles

def generate_flood_grid(min_lat: float = None, max_lat: float = None,
                       min_lon: float = None, max_lon: float = None,
                       seed: int = 0) -> List[Dict[str, Any]]:
    """
    Generates a grid of flood status tiles for specified region
    Tile size automatically scales based on region size (500m to 11km)
    
    Uses seed for deterministic generation - same inputs = same outputs (RESTful)
    
    Flood patterns are logically organized:
    - Red (Flooded): Core flood zones near epicenters
    - Yellow (At Risk): Buffer zones around flooded areas
    - Green (Safe): Areas far from flood zones
    
    Tiles are spatially smoothed for cleaner visualization.
    """
    tiles = []
    
    # Use provided bounds or default to all Pakistan
    lat_min = min_lat if min_lat is not None else PAKISTAN_BOUNDS["min_lat"]
    lat_max = max_lat if max_lat is not None else PAKISTAN_BOUNDS["max_lat"]
    lon_min = min_lon if min_lon is not None else PAKISTAN_BOUNDS["min_lon"]
    lon_max = max_lon if max_lon is not None else PAKISTAN_BOUNDS["max_lon"]
    
    # Calculate optimal tile size for this region
    tile_size = calculate_optimal_tile_size(lat_min, lat_max, lon_min, lon_max)
    
    # Use bounds for tile ID generation (deterministic)
    tile_id_base = int(hashlib.md5(f"{lat_min}{lat_max}{lon_min}{lon_max}{seed}".encode()).hexdigest(), 16) % 1000000
    
    # Generate grid covering the specified region
    lat = lat_min
    tile_count = 0
    while lat < lat_max:
        lon = lon_min
        while lon < lon_max:
            # Calculate flood status based on proximity to epicenters (deterministic)
            status, risk_score = calculate_flood_risk(lat, lon, seed)
            
            # Create tile polygon
            tile = {
                "id": f"tile-{tile_id_base + tile_count}",
                "status": status,
                "risk_score": round(risk_score, 1),
                "coordinates": [
                    [lon, lat],
                    [lon + tile_size, lat],
                    [lon + tile_size, lat + tile_size],
                    [lon, lat + tile_size],
                    [lon, lat]
                ],
                "last_updated": datetime.now().isoformat()
            }
            
            tiles.append(tile)
            tile_count += 1
            lon += tile_size
        lat += tile_size
    
    # Apply spatial smoothing for cleaner visualization
    tiles = smooth_tiles(tiles, tile_size)
    
    return tiles

def get_flood_risk_areas(min_lat: float = None, max_lat: float = None, 
                         min_lon: float = None, max_lon: float = None,
                         seed: int = None) -> List[Dict[str, Any]]:
    """
    Returns the current flood status grid for visible region
    If no bounds provided, returns all of Pakistan
    
    RESTful: Same parameters always return same data (deterministic)
    Seed changes daily to simulate evolving flood conditions
    
    Args:
        min_lat, max_lat, min_lon, max_lon: Bounding box of visible region
        seed: Optional seed for deterministic generation (defaults to current date)
    
    In production, this would query real-time satellite data and sensor networks
    """
    # Use current date as seed if not provided (changes daily)
    if seed is None:
        seed = int(datetime.now().strftime("%Y%m%d"))
    
    # Use provided bounds or default to all Pakistan
    bounds = {
        "min_lat": min_lat if min_lat is not None else PAKISTAN_BOUNDS["min_lat"],
        "max_lat": max_lat if max_lat is not None else PAKISTAN_BOUNDS["max_lat"],
        "min_lon": min_lon if min_lon is not None else PAKISTAN_BOUNDS["min_lon"],
        "max_lon": max_lon if max_lon is not None else PAKISTAN_BOUNDS["max_lon"],
    }
    
    return generate_flood_grid(bounds["min_lat"], bounds["max_lat"], 
                               bounds["min_lon"], bounds["max_lon"], seed)


def get_risk_by_location(lat: float, lng: float) -> Dict:
    """
    Get flood risk information for a specific location
    
    Args:
        lat: Latitude coordinate
        lng: Longitude coordinate
    
    Returns:
        Risk assessment for the location
    """
    # Dummy implementation - in reality, would query GIS database
    # with point-in-polygon checks
    return {
        "location": f"Location ({lat:.4f}, {lng:.4f})",
        "coordinates": [lng, lat],
        "risk_level": "moderate",
        "risk_percentage": 45,
        "water_level": "1.5m above normal",
        "nearest_station": "Station Alpha",
        "last_updated": "2024-11-24T10:30:00Z"
    }


# ============================================================================
# WEATHER-ENHANCED FLOOD RISK
# ============================================================================

async def get_weather_enhanced_tiles(
    min_lat: float = None, 
    max_lat: float = None, 
    min_lon: float = None, 
    max_lon: float = None,
    seed: int = None,
    demo_mode: bool = False
) -> Dict[str, Any]:
    """
    Get flood tiles enhanced with real-time weather data.
    
    For a zoomed-in view, fetches weather for the center point and uses it
    to influence tile status. Larger regions use base geographical factors
    for performance.
    
    Args:
        min_lat, max_lat, min_lon, max_lon: Bounding box of visible region
        seed: Optional seed for deterministic base generation
        demo_mode: If True, simulates heavy rain conditions
    
    Returns:
        tiles: List of flood tiles
        weather_influence: Weather data used (if available)
    """
    from flood_prediction import get_weather_based_flood_risk
    
    # Use current date as seed if not provided
    if seed is None:
        seed = int(datetime.now().strftime("%Y%m%d"))
    
    # Get base tiles (geographical factors)
    base_tiles = get_flood_risk_areas(min_lat, max_lat, min_lon, max_lon, seed)
    
    # Calculate region size to determine if we should fetch weather
    lat_span = (max_lat or PAKISTAN_BOUNDS["max_lat"]) - (min_lat or PAKISTAN_BOUNDS["min_lat"])
    lon_span = (max_lon or PAKISTAN_BOUNDS["max_lon"]) - (min_lon or PAKISTAN_BOUNDS["min_lon"])
    region_area = lat_span * lon_span
    
    weather_influence = None
    
    # Only fetch weather for smaller regions (zoomed in views)
    # to avoid excessive API calls
    if region_area < 1.0 or demo_mode:  # Less than ~100km x 100km, or demo mode
        # Calculate center of visible region
        center_lat = ((min_lat or PAKISTAN_BOUNDS["min_lat"]) + 
                      (max_lat or PAKISTAN_BOUNDS["max_lat"])) / 2
        center_lon = ((min_lon or PAKISTAN_BOUNDS["min_lon"]) + 
                      (max_lon or PAKISTAN_BOUNDS["max_lon"])) / 2
        
        try:
            # Get weather-based prediction for center point
            weather_prediction = await get_weather_based_flood_risk(center_lat, center_lon, demo_mode=demo_mode)
            
            if weather_prediction.get("weather_available"):
                weather_influence = {
                    "location": {"lat": center_lat, "lon": center_lon},
                    "risk_level": weather_prediction.get("risk_level"),
                    "severity": weather_prediction.get("severity"),
                    "total_score": weather_prediction.get("total_score"),
                    "current_conditions": weather_prediction.get("current_conditions"),
                    "timestamp": weather_prediction.get("timestamp")
                }
                
                # Adjust tile statuses based on weather
                weather_severity = weather_prediction.get("severity", "low_risk")
                weather_score = weather_prediction.get("total_score", 0)
                
                # Weather influence matrix:
                # - high_risk weather: upgrades At Risk -> Flooded, Safe -> At Risk
                # - medium_risk weather: upgrades some Safe -> At Risk
                # - low_risk weather (no rain): downgrades Flooded -> At Risk, At Risk -> Safe
                
                for tile in base_tiles:
                    base_status = tile["status"]
                    
                    if weather_severity == "high_risk" and weather_score > 60:
                        # Heavy rain: increase risk levels
                        if base_status == "At Risk":
                            tile["status"] = "Flooded"
                            tile["weather_upgraded"] = True
                        elif base_status == "Safe":
                            tile["status"] = "At Risk"
                            tile["weather_upgraded"] = True
                    
                    elif weather_severity == "medium_risk" and weather_score > 40:
                        # Moderate rain: slight increase
                        if base_status == "Safe":
                            coords = tile["coordinates"][0]
                            loc_hash = int(hashlib.md5(
                                f"{coords[0]:.6f}{coords[1]:.6f}".encode()
                            ).hexdigest(), 16) % 100
                            if loc_hash < 30:  # 30% of Safe tiles
                                tile["status"] = "At Risk"
                                tile["weather_upgraded"] = True
                    
                    elif weather_severity == "low_risk" and weather_score < 20:
                        # Clear/dry weather: significantly reduce risk since no actual rain
                        if base_status == "Flooded":
                            # No rain = no flooding, downgrade to At Risk (still flood-prone area)
                            tile["status"] = "At Risk"
                            tile["weather_downgraded"] = True
                        elif base_status == "At Risk":
                            # Most At Risk tiles become Safe when no rain
                            coords = tile["coordinates"][0]
                            loc_hash = int(hashlib.md5(
                                f"{coords[0]:.6f}{coords[1]:.6f}".encode()
                            ).hexdigest(), 16) % 100
                            if loc_hash < 70:  # 70% of At Risk tiles become Safe
                                tile["status"] = "Safe"
                                tile["weather_downgraded"] = True
                    
                    # Add weather metadata
                    tile["weather_influenced"] = True
        
        except Exception as e:
            print(f"Weather fetch failed, using base tiles: {e}")
    
    return {
        "tiles": base_tiles,
        "weather_influence": weather_influence,
        "data_source": "weather_enhanced" if weather_influence else "geographical"
    }


def get_historical_data(location_id: str, days: int = 7) -> List[Dict]:
    """
    Get historical flood risk data for a location
    
    Args:
        location_id: ID of the location
        days: Number of days of historical data
    
    Returns:
        List of historical risk assessments
    """
    # Dummy implementation
    return [
        {
            "date": "2024-11-24",
            "risk_level": "moderate",
            "water_level": "2.1m",
            "risk_percentage": 55
        },
        {
            "date": "2024-11-23",
            "risk_level": "low",
            "water_level": "1.2m",
            "risk_percentage": 30
        },
        {
            "date": "2024-11-22",
            "risk_level": "low",
            "water_level": "0.9m",
            "risk_percentage": 20
        }
    ]
