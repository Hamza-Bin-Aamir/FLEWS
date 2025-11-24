"""
Flood risk assessment module for FLEWS
Generates a grid of flood status tiles covering Pakistan
"""

from typing import List, Dict, Any
from datetime import datetime
import random
import math
import hashlib

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

def calculate_flood_risk(lat: float, lon: float, seed: int = 0) -> str:
    """
    Calculate flood risk based on distance from flood epicenters
    Uses seed for deterministic randomness (same location = same result)
    
    Returns:
        'Flooded' for areas within epicenter radius
        'At Risk' for areas in buffer zone
        'Safe' for areas far from flood zones
    """
    min_distance = float('inf')
    max_intensity = 0.0
    
    for epi_lon, epi_lat, intensity in FLOOD_EPICENTERS:
        # Calculate distance (approximate, using euclidean distance)
        distance = math.sqrt((lat - epi_lat)**2 + (lon - epi_lon)**2)
        
        if distance < min_distance:
            min_distance = distance
            max_intensity = intensity
    
    # Adjust thresholds based on intensity
    flooded_radius = 0.15 * max_intensity  # Core flood zone
    at_risk_radius = 0.4 * max_intensity   # Buffer zone
    
    if min_distance < flooded_radius:
        return "Flooded"
    elif min_distance < at_risk_radius:
        return "At Risk"
    else:
        # Use deterministic pseudo-random for isolated incidents
        # Seed based on location coordinates for consistency
        location_hash = int(hashlib.md5(f"{lat:.6f}{lon:.6f}{seed}".encode()).hexdigest(), 16) % 1000
        rand_value = location_hash / 1000.0
        
        if rand_value < 0.02:
            return "At Risk"
        elif rand_value < 0.03:
            return "Flooded"
        return "Safe"

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
            status = calculate_flood_risk(lat, lon, seed)
            
            # Create tile polygon
            tile = {
                "id": f"tile-{tile_id_base + tile_count}",
                "status": status,
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
