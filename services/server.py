"""
FastAPI server for FLEWS (Flood Early Warning System)
Provides REST API endpoints for flood risk data and alerts
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import service modules
from alert import get_alert_summary
from flood_risk import get_flood_risk_areas

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
            "health": "/api/health"
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
def get_alerts(seed: int = None):
    """
    Get live alert notifications
    
    RESTful: Same seed parameter returns same alerts (stateless, deterministic)
    Omit seed to use current 5-minute window (alerts change every 5 minutes)
    
    Query parameters:
        seed: Optional seed for deterministic generation
    """
    try:
        return get_alert_summary(seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FLOOD RISK ENDPOINTS
# ============================================================================

@app.get("/api/flood-risk")
def get_flood_risk(min_lat: float = None, max_lat: float = None, 
                   min_lon: float = None, max_lon: float = None,
                   seed: int = None):
    """
    Get flood status grid for visible region
    
    RESTful: Same parameters return same tiles (stateless, deterministic)
    Omit seed to use current date (tiles change daily)
    
    Query parameters:
        min_lat, max_lat, min_lon, max_lon: Bounding box of visible map region
        seed: Optional seed for deterministic generation (defaults to current date)
    
    Returns tiles with status: Safe (green), At Risk (yellow), Flooded (red)
    """
    try:
        tiles = get_flood_risk_areas(min_lat, max_lat, min_lon, max_lon, seed)
        return {"tiles": tiles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SERVER STARTUP
# ============================================================================

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


# ============================================================================
# SERVER STARTUP
# ============================================================================

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
