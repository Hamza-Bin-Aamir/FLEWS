# FLEWS Backend Services

Python FastAPI server providing REST endpoints for the FLEWS frontend.

## Setup

### 1. Install Dependencies

```bash
cd services
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python server.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### Health Check
- `GET /api/health` - Check server status

### Alerts
- `GET /api/alerts` - Get all active flood alerts
- `POST /api/alerts` - Create a new alert
- `DELETE /api/alerts/{alert_id}` - Dismiss an alert

### Flood Risk
- `GET /api/flood-risk` - Get all flood risk zones with map overlays
- `GET /api/flood-risk/location?lat={lat}&lng={lng}` - Get risk for specific location
- `GET /api/flood-risk/{location_id}/history?days={days}` - Get historical data

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Color-Coded System

### Alerts
- **Red (High/Danger)**: Immediate evacuation required
- **Yellow (Moderate/Caution)**: Monitor closely, prepare for evacuation
- **Green (Low/Safe)**: Normal conditions

### Risk Levels
- **high**: Red overlay on map
- **moderate**: Yellow overlay on map
- **low**: Green overlay on map

## Data Format

### Alert Object
```json
{
  "id": "alert_001",
  "location": "Topi, Swabi",
  "coordinates": [72.6234, 34.0705],
  "severity": "high",
  "risk_level": "Danger",
  "message": "Severe flooding expected...",
  "timestamp": "2024-11-24T10:30:00Z",
  "affected_area": "5km radius"
}
```

### Flood Risk Zone
```json
{
  "id": "risk_zone_001",
  "name": "Topi Basin",
  "coordinates": [72.6234, 34.0705],
  "risk_level": "high",
  "risk_percentage": 85,
  "water_level": "4.2m above normal",
  "prediction": "Rising rapidly",
  "affected_population": 15000,
  "polygon": [[lng, lat], ...]
}
```

## CORS Configuration

The server is configured to accept requests from:
- http://localhost:5173 (Vite dev server)
- http://localhost:3000 (Alternative dev port)

To add more origins, edit the `allow_origins` list in `server.py`.
