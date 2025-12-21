import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import MapSearch from './MapSearch';
import AlertNotifications from './AlertNotifications';
import MLPredictionPanel from './MLPredictionPanel';
import DataTrendsPanel from './DataTrendsPanel';  // Add this import
import { useFloodData } from '../hooks/useFloodData';
import './MapBackground.scss';

// Add your Mapbox access token here
// You can get one from https://www.mapbox.com/
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || 'your-mapbox-token-here';

// Weather severity colors for the badge
const SEVERITY_COLORS = {
  'low_risk': '#22c55e',
  'medium_risk': '#f59e0b',
  'high_risk': '#ef4444'
};

const MapBackground: React.FC = () => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const searchMarkerRef = useRef<mapboxgl.Marker | null>(null);
  const [mapInstance, setMapInstance] = useState<mapboxgl.Map | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [mapCenter, setMapCenter] = useState<[number, number]>([72.643374, 34.069335]);
  const [demoMode, setDemoMode] = useState(false);
  const [showMLPanel, setShowMLPanel] = useState(false);
  const [showTrendsPanel, setShowTrendsPanel] = useState(false);  // Add this state

  // Use flood data hook to fetch and display tiles
  const { weatherInfluence, dataSource, alertSummary, error: dataError } = useFloodData(mapInstance, demoMode);

  const handleLocationSelect = (coordinates: [number, number], placeName: string) => {
    if (!mapRef.current) return;

    // Fly to the selected location
    mapRef.current.flyTo({
      center: coordinates,
      zoom: 14,
      duration: 2000,
    });

    // Remove existing search marker if any
    if (searchMarkerRef.current) {
      searchMarkerRef.current.remove();
    }

    // Create a custom search marker element
    const searchMarkerEl = document.createElement('div');
    searchMarkerEl.className = 'search-marker';
    searchMarkerEl.innerHTML = '📍';
    searchMarkerEl.style.fontSize = '32px';
    searchMarkerEl.style.cursor = 'pointer';

    // Add new marker at searched location
    searchMarkerRef.current = new mapboxgl.Marker({
      element: searchMarkerEl,
      anchor: 'bottom',
    })
      .setLngLat(coordinates)
      .setPopup(
        new mapboxgl.Popup({ offset: 25 })
          .setHTML(`<div style="padding: 0.5rem; color: #1a202c;"><strong>${placeName}</strong></div>`)
      )
      .addTo(mapRef.current);

    // Show popup automatically
    searchMarkerRef.current.togglePopup();
  };

  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Set Mapbox access token
    mapboxgl.accessToken = MAPBOX_TOKEN;

    console.log('Initializing map with token:', MAPBOX_TOKEN.substring(0, 10) + '...');

    // Convert coordinates to decimal degrees
    // 34° 4' 9.606'' N = 34.069335°
    // 72° 38' 36.1464'' E = 72.643374°
    const lat = 34.069335;
    const lng = 72.643374;

    try {
      // Initialize map
      mapRef.current = new mapboxgl.Map({
        container: mapContainerRef.current,
        style: 'mapbox://styles/mapbox/dark-v11',
        center: [lng, lat], // [longitude, latitude]
        zoom: 13,
        pitch: 0,
        bearing: 0,
      });

      // Add event listener for map load
      mapRef.current.on('load', () => {
        console.log('Map loaded successfully');
        setIsLoading(false);
        setMapError(null);
        setMapInstance(mapRef.current);
      });

      // Update map center when user moves the map
      mapRef.current.on('moveend', () => {
        if (mapRef.current) {
          const center = mapRef.current.getCenter();
          setMapCenter([center.lng, center.lat]);
        }
      });

      // Add error handler
      mapRef.current.on('error', (e) => {
        console.error('Map error:', e);
        setIsLoading(false);
        setMapError('Failed to fetch map. Are you connected to the internet?');
      });

      // Add navigation controls
      const nav = new mapboxgl.NavigationControl();
      mapRef.current.addControl(nav, 'top-right');

      // Add fullscreen control
      const fullscreen = new mapboxgl.FullscreenControl();
      mapRef.current.addControl(fullscreen, 'top-right');

      // Create a custom home marker element
      const homeMarker = document.createElement('div');
      homeMarker.className = 'home-marker';
      homeMarker.innerHTML = '🏠';
      homeMarker.style.fontSize = '32px';
      homeMarker.style.cursor = 'pointer';

      // Add a home marker at the center
      new mapboxgl.Marker({
        element: homeMarker,
        anchor: 'bottom',
      })
        .setLngLat([lng, lat])
        .addTo(mapRef.current);
    } catch (error) {
      console.error('Error initializing map:', error);
      setIsLoading(false);
      setMapError('Failed to initialize map. Please check your connection.');
    }

    // Cleanup on unmount
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
      }
    };
  }, []);

  return (
    <div className="map-wrapper">
      <div ref={mapContainerRef} className="map-container" />
      
      {/* Search Component */}
      {!mapError && !isLoading && (
        <MapSearch 
          onLocationSelect={handleLocationSelect}
          mapboxToken={MAPBOX_TOKEN}
          currentCenter={mapCenter}
        />
      )}

      {/* Search Component */}
      {!mapError && !isLoading && (
        <MapSearch 
          onLocationSelect={handleLocationSelect}
          mapboxToken={MAPBOX_TOKEN}
          currentCenter={mapCenter}
        />
      )}

      {/* Live Alert Notifications */}
      {!mapError && !isLoading && alertSummary && alertSummary.alerts && (
        <AlertNotifications 
          alerts={alertSummary.alerts}
          demoMode={demoMode}
        />
      )}

      {/* Weather Data Indicator */}
      {!mapError && !isLoading && weatherInfluence && (
        <div className="weather-indicator">
          <div 
            className="weather-badge"
            style={{ 
              borderLeftColor: SEVERITY_COLORS[weatherInfluence.severity as keyof typeof SEVERITY_COLORS] 
            }}
          >
            <div className="weather-badge-header">
              <span className="weather-icon">🌧️</span>
              <span className="weather-title">Live Weather</span>
            </div>
            <div className="weather-details">
              {weatherInfluence.current_conditions && (
                <>
                  <p>{weatherInfluence.current_conditions.condition}</p>
                  <p className="weather-stat">
                    💧 Rain: {weatherInfluence.current_conditions.rain_1h_mm.toFixed(1)}mm/h
                  </p>
                  <p className="weather-stat">
                    💦 Humidity: {weatherInfluence.current_conditions.humidity}%
                  </p>
                </>
              )}
              <p className="risk-level" style={{ color: SEVERITY_COLORS[weatherInfluence.severity as keyof typeof SEVERITY_COLORS] }}>
                Risk: {weatherInfluence.risk_level}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Data Source Indicator */}
      {!mapError && !isLoading && dataSource && (
        <div className="data-source-indicator">
          {dataSource === 'weather_enhanced' ? '📡 Weather-Enhanced Data' : '🗺️ Geographical Data'}
        </div>
      )}

      {/* Demo Mode Toggle */}
      {!mapError && !isLoading && (
        <div className="demo-mode-toggle">
          <button 
            className={`demo-button ${demoMode ? 'active' : ''}`}
            onClick={() => setDemoMode(!demoMode)}
            title={demoMode ? 'Switch to real weather data' : 'Enable flood simulation demo'}
          >
            {demoMode ? '🌧️ Demo Mode ON' : '☀️ Real Weather'}
          </button>
          {demoMode && (
            <span className="demo-label">Simulating heavy rainfall</span>
          )}
        </div>
      )}

      {/* ML Prediction Toggle Button */}
      {!mapError && !isLoading && (
        <button 
          className={`ml-toggle-button ${showMLPanel ? 'active' : ''}`}
          onClick={() => setShowMLPanel(!showMLPanel)}
          title="Toggle ML Flood Prediction"
        >
          🤖 {showMLPanel ? 'Hide' : 'ML'} Prediction
        </button>
      )}

      {/* Data Trends Toggle Button - Add this */}
      {!mapError && !isLoading && (
        <button 
          className={`trends-toggle-button ${showTrendsPanel ? 'active' : ''}`}
          onClick={() => setShowTrendsPanel(!showTrendsPanel)}
          title="Toggle Data Trends"
        >
          📊 {showTrendsPanel ? 'Hide' : 'Data'} Trends
        </button>
      )}

      {/* ML Prediction Panel */}
      {!mapError && !isLoading && showMLPanel && (
        <div className="ml-panel-container">
          <MLPredictionPanel
            lat={mapCenter[1]}
            lon={mapCenter[0]}
            demoMode={demoMode}
            onClose={() => setShowMLPanel(false)}
          />
        </div>
      )}

      {/* Data Trends Panel - Add this */}
      {!mapError && !isLoading && showTrendsPanel && (
        <div className="trends-panel-container">
          <DataTrendsPanel
            lat={mapCenter[1]}
            lon={mapCenter[0]}
            onClose={() => setShowTrendsPanel(false)}
          />
        </div>
      )}

      {/* Flood Data Error */}
      {!mapError && !isLoading && dataError && (
        <div className="data-error-banner">
          ⚠️ Unable to load flood data: {dataError}
        </div>
      )}
      
      {/* Loading Indicator */}
      {isLoading && !mapError && (
        <div className="map-overlay">
          <div className="loading-message">
            <div className="spinner"></div>
            <p>Loading map...</p>
          </div>
        </div>
      )}

      {/* Error Message */}
      {mapError && (
        <div className="map-overlay error">
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            <h3>Map Error</h3>
            <p>{mapError}</p>
            <button 
              className="retry-button"
              onClick={() => window.location.reload()}
            >
              Retry
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default MapBackground;
