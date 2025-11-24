import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import './MapBackground.scss';

// Add your Mapbox access token here
// You can get one from https://www.mapbox.com/
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || 'your-mapbox-token-here';

const MapBackground: React.FC = () => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

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
