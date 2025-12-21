import { useEffect, useState, useRef } from 'react';
import mapboxgl from 'mapbox-gl';

interface FloodTile {
  id: string;
  status: 'Safe' | 'At Risk' | 'Flooded';
  coordinates: [number, number][];
  last_updated: string;
  weather_influenced?: boolean;
  weather_upgraded?: boolean;
  weather_downgraded?: boolean;
}

// Weather influence data from the API
interface WeatherInfluence {
  location: { lat: number; lon: number };
  risk_level: 'Safe' | 'At Risk' | 'Flooded';
  severity: 'low_risk' | 'medium_risk' | 'high_risk';
  total_score: number;
  current_conditions?: {
    condition: string;
    description: string;
    temperature: number;
    humidity: number;
    rain_1h_mm: number;
    rain_3h_mm: number;
  };
  timestamp: string;
}

// Weather data included in alerts when using real-time weather API
interface AlertWeatherData {
  rain_1h_mm: number;
  rain_3h_mm: number;
  humidity: number;
  temperature: number;
  condition: string;
  forecast_rain_24h_mm: number;
}

interface Alert {
  id: string;
  severity: 'high_risk' | 'medium_risk' | 'low_risk';
  color: 'red' | 'yellow' | 'green';
  location: string;
  region: string;
  coordinates: [number, number];
  message: string;
  issued_at: string;
  issued_minutes_ago: number;
  // Weather data (only present when using real-time weather API)
  weather_data?: AlertWeatherData;
  risk_score?: number;
}

interface AlertSummary {
  alerts: Alert[];
  summary: {
    high_risk: number;
    medium_risk: number;
    low_risk: number;
    total: number;
  };
  last_updated: string;
  data_source?: 'openweathermap_realtime' | 'demo_simulation' | 'simulated';
}

const API_BASE_URL = '/api';

// Color mapping for flood status
const STATUS_COLORS = {
  'Safe': '#22c55e',      // Green
  'At Risk': '#f59e0b',   // Yellow
  'Flooded': '#ef4444'    // Red
};

const STATUS_OPACITY = 0.1; // Balanced opacity - visible but not blocking map

export const useFloodData = (map: mapboxgl.Map | null, demoMode: boolean = false) => {
  const [floodTiles, setFloodTiles] = useState<FloodTile[]>([]);
  const [weatherInfluence, setWeatherInfluence] = useState<WeatherInfluence | null>(null);
  const [dataSource, setDataSource] = useState<'weather_enhanced' | 'geographical' | null>(null);
  const [alertSummary, setAlertSummary] = useState<AlertSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollIntervalRef = useRef<number | undefined>(undefined);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Fetch flood risk tiles for visible region
  useEffect(() => {
    if (!map) return;
    
    const fetchFloodTiles = async () => {
      // Cancel any ongoing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      
      try {
        // Get visible map bounds
        const bounds = map.getBounds();
        if (!bounds) return;
        
        const params = new URLSearchParams({
          min_lat: bounds.getSouth().toFixed(4),
          max_lat: bounds.getNorth().toFixed(4),
          min_lon: bounds.getWest().toFixed(4),
          max_lon: bounds.getEast().toFixed(4)
        });
        
        // Add demo mode parameter
        if (demoMode) {
          params.append('demo', 'true');
        }
        
        console.log(`Fetching flood tiles (demo=${demoMode})`);
        
        // Create new abort controller for this request
        abortControllerRef.current = new AbortController();
        
        const response = await fetch(`${API_BASE_URL}/flood-risk?${params}`, {
          signal: abortControllerRef.current.signal
        });
        
        if (!response.ok) throw new Error('Failed to fetch flood risk data');
        const data = await response.json();
        console.log(`Received ${data.tiles?.length || 0} flood tiles (demo=${demoMode})`);
        
        const tiles = data.tiles || [];
        setFloodTiles(tiles);
        
        // Handle weather influence data
        if (data.weather_influence) {
          setWeatherInfluence(data.weather_influence);
          console.log('Weather influence applied:', data.weather_influence.severity);
        } else {
          setWeatherInfluence(null);
        }
        
        // Track data source
        if (data.data_source) {
          setDataSource(data.data_source);
        }
        
        setError(null);
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          console.log('Request cancelled');
          return;
        }
        console.error('Error fetching flood tiles:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    };

    // Initial fetch
    fetchFloodTiles();
    
    // Refetch when map moves or zooms
    const handleMapMove = () => {
      fetchFloodTiles();
    };
    
    map.on('moveend', handleMapMove);
    map.on('zoomend', handleMapMove);
    
    // Poll for updates every 60 seconds
    pollIntervalRef.current = window.setInterval(fetchFloodTiles, 60000);
    
    // Cleanup
    return () => {
      map.off('moveend', handleMapMove);
      map.off('zoomend', handleMapMove);
      if (pollIntervalRef.current) window.clearInterval(pollIntervalRef.current);
      if (abortControllerRef.current) abortControllerRef.current.abort();
    };
  }, [map, demoMode]);

  // Fetch alert summary (using real-time weather data)
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        // Build URL with demo parameter if needed
        const params = new URLSearchParams();
        if (demoMode) {
          params.append('demo', 'true');
        }
        const url = `${API_BASE_URL}/alerts${params.toString() ? '?' + params.toString() : ''}`;
        
        // Fetch alerts with real-time weather data (use_weather=true is default)
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to fetch alerts');
        const data = await response.json();
        setAlertSummary(data);
        
        // Log data source for debugging
        console.log(`Alerts loaded from: ${data.data_source || 'unknown'} (demo=${demoMode})`);
      } catch (err) {
        console.error('Error fetching alerts:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchAlerts();
    // Poll for updates every 60 seconds (weather data updates every 10 min on API side)
    const interval = window.setInterval(fetchAlerts, 60000);
    return () => window.clearInterval(interval);
  }, [demoMode]);

  // Add tiles to map as layers
  useEffect(() => {
    if (!map || floodTiles.length === 0) {
      console.log('Skipping tile rendering:', { hasMap: !!map, tileCount: floodTiles.length });
      return;
    }

    console.log(`Rendering ${floodTiles.length} tiles on map`);

    const addTilesToMap = () => {
      // Group tiles by status for efficiency
      const tilesByStatus: Record<string, FloodTile[]> = {
        'Safe': [],
        'At Risk': [],
        'Flooded': []
      };

      floodTiles.forEach(tile => {
        tilesByStatus[tile.status].push(tile);
      });

      console.log('Tiles by status:', {
        safe: tilesByStatus['Safe'].length,
        atRisk: tilesByStatus['At Risk'].length,
        flooded: tilesByStatus['Flooded'].length
      });

      // Add a layer for each status type
      Object.entries(tilesByStatus).forEach(([status, tiles]) => {
        const sourceId = `flood-tiles-${status.toLowerCase().replace(' ', '-')}`;
        const layerId = `${sourceId}-layer`;

        // Remove existing layer and source if present
        if (map.getLayer(layerId)) {
          map.removeLayer(layerId);
        }
        if (map.getSource(sourceId)) {
          map.removeSource(sourceId);
        }

        if (tiles.length === 0) return;

        console.log(`Adding ${tiles.length} ${status} tiles as layer ${layerId}`);

        // Create GeoJSON for this status type
        const geojson: GeoJSON.FeatureCollection = {
          type: 'FeatureCollection',
          features: tiles.map(tile => ({
            type: 'Feature',
            geometry: {
              type: 'Polygon',
              coordinates: [tile.coordinates]
            },
            properties: {
              id: tile.id,
              status: tile.status,
              last_updated: tile.last_updated,
              weather_influenced: tile.weather_influenced || false,
              weather_upgraded: tile.weather_upgraded || false,
              weather_downgraded: tile.weather_downgraded || false
            }
          }))
        };

        // Add source
        map.addSource(sourceId, {
          type: 'geojson',
          data: geojson
        });

        // Add fill layer
        map.addLayer({
          id: layerId,
          type: 'fill',
          source: sourceId,
          paint: {
            'fill-color': STATUS_COLORS[status as keyof typeof STATUS_COLORS],
            'fill-opacity': STATUS_OPACITY
          }
        });

        // Add hover effect
        map.on('mouseenter', layerId, () => {
          map.getCanvas().style.cursor = 'pointer';
        });

        map.on('mouseleave', layerId, () => {
          map.getCanvas().style.cursor = '';
        });

        // Add click handler to show popup
        map.on('click', layerId, (e) => {
          if (!e.features || e.features.length === 0) return;
          
          const feature = e.features[0];
          const properties = feature.properties;
          
          // Build weather badge if applicable
          let weatherBadge = '';
          if (properties?.weather_influenced) {
            if (properties?.weather_upgraded) {
              weatherBadge = '<span style="background: #ef4444; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-left: 4px;">⬆ Weather Risk</span>';
            } else if (properties?.weather_downgraded) {
              weatherBadge = '<span style="background: #22c55e; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-left: 4px;">⬇ Clear Weather</span>';
            } else {
              weatherBadge = '<span style="background: #3b82f6; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-left: 4px;">🌧 Weather Data</span>';
            }
          }

          new mapboxgl.Popup()
            .setLngLat(e.lngLat)
            .setHTML(`
              <div style="padding: 8px; min-width: 150px;">
                <h3 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: ${STATUS_COLORS[properties?.status as keyof typeof STATUS_COLORS]};">
                  Status: ${properties?.status || 'Unknown'} ${weatherBadge}
                </h3>
                <p style="margin: 4px 0; font-size: 12px; color: #666;">
                  Tile ID: ${properties?.id || 'N/A'}
                </p>
                <p style="margin: 4px 0; font-size: 11px; color: #999;">
                  Updated: ${properties?.last_updated ? new Date(properties.last_updated).toLocaleString() : 'N/A'}
                </p>
                ${properties?.weather_influenced ? '<p style="margin: 4px 0; font-size: 10px; color: #3b82f6;">📡 Real-time weather data applied</p>' : ''}
              </div>
            `)
            .addTo(map);
        });
      });
    };

    if (map.isStyleLoaded()) {
      addTilesToMap();
    } else {
      map.once('load', addTilesToMap);
    }

    // Cleanup
    return () => {
      ['Safe', 'At Risk', 'Flooded'].forEach(status => {
        const sourceId = `flood-tiles-${status.toLowerCase().replace(' ', '-')}`;
        const layerId = `${sourceId}-layer`;
        
        if (map.getLayer(layerId)) {
          map.removeLayer(layerId);
        }
        if (map.getSource(sourceId)) {
          map.removeSource(sourceId);
        }
      });
    };
  }, [map, floodTiles]);

  return {
    floodTiles,
    weatherInfluence,
    dataSource,
    alertSummary,
    loading,
    error
  };
};
