import { useEffect, useState, useRef } from 'react';
import mapboxgl from 'mapbox-gl';

interface FloodTile {
  id: string;
  status: 'Safe' | 'At Risk' | 'Flooded';
  coordinates: [number, number][];
  last_updated: string;
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
}

const API_BASE_URL = '/api';

// Color mapping for flood status
const STATUS_COLORS = {
  'Safe': '#22c55e',      // Green
  'At Risk': '#f59e0b',   // Yellow
  'Flooded': '#ef4444'    // Red
};

const STATUS_OPACITY = 0.1; // Balanced opacity - visible but not blocking map

export const useFloodData = (map: mapboxgl.Map | null) => {
  const [floodTiles, setFloodTiles] = useState<FloodTile[]>([]);
  const [alertSummary, setAlertSummary] = useState<AlertSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollIntervalRef = useRef<number | undefined>(undefined);
  const abortControllerRef = useRef<AbortController | null>(null);
  const tilesCacheRef = useRef<Map<string, FloodTile[]>>(new Map());

  // Fetch flood risk tiles for visible region
  useEffect(() => {
    const fetchFloodTiles = async () => {
      if (!map) return;
      
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
        
        // Create cache key from bounds
        const cacheKey = params.toString();
        
        // Check cache first
        if (tilesCacheRef.current.has(cacheKey)) {
          console.log('Using cached tiles');
          setFloodTiles(tilesCacheRef.current.get(cacheKey)!);
          return;
        }
        
        console.log('Fetching flood tiles for bounds:', {
          south: bounds.getSouth(),
          north: bounds.getNorth(),
          west: bounds.getWest(),
          east: bounds.getEast()
        });
        
        // Create new abort controller for this request
        abortControllerRef.current = new AbortController();
        
        const response = await fetch(`${API_BASE_URL}/flood-risk?${params}`, {
          signal: abortControllerRef.current.signal
        });
        
        if (!response.ok) throw new Error('Failed to fetch flood risk data');
        const data = await response.json();
        console.log(`Received ${data.tiles?.length || 0} flood tiles`);
        
        const tiles = data.tiles || [];
        setFloodTiles(tiles);
        
        // Cache the tiles
        tilesCacheRef.current.set(cacheKey, tiles);
        
        // Limit cache size to 10 entries (keep most recent)
        if (tilesCacheRef.current.size > 10) {
          const firstKey = tilesCacheRef.current.keys().next().value;
          if (firstKey) {
            tilesCacheRef.current.delete(firstKey);
          }
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

    if (map) {
      const initFetch = () => {
        fetchFloodTiles();
        
        // Refetch IMMEDIATELY when map moves or zooms (no debounce)
        const handleMapMove = () => {
          fetchFloodTiles();
        };
        
        // Use 'move' instead of 'moveend' for instant response
        map.on('moveend', handleMapMove);
        map.on('zoomend', handleMapMove);
        
        // Poll for updates every 60 seconds (increased from 30)
        pollIntervalRef.current = window.setInterval(fetchFloodTiles, 60000);
        
        return () => {
          map.off('moveend', handleMapMove);
          map.off('zoomend', handleMapMove);
          if (pollIntervalRef.current) window.clearInterval(pollIntervalRef.current);
          if (abortControllerRef.current) abortControllerRef.current.abort();
        };
      };

      if (map.isStyleLoaded()) {
        return initFetch();
      } else {
        map.once('load', initFetch);
      }
    }
  }, [map]);

  // Fetch alert summary
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/alerts`);
        if (!response.ok) throw new Error('Failed to fetch alerts');
        const data = await response.json();
        setAlertSummary(data);
      } catch (err) {
        console.error('Error fetching alerts:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchAlerts();
    // Poll for updates every 30 seconds
    const interval = window.setInterval(fetchAlerts, 30000);
    return () => window.clearInterval(interval);
  }, []);

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
              last_updated: tile.last_updated
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

          new mapboxgl.Popup()
            .setLngLat(e.lngLat)
            .setHTML(`
              <div style="padding: 8px; min-width: 150px;">
                <h3 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: ${STATUS_COLORS[properties?.status as keyof typeof STATUS_COLORS]};">
                  Status: ${properties?.status || 'Unknown'}
                </h3>
                <p style="margin: 4px 0; font-size: 12px; color: #666;">
                  Tile ID: ${properties?.id || 'N/A'}
                </p>
                <p style="margin: 4px 0; font-size: 11px; color: #999;">
                  Updated: ${properties?.last_updated ? new Date(properties.last_updated).toLocaleString() : 'N/A'}
                </p>
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
    alertSummary,
    loading,
    error
  };
};
