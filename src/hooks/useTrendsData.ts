import { useState, useCallback } from 'react';

const API_BASE_URL = '/api';

// Types for trend data
export interface RainfallDataPoint {
  timestamp: string;
  rainfall_1h: number;
  rainfall_3h: number;
  humidity: number;
  risk_score: number;
  flood_status: 'Safe' | 'At Risk' | 'Flooded';
}

export interface RainfallForecastPoint {
  timestamp: string;
  rainfall_3h: number;
  probability: number;
  humidity: number;
  condition: string;
  temperature: number;
}

export interface RainfallTrends {
  success: boolean;
  location: { lat: number; lon: number };
  historical: RainfallDataPoint[];
  forecast: RainfallForecastPoint[];
  summary: {
    avg_rainfall: number;
    max_rainfall: number;
    min_rainfall: number;
    total_rainfall: number;
    flood_events: number;
    at_risk_events: number;
  };
  timestamp: string;
}

export interface RiverLevelDataPoint {
  timestamp: string;
  level: number;
  flow_rate: number;
  status: 'Normal' | 'Warning' | 'Danger' | 'Extreme';
}

export interface RiverLevelForecastPoint {
  timestamp: string;
  level: number;
  level_min: number;
  level_max: number;
  status: 'Normal' | 'Warning' | 'Danger' | 'Extreme';
  confidence: number;
}

export interface RiverLevelTrends {
  success: boolean;
  location: { lat: number; lon: number };
  river_name: string;
  historical: RiverLevelDataPoint[];
  forecast: RiverLevelForecastPoint[];
  thresholds: {
    normal: number;
    warning: number;
    danger: number;
    extreme: number;
  };
  unit: string;
  summary: {
    current_level: number;
    current_status: string;
    avg_level: number;
    max_level: number;
    min_level: number;
    trend: 'rising' | 'falling' | 'stable';
  };
  timestamp: string;
}

export interface CombinedTrends {
  success: boolean;
  location: { lat: number; lon: number };
  rainfall: RainfallTrends;
  river_level: RiverLevelTrends;
  correlation: {
    coefficient: number;
    description: string;
  };
  timestamp: string;
}

// Hook for rainfall trends
export const useRainfallTrends = () => {
  const [data, setData] = useState<RainfallTrends | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchRainfallTrends = useCallback(async (lat: number, lon: number, days: number = 7) => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        lat: lat.toString(),
        lon: lon.toString(),
        days: days.toString()
      });

      const response = await fetch(`${API_BASE_URL}/trends/rainfall?${params}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setData(result);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch rainfall trends';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { data, loading, error, fetchRainfallTrends };
};

// Hook for river level trends
export const useRiverLevelTrends = () => {
  const [data, setData] = useState<RiverLevelTrends | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchRiverLevelTrends = useCallback(async (lat: number, lon: number, days: number = 7) => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        lat: lat.toString(),
        lon: lon.toString(),
        days: days.toString()
      });

      const response = await fetch(`${API_BASE_URL}/trends/river-level?${params}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setData(result);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch river level trends';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { data, loading, error, fetchRiverLevelTrends };
};

// Hook for combined trends
export const useCombinedTrends = () => {
  const [data, setData] = useState<CombinedTrends | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchCombinedTrends = useCallback(async (lat: number, lon: number, days: number = 7) => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        lat: lat.toString(),
        lon: lon.toString(),
        days: days.toString()
      });

      const response = await fetch(`${API_BASE_URL}/trends/combined?${params}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setData(result);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch combined trends';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { data, loading, error, fetchCombinedTrends };
};