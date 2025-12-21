import { useState, useEffect, useCallback } from 'react';

// API base URL - same as other hooks
const API_BASE_URL = '/api';

// Types for weather data
export interface WeatherLocation {
  lat: number;
  lon: number;
  name: string;
  country: string;
}

export interface WeatherConditions {
  condition: string;
  description: string;
  icon: string;
  temperature: number;
  feels_like: number;
  humidity: number;
  pressure: number;
  visibility: number;
  clouds: number;
  wind: {
    speed: number;
    direction: number;
    gust?: number;
  };
  rain?: {
    '1h'?: number;
    '3h'?: number;
  };
  snow?: {
    '1h'?: number;
    '3h'?: number;
  };
}

export interface CurrentWeather {
  success: boolean;
  location: WeatherLocation;
  weather: WeatherConditions;
  timestamp: string;
  source: string;
  error?: string;
  message?: string;
}

export interface ForecastItem {
  datetime: string;
  timestamp: number;
  temperature: number;
  feels_like: number;
  humidity: number;
  pressure: number;
  condition: string;
  description: string;
  icon: string;
  clouds: number;
  wind_speed: number;
  rain_3h: number;
  snow_3h: number;
  pop: number; // Probability of precipitation
}

export interface WeatherForecast {
  success: boolean;
  location: WeatherLocation;
  forecasts: ForecastItem[];
  total_forecasts: number;
  timestamp: string;
  source: string;
  error?: string;
  message?: string;
}

export interface PrecipitationData {
  success: boolean;
  location: WeatherLocation;
  current_precipitation: {
    rain_1h_mm: number;
    rain_3h_mm: number;
    humidity: number;
    clouds: number;
  };
  forecast_precipitation: {
    total_rain_mm: number;
    max_probability: number;
    rainy_periods: number;
  };
  risk_assessment: {
    precipitation_risk: 'low' | 'medium' | 'high';
    flood_contribution: 'low' | 'moderate' | 'high';
  };
  timestamp: string;
  error?: string;
}

export interface FloodRiskWeather {
  success: boolean;
  location: {
    lat: number;
    lon: number;
  };
  current_weather: CurrentWeather | null;
  forecast: WeatherForecast | null;
  precipitation: PrecipitationData | null;
  flood_risk_assessment: {
    risk_level: 'low' | 'medium' | 'high';
    risk_score: number;
    risk_factors: string[];
  };
  timestamp: string;
  error?: string;
}

export interface CityWeatherData {
  city: string;
  success: boolean;
  location?: WeatherLocation;
  weather?: WeatherConditions;
  timestamp?: string;
  source?: string;
  error?: string;
}

export interface RegionalWeather {
  success: boolean;
  region: string;
  cities: CityWeatherData[];
  timestamp: string;
  error?: string;
}

// Main hook for weather data
export const useWeatherData = (lat?: number, lon?: number) => {
  const [currentWeather, setCurrentWeather] = useState<CurrentWeather | null>(null);
  const [forecast, setForecast] = useState<WeatherForecast | null>(null);
  const [precipitation, setPrecipitation] = useState<PrecipitationData | null>(null);
  const [floodRiskWeather, setFloodRiskWeather] = useState<FloodRiskWeather | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch current weather
  const fetchCurrentWeather = useCallback(async (latitude: number, longitude: number) => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams({
        lat: latitude.toString(),
        lon: longitude.toString()
      });
      
      const response = await fetch(`${API_BASE_URL}/weather/current?${params}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch weather: ${response.statusText}`);
      }
      
      const data: CurrentWeather = await response.json();
      setCurrentWeather(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch weather data';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch weather forecast
  const fetchForecast = useCallback(async (latitude: number, longitude: number, days: number = 5) => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams({
        lat: latitude.toString(),
        lon: longitude.toString(),
        days: days.toString()
      });
      
      const response = await fetch(`${API_BASE_URL}/weather/forecast?${params}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch forecast: ${response.statusText}`);
      }
      
      const data: WeatherForecast = await response.json();
      setForecast(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch forecast data';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch precipitation data
  const fetchPrecipitation = useCallback(async (latitude: number, longitude: number) => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams({
        lat: latitude.toString(),
        lon: longitude.toString()
      });
      
      const response = await fetch(`${API_BASE_URL}/weather/precipitation?${params}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch precipitation: ${response.statusText}`);
      }
      
      const data: PrecipitationData = await response.json();
      setPrecipitation(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch precipitation data';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch comprehensive flood risk weather data
  const fetchFloodRiskWeather = useCallback(async (latitude: number, longitude: number) => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams({
        lat: latitude.toString(),
        lon: longitude.toString()
      });
      
      const response = await fetch(`${API_BASE_URL}/weather/flood-risk?${params}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch flood risk weather: ${response.statusText}`);
      }
      
      const data: FloodRiskWeather = await response.json();
      setFloodRiskWeather(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch flood risk weather data';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch all weather data at once
  const fetchAllWeatherData = useCallback(async (latitude: number, longitude: number) => {
    setLoading(true);
    setError(null);
    
    try {
      // Fetch all data in parallel
      const [weatherRes, forecastRes, precipRes, floodRiskRes] = await Promise.all([
        fetch(`${API_BASE_URL}/weather/current?lat=${latitude}&lon=${longitude}`),
        fetch(`${API_BASE_URL}/weather/forecast?lat=${latitude}&lon=${longitude}`),
        fetch(`${API_BASE_URL}/weather/precipitation?lat=${latitude}&lon=${longitude}`),
        fetch(`${API_BASE_URL}/weather/flood-risk?lat=${latitude}&lon=${longitude}`)
      ]);
      
      const [weather, forecastData, precipData, floodRiskData] = await Promise.all([
        weatherRes.json(),
        forecastRes.json(),
        precipRes.json(),
        floodRiskRes.json()
      ]);
      
      setCurrentWeather(weather);
      setForecast(forecastData);
      setPrecipitation(precipData);
      setFloodRiskWeather(floodRiskData);
      
      return {
        currentWeather: weather,
        forecast: forecastData,
        precipitation: precipData,
        floodRiskWeather: floodRiskData
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch weather data';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-fetch when coordinates are provided
  useEffect(() => {
    if (lat !== undefined && lon !== undefined) {
      fetchCurrentWeather(lat, lon);
    }
  }, [lat, lon, fetchCurrentWeather]);

  return {
    currentWeather,
    forecast,
    precipitation,
    floodRiskWeather,
    loading,
    error,
    fetchCurrentWeather,
    fetchForecast,
    fetchPrecipitation,
    fetchFloodRiskWeather,
    fetchAllWeatherData
  };
};

// Hook for city-specific weather in Pakistan
export const useCityWeather = (cityName?: string) => {
  const [cityWeather, setCityWeather] = useState<CurrentWeather | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchCityWeather = useCallback(async (city: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/weather/city/${encodeURIComponent(city)}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch city weather: ${response.statusText}`);
      }
      
      const data: CurrentWeather = await response.json();
      setCityWeather(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch city weather';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (cityName) {
      fetchCityWeather(cityName);
    }
  }, [cityName, fetchCityWeather]);

  return {
    cityWeather,
    loading,
    error,
    fetchCityWeather
  };
};

// Hook for regional weather data
export const useRegionalWeather = (region?: string) => {
  const [regionalWeather, setRegionalWeather] = useState<RegionalWeather | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchRegionalWeather = useCallback(async (regionName: string = 'all') => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/weather/region/${encodeURIComponent(regionName)}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch regional weather: ${response.statusText}`);
      }
      
      const data: RegionalWeather = await response.json();
      setRegionalWeather(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch regional weather';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (region) {
      fetchRegionalWeather(region);
    }
  }, [region, fetchRegionalWeather]);

  return {
    regionalWeather,
    loading,
    error,
    fetchRegionalWeather
  };
};

// ============================================================================
// FLOOD PREDICTION HOOK - Uses real-time weather for flood risk assessment
// ============================================================================

export interface FloodPrediction {
  success: boolean;
  location: {
    lat: number;
    lon: number;
  };
  risk_level: 'Safe' | 'At Risk' | 'Flooded';
  severity: 'low_risk' | 'medium_risk' | 'high_risk';
  total_score: number;
  weather_available: boolean;
  forecast_available: boolean;
  current_conditions: {
    condition: string;
    description: string;
    temperature: number | null;
    humidity: number;
    rain_1h_mm: number;
    rain_3h_mm: number;
  };
  forecast: {
    total_rain_24h_mm: number;
    max_rain_3h_mm: number;
    rainy_periods_24h: number;
  };
  risk_factors: {
    rainfall_current: number;
    rainfall_accumulated: number;
    humidity: number;
    forecast: number;
    geographical: number;
  };
  timestamp: string;
}

// ============================================================================
// ML PREDICTION TYPES AND HOOKS
// ============================================================================

export interface MLModelInfo {
  ml_available: boolean;
  available_models: string[];
  sklearn_available: boolean;
  tensorflow_available: boolean;
  feature_names: string[];
  risk_levels: string[];
  model_directory: string;
}

export interface MLPrediction {
  success: boolean;
  demo_mode?: boolean;
  prediction_method: 'machine_learning';
  model_type: string;
  location: {
    lat: number;
    lon: number;
  };
  risk_level: 'Safe' | 'At Risk' | 'Flooded';
  risk_index: number;
  confidence: number;
  probabilities: {
    Safe: number;
    'At Risk': number;
    Flooded: number;
  };
  current_conditions: {
    condition: string;
    description: string;
    temperature: number;
    humidity: number;
    rain_1h_mm: number;
    rain_3h_mm: number;
  };
  ml_features: {
    rainfall_1h: number;
    rainfall_3h: number;
    rainfall_24h: number;
    humidity: number;
    temperature: number;
    pressure: number;
    wind_speed: number;
    cloud_cover: number;
    forecast_rain_24h: number;
    river_proximity: number;
    elevation: number;
    soil_saturation: number;
    previous_flood: number;
    monsoon_season: number;
  };
  individual_predictions: {
    [modelName: string]: {
      model: string;
      risk_level: string;
      risk_index: number;
      confidence: number;
      probabilities: Record<string, number>;
      feature_importance?: Record<string, number>;
    };
  };
  models_used: string[];
  feature_importance: Record<string, number>;
  timestamp: string;
}

export interface MLComparisonResult {
  location: { lat: number; lon: number };
  demo_mode: boolean;
  rule_based_prediction: {
    risk_level: string;
    total_score: number;
    method: string;
  };
  ml_prediction: {
    risk_level: string;
    confidence: number;
    probabilities: Record<string, number>;
    models_used: string[];
    method: string;
  };
  agreement: boolean;
  timestamp: string;
}

// Hook for ML Model Info
export const useMLModelInfo = () => {
  const [modelInfo, setModelInfo] = useState<MLModelInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchModelInfo = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/ml/models`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch ML model info: ${response.statusText}`);
      }
      
      const data: MLModelInfo = await response.json();
      setModelInfo(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch ML model info';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModelInfo();
  }, [fetchModelInfo]);

  return {
    modelInfo,
    loading,
    error,
    fetchModelInfo
  };
};

// Hook for ML Flood Predictions
export const useMLPrediction = (lat?: number, lon?: number, modelType: string = 'ensemble', demoMode: boolean = false) => {
  const [prediction, setPrediction] = useState<MLPrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchMLPrediction = useCallback(async (
    latitude: number, 
    longitude: number,
    model: string = 'ensemble',
    demo: boolean = false
  ) => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams({
        lat: latitude.toString(),
        lon: longitude.toString(),
        model: model,
        demo: demo.toString()
      });
      
      const response = await fetch(`${API_BASE_URL}/ml/predict?${params}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch ML prediction: ${response.statusText}`);
      }
      
      const data: MLPrediction = await response.json();
      setPrediction(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch ML prediction';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-fetch when coordinates are provided
  useEffect(() => {
    if (lat !== undefined && lon !== undefined) {
      fetchMLPrediction(lat, lon, modelType, demoMode);
    }
  }, [lat, lon, modelType, demoMode, fetchMLPrediction]);

  return {
    prediction,
    loading,
    error,
    fetchMLPrediction
  };
};

// Hook for comparing rule-based vs ML predictions
export const useMLComparison = () => {
  const [comparison, setComparison] = useState<MLComparisonResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchComparison = useCallback(async (
    latitude: number, 
    longitude: number,
    demo: boolean = false
  ) => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams({
        lat: latitude.toString(),
        lon: longitude.toString(),
        demo: demo.toString()
      });
      
      const response = await fetch(`${API_BASE_URL}/ml/compare?${params}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch comparison: ${response.statusText}`);
      }
      
      const data: MLComparisonResult = await response.json();
      setComparison(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch prediction comparison';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    comparison,
    loading,
    error,
    fetchComparison
  };
};

// Hook for training ML models
export const useMLTraining = () => {
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<{ success: boolean; accuracies?: Record<string, number>; message?: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const trainModels = useCallback(async () => {
    setTraining(true);
    setError(null);
    setResult(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/ml/train`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error(`Failed to train models: ${response.statusText}`);
      }
      
      const data = await response.json();
      setResult(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to train ML models';
      setError(message);
      return null;
    } finally {
      setTraining(false);
    }
  }, []);

  return {
    training,
    result,
    error,
    trainModels
  };
};

export const useFloodPrediction = (lat?: number, lon?: number) => {
  const [prediction, setPrediction] = useState<FloodPrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchFloodPrediction = useCallback(async (latitude: number, longitude: number) => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams({
        lat: latitude.toString(),
        lon: longitude.toString()
      });
      
      const response = await fetch(`${API_BASE_URL}/flood-prediction?${params}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch flood prediction: ${response.statusText}`);
      }
      
      const data: FloodPrediction = await response.json();
      setPrediction(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch flood prediction';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-fetch when coordinates are provided
  useEffect(() => {
    if (lat !== undefined && lon !== undefined) {
      fetchFloodPrediction(lat, lon);
    }
  }, [lat, lon, fetchFloodPrediction]);

  return {
    prediction,
    loading,
    error,
    fetchFloodPrediction
  };
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Utility function to get weather icon URL from OpenWeatherMap
export const getWeatherIconUrl = (iconCode: string, size: '1x' | '2x' | '4x' = '2x'): string => {
  const sizeMap = {
    '1x': '',
    '2x': '@2x',
    '4x': '@4x'
  };
  return `https://openweathermap.org/img/wn/${iconCode}${sizeMap[size]}.png`;
};

// Utility function to format temperature
export const formatTemperature = (temp: number, unit: 'C' | 'F' = 'C'): string => {
  if (unit === 'F') {
    return `${Math.round(temp * 9/5 + 32)}°F`;
  }
  return `${Math.round(temp)}°C`;
};

// Utility function to get wind direction as compass direction
export const getWindDirection = (degrees: number): string => {
  const directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
  const index = Math.round(degrees / 22.5) % 16;
  return directions[index];
};

// Utility function to determine weather severity for flood risk
export const getWeatherSeverity = (weather: WeatherConditions): 'low' | 'medium' | 'high' => {
  const rain1h = weather.rain?.['1h'] || 0;
  const rain3h = weather.rain?.['3h'] || 0;
  
  if (rain1h > 10 || rain3h > 30) {
    return 'high';
  } else if (rain1h > 5 || rain3h > 15 || weather.humidity > 85) {
    return 'medium';
  }
  return 'low';
};

// Get color for risk level (matches map tile colors)
export const getRiskColor = (riskLevel: 'Safe' | 'At Risk' | 'Flooded'): string => {
  const colors = {
    'Safe': '#22c55e',      // Green
    'At Risk': '#f59e0b',   // Yellow/Amber
    'Flooded': '#ef4444'    // Red
  };
  return colors[riskLevel];
};

// Get severity color (matches alert colors)
export const getSeverityColor = (severity: 'low_risk' | 'medium_risk' | 'high_risk'): string => {
  const colors = {
    'low_risk': '#22c55e',    // Green
    'medium_risk': '#f59e0b', // Yellow
    'high_risk': '#ef4444'    // Red
  };
  return colors[severity];
};

export default useWeatherData;
