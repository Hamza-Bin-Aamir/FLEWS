import React, { useState, useEffect } from 'react';
import { 
  useMLPrediction, 
  useMLModelInfo, 
  useMLComparison,
  getRiskColor 
} from '../hooks/useWeatherData';
import './MLPredictionPanel.scss';

interface MLPredictionPanelProps {
  lat: number;
  lon: number;
  demoMode?: boolean;
  onClose?: () => void;
}

const MLPredictionPanel: React.FC<MLPredictionPanelProps> = ({ 
  lat, 
  lon, 
  demoMode = false,
  onClose 
}) => {
  const [selectedModel, setSelectedModel] = useState<string>('ensemble');
  const [showComparison, setShowComparison] = useState(false);
  const [showFeatures, setShowFeatures] = useState(false);
  
  const { modelInfo } = useMLModelInfo();
  const { prediction, loading, error, fetchMLPrediction } = useMLPrediction();
  const { comparison, fetchComparison } = useMLComparison();

  useEffect(() => {
    if (lat && lon) {
      fetchMLPrediction(lat, lon, selectedModel, demoMode);
    }
  }, [lat, lon, selectedModel, demoMode, fetchMLPrediction]);

  useEffect(() => {
    if (showComparison && lat && lon) {
      fetchComparison(lat, lon, demoMode);
    }
  }, [showComparison, lat, lon, demoMode, fetchComparison]);

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;
  
  const getConfidenceClass = (confidence: number) => {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.6) return 'medium';
    return 'low';
  };

  const getTopFeatures = (importance: Record<string, number>, count: number = 5) => {
    return Object.entries(importance)
      .sort(([, a], [, b]) => b - a)
      .slice(0, count);
  };

  if (loading) {
    return (
      <div className="ml-prediction-panel">
        <div className="panel-header">
          <h3>🤖 ML Flood Prediction</h3>
          {onClose && <button className="close-btn" onClick={onClose}>×</button>}
        </div>
        <div className="loading">
          <div className="spinner"></div>
          <p>Analyzing with Machine Learning...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="ml-prediction-panel">
        <div className="panel-header">
          <h3>🤖 ML Flood Prediction</h3>
          {onClose && <button className="close-btn" onClick={onClose}>×</button>}
        </div>
        <div className="error">
          <p>⚠️ {error}</p>
          <button onClick={() => fetchMLPrediction(lat, lon, selectedModel, demoMode)}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="ml-prediction-panel">
      <div className="panel-header">
        <h3>🤖 ML Flood Prediction</h3>
        {onClose && <button className="close-btn" onClick={onClose}>×</button>}
      </div>

      {/* Model Selector */}
      <div className="model-selector">
        <label>Model:</label>
        <select 
          value={selectedModel} 
          onChange={(e) => setSelectedModel(e.target.value)}
        >
          {modelInfo?.available_models.map(model => (
            <option key={model} value={model}>
              {model.charAt(0).toUpperCase() + model.slice(1).replace('_', ' ')}
            </option>
          ))}
        </select>
      </div>

      {prediction && (
        <>
          {/* Main Prediction */}
          <div 
            className="prediction-result"
            style={{ borderColor: getRiskColor(prediction.risk_level) }}
          >
            <div 
              className="risk-badge"
              style={{ backgroundColor: getRiskColor(prediction.risk_level) }}
            >
              {prediction.risk_level}
            </div>
            
            <div className={`confidence confidence-${getConfidenceClass(prediction.confidence)}`}>
              <span className="label">Confidence:</span>
              <span className="value">{formatPercentage(prediction.confidence)}</span>
            </div>

            {prediction.demo_mode && (
              <div className="demo-badge">Demo Mode</div>
            )}
          </div>

          {/* Probability Distribution */}
          <div className="probability-section">
            <h4>Risk Probabilities</h4>
            <div className="probability-bars">
              {Object.entries(prediction.probabilities).map(([level, prob]) => (
                <div key={level} className="prob-item">
                  <div className="prob-label">{level}</div>
                  <div className="prob-bar-container">
                    <div 
                      className="prob-bar"
                      style={{ 
                        width: `${prob * 100}%`,
                        backgroundColor: getRiskColor(level as 'Safe' | 'At Risk' | 'Flooded')
                      }}
                    />
                  </div>
                  <div className="prob-value">{formatPercentage(prob)}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Current Conditions */}
          <div className="conditions-section">
            <h4>Current Conditions</h4>
            <div className="conditions-grid">
              <div className="condition-item">
                <span className="icon">🌡️</span>
                <span className="value">{prediction.current_conditions.temperature}°C</span>
                <span className="label">Temperature</span>
              </div>
              <div className="condition-item">
                <span className="icon">💧</span>
                <span className="value">{prediction.current_conditions.humidity}%</span>
                <span className="label">Humidity</span>
              </div>
              <div className="condition-item">
                <span className="icon">🌧️</span>
                <span className="value">{prediction.current_conditions.rain_1h_mm}mm</span>
                <span className="label">Rain (1h)</span>
              </div>
              <div className="condition-item">
                <span className="icon">⛈️</span>
                <span className="value">{prediction.current_conditions.rain_3h_mm}mm</span>
                <span className="label">Rain (3h)</span>
              </div>
            </div>
          </div>

          {/* Models Used */}
          <div className="models-section">
            <h4>Models Used</h4>
            <div className="model-tags">
              {prediction.models_used.map(model => (
                <span key={model} className="model-tag">{model}</span>
              ))}
            </div>
          </div>

          {/* Toggle Buttons */}
          <div className="toggle-buttons">
            <button 
              className={`toggle-btn ${showFeatures ? 'active' : ''}`}
              onClick={() => setShowFeatures(!showFeatures)}
            >
              {showFeatures ? 'Hide' : 'Show'} Features
            </button>
            <button 
              className={`toggle-btn ${showComparison ? 'active' : ''}`}
              onClick={() => setShowComparison(!showComparison)}
            >
              {showComparison ? 'Hide' : 'Compare'} with Rules
            </button>
          </div>

          {/* Feature Importance */}
          {showFeatures && prediction.individual_predictions && (
            <div className="features-section">
              <h4>Top Feature Importance</h4>
              {Object.entries(prediction.individual_predictions).map(([modelName, modelPred]) => (
                modelPred.feature_importance && (
                  <div key={modelName} className="model-features">
                    <h5>{modelName}</h5>
                    <div className="feature-bars">
                      {getTopFeatures(modelPred.feature_importance).map(([feature, importance]) => (
                        <div key={feature} className="feature-item">
                          <span className="feature-name">{feature.replace('_', ' ')}</span>
                          <div className="feature-bar-container">
                            <div 
                              className="feature-bar"
                              style={{ width: `${importance * 100}%` }}
                            />
                          </div>
                          <span className="feature-value">{(importance * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              ))}
            </div>
          )}

          {/* Comparison with Rule-Based */}
          {showComparison && comparison && (
            <div className="comparison-section">
              <h4>Rule-Based vs ML Comparison</h4>
              <div className="comparison-grid">
                <div className="comparison-item">
                  <h5>Rule-Based</h5>
                  <div 
                    className="comp-risk"
                    style={{ color: getRiskColor(comparison.rule_based_prediction.risk_level as 'Safe' | 'At Risk' | 'Flooded') }}
                  >
                    {comparison.rule_based_prediction.risk_level}
                  </div>
                  <div className="comp-score">
                    Score: {comparison.rule_based_prediction.total_score}
                  </div>
                </div>
                <div className="comparison-item">
                  <h5>ML Ensemble</h5>
                  <div 
                    className="comp-risk"
                    style={{ color: getRiskColor(comparison.ml_prediction.risk_level as 'Safe' | 'At Risk' | 'Flooded') }}
                  >
                    {comparison.ml_prediction.risk_level}
                  </div>
                  <div className="comp-score">
                    Confidence: {formatPercentage(comparison.ml_prediction.confidence)}
                  </div>
                </div>
              </div>
              <div className={`agreement-badge ${comparison.agreement ? 'agree' : 'disagree'}`}>
                {comparison.agreement ? '✓ Models Agree' : '⚠ Models Disagree'}
              </div>
            </div>
          )}
        </>
      )}

      {/* ML Info */}
      {modelInfo && (
        <div className="ml-info">
          <span className="info-item">
            sklearn: {modelInfo.sklearn_available ? '✓' : '✗'}
          </span>
          <span className="info-item">
            TensorFlow: {modelInfo.tensorflow_available ? '✓' : '✗'}
          </span>
        </div>
      )}
    </div>
  );
};

export default MLPredictionPanel;
