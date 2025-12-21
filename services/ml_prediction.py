"""
Machine Learning Flood Prediction Module for FLEWS
Implements Random Forest, Gradient Boosting, and LSTM models
for high-accuracy flood risk prediction
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. ML features will be limited.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not installed. LSTM features will be disabled.")


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Feature names for the models
FEATURE_NAMES = [
    'rainfall_1h',      # Current hourly rainfall (mm)
    'rainfall_3h',      # 3-hour accumulated rainfall (mm)
    'rainfall_24h',     # 24-hour accumulated rainfall (mm)
    'humidity',         # Relative humidity (%)
    'temperature',      # Temperature (°C)
    'pressure',         # Atmospheric pressure (hPa)
    'wind_speed',       # Wind speed (m/s)
    'cloud_cover',      # Cloud coverage (%)
    'forecast_rain_24h', # Forecasted rain in next 24h (mm)
    'river_proximity',  # Distance to nearest river (km, normalized 0-1)
    'elevation',        # Elevation above sea level (m, normalized)
    'soil_saturation',  # Estimated soil saturation (0-1)
    'previous_flood',   # Was there flooding in past 7 days (0/1)
    'monsoon_season',   # Is it monsoon season (0/1)
]

# Risk levels
RISK_LEVELS = ['Safe', 'At Risk', 'Flooded']


# ============================================================================
# SYNTHETIC DATA GENERATOR (for training when no real data available)
# ============================================================================

def generate_synthetic_training_data(n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data based on realistic flood patterns.
    In production, this should be replaced with real historical data.
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Generate features with realistic correlations
        monsoon = np.random.choice([0, 1], p=[0.6, 0.4])
        
        # Rainfall is higher during monsoon
        if monsoon:
            rainfall_1h = np.random.exponential(5) * np.random.uniform(0.5, 2)
            rainfall_3h = rainfall_1h * np.random.uniform(2, 4)
            rainfall_24h = rainfall_3h * np.random.uniform(3, 6)
        else:
            rainfall_1h = np.random.exponential(1) * np.random.uniform(0, 1)
            rainfall_3h = rainfall_1h * np.random.uniform(1, 3)
            rainfall_24h = rainfall_3h * np.random.uniform(2, 4)
        
        humidity = min(100, 50 + rainfall_1h * 3 + np.random.normal(0, 10))
        temperature = 25 + np.random.normal(0, 8) - (rainfall_1h > 10) * 5
        pressure = 1013 + np.random.normal(0, 10) - (rainfall_1h > 5) * 5
        wind_speed = np.abs(np.random.normal(5, 3) + rainfall_1h * 0.3)
        cloud_cover = min(100, max(0, rainfall_1h * 5 + np.random.normal(40, 20)))
        forecast_rain = rainfall_24h * np.random.uniform(0.5, 1.5)
        river_proximity = np.random.uniform(0, 1)
        elevation = np.random.uniform(0, 1)
        soil_saturation = min(1, max(0, (rainfall_24h / 100) + np.random.normal(0.3, 0.2)))
        previous_flood = np.random.choice([0, 1], p=[0.85, 0.15])
        
        features = [
            rainfall_1h, rainfall_3h, rainfall_24h,
            humidity, temperature, pressure, wind_speed, cloud_cover,
            forecast_rain, river_proximity, elevation, soil_saturation,
            previous_flood, monsoon
        ]
        
        # Determine risk level based on realistic thresholds
        risk_score = (
            rainfall_1h * 0.25 +
            rainfall_3h * 0.15 +
            rainfall_24h * 0.10 +
            (humidity / 100) * 10 +
            (1 - elevation) * 15 +  # Lower elevation = higher risk
            (1 - river_proximity) * 20 +  # Closer to river = higher risk
            soil_saturation * 15 +
            previous_flood * 10 +
            monsoon * 10 +
            forecast_rain * 0.05
        )
        
        # Add some noise
        risk_score += np.random.normal(0, 5)
        
        # Classify
        if risk_score > 60:
            label = 2  # Flooded
        elif risk_score > 35:
            label = 1  # At Risk
        else:
            label = 0  # Safe
        
        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y)


def generate_time_series_data(n_sequences: int = 1000, seq_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate time series data for LSTM training.
    Each sequence represents 24 hours of weather data.
    
    Returns:
        X: Sequence data (n_sequences, seq_length, n_features)
        y: Labels for the next time step (n_sequences,)
    """
    np.random.seed(42)
    
    # Use subset of features for time series
    n_ts_features = 6  # rainfall, humidity, temp, pressure, wind, clouds
    
    X = []
    y = []
    
    for _ in range(n_sequences):
        sequence = []
        
        # Generate a realistic weather sequence
        base_rainfall = np.random.exponential(2)
        trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])  # decreasing, stable, increasing
        
        for t in range(seq_length):
            rainfall = max(0, base_rainfall + trend * t * 0.5 + np.random.normal(0, 1))
            humidity = min(100, max(30, 60 + rainfall * 2 + np.random.normal(0, 5)))
            temperature = 28 - rainfall * 0.3 + np.random.normal(0, 2)
            pressure = 1013 - rainfall * 0.5 + np.random.normal(0, 3)
            wind_speed = max(0, 5 + rainfall * 0.2 + np.random.normal(0, 2))
            cloud_cover = min(100, max(0, rainfall * 5 + np.random.normal(50, 15)))
            
            sequence.append([rainfall, humidity, temperature, pressure, wind_speed, cloud_cover])
        
        X.append(sequence)
        
        # Predict risk based on sequence pattern
        last_rainfall = sequence[-1][0]
        avg_rainfall = np.mean([s[0] for s in sequence])
        rainfall_trend = sequence[-1][0] - sequence[0][0]
        
        risk_score = last_rainfall * 3 + avg_rainfall * 2 + max(0, rainfall_trend) * 5
        
        if risk_score > 40:
            y.append(2)  # Flooded
        elif risk_score > 20:
            y.append(1)  # At Risk
        else:
            y.append(0)  # Safe
    
    return np.array(X), np.array(y)


# ============================================================================
# RANDOM FOREST MODEL
# ============================================================================

class RandomForestFloodModel:
    """
    Random Forest classifier for flood risk prediction.
    Good for handling mixed feature types and providing feature importance.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = MODEL_DIR / "random_forest_flood.pkl"
        self.scaler_path = MODEL_DIR / "rf_scaler.pkl"
        self.feature_importance = {}
    
    def train(self, X: np.ndarray = None, y: np.ndarray = None):
        """Train the Random Forest model"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Random Forest")
        
        # Generate synthetic data if none provided
        if X is None or y is None:
            print("Generating synthetic training data...")
            X, y = generate_synthetic_training_data(10000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Random Forest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=RISK_LEVELS))
        
        # Store feature importance
        for i, name in enumerate(FEATURE_NAMES):
            self.feature_importance[name] = self.model.feature_importances_[i]
        
        print("\nTop 5 Feature Importances:")
        sorted_importance = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        for name, importance in sorted_importance[:5]:
            print(f"  {name}: {importance:.4f}")
        
        self.is_trained = True
        self.save()
        
        return accuracy
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict flood risk for given features.
        
        Args:
            features: Dict with feature names and values
            
        Returns:
            Prediction result with risk level and probabilities
        """
        if not self.is_trained:
            self.load()
        
        if not self.is_trained:
            # Train on-the-fly if no saved model
            self.train()
        
        # Prepare feature vector
        X = np.array([[features.get(name, 0) for name in FEATURE_NAMES]])
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        return {
            "model": "RandomForest",
            "risk_level": RISK_LEVELS[prediction],
            "risk_index": int(prediction),
            "confidence": float(max(probabilities)),
            "probabilities": {
                RISK_LEVELS[i]: float(p) for i, p in enumerate(probabilities)
            },
            "feature_importance": self.feature_importance
        }
    
    def save(self):
        """Save model and scaler to disk"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Random Forest model saved to {self.model_path}")
    
    def load(self):
        """Load model and scaler from disk"""
        if self.model_path.exists() and self.scaler_path.exists():
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
            print("Random Forest model loaded from disk")
            return True
        return False


# ============================================================================
# GRADIENT BOOSTING MODEL
# ============================================================================

class GradientBoostingFloodModel:
    """
    Gradient Boosting classifier for flood risk prediction.
    Often achieves higher accuracy than Random Forest.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = MODEL_DIR / "gradient_boosting_flood.pkl"
        self.scaler_path = MODEL_DIR / "gb_scaler.pkl"
        self.feature_importance = {}
    
    def train(self, X: np.ndarray = None, y: np.ndarray = None):
        """Train the Gradient Boosting model"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Gradient Boosting")
        
        # Generate synthetic data if none provided
        if X is None or y is None:
            print("Generating synthetic training data...")
            X, y = generate_synthetic_training_data(10000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training Gradient Boosting model...")
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Gradient Boosting Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=RISK_LEVELS))
        
        # Store feature importance
        for i, name in enumerate(FEATURE_NAMES):
            self.feature_importance[name] = self.model.feature_importances_[i]
        
        print("\nTop 5 Feature Importances:")
        sorted_importance = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        for name, importance in sorted_importance[:5]:
            print(f"  {name}: {importance:.4f}")
        
        self.is_trained = True
        self.save()
        
        return accuracy
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict flood risk for given features"""
        if not self.is_trained:
            self.load()
        
        if not self.is_trained:
            self.train()
        
        # Prepare feature vector
        X = np.array([[features.get(name, 0) for name in FEATURE_NAMES]])
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        return {
            "model": "GradientBoosting",
            "risk_level": RISK_LEVELS[prediction],
            "risk_index": int(prediction),
            "confidence": float(max(probabilities)),
            "probabilities": {
                RISK_LEVELS[i]: float(p) for i, p in enumerate(probabilities)
            },
            "feature_importance": self.feature_importance
        }
    
    def save(self):
        """Save model and scaler to disk"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Gradient Boosting model saved to {self.model_path}")
    
    def load(self):
        """Load model and scaler from disk"""
        if self.model_path.exists() and self.scaler_path.exists():
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
            print("Gradient Boosting model loaded from disk")
            return True
        return False


# ============================================================================
# LSTM MODEL (Time Series)
# ============================================================================

class LSTMFloodModel:
    """
    LSTM neural network for time-series flood prediction.
    Uses sequence of weather data to predict future flood risk.
    """
    
    def __init__(self, sequence_length: int = 24):
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.sequence_length = sequence_length
        self.n_features = 6  # rainfall, humidity, temp, pressure, wind, clouds
        self.is_trained = False
        self.model_path = MODEL_DIR / "lstm_flood.keras"
        self.scaler_path = MODEL_DIR / "lstm_scaler.pkl"
    
    def build_model(self):
        """Build LSTM architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM")
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes: Safe, At Risk, Flooded
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X: np.ndarray = None, y: np.ndarray = None, epochs: int = 50):
        """Train the LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM training")
        
        # Generate synthetic data if none provided
        if X is None or y is None:
            print("Generating synthetic time series data...")
            X, y = generate_time_series_data(2000, self.sequence_length)
        
        # Reshape and scale
        original_shape = X.shape
        X_reshaped = X.reshape(-1, self.n_features)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Build and train model
        print("Training LSTM model...")
        self.model = self.build_model()
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nLSTM Accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        self.save()
        
        return accuracy
    
    def predict(self, sequence: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Predict flood risk from a sequence of weather data.
        
        Args:
            sequence: List of 24 hourly weather readings
            
        Returns:
            Prediction result with risk level and probabilities
        """
        if not TENSORFLOW_AVAILABLE:
            return {"error": "TensorFlow not available", "model": "LSTM"}
        
        if not self.is_trained:
            self.load()
        
        if not self.is_trained:
            self.train()
        
        # Prepare sequence
        feature_names = ['rainfall_1h', 'humidity', 'temperature', 'pressure', 'wind_speed', 'cloud_cover']
        X = np.array([[
            [reading.get(name, 0) for name in feature_names]
            for reading in sequence
        ]])
        
        # Pad if sequence is too short
        if X.shape[1] < self.sequence_length:
            padding = np.zeros((1, self.sequence_length - X.shape[1], self.n_features))
            X = np.concatenate([padding, X], axis=1)
        elif X.shape[1] > self.sequence_length:
            X = X[:, -self.sequence_length:, :]
        
        # Scale
        X_reshaped = X.reshape(-1, self.n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(1, self.sequence_length, self.n_features)
        
        # Predict
        probabilities = self.model.predict(X_scaled, verbose=0)[0]
        prediction = np.argmax(probabilities)
        
        return {
            "model": "LSTM",
            "risk_level": RISK_LEVELS[prediction],
            "risk_index": int(prediction),
            "confidence": float(max(probabilities)),
            "probabilities": {
                RISK_LEVELS[i]: float(p) for i, p in enumerate(probabilities)
            },
            "sequence_length": len(sequence)
        }
    
    def save(self):
        """Save model and scaler to disk"""
        if self.model:
            self.model.save(self.model_path)
        if self.scaler and SKLEARN_AVAILABLE:
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        print(f"LSTM model saved to {self.model_path}")
    
    def load(self):
        """Load model and scaler from disk"""
        if self.model_path.exists():
            try:
                self.model = load_model(self.model_path)
                if self.scaler_path.exists() and SKLEARN_AVAILABLE:
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                self.is_trained = True
                print("LSTM model loaded from disk")
                return True
            except Exception as e:
                print(f"Error loading LSTM model: {e}")
        return False


# ============================================================================
# ENSEMBLE MODEL
# ============================================================================

class EnsembleFloodModel:
    """
    Ensemble model that combines predictions from Random Forest,
    Gradient Boosting, and LSTM for improved accuracy.
    """
    
    def __init__(self):
        self.rf_model = RandomForestFloodModel()
        self.gb_model = GradientBoostingFloodModel()
        self.lstm_model = LSTMFloodModel() if TENSORFLOW_AVAILABLE else None
        
        # Weights for ensemble (can be tuned based on validation performance)
        self.weights = {
            'RandomForest': 0.3,
            'GradientBoosting': 0.4,
            'LSTM': 0.3
        }
    
    def train_all(self):
        """Train all models in the ensemble"""
        print("=" * 60)
        print("TRAINING ENSEMBLE MODELS")
        print("=" * 60)
        
        accuracies = {}
        
        print("\n" + "-" * 40)
        print("Training Random Forest...")
        print("-" * 40)
        accuracies['RandomForest'] = self.rf_model.train()
        
        print("\n" + "-" * 40)
        print("Training Gradient Boosting...")
        print("-" * 40)
        accuracies['GradientBoosting'] = self.gb_model.train()
        
        if TENSORFLOW_AVAILABLE and self.lstm_model:
            print("\n" + "-" * 40)
            print("Training LSTM...")
            print("-" * 40)
            accuracies['LSTM'] = self.lstm_model.train()
        
        print("\n" + "=" * 60)
        print("ENSEMBLE TRAINING COMPLETE")
        print("=" * 60)
        print("\nModel Accuracies:")
        for model, acc in accuracies.items():
            print(f"  {model}: {acc:.4f}")
        
        return accuracies
    
    def predict(self, features: Dict[str, float], weather_sequence: List[Dict] = None) -> Dict[str, Any]:
        """
        Make ensemble prediction by combining all models.
        
        Args:
            features: Current weather features
            weather_sequence: Optional 24-hour weather sequence for LSTM
            
        Returns:
            Combined prediction with individual model outputs
        """
        predictions = {}
        weighted_probs = np.zeros(3)
        total_weight = 0
        
        # Random Forest prediction
        try:
            rf_pred = self.rf_model.predict(features)
            predictions['RandomForest'] = rf_pred
            for i, level in enumerate(RISK_LEVELS):
                weighted_probs[i] += rf_pred['probabilities'][level] * self.weights['RandomForest']
            total_weight += self.weights['RandomForest']
        except Exception as e:
            print(f"RF prediction error: {e}")
        
        # Gradient Boosting prediction
        try:
            gb_pred = self.gb_model.predict(features)
            predictions['GradientBoosting'] = gb_pred
            for i, level in enumerate(RISK_LEVELS):
                weighted_probs[i] += gb_pred['probabilities'][level] * self.weights['GradientBoosting']
            total_weight += self.weights['GradientBoosting']
        except Exception as e:
            print(f"GB prediction error: {e}")
        
        # LSTM prediction (if sequence available)
        if self.lstm_model and weather_sequence and len(weather_sequence) >= 6:
            try:
                lstm_pred = self.lstm_model.predict(weather_sequence)
                predictions['LSTM'] = lstm_pred
                for i, level in enumerate(RISK_LEVELS):
                    weighted_probs[i] += lstm_pred['probabilities'][level] * self.weights['LSTM']
                total_weight += self.weights['LSTM']
            except Exception as e:
                print(f"LSTM prediction error: {e}")
        
        # Normalize probabilities
        if total_weight > 0:
            weighted_probs /= total_weight
        
        # Final prediction
        final_prediction = np.argmax(weighted_probs)
        
        return {
            "model": "Ensemble",
            "risk_level": RISK_LEVELS[final_prediction],
            "risk_index": int(final_prediction),
            "confidence": float(max(weighted_probs)),
            "probabilities": {
                RISK_LEVELS[i]: float(p) for i, p in enumerate(weighted_probs)
            },
            "individual_predictions": predictions,
            "models_used": list(predictions.keys()),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# SINGLETON INSTANCES
# ============================================================================

# Initialize models (lazy loading)
_ensemble_model = None

def get_ensemble_model() -> EnsembleFloodModel:
    """Get or create the singleton ensemble model"""
    global _ensemble_model
    if _ensemble_model is None:
        _ensemble_model = EnsembleFloodModel()
    return _ensemble_model


# ============================================================================
# PUBLIC API
# ============================================================================

def predict_flood_risk_ml(
    features: Dict[str, float],
    weather_sequence: List[Dict] = None,
    model_type: str = "ensemble"
) -> Dict[str, Any]:
    """
    Predict flood risk using machine learning models.
    
    Args:
        features: Dict with weather/environmental features
        weather_sequence: Optional list of historical weather readings for LSTM
        model_type: "ensemble", "random_forest", "gradient_boosting", or "lstm"
        
    Returns:
        Prediction result with risk level, confidence, and probabilities
    """
    ensemble = get_ensemble_model()
    
    if model_type == "ensemble":
        return ensemble.predict(features, weather_sequence)
    elif model_type == "random_forest":
        return ensemble.rf_model.predict(features)
    elif model_type == "gradient_boosting":
        return ensemble.gb_model.predict(features)
    elif model_type == "lstm" and ensemble.lstm_model:
        if weather_sequence:
            return ensemble.lstm_model.predict(weather_sequence)
        else:
            return {"error": "LSTM requires weather_sequence parameter"}
    else:
        return {"error": f"Unknown model type: {model_type}"}


def train_models():
    """Train all ML models"""
    ensemble = get_ensemble_model()
    return ensemble.train_all()


def get_model_info() -> Dict[str, Any]:
    """Get information about available models"""
    return {
        "available_models": ["ensemble", "random_forest", "gradient_boosting"] + 
                           (["lstm"] if TENSORFLOW_AVAILABLE else []),
        "sklearn_available": SKLEARN_AVAILABLE,
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "feature_names": FEATURE_NAMES,
        "risk_levels": RISK_LEVELS,
        "model_directory": str(MODEL_DIR)
    }


# ============================================================================
# CLI for training
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        print("Starting model training...")
        train_models()
    else:
        print("ML Flood Prediction Module")
        print("-" * 40)
        info = get_model_info()
        print(f"Available models: {info['available_models']}")
        print(f"scikit-learn: {'✓' if info['sklearn_available'] else '✗'}")
        print(f"TensorFlow: {'✓' if info['tensorflow_available'] else '✗'}")
        print(f"\nTo train models, run: python ml_prediction.py train")
