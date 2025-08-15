# agents/touch_pattern_agent.py
"""
Touch Pattern Agent for detecting anomalous touch behavior.
Uses Isolation Forest for swipe speed, touch pressure, and tap interval analysis.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import time
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResult, RiskLevel

@dataclass
class TouchEvent:
    """Represents a single touch event"""
    timestamp: float
    x: float
    y: float
    pressure: float
    touch_major: float
    touch_minor: float
    action: str  # 'down', 'move', 'up'

class TouchPatternAgent(BaseAgent):
    """
    Agent for analyzing touch patterns and detecting anomalies.
    Uses Isolation Forest to detect unusual swipe speeds, pressures, and tap patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("TouchPatternAgent", config)
        
        # Touch pattern parameters
        self.min_swipe_points = config.get('min_swipe_points', 3)
        self.tap_timeout_ms = config.get('tap_timeout_ms', 500)
        self.pressure_sensitivity = config.get('pressure_sensitivity', 0.1)
        
        # Model parameters
        self.contamination = config.get('contamination', 0.1)
        self.n_estimators = config.get('n_estimators', 50)
        self.max_samples = config.get('max_samples', 256)
        
        # Initialize models
        self.isolation_forest = None
        self.scaler = StandardScaler()
        
        # Feature extraction buffers
        self.current_gesture = []
        self.recent_taps = []
        self.gesture_history = []
    
    def capture_data(self, sensor_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Capture and preprocess touch sensor data.
        
        Expected sensor_data format:
        {
            'touch_events': [
                {
                    'timestamp': float,
                    'x': float, 'y': float,
                    'pressure': float,
                    'touch_major': float, 'touch_minor': float,
                    'action': str  # 'down', 'move', 'up'
                }
            ]
        }
        
        Returns:
            Feature vector: [swipe_speed, avg_pressure, pressure_variance, 
                           tap_interval, gesture_complexity, tremor_score]
        """
        try:
            touch_events = sensor_data.get('touch_events', [])
            if not touch_events:
                return None
            
            # Convert to TouchEvent objects
            events = [
                TouchEvent(
                    timestamp=event['timestamp'],
                    x=event['x'], y=event['y'],
                    pressure=event['pressure'],
                    touch_major=event.get('touch_major', 0),
                    touch_minor=event.get('touch_minor', 0),
                    action=event['action']
                ) for event in touch_events
            ]
            
            # Process gesture based on action types
            if events[0].action == 'down':
                self.current_gesture = [events[0]]
            elif events[-1].action == 'up' and self.current_gesture:
                self.current_gesture.extend(events)
                features = self._extract_gesture_features(self.current_gesture)
                self.current_gesture = []  # Reset for next gesture
                return features
            else:
                if self.current_gesture:
                    self.current_gesture.extend(events)
            
            return None  # Gesture not complete yet
            
        except Exception as e:
            print(f"Error capturing touch data: {e}")
            return None
    
    def _extract_gesture_features(self, gesture: List[TouchEvent]) -> np.ndarray:
        """
        Extract features from a complete gesture.
        
        Returns:
            Feature vector with 8 dimensions
        """
        if len(gesture) < 2:
            return np.zeros(8)
        
        # Convert to arrays for easier processing
        timestamps = np.array([e.timestamp for e in gesture])
        positions = np.array([(e.x, e.y) for e in gesture])
        pressures = np.array([e.pressure for e in gesture])
        
        # Feature 1: Average swipe speed (pixels/second)
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        time_diffs = np.diff(timestamps)
        speeds = distances / (time_diffs + 1e-6)  # Avoid division by zero
        avg_speed = np.mean(speeds) if len(speeds) > 0 else 0
        
        # Feature 2: Average pressure
        avg_pressure = np.mean(pressures)
        
        # Feature 3: Pressure variance (stability)
        pressure_variance = np.var(pressures)
        
        # Feature 4: Gesture duration
        gesture_duration = timestamps[-1] - timestamps[0]
        
        # Feature 5: Path complexity (total path length / straight line distance)
        total_path_length = np.sum(distances)
        straight_line_distance = np.sqrt(np.sum((positions[-1] - positions[0])**2))
        path_complexity = total_path_length / (straight_line_distance + 1e-6)
        
        # Feature 6: Tremor score (high-frequency movements)
        if len(positions) > 4:
            # Calculate second derivative (acceleration)
            velocity = np.diff(positions, axis=0)
            acceleration = np.diff(velocity, axis=0)
            tremor_score = np.mean(np.sqrt(np.sum(acceleration**2, axis=1)))
        else:
            tremor_score = 0
        
        # Feature 7: Pressure dynamics (how much pressure changes)
        if len(pressures) > 1:
            pressure_changes = np.abs(np.diff(pressures))
            pressure_dynamics = np.mean(pressure_changes)
        else:
            pressure_dynamics = 0
        
        # Feature 8: Tap interval (if this is part of multi-tap sequence)
        tap_interval = self._calculate_tap_interval(gesture[0].timestamp)
        
        features = np.array([
            avg_speed,
            avg_pressure,
            pressure_variance,
            gesture_duration,
            path_complexity,
            tremor_score,
            pressure_dynamics,
            tap_interval
        ])
        
        # Store for tap interval calculation
        if self._is_tap_gesture(gesture):
            self.recent_taps.append(gesture[0].timestamp)
            # Keep only recent taps (last 5 seconds)
            cutoff_time = gesture[0].timestamp - 5.0
            self.recent_taps = [t for t in self.recent_taps if t > cutoff_time]
        
        return features
    
    def _is_tap_gesture(self, gesture: List[TouchEvent]) -> bool:
        """Check if gesture is a tap (short duration, minimal movement)"""
        if len(gesture) < 2:
            return True
        
        duration = gesture[-1].timestamp - gesture[0].timestamp
        positions = np.array([(e.x, e.y) for e in gesture])
        movement = np.sqrt(np.sum((positions[-1] - positions[0])**2))
        
        return duration < (self.tap_timeout_ms / 1000.0) and movement < 20  # pixels
    
    def _calculate_tap_interval(self, current_timestamp: float) -> float:
        """Calculate average interval between recent taps"""
        if len(self.recent_taps) < 2:
            return 1.0  # Default interval
        
        intervals = np.diff(sorted(self.recent_taps))
        return np.mean(intervals) if len(intervals) > 0 else 1.0
    
    def train_initial(self, training_data: List[np.ndarray]) -> bool:
        """
        Train initial Isolation Forest model on baseline touch patterns.
        
        Args:
            training_data: List of feature vectors from normal touch behavior
            
        Returns:
            True if training successful
        """
        try:
            if len(training_data) < 10:
                print("Insufficient training data for TouchPatternAgent")
                return False
            
            # Convert to numpy array
            X = np.array(training_data)
            
            # Handle any invalid values
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Fit scaler on training data
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                max_samples=min(self.max_samples, len(X)),
                random_state=42,
                n_jobs=1  # Important for mobile deployment
            )
            
            self.isolation_forest.fit(X_scaled)
            
            self.is_trained = True
            print(f"TouchPatternAgent trained on {len(training_data)} samples")
            return True
            
        except Exception as e:
            print(f"Error training TouchPatternAgent: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> AgentResult:
        """
        Predict anomaly score for touch pattern features.
        
        Args:
            features: Preprocessed feature vector
            
        Returns:
            AgentResult with anomaly score and metadata
        """
        start_time = time.time()
        
        try:
            if not self.is_trained or self.isolation_forest is None:
                return AgentResult(
                    agent_name=self.agent_name,
                    anomaly_score=0.0,
                    risk_level=RiskLevel.LOW,
                    confidence=0.0,
                    features_used=['untrained'],
                    processing_time_ms=0,
                    metadata={'error': 'Model not trained'}
                )
            
            # Handle invalid values
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get anomaly score (Isolation Forest returns -1 for anomalies, 1 for normal)
            isolation_score = self.isolation_forest.decision_function(features_scaled)[0]
            
            # Convert to 0-1 range (0 = normal, 1 = highly anomalous)
            # Isolation Forest decision function typically ranges from -0.5 to 0.5
            anomaly_score = max(0.0, min(1.0, (0.5 - isolation_score)))
            
            # Calculate confidence based on distance from decision boundary
            confidence = min(1.0, abs(isolation_score) * 2)
            
            # Determine risk level
            risk_level = self.get_risk_level_from_score(anomaly_score)
            
            # Feature importance (simplified)
            feature_names = [
                'swipe_speed', 'avg_pressure', 'pressure_variance',
                'gesture_duration', 'path_complexity', 'tremor_score',
                'pressure_dynamics', 'tap_interval'
            ]
            
            processing_time = (time.time() - start_time) * 1000
            
            metadata = {
                'raw_isolation_score': isolation_score,
                'feature_values': {
                    name: float(val) for name, val in zip(feature_names, features)
                },
                'model_type': 'isolation_forest'
            }
            
            return AgentResult(
                agent_name=self.agent_name,
                anomaly_score=anomaly_score,
                risk_level=risk_level,
                confidence=confidence,
                features_used=feature_names,
                processing_time_ms=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            return AgentResult(
                agent_name=self.agent_name,
                anomaly_score=0.0,
                risk_level=RiskLevel.LOW,
                confidence=0.0,
                features_used=['error'],
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(e)}
            )
    
    def incremental_update(self, new_data: List[np.ndarray], 
                          is_anomaly: List[bool] = None) -> bool:
        """
        Update model with new normal touch patterns.
        
        Args:
            new_data: List of new feature vectors
            is_anomaly: Optional labels (ignored for unsupervised method)
            
        Returns:
            True if update successful
        """
        try:
            if not self.is_trained or len(new_data) < 5:
                return False
            
            # Filter to only normal data for unsupervised learning
            if is_anomaly is not None:
                normal_data = [data for data, anomaly in zip(new_data, is_anomaly) if not anomaly]
            else:
                # Assume all data is normal for incremental learning
                normal_data = new_data
            
            if len(normal_data) < 3:
                return False
            
            # Combine with existing baseline data for retraining
            all_data = self.baseline_data + normal_data
            
            # Limit dataset size for efficiency
            if len(all_data) > self.max_baseline_samples:
                # Keep most recent data
                all_data = all_data[-self.max_baseline_samples:]
            
            # Retrain model with updated data
            X = np.array(all_data)
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Update scaler and model
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            self.isolation_forest.fit(X_scaled)
            
            # Update baseline data
            self.baseline_data = all_data
            
            print(f"TouchPatternAgent incrementally updated with {len(normal_data)} new samples")
            return True
            
        except Exception as e:
            print(f"Error in incremental update for TouchPatternAgent: {e}")
            return False
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        try:
            model_data = {
                'isolation_forest': self.isolation_forest,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'baseline_data': self.baseline_data,
                'config': self.config
            }
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            print(f"Error saving TouchPatternAgent model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            model_data = joblib.load(filepath)
            self.isolation_forest = model_data['isolation_forest']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.baseline_data = model_data.get('baseline_data', [])
            self.config.update(model_data.get('config', {}))
            return True
        except Exception as e:
            print(f"Error loading TouchPatternAgent model: {e}")
            return False