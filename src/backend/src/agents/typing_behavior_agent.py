# agents/typing_behavior_agent.py
"""
Typing Behavior Agent for keystroke dynamics analysis.
Uses LSTM Autoencoder to learn typing patterns and detect anomalies.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import time
import json
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResult, RiskLevel

@dataclass
class KeystrokeEvent:
    """Represents a single keystroke event"""
    timestamp: float
    key_code: int
    action: str  # 'down' or 'up'
    pressure: float = 0.0

class TypingBehaviorAgent(BaseAgent):
    """
    Agent for analyzing typing patterns and detecting anomalies.
    Uses LSTM Autoencoder to model keystroke dynamics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("TypingBehaviorAgent", config)
        
        # Typing analysis parameters
        self.sequence_length = config.get('sequence_length', 20)  # Number of keystrokes to analyze
        self.feature_dim = config.get('feature_dim', 6)  # Features per keystroke
        self.min_typing_speed = config.get('min_typing_speed', 0.1)  # Minimum WPM
        
        # Model parameters
        self.lstm_units = config.get('lstm_units', 32)
        self.encoding_dim = config.get('encoding_dim', 16)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.reconstruction_threshold = config.get('reconstruction_threshold', 0.1)
        
        # Initialize model components
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.scaler = StandardScaler()
        
        # Keystroke buffers
        self.current_sequence = []
        self.keystroke_buffer = []
        self.reconstruction_errors = []
    
    def _build_lstm_autoencoder(self) -> Tuple[Model, Model, Model]:
        """
        Build LSTM Autoencoder for keystroke dynamics.
        
        Returns:
            Tuple of (autoencoder, encoder, decoder) models
        """
        # Input layer
        input_layer = layers.Input(shape=(self.sequence_length, self.feature_dim))
        
        # Encoder
        encoded = layers.LSTM(self.lstm_units, return_sequences=True)(input_layer)
        encoded = layers.LSTM(self.encoding_dim, return_sequences=False)(encoded)
        
        # Decoder
        decoded = layers.RepeatVector(self.sequence_length)(encoded)
        decoded = layers.LSTM(self.encoding_dim, return_sequences=True)(decoded)
        decoded = layers.LSTM(self.lstm_units, return_sequences=True)(decoded)
        decoded = layers.TimeDistributed(layers.Dense(self.feature_dim))(decoded)
        
        # Create models
        autoencoder = Model(input_layer, decoded, name='keystroke_autoencoder')
        encoder = Model(input_layer, encoded, name='keystroke_encoder')
        
        # Decoder model (for generating reconstructions)
        encoded_input = layers.Input(shape=(self.encoding_dim,))
        decoded_layer = autoencoder.layers[-4](encoded_input)
        for layer in autoencoder.layers[-3:]:
            decoded_layer = layer(decoded_layer)
        decoder = Model(encoded_input, decoded_layer, name='keystroke_decoder')
        
        # Compile autoencoder
        autoencoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder, encoder, decoder
    
    def capture_data(self, sensor_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Capture and preprocess keystroke data.
        
        Expected sensor_data format:
        {
            'keystroke_events': [
                {
                    'timestamp': float,
                    'key_code': int,
                    'action': str,  # 'down' or 'up'
                    'pressure': float  # optional
                }
            ]
        }
        
        Returns:
            Sequence of keystroke features or None if sequence not complete
        """
        try:
            keystroke_events = sensor_data.get('keystroke_events', [])
            if not keystroke_events:
                return None
            
            # Convert to KeystrokeEvent objects
            events = [
                KeystrokeEvent(
                    timestamp=event['timestamp'],
                    key_code=event['key_code'],
                    action=event['action'],
                    pressure=event.get('pressure', 0.0)
                ) for event in keystroke_events
            ]
            
            # Add events to buffer
            self.keystroke_buffer.extend(events)
            
            # Extract features from complete keystroke sequences
            sequences = self._extract_keystroke_sequences()
            
            if sequences:
                # Return the most recent complete sequence
                return sequences[-1]
            
            return None
            
        except Exception as e:
            print(f"Error capturing keystroke data: {e}")
            return None
    
    def _extract_keystroke_sequences(self) -> List[np.ndarray]:
        """
        Extract keystroke feature sequences from buffered events.
        
        Returns:
            List of feature sequences ready for model input
        """
        sequences = []
        
        # Remove old events (keep last 1000 keystrokes)
        if len(self.keystroke_buffer) > 1000:
            self.keystroke_buffer = self.keystroke_buffer[-1000:]
        
        # Group events into key press pairs (down + up)
        key_presses = self._group_keystrokes()
        
        if len(key_presses) < self.sequence_length:
            return sequences
        
        # Create sliding windows of keystroke sequences
        for i in range(len(key_presses) - self.sequence_length + 1):
            sequence_presses = key_presses[i:i + self.sequence_length]
            feature_sequence = self._extract_sequence_features(sequence_presses)
            
            if feature_sequence is not None:
                sequences.append(feature_sequence)
        
        return sequences
    
    def _group_keystrokes(self) -> List[Tuple[KeystrokeEvent, KeystrokeEvent]]:
        """
        Group keystroke events into (key_down, key_up) pairs.
        
        Returns:
            List of (down_event, up_event) tuples
        """
        key_presses = []
        pending_downs = {}  # key_code -> down_event
        
        for event in self.keystroke_buffer:
            if event.action == 'down':
                pending_downs[event.key_code] = event
            elif event.action == 'up' and event.key_code in pending_downs:
                down_event = pending_downs.pop(event.key_code)
                key_presses.append((down_event, event))
        
        return key_presses
    
    def _extract_sequence_features(self, key_presses: List[Tuple[KeystrokeEvent, KeystrokeEvent]]) -> Optional[np.ndarray]:
        """
        Extract features from a sequence of key presses.
        
        Args:
            key_presses: List of (down_event, up_event) tuples
            
        Returns:
            Feature array of shape (sequence_length, feature_dim)
        """
        if len(key_presses) != self.sequence_length:
            return None
        
        features = []
        
        for i, (down_event, up_event) in enumerate(key_presses):
            # Feature 1: Dwell time (how long key is held down)
            dwell_time = up_event.timestamp - down_event.timestamp
            
            # Feature 2: Flight time (time between key releases)
            if i > 0:
                prev_up = key_presses[i-1][1]
                flight_time = down_event.timestamp - prev_up.timestamp
            else:
                flight_time = 0.0
            
            # Feature 3: Key code (normalized)
            key_code_normalized = down_event.key_code / 255.0
            
            # Feature 4: Average pressure during key press
            avg_pressure = (down_event.pressure + up_event.pressure) / 2.0
            
            # Feature 5: Pressure change during key press
            pressure_change = abs(up_event.pressure - down_event.pressure)
            
            # Feature 6: Inter-keystroke interval (time from this key down to next key down)
            if i < len(key_presses) - 1:
                next_down = key_presses[i+1][0]
                inter_keystroke_interval = next_down.timestamp - down_event.timestamp
            else:
                inter_keystroke_interval = dwell_time  # Use dwell time as fallback
            
            keystroke_features = np.array([
                dwell_time,
                flight_time,
                key_code_normalized,
                avg_pressure,
                pressure_change,
                inter_keystroke_interval
            ])
            
            features.append(keystroke_features)
        
        return np.array(features)
    
    def train_initial(self, training_data: List[np.ndarray]) -> bool:
        """
        Train initial LSTM Autoencoder on normal typing patterns.
        
        Args:
            training_data: List of keystroke sequences (each of shape sequence_length x feature_dim)
            
        Returns:
            True if training successful
        """
        try:
            if len(training_data) < 50:
                print("Insufficient training data for TypingBehaviorAgent")
                return False
            
            # Convert to numpy array
            X = np.array(training_data)
            print(f"Training data shape: {X.shape}")
            
            # Handle invalid values
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Normalize features
            X_reshaped = X.reshape(-1, self.feature_dim)
            self.scaler.fit(X_reshaped)
            X_scaled = self.scaler.transform(X_reshaped).reshape(X.shape)
            
            # Build model
            self.autoencoder, self.encoder, self.decoder = self._build_lstm_autoencoder()
            
            # Train autoencoder
            history = self.autoencoder.fit(
                X_scaled, X_scaled,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0,
                shuffle=True
            )
            
            # Calculate reconstruction threshold based on training data
            train_predictions = self.autoencoder.predict(X_scaled, verbose=0)
            reconstruction_errors = np.mean(np.square(X_scaled - train_predictions), axis=(1, 2))
            self.reconstruction_threshold = np.percentile(reconstruction_errors, 95)
            
            self.is_trained = True
            print(f"TypingBehaviorAgent trained on {len(training_data)} sequences")
            print(f"Reconstruction threshold: {self.reconstruction_threshold:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error training TypingBehaviorAgent: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> AgentResult:
        """
        Predict anomaly score for keystroke sequence.
        
        Args:
            features: Keystroke sequence features (sequence_length x feature_dim)
            
        Returns:
            AgentResult with reconstruction error as anomaly score
        """
        start_time = time.time()
        
        try:
            if not self.is_trained or self.autoencoder is None:
                return AgentResult(
                    agent_name=self.agent_name,
                    anomaly_score=0.0,
                    risk_level=RiskLevel.LOW,
                    confidence=0.0,
                    features_used=['untrained'],
                    processing_time_ms=0,
                    metadata={'error': 'Model not trained'}
                )
            
            # Ensure correct shape
            if features.shape != (self.sequence_length, self.feature_dim):
                return AgentResult(
                    agent_name=self.agent_name,
                    anomaly_score=0.0,
                    risk_level=RiskLevel.LOW,
                    confidence=0.0,
                    features_used=['invalid_shape'],
                    processing_time_ms=0,
                    metadata={'error': f'Invalid shape: {features.shape}'}
                )
            
            # Normalize features
            features_flat = features.reshape(-1, self.feature_dim)
            features_scaled = self.scaler.transform(features_flat).reshape(features.shape)
            features_batch = features_scaled[np.newaxis, :, :]  # Add batch dimension
            
            # Get reconstruction
            reconstruction = self.autoencoder.predict(features_batch, verbose=0)[0]
            
            # Calculate reconstruction error
            reconstruction_error = np.mean(np.square(features_scaled - reconstruction))
            
            # Normalize anomaly score (0-1 range)
            anomaly_score = min(1.0, reconstruction_error / (self.reconstruction_threshold + 1e-6))
            
            # Calculate confidence (higher error = higher confidence in anomaly detection)
            confidence = min(1.0, reconstruction_error / 0.5)
            
            # Determine risk level
            risk_level = self.get_risk_level_from_score(anomaly_score)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Extract typing characteristics for metadata
            dwell_times = features[:, 0]
            flight_times = features[:, 1]
            inter_keystroke_intervals = features[:, 5]
            
            metadata = {
                'reconstruction_error': float(reconstruction_error),
                'reconstruction_threshold': float(self.reconstruction_threshold),
                'avg_dwell_time': float(np.mean(dwell_times)),
                'avg_flight_time': float(np.mean(flight_times[flight_times > 0])),
                'avg_inter_keystroke_interval': float(np.mean(inter_keystroke_intervals)),
                'typing_rhythm_variance': float(np.var(inter_keystroke_intervals)),
                'model_type': 'lstm_autoencoder'
            }
            
            feature_names = [
                'dwell_times', 'flight_times', 'key_codes', 
                'pressures', 'pressure_changes', 'inter_keystroke_intervals'
            ]
            
            # Store reconstruction error for incremental learning
            self.reconstruction_errors.append(reconstruction_error)
            if len(self.reconstruction_errors) > 1000:
                self.reconstruction_errors = self.reconstruction_errors[-1000:]
            
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
        Update model with new normal typing patterns.
        
        Args:
            new_data: List of new keystroke sequences
            is_anomaly: Optional labels for supervised updates
            
        Returns:
            True if update successful
        """
        try:
            if not self.is_trained or len(new_data) < 5:
                return False
            
            # Filter to only normal data
            if is_anomaly is not None:
                normal_data = [data for data, anomaly in zip(new_data, is_anomaly) if not anomaly]
            else:
                # Use reconstruction error to filter likely normal sequences
                normal_data = []
                for sequence in new_data:
                    result = self.predict(sequence)
                    if result.anomaly_score < 0.5:  # Threshold for "normal"
                        normal_data.append(sequence)
            
            if len(normal_data) < 3:
                return False
            
            # Combine with existing baseline for fine-tuning
            X_new = np.array(normal_data)
            X_new = np.nan_to_num(X_new, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Normalize new data
            X_new_flat = X_new.reshape(-1, self.feature_dim)
            X_new_scaled = self.scaler.transform(X_new_flat).reshape(X_new.shape)
            
            # Fine-tune model with new data (few epochs to avoid catastrophic forgetting)
            self.autoencoder.fit(
                X_new_scaled, X_new_scaled,
                epochs=5,
                batch_size=16,
                verbose=0
            )
            
            # Update reconstruction threshold
            new_predictions = self.autoencoder.predict(X_new_scaled, verbose=0)
            new_errors = np.mean(np.square(X_new_scaled - new_predictions), axis=(1, 2))
            
            # Exponential moving average for threshold update
            alpha = 0.1
            new_threshold = np.percentile(new_errors, 90)
            self.reconstruction_threshold = (1 - alpha) * self.reconstruction_threshold + alpha * new_threshold
            
            print(f"TypingBehaviorAgent incrementally updated with {len(normal_data)} sequences")
            return True
            
        except Exception as e:
            print(f"Error in incremental update for TypingBehaviorAgent: {e}")
            return False
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        try:
            # Save Keras model
            if self.autoencoder is not None:
                self.autoencoder.save(f"{filepath}_autoencoder.h5")
            
            # Save other components
            model_data = {
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'reconstruction_threshold': self.reconstruction_threshold,
                'baseline_data': self.baseline_data,
                'config': self.config,
                'reconstruction_errors': self.reconstruction_errors
            }
            
            import joblib
            joblib.dump(model_data, f"{filepath}_components.joblib")
            
            return True
        except Exception as e:
            print(f"Error saving TypingBehaviorAgent model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            # Load Keras model
            self.autoencoder = tf.keras.models.load_model(f"{filepath}_autoencoder.h5")
            
            # Rebuild encoder and decoder from loaded autoencoder
            # Encoder
            encoder_output = self.autoencoder.layers[2].output  # After second LSTM
            self.encoder = Model(self.autoencoder.input, encoder_output)
            
            # Decoder (simplified - would need full reconstruction for complex cases)
            
            # Load other components
            import joblib
            model_data = joblib.load(f"{filepath}_components.joblib")
            
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.reconstruction_threshold = model_data['reconstruction_threshold']
            self.baseline_data = model_data.get('baseline_data', [])
            self.reconstruction_errors = model_data.get('reconstruction_errors', [])
            self.config.update(model_data.get('config', {}))
            
            return True
        except Exception as e:
            print(f"Error loading TypingBehaviorAgent model: {e}")
            return False