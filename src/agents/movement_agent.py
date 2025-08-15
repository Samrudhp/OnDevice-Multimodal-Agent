# agents/movement_agent.py
"""
Movement Agent for analyzing motion and sensor patterns.
Uses CNN for motion pattern analysis and statistical methods for sensor data.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from collections import deque
import time
import json

from .base_agent import BaseAgent, AgentResult, RiskLevel

class MovementAgent(BaseAgent):
    """
    Agent for analyzing movement and motion patterns.
    Uses CNN for motion analysis and statistical methods for sensor data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MovementAgent", config)
        
        # Motion analysis parameters
        self.sequence_length = config.get('sequence_length', 100)  # Number of sensor readings
        self.sensor_features = config.get('sensor_features', 6)  # ax, ay, az, gx, gy, gz
        self.sampling_rate = config.get('sampling_rate', 50)  # Hz
        self.analysis_window = config.get('analysis_window', 5.0)  # seconds
        
        # CNN model parameters
        self.conv_filters = config.get('conv_filters', [32, 64, 128])
        self.kernel_size = config.get('kernel_size', 3)
        self.pool_size = config.get('pool_size', 2)
        self.dense_units = config.get('dense_units', 128)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Anomaly detection parameters
        self.reconstruction_threshold = config.get('reconstruction_threshold', 0.1)
        self.movement_threshold = config.get('movement_threshold', 0.01)
        
        # Data buffers
        self.sensor_buffer = deque(maxlen=self.sequence_length * 2)
        self.baseline_patterns = []
        
        # Model components
        self.motion_cnn = None
        self.encoder = None
        self.decoder = None
        self.scaler = StandardScaler()
        
        # Training state
        self.is_trained = False
        self.last_update_time = time.time()
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the CNN motion analysis model"""
        try:
            # Build CNN autoencoder for motion pattern learning
            input_shape = (self.sequence_length, self.sensor_features)
            
            # Encoder
            encoder_input = layers.Input(shape=input_shape)
            x = encoder_input
            
            # Convolutional layers
            for filters in self.conv_filters:
                x = layers.Conv1D(filters, self.kernel_size, activation='relu', padding='same')(x)
                x = layers.MaxPooling1D(self.pool_size, padding='same')(x)
            
            # Flatten and dense layers
            x = layers.Flatten()(x)
            encoded = layers.Dense(self.dense_units, activation='relu')(x)
            
            self.encoder = Model(encoder_input, encoded)
            
            # Decoder
            decoder_input = layers.Input(shape=(self.dense_units,))
            x = decoder_input
            
            # Calculate the shape after encoding
            temp_shape = input_shape[0]
            for _ in self.conv_filters:
                temp_shape = temp_shape // self.pool_size
            
            # Dense layers
            x = layers.Dense(temp_shape * self.conv_filters[-1], activation='relu')(x)
            x = layers.Reshape((temp_shape, self.conv_filters[-1]))(x)
            
            # Deconvolutional layers
            for i, filters in enumerate(reversed(self.conv_filters[:-1])):
                x = layers.UpSampling1D(self.pool_size)(x)
                x = layers.Conv1D(filters, self.kernel_size, activation='relu', padding='same')(x)
            
            # Final layer
            x = layers.UpSampling1D(self.pool_size)(x)
            decoded = layers.Conv1D(self.sensor_features, self.kernel_size, activation='linear', padding='same')(x)
            
            self.decoder = Model(decoder_input, decoded)
            
            # Complete autoencoder
            autoencoder_output = self.decoder(self.encoder(encoder_input))
            self.motion_cnn = Model(encoder_input, autoencoder_output)
            
            # Compile model
            self.motion_cnn.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            print(f"[{self.agent_name}] CNN model initialized successfully")
            
        except Exception as e:
            print(f"[{self.agent_name}] Model initialization error: {e}")
            print(f"[{self.agent_name}] Running in simplified mode")
    
    def add_sensor_data(self, accelerometer: Tuple[float, float, float], 
                       gyroscope: Tuple[float, float, float], 
                       timestamp: float = None) -> None:
        """
        Add new sensor data to the buffer
        
        Args:
            accelerometer: (ax, ay, az) values
            gyroscope: (gx, gy, gz) values
            timestamp: Data timestamp (current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        try:
            # Combine sensor data
            sensor_reading = list(accelerometer) + list(gyroscope) + [timestamp]
            self.sensor_buffer.append(sensor_reading)
            
        except Exception as e:
            print(f"[{self.agent_name}] Error adding sensor data: {e}")
    
    def _extract_motion_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from motion data
        
        Args:
            data: Motion data array (sequence_length, sensor_features)
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {}
            
            # Split accelerometer and gyroscope data
            acc_data = data[:, :3]  # ax, ay, az
            gyro_data = data[:, 3:6]  # gx, gy, gz
            
            # Accelerometer features
            features['acc_mean_x'] = float(np.mean(acc_data[:, 0]))
            features['acc_mean_y'] = float(np.mean(acc_data[:, 1]))
            features['acc_mean_z'] = float(np.mean(acc_data[:, 2]))
            features['acc_std_x'] = float(np.std(acc_data[:, 0]))
            features['acc_std_y'] = float(np.std(acc_data[:, 1]))
            features['acc_std_z'] = float(np.std(acc_data[:, 2]))
            
            # Gyroscope features
            features['gyro_mean_x'] = float(np.mean(gyro_data[:, 0]))
            features['gyro_mean_y'] = float(np.mean(gyro_data[:, 1]))
            features['gyro_mean_z'] = float(np.mean(gyro_data[:, 2]))
            features['gyro_std_x'] = float(np.std(gyro_data[:, 0]))
            features['gyro_std_y'] = float(np.std(gyro_data[:, 1]))
            features['gyro_std_z'] = float(np.std(gyro_data[:, 2]))
            
            # Magnitude features
            acc_magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
            gyro_magnitude = np.sqrt(np.sum(gyro_data**2, axis=1))
            
            features['acc_magnitude_mean'] = float(np.mean(acc_magnitude))
            features['acc_magnitude_std'] = float(np.std(acc_magnitude))
            features['gyro_magnitude_mean'] = float(np.mean(gyro_magnitude))
            features['gyro_magnitude_std'] = float(np.std(gyro_magnitude))
            
            # Activity level
            features['activity_level'] = float(np.mean(acc_magnitude + gyro_magnitude))
            
            # Jerk (rate of acceleration change)
            acc_jerk = np.diff(acc_data, axis=0)
            features['jerk_magnitude'] = float(np.mean(np.sqrt(np.sum(acc_jerk**2, axis=1))))
            
            # Zero-crossing rate
            features['acc_zcr'] = float(np.mean([
                np.sum(np.diff(np.sign(acc_data[:, i])) != 0) for i in range(3)
            ]))
            
            return features
            
        except Exception as e:
            print(f"[{self.agent_name}] Feature extraction error: {e}")
            return {}
    
    def train_baseline(self, training_data: List[np.ndarray]) -> bool:
        """
        Train baseline movement patterns
        
        Args:
            training_data: List of motion sequences for training
            
        Returns:
            True if training successful
        """
        try:
            print(f"[{self.agent_name}] Training baseline with {len(training_data)} sequences")
            
            if len(training_data) < 10:
                print(f"[{self.agent_name}] Insufficient training data")
                return False
            
            # Prepare training data
            X_train = np.array(training_data)
            
            # Normalize data
            original_shape = X_train.shape
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
            X_train = X_train_scaled.reshape(original_shape)
            
            if self.motion_cnn is not None:
                # Train CNN autoencoder
                history = self.motion_cnn.fit(
                    X_train, X_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
                
                print(f"[{self.agent_name}] CNN training completed with final loss: {history.history['loss'][-1]:.4f}")
            
            # Store baseline patterns for statistical analysis
            self.baseline_patterns = training_data
            
            self.is_trained = True
            self.last_update_time = time.time()
            
            return True
            
        except Exception as e:
            print(f"[{self.agent_name}] Training error: {e}")
            return False
    
    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        """
        Analyze motion data for anomalies
        
        Args:
            data: Dictionary containing motion sensor data
            
        Returns:
            AgentResult with anomaly detection results
        """
        start_time = time.time()
        
        try:
            if not self.is_trained:
                return self._create_error_result("Agent not trained", start_time)
            
            # Extract motion data
            motion_sequence = data.get('motion_sequence')
            if motion_sequence is None:
                # Try to build from buffer
                if len(self.sensor_buffer) >= self.sequence_length:
                    buffer_data = list(self.sensor_buffer)[-self.sequence_length:]
                    motion_sequence = np.array(buffer_data)[:, :6]  # Remove timestamp
                else:
                    return self._create_error_result("Insufficient motion data", start_time)
            
            motion_sequence = np.array(motion_sequence)
            
            if motion_sequence.shape[0] < self.sequence_length:
                return self._create_error_result(f"Sequence too short: {motion_sequence.shape[0]} < {self.sequence_length}", start_time)
            
            # Take the most recent sequence_length readings
            if motion_sequence.shape[0] > self.sequence_length:
                motion_sequence = motion_sequence[-self.sequence_length:]
            
            # Normalize data
            motion_sequence_reshaped = motion_sequence.reshape(-1, motion_sequence.shape[-1])
            motion_sequence_scaled = self.scaler.transform(motion_sequence_reshaped)
            motion_sequence = motion_sequence_scaled.reshape(motion_sequence.shape)
            
            anomaly_scores = []
            features_used = []
            
            # CNN-based anomaly detection
            if self.motion_cnn is not None:
                try:
                    # Reshape for CNN input
                    X_test = motion_sequence.reshape(1, self.sequence_length, self.sensor_features)
                    
                    # Get reconstruction
                    reconstruction = self.motion_cnn.predict(X_test, verbose=0)
                    
                    # Calculate reconstruction error
                    reconstruction_error = np.mean(np.square(X_test - reconstruction))
                    cnn_anomaly_score = min(reconstruction_error / self.reconstruction_threshold, 1.0)
                    
                    anomaly_scores.append(cnn_anomaly_score)
                    features_used.append('cnn_reconstruction')
                    
                except Exception as e:
                    print(f"[{self.agent_name}] CNN analysis error: {e}")
            
            # Statistical feature analysis
            motion_features = self._extract_motion_features(motion_sequence)
            
            if motion_features and self.baseline_patterns:
                # Compare with baseline patterns
                baseline_features = []
                for baseline in self.baseline_patterns[:50]:  # Use subset for efficiency
                    baseline_feat = self._extract_motion_features(baseline)
                    if baseline_feat:
                        baseline_features.append(baseline_feat)
                
                if baseline_features:
                    feature_anomalies = []
                    
                    for feature_name, current_value in motion_features.items():
                        if feature_name in baseline_features[0]:
                            baseline_values = [bf[feature_name] for bf in baseline_features if feature_name in bf]
                            
                            if len(baseline_values) > 1:
                                baseline_mean = np.mean(baseline_values)
                                baseline_std = np.std(baseline_values)
                                
                                if baseline_std > 0:
                                    z_score = abs(current_value - baseline_mean) / baseline_std
                                    if z_score > 2.5:  # Anomaly threshold
                                        feature_anomalies.append(feature_name)
                                        anomaly_scores.append(min(z_score / 5.0, 1.0))
                    
                    features_used.extend(['statistical_features'])
            
            # Activity level check
            if motion_features:
                activity_level = motion_features.get('activity_level', 0)
                if activity_level < self.movement_threshold:
                    anomaly_scores.append(0.3)  # Low activity anomaly
                    features_used.append('activity_level')
            
            # Calculate overall anomaly score
            if anomaly_scores:
                overall_anomaly = np.mean(anomaly_scores)
            else:
                overall_anomaly = 0.0
            
            # Determine risk level
            if overall_anomaly > 0.7:
                risk_level = RiskLevel.HIGH
            elif overall_anomaly > 0.4:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Calculate confidence
            confidence = min(0.9, 0.5 + (len(self.baseline_patterns) / 100.0))
            
            # Create metadata
            metadata = {
                'sequence_length': int(motion_sequence.shape[0]),
                'features_extracted': len(motion_features),
                'baseline_patterns': len(self.baseline_patterns),
                'activity_level': motion_features.get('activity_level', 0) if motion_features else 0,
                'motion_features': motion_features
            }
            
            processing_time = (time.time() - start_time) * 1000
            
            return AgentResult(
                agent_name=self.agent_name,
                anomaly_score=float(overall_anomaly),
                risk_level=risk_level,
                confidence=float(confidence),
                features_used=features_used,
                processing_time_ms=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            return self._create_error_result(f"Analysis error: {str(e)}", start_time)
    
    def update_model(self, new_data: List[np.ndarray]) -> bool:
        """
        Update the model with new motion data
        
        Args:
            new_data: List of new motion sequences
            
        Returns:
            True if update successful
        """
        try:
            if not self.is_trained:
                print(f"[{self.agent_name}] Cannot update - agent not initially trained")
                return False
            
            # Add new patterns to baseline
            self.baseline_patterns.extend(new_data)
            
            # Keep only recent patterns to prevent memory issues
            max_patterns = 500
            if len(self.baseline_patterns) > max_patterns:
                self.baseline_patterns = self.baseline_patterns[-max_patterns:]
            
            # If enough new data, retrain
            if len(new_data) > 20 and self.motion_cnn is not None:
                # Incremental training
                all_data = self.baseline_patterns[-100:]  # Use recent data
                
                # Prepare data
                X_train = np.array(all_data)
                original_shape = X_train.shape
                X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
                X_train_scaled = self.scaler.transform(X_train_reshaped)
                X_train = X_train_scaled.reshape(original_shape)
                
                # Incremental training
                self.motion_cnn.fit(
                    X_train, X_train,
                    epochs=10,
                    batch_size=16,
                    verbose=0
                )
                
                print(f"[{self.agent_name}] Model updated with {len(new_data)} new patterns")
            
            self.last_update_time = time.time()
            return True
            
        except Exception as e:
            print(f"[{self.agent_name}] Update error: {e}")
            return False
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to file"""
        try:
            model_data = {
                'baseline_patterns': [pattern.tolist() for pattern in self.baseline_patterns],
                'is_trained': self.is_trained,
                'last_update_time': self.last_update_time,
                'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                'config': {
                    'sequence_length': self.sequence_length,
                    'sensor_features': self.sensor_features,
                    'reconstruction_threshold': self.reconstruction_threshold
                }
            }
            
            # Save configuration and baseline patterns
            with open(f"{filepath}_data.json", 'w') as f:
                json.dump(model_data, f)
            
            # Save CNN model if available
            if self.motion_cnn is not None:
                self.motion_cnn.save(f"{filepath}_cnn.h5")
            
            print(f"[{self.agent_name}] Model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"[{self.agent_name}] Save error: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from file"""
        try:
            # Load configuration and baseline patterns
            with open(f"{filepath}_data.json", 'r') as f:
                model_data = json.load(f)
            
            self.baseline_patterns = [np.array(pattern) for pattern in model_data['baseline_patterns']]
            self.is_trained = model_data['is_trained']
            self.last_update_time = model_data['last_update_time']
            
            # Restore scaler
            if model_data['scaler_mean'] is not None:
                self.scaler.mean_ = np.array(model_data['scaler_mean'])
                self.scaler.scale_ = np.array(model_data['scaler_scale'])
            
            # Load CNN model if available
            try:
                self.motion_cnn = tf.keras.models.load_model(f"{filepath}_cnn.h5")
                print(f"[{self.agent_name}] CNN model loaded successfully")
            except:
                print(f"[{self.agent_name}] CNN model not found, using baseline patterns only")
            
            print(f"[{self.agent_name}] Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"[{self.agent_name}] Load error: {e}")
            return False
    
    def _create_error_result(self, error_message: str, start_time: float) -> AgentResult:
        """Create an error result"""
        processing_time = (time.time() - start_time) * 1000
        return AgentResult(
            agent_name=self.agent_name,
            anomaly_score=1.0,
            risk_level=RiskLevel.HIGH,
            confidence=0.0,
            features_used=[],
            processing_time_ms=processing_time,
            metadata={'error': error_message}
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model state"""
        return {
            'agent_name': self.agent_name,
            'is_trained': self.is_trained,
            'baseline_patterns': len(self.baseline_patterns),
            'buffer_size': len(self.sensor_buffer),
            'last_update_time': self.last_update_time,
            'model_parameters': {
                'sequence_length': self.sequence_length,
                'sensor_features': self.sensor_features,
                'reconstruction_threshold': self.reconstruction_threshold,
                'sampling_rate': self.sampling_rate
            }
        }
