# models/lstm_autoencoder.py
"""
LSTM Autoencoder for keystroke dynamics analysis.
Mobile-optimized implementation for typing behavior anomaly detection.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time
import json

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available, using simplified implementation")

class LSTMAutoencoder:
    """
    LSTM Autoencoder for analyzing keystroke dynamics patterns.
    Optimized for mobile deployment with quantization support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Model architecture parameters
        self.sequence_length = config.get('sequence_length', 20)
        self.feature_dim = config.get('feature_dim', 6)
        self.lstm_units = config.get('lstm_units', 32)
        self.encoding_dim = config.get('encoding_dim', 16)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 50)
        self.validation_split = config.get('validation_split', 0.2)
        
        # Mobile optimization parameters
        self.quantize_model = config.get('quantize_model', True)
        self.use_pruning = config.get('use_pruning', True)
        self.target_model_size_mb = config.get('target_model_size_mb', 5.0)
        
        # Anomaly detection parameters
        self.reconstruction_threshold = config.get('reconstruction_threshold', 0.1)
        self.adaptive_threshold = config.get('adaptive_threshold', True)
        
        # Model components
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.is_fitted = False
        
        # Training state
        self.training_history = {}
        self.reconstruction_errors = []
        self.threshold_value = self.reconstruction_threshold
        
        # Performance tracking
        self.training_time = 0.0
        self.inference_times = []
        
        if TENSORFLOW_AVAILABLE:
            self._build_model()
        else:
            self._build_simple_model()
    
    def _build_model(self):
        """Build LSTM autoencoder architecture"""
        try:
            # Encoder
            encoder_inputs = layers.Input(shape=(self.sequence_length, self.feature_dim))
            
            # LSTM layers for encoding
            x = layers.LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate)(encoder_inputs)
            x = layers.LSTM(self.encoding_dim, return_sequences=False, dropout=self.dropout_rate)(x)
            
            encoded = layers.Dense(self.encoding_dim, activation='relu')(x)
            
            # Create encoder model
            self.encoder = Model(encoder_inputs, encoded, name='encoder')
            
            # Decoder
            decoder_inputs = layers.Input(shape=(self.encoding_dim,))
            
            # Repeat vector to match sequence length
            x = layers.RepeatVector(self.sequence_length)(decoder_inputs)
            
            # LSTM layers for decoding
            x = layers.LSTM(self.encoding_dim, return_sequences=True, dropout=self.dropout_rate)(x)
            x = layers.LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate)(x)
            
            decoded = layers.TimeDistributed(layers.Dense(self.feature_dim))(x)
            
            # Create decoder model
            self.decoder = Model(decoder_inputs, decoded, name='decoder')
            
            # Complete autoencoder
            autoencoder_outputs = self.decoder(self.encoder(encoder_inputs))
            self.autoencoder = Model(encoder_inputs, autoencoder_outputs, name='autoencoder')
            
            # Compile model
            optimizer = Adam(learning_rate=self.learning_rate)
            self.autoencoder.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            print(f"LSTM Autoencoder built: {self.autoencoder.count_params()} parameters")
            
        except Exception as e:
            print(f"Model building error: {e}")
            self._build_simple_model()
    
    def _build_simple_model(self):
        """Build simplified model when TensorFlow is not available"""
        print("Building simplified LSTM Autoencoder (no TensorFlow)")
        
        # Simple statistical model for keystroke analysis
        self.simple_model = {
            'mean_features': None,
            'std_features': None,
            'feature_correlations': None
        }
        
        self.is_simple_model = True
    
    def _prepare_data(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare input data for training/inference
        
        Args:
            X: Input sequences
            
        Returns:
            Prepared data
        """
        X = np.array(X)
        
        # Ensure correct shape
        if len(X.shape) == 2:
            # Reshape to (batch_size, sequence_length, features)
            if X.shape[1] == self.sequence_length * self.feature_dim:
                X = X.reshape(-1, self.sequence_length, self.feature_dim)
        
        # Normalize data
        X = self._normalize_sequences(X)
        
        return X.astype(np.float32)
    
    def _normalize_sequences(self, X: np.ndarray) -> np.ndarray:
        """Normalize keystroke sequences"""
        try:
            # Normalize each feature across the sequence
            X_normalized = np.zeros_like(X)
            
            for i in range(X.shape[0]):  # For each sequence
                sequence = X[i]
                
                # Normalize each feature dimension
                for j in range(X.shape[2]):  # For each feature
                    feature_values = sequence[:, j]
                    
                    if np.std(feature_values) > 0:
                        X_normalized[i, :, j] = (feature_values - np.mean(feature_values)) / np.std(feature_values)
                    else:
                        X_normalized[i, :, j] = feature_values
            
            return X_normalized
            
        except Exception as e:
            print(f"Normalization error: {e}")
            return X
    
    def fit(self, X: np.ndarray, validation_data: Optional[np.ndarray] = None) -> 'LSTMAutoencoder':
        """
        Train the LSTM autoencoder
        
        Args:
            X: Training sequences
            validation_data: Optional validation data
            
        Returns:
            Self
        """
        start_time = time.time()
        
        try:
            print(f"Training LSTM Autoencoder on {X.shape[0]} sequences")
            
            # Prepare data
            X_prepared = self._prepare_data(X)
            
            if TENSORFLOW_AVAILABLE and self.autoencoder is not None:
                # TensorFlow training
                callbacks = [
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
                ]
                
                # Prepare validation data
                validation_data_prepared = None
                if validation_data is not None:
                    validation_data_prepared = (self._prepare_data(validation_data), self._prepare_data(validation_data))
                
                # Train model
                history = self.autoencoder.fit(
                    X_prepared, X_prepared,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    validation_split=self.validation_split if validation_data_prepared is None else 0.0,
                    validation_data=validation_data_prepared,
                    callbacks=callbacks,
                    verbose=1
                )
                
                self.training_history = history.history
                
                # Calculate reconstruction errors for threshold
                reconstructions = self.autoencoder.predict(X_prepared, verbose=0)
                self.reconstruction_errors = np.mean(np.square(X_prepared - reconstructions), axis=(1, 2))
                
            else:
                # Simple model training
                self._fit_simple_model(X_prepared)
            
            # Set adaptive threshold
            if self.adaptive_threshold and len(self.reconstruction_errors) > 0:
                self.threshold_value = np.percentile(self.reconstruction_errors, 95)
            
            # Apply mobile optimizations
            if TENSORFLOW_AVAILABLE and self.quantize_model:
                self._optimize_for_mobile()
            
            self.is_fitted = True
            self.training_time = time.time() - start_time
            
            print(f"Training completed in {self.training_time:.2f} seconds")
            print(f"Reconstruction threshold: {self.threshold_value:.4f}")
            
            return self
            
        except Exception as e:
            print(f"Training error: {e}")
            raise
    
    def _fit_simple_model(self, X: np.ndarray):
        """Fit simple statistical model"""
        try:
            # Calculate basic statistics
            self.simple_model['mean_features'] = np.mean(X, axis=(0, 1))
            self.simple_model['std_features'] = np.std(X, axis=(0, 1))
            
            # Calculate feature correlations
            X_flat = X.reshape(X.shape[0], -1)
            if X_flat.shape[1] > 1:
                correlation_matrix = np.corrcoef(X_flat.T)
                self.simple_model['feature_correlations'] = correlation_matrix
            
            # Simple reconstruction errors
            self.reconstruction_errors = []
            for i in range(X.shape[0]):
                sequence = X[i]
                error = np.mean(np.square(sequence - self.simple_model['mean_features']))
                self.reconstruction_errors.append(error)
            
            print("Simple model fitted successfully")
            
        except Exception as e:
            print(f"Simple model fitting error: {e}")
    
    def predict_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Predict reconstruction errors for input sequences
        
        Args:
            X: Input sequences
            
        Returns:
            Reconstruction errors
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        start_time = time.time()
        
        try:
            X_prepared = self._prepare_data(X)
            
            if TENSORFLOW_AVAILABLE and self.autoencoder is not None:
                # TensorFlow prediction
                reconstructions = self.autoencoder.predict(X_prepared, verbose=0)
                errors = np.mean(np.square(X_prepared - reconstructions), axis=(1, 2))
            else:
                # Simple model prediction
                errors = []
                for i in range(X_prepared.shape[0]):
                    sequence = X_prepared[i]
                    error = np.mean(np.square(sequence - self.simple_model['mean_features']))
                    errors.append(error)
                errors = np.array(errors)
            
            # Track inference time
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_times.append(inference_time)
            
            # Keep only recent inference times
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            return errors
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.array([1.0] * X.shape[0])  # Default to high error
    
    def predict_anomaly(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies in keystroke sequences
        
        Args:
            X: Input sequences
            
        Returns:
            Tuple of (anomaly_labels, anomaly_scores)
        """
        try:
            # Get reconstruction errors
            errors = self.predict_reconstruction_error(X)
            
            # Normalize errors to [0, 1] range
            if len(self.reconstruction_errors) > 0:
                max_error = max(np.max(self.reconstruction_errors), np.max(errors))
                normalized_scores = errors / max(max_error, 1e-8)
            else:
                normalized_scores = errors / np.max(errors) if np.max(errors) > 0 else errors
            
            # Determine anomalies based on threshold
            anomaly_labels = (errors > self.threshold_value).astype(int)
            
            return anomaly_labels, normalized_scores
            
        except Exception as e:
            print(f"Anomaly prediction error: {e}")
            return np.ones(X.shape[0]), np.ones(X.shape[0])
    
    def update_threshold(self, new_data: np.ndarray, percentile: float = 95) -> float:
        """
        Update anomaly threshold based on new data
        
        Args:
            new_data: New keystroke sequences
            percentile: Percentile for threshold calculation
            
        Returns:
            New threshold value
        """
        try:
            new_errors = self.predict_reconstruction_error(new_data)
            
            # Combine with existing errors
            all_errors = np.concatenate([self.reconstruction_errors, new_errors])
            
            # Calculate new threshold
            new_threshold = np.percentile(all_errors, percentile)
            
            # Update threshold and error history
            self.threshold_value = new_threshold
            self.reconstruction_errors = all_errors[-1000:]  # Keep recent errors
            
            print(f"Threshold updated to {new_threshold:.4f}")
            return new_threshold
            
        except Exception as e:
            print(f"Threshold update error: {e}")
            return self.threshold_value
    
    def _optimize_for_mobile(self):
        """Apply mobile-specific optimizations"""
        try:
            if not TENSORFLOW_AVAILABLE or self.autoencoder is None:
                return
            
            print("Applying mobile optimizations...")
            
            # Model quantization (simplified)
            if self.quantize_model:
                # This would typically involve TensorFlow Lite conversion
                # For now, just print that optimization would occur
                print("Model quantization would be applied here")
            
            # Model pruning (simplified)
            if self.use_pruning:
                print("Model pruning would be applied here")
            
        except Exception as e:
            print(f"Mobile optimization error: {e}")
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size information"""
        size_info = {
            'is_fitted': self.is_fitted,
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'lstm_units': self.lstm_units,
            'encoding_dim': self.encoding_dim
        }
        
        if TENSORFLOW_AVAILABLE and self.autoencoder is not None:
            size_info.update({
                'total_parameters': self.autoencoder.count_params(),
                'trainable_parameters': sum([tf.keras.backend.count_params(w) for w in self.autoencoder.trainable_weights]),
                'model_type': 'tensorflow'
            })
        else:
            size_info.update({
                'model_type': 'simple_statistical',
                'total_parameters': self.sequence_length * self.feature_dim * 2  # Rough estimate
            })
        
        return size_info
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'training_time_seconds': self.training_time,
            'is_fitted': self.is_fitted,
            'threshold_value': self.threshold_value,
            'training_samples': len(self.reconstruction_errors)
        }
        
        if self.inference_times:
            stats.update({
                'avg_inference_time_ms': np.mean(self.inference_times),
                'max_inference_time_ms': np.max(self.inference_times),
                'min_inference_time_ms': np.min(self.inference_times),
                'total_inferences': len(self.inference_times)
            })
        
        if self.training_history:
            final_loss = self.training_history.get('loss', [])
            if final_loss:
                stats['final_training_loss'] = final_loss[-1]
            
            val_loss = self.training_history.get('val_loss', [])
            if val_loss:
                stats['final_validation_loss'] = val_loss[-1]
        
        return stats
    
    def save_model(self, filepath: str) -> bool:
        """Save the model to file"""
        try:
            # Save metadata
            model_metadata = {
                'config': {
                    'sequence_length': self.sequence_length,
                    'feature_dim': self.feature_dim,
                    'lstm_units': self.lstm_units,
                    'encoding_dim': self.encoding_dim,
                    'learning_rate': self.learning_rate
                },
                'training_state': {
                    'is_fitted': self.is_fitted,
                    'threshold_value': self.threshold_value,
                    'reconstruction_errors': self.reconstruction_errors[-100:],  # Keep recent
                    'training_time': self.training_time
                },
                'performance_stats': self.get_performance_stats(),
                'model_size': self.get_model_size()
            }
            
            # Save metadata
            with open(f"{filepath}_metadata.json", 'w') as f:
                json.dump(model_metadata, f, default=str, indent=2)
            
            # Save TensorFlow model if available
            if TENSORFLOW_AVAILABLE and self.autoencoder is not None:
                self.autoencoder.save(f"{filepath}_autoencoder.h5")
                if self.encoder is not None:
                    self.encoder.save(f"{filepath}_encoder.h5")
                if self.decoder is not None:
                    self.decoder.save(f"{filepath}_decoder.h5")
            else:
                # Save simple model
                with open(f"{filepath}_simple_model.json", 'w') as f:
                    json.dump(self.simple_model, f, default=str, indent=2)
            
            print(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Save error: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load the model from file"""
        try:
            # Load metadata
            with open(f"{filepath}_metadata.json", 'r') as f:
                model_metadata = json.load(f)
            
            # Restore configuration
            config = model_metadata['config']
            self.sequence_length = config['sequence_length']
            self.feature_dim = config['feature_dim']
            self.lstm_units = config['lstm_units']
            self.encoding_dim = config['encoding_dim']
            self.learning_rate = config['learning_rate']
            
            # Restore training state
            training_state = model_metadata['training_state']
            self.is_fitted = training_state['is_fitted']
            self.threshold_value = training_state['threshold_value']
            self.reconstruction_errors = training_state['reconstruction_errors']
            self.training_time = training_state['training_time']
            
            # Load TensorFlow model if available
            if TENSORFLOW_AVAILABLE:
                try:
                    self.autoencoder = tf.keras.models.load_model(f"{filepath}_autoencoder.h5")
                    self.encoder = tf.keras.models.load_model(f"{filepath}_encoder.h5")
                    self.decoder = tf.keras.models.load_model(f"{filepath}_decoder.h5")
                    print("TensorFlow models loaded successfully")
                except:
                    print("TensorFlow models not found, checking for simple model")
                    self._load_simple_model(filepath)
            else:
                self._load_simple_model(filepath)
            
            print(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Load error: {e}")
            return False
    
    def _load_simple_model(self, filepath: str):
        """Load simple statistical model"""
        try:
            with open(f"{filepath}_simple_model.json", 'r') as f:
                self.simple_model = json.load(f)
            
            # Convert lists back to numpy arrays
            if self.simple_model['mean_features'] is not None:
                self.simple_model['mean_features'] = np.array(self.simple_model['mean_features'])
            if self.simple_model['std_features'] is not None:
                self.simple_model['std_features'] = np.array(self.simple_model['std_features'])
            if self.simple_model['feature_correlations'] is not None:
                self.simple_model['feature_correlations'] = np.array(self.simple_model['feature_correlations'])
            
            self.is_simple_model = True
            print("Simple model loaded successfully")
            
        except Exception as e:
            print(f"Simple model load error: {e}")

# Utility functions
def create_mobile_lstm_autoencoder(config: Dict[str, Any]) -> LSTMAutoencoder:
    """
    Create mobile-optimized LSTM autoencoder
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LSTMAutoencoder instance
    """
    # Mobile-optimized defaults
    mobile_config = {
        'sequence_length': 15,  # Shorter sequences
        'feature_dim': 6,
        'lstm_units': 24,       # Smaller LSTM
        'encoding_dim': 12,     # Smaller encoding
        'dropout_rate': 0.1,
        'learning_rate': 0.001,
        'batch_size': 16,       # Smaller batches
        'epochs': 30,           # Fewer epochs
        'quantize_model': True,
        'use_pruning': True,
        'target_model_size_mb': 3.0
    }
    
    mobile_config.update(config)
    return LSTMAutoencoder(mobile_config)
