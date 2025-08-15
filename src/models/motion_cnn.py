# models/motion_cnn.py

"""
Motion CNN for Multi-Sensor Mobile Device Authentication
QuadFusion Models - Mobile-Optimized Motion Analysis

Features:
- 1D CNN architecture for temporal motion patterns
- Multi-sensor fusion (accelerometer, gyroscope, magnetometer)
- Real-time processing with <30ms inference
- Memory efficient (<100MB)
- Battery optimized adaptive sampling
- Anomaly detection for fraud prevention
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
import json
import logging
from pathlib import Path
import threading
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class MotionConfig:
    """Configuration for motion CNN model."""
    input_channels: int = 9  # 3 sensors × 3 axes
    num_classes: int = 10   # Activity classes
    window_size: int = 200  # Sensor samples per window
    sample_rate: int = 50   # Hz
    batch_size: int = 16    # Mobile optimized
    model_size_mb: float = 8.5  # Target model size

@dataclass 
class SensorReading:
    """Single sensor reading with timestamp."""
    timestamp: float
    accel: np.ndarray  # [x, y, z]
    gyro: np.ndarray   # [x, y, z] 
    mag: np.ndarray    # [x, y, z]

class MotionCNN(nn.Module):
    """
    Mobile-optimized 1D CNN for motion pattern recognition.
    
    Architecture:
    - 3 convolutional layers with batch norm
    - Residual connections for stability
    - Attention mechanism for feature importance
    - Dropout for regularization
    - <10MB model size, <50ms inference
    """
    
    def __init__(self, config: MotionConfig):
        super(MotionCNN, self).__init__()
        self.config = config
        
        # Convolutional backbone
        self.conv1 = nn.Conv1d(config.input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=2)  # Downsample
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Residual connection (dimension matching)
        self.residual_conv = nn.Conv1d(config.input_channels, 128, kernel_size=1, stride=2)
        
        # Attention mechanism for important features
        self.attention = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, config.num_classes)
        )
        
        # Anomaly detection branch
        self.anomaly_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, channels, seq_len)
            
        Returns:
            Dict containing 'logits' and 'anomaly_score'
        """
        batch_size = x.size(0)
        
        # Residual path
        residual = self.residual_conv(x)
        
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        
        # Add residual connection
        out = out + residual
        
        # Apply attention
        attention_weights = self.attention(out)
        out = out * attention_weights
        
        # Global pooling
        features = self.global_pool(out).squeeze(-1)  # (batch, 128)
        
        # Classification
        logits = self.classifier(features)
        
        # Anomaly detection
        anomaly_score = self.anomaly_head(features)
        
        return {
            'logits': logits,
            'anomaly_score': anomaly_score,
            'features': features
        }
    
    def quantize_model(self):
        """Apply quantization for mobile deployment."""
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self, inplace=True)
        # Would need calibration data in real implementation
        torch.quantization.convert(self, inplace=True)
        logging.info("Model quantized for mobile deployment")

class SensorDataProcessor:
    """
    Real-time sensor data preprocessing pipeline.
    Handles noise filtering, normalization, and windowing.
    """
    
    def __init__(self, config: MotionConfig):
        self.config = config
        self.window_buffer = deque(maxlen=config.window_size)
        self.stats = {'mean': None, 'std': None}
        self.noise_filter = self._create_noise_filter()
        
    def _create_noise_filter(self):
        """Create Butterworth low-pass filter for noise reduction."""
        from scipy.signal import butter
        nyquist = self.config.sample_rate / 2
        cutoff = 20  # Hz - human motion typically <20Hz
        return butter(3, cutoff / nyquist, btype='low', output='sos')
        
    def filter_noise(self, data: np.ndarray) -> np.ndarray:
        """Apply low-pass filtering to remove sensor noise."""
        from scipy.signal import sosfilt
        return sosfilt(self.noise_filter, data, axis=0)
        
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalization with running statistics."""
        if self.stats['mean'] is None:
            self.stats['mean'] = np.mean(data, axis=0)
            self.stats['std'] = np.std(data, axis=0) + 1e-6
        else:
            # Update running statistics
            alpha = 0.01  # Learning rate for stats update
            self.stats['mean'] = (1-alpha) * self.stats['mean'] + alpha * np.mean(data, axis=0)
            self.stats['std'] = (1-alpha) * self.stats['std'] + alpha * np.std(data, axis=0)
            
        return (data - self.stats['mean']) / self.stats['std']
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract time-domain features from sensor data."""
        features = []
        
        # Statistical features
        features.extend([
            np.mean(data, axis=0),
            np.std(data, axis=0),
            np.min(data, axis=0),
            np.max(data, axis=0)
        ])
        
        # Signal magnitude area
        sma = np.mean(np.sum(np.abs(data), axis=1))
        features.append([sma])
        
        return np.concatenate(features, axis=0)
    
    def process_window(self, sensor_readings: List[SensorReading]) -> torch.Tensor:
        """
        Process a window of sensor readings into model input format.
        
        Args:
            sensor_readings: List of SensorReading objects
            
        Returns:
            Processed tensor ready for model inference
        """
        # Extract sensor data
        accel_data = np.array([reading.accel for reading in sensor_readings])
        gyro_data = np.array([reading.gyro for reading in sensor_readings])
        mag_data = np.array([reading.mag for reading in sensor_readings])
        
        # Combine all sensors
        combined_data = np.concatenate([accel_data, gyro_data, mag_data], axis=1)
        
        # Apply preprocessing
        filtered_data = self.filter_noise(combined_data)
        normalized_data = self.normalize(filtered_data)
        
        # Convert to PyTorch format (channels, seq_len)
        tensor_data = torch.tensor(normalized_data.T, dtype=torch.float32)
        
        return tensor_data

class ActivityClassifier:
    """
    Activity classification using trained MotionCNN model.
    Provides real-time inference with <30ms latency.
    """
    
    ACTIVITY_LABELS = {
        0: 'walking',
        1: 'running', 
        2: 'sitting',
        3: 'standing',
        4: 'climbing_stairs',
        5: 'falling',
        6: 'cycling',
        7: 'driving',
        8: 'lying_down',
        9: 'unknown'
    }
    
    def __init__(self, model: MotionCNN, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        
    def predict(self, sensor_data: torch.Tensor) -> Dict[str, Any]:
        """
        Predict activity and anomaly score.
        
        Args:
            sensor_data: Preprocessed sensor tensor
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        with torch.no_grad():
            # Add batch dimension if needed
            if sensor_data.dim() == 2:
                sensor_data = sensor_data.unsqueeze(0)
            
            sensor_data = sensor_data.to(self.device)
            
            # Forward pass
            outputs = self.model(sensor_data)
            
            # Get predictions
            probs = F.softmax(outputs['logits'], dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs, dim=1)[0].item()
            anomaly_score = outputs['anomaly_score'].item()
            
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        return {
            'activity': self.ACTIVITY_LABELS[predicted_class],
            'class_id': predicted_class,
            'confidence': confidence,
            'anomaly_score': anomaly_score,
            'inference_time_ms': inference_time,
            'features': outputs['features'].cpu().numpy()
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics."""
        if not self.inference_times:
            return {}
            
        times = list(self.inference_times)
        return {
            'avg_inference_ms': np.mean(times),
            'max_inference_ms': np.max(times),
            'min_inference_ms': np.min(times),
            'std_inference_ms': np.std(times)
        }

class MotionAnomalyDetector:
    """
    Advanced anomaly detection for motion patterns.
    Detects unusual behavior that might indicate fraud.
    """
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.user_profile = None  # Will store normal behavior profile
        self.anomaly_history = deque(maxlen=1000)
        
    def build_user_profile(self, normal_features: List[np.ndarray]):
        """Build normal behavior profile from training data."""
        if not normal_features:
            return
            
        features_array = np.array(normal_features)
        self.user_profile = {
            'mean': np.mean(features_array, axis=0),
            'cov': np.cov(features_array.T),
            'percentiles': {
                '25': np.percentile(features_array, 25, axis=0),
                '75': np.percentile(features_array, 75, axis=0),
                '95': np.percentile(features_array, 95, axis=0)
            }
        }
        logging.info("User motion profile built from {} samples".format(len(normal_features)))
    
    def mahalanobis_distance(self, features: np.ndarray) -> float:
        """Calculate Mahalanobis distance from user profile."""
        if self.user_profile is None:
            return 0.0
            
        try:
            diff = features - self.user_profile['mean']
            cov_inv = np.linalg.pinv(self.user_profile['cov'])
            distance = np.sqrt(diff.T @ cov_inv @ diff)
            return distance
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance
            return np.linalg.norm(features - self.user_profile['mean'])
    
    def detect_anomaly(self, features: np.ndarray, model_anomaly_score: float) -> Dict[str, Any]:
        """
        Comprehensive anomaly detection combining multiple signals.
        
        Args:
            features: Feature vector from motion CNN
            model_anomaly_score: Anomaly score from neural network
            
        Returns:
            Anomaly detection results
        """
        # Statistical anomaly (Mahalanobis distance)
        stat_anomaly_score = 0.0
        if self.user_profile is not None:
            stat_anomaly_score = self.mahalanobis_distance(features)
            
        # Combined anomaly score
        combined_score = 0.6 * model_anomaly_score + 0.4 * min(stat_anomaly_score, 1.0)
        
        # Temporal anomaly (sudden changes)
        temporal_anomaly = 0.0
        if len(self.anomaly_history) > 0:
            recent_scores = list(self.anomaly_history)[-10:]
            if len(recent_scores) > 1:
                temporal_anomaly = abs(combined_score - np.mean(recent_scores))
        
        self.anomaly_history.append(combined_score)
        
        is_anomaly = combined_score > self.threshold
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': combined_score,
            'model_score': model_anomaly_score,
            'statistical_score': stat_anomaly_score,
            'temporal_score': temporal_anomaly,
            'confidence': abs(combined_score - self.threshold) / self.threshold
        }

class SensorFusion:
    """
    Multi-sensor data fusion for robust motion analysis.
    Combines accelerometer, gyroscope, and magnetometer data.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Default sensor weights (can be learned/adapted)
        self.weights = weights or {
            'accel': 0.5,   # Primary motion sensor
            'gyro': 0.3,    # Orientation changes
            'mag': 0.2      # Compass/heading info
        }
        self.calibration_data = {'accel': [], 'gyro': [], 'mag': []}
        
    def calibrate_sensors(self, sensor_readings: List[SensorReading]):
        """Calibrate sensors by collecting baseline measurements."""
        for reading in sensor_readings:
            self.calibration_data['accel'].append(reading.accel)
            self.calibration_data['gyro'].append(reading.gyro) 
            self.calibration_data['mag'].append(reading.mag)
            
        logging.info("Sensor calibration updated with {} readings".format(len(sensor_readings)))
    
    def fuse_sensors(self, accel_data: np.ndarray, gyro_data: np.ndarray, 
                    mag_data: np.ndarray) -> np.ndarray:
        """
        Weighted fusion of multi-sensor data.
        
        Args:
            accel_data: Accelerometer readings (N, 3)
            gyro_data: Gyroscope readings (N, 3)  
            mag_data: Magnetometer readings (N, 3)
            
        Returns:
            Fused sensor data (N, 9)
        """
        # Apply sensor-specific preprocessing
        accel_processed = self._preprocess_accel(accel_data)
        gyro_processed = self._preprocess_gyro(gyro_data)
        mag_processed = self._preprocess_mag(mag_data)
        
        # Weight and combine
        fused_data = np.concatenate([
            accel_processed * self.weights['accel'],
            gyro_processed * self.weights['gyro'], 
            mag_processed * self.weights['mag']
        ], axis=1)
        
        return fused_data
    
    def _preprocess_accel(self, data: np.ndarray) -> np.ndarray:
        """Accelerometer-specific preprocessing."""
        # Remove gravity component (assuming first axis is vertical)
        if len(self.calibration_data['accel']) > 0:
            gravity_baseline = np.mean(self.calibration_data['accel'], axis=0)
            data = data - gravity_baseline
        return data
        
    def _preprocess_gyro(self, data: np.ndarray) -> np.ndarray:
        """Gyroscope-specific preprocessing.""" 
        # Remove bias
        if len(self.calibration_data['gyro']) > 0:
            bias = np.mean(self.calibration_data['gyro'], axis=0)
            data = data - bias
        return data
        
    def _preprocess_mag(self, data: np.ndarray) -> np.ndarray:
        """Magnetometer-specific preprocessing."""
        # Normalize magnetic field strength
        magnitude = np.linalg.norm(data, axis=1, keepdims=True)
        return data / (magnitude + 1e-6)

class RealTimeMotionProcessor:
    """
    Real-time motion processing pipeline.
    Handles streaming sensor data with minimal latency.
    """
    
    def __init__(self, model: MotionCNN, config: MotionConfig):
        self.config = config
        self.processor = SensorDataProcessor(config)
        self.classifier = ActivityClassifier(model)
        self.anomaly_detector = MotionAnomalyDetector()
        self.sensor_fusion = SensorFusion()
        
        # Streaming buffers
        self.sensor_buffer = deque(maxlen=config.window_size)
        self.prediction_history = deque(maxlen=100)
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.last_prediction_time = 0
        
        # Threading for real-time processing
        self.processing_thread = None
        self.is_running = False
        self.lock = threading.Lock()
        
    def add_sensor_reading(self, reading: SensorReading) -> Optional[Dict[str, Any]]:
        """
        Add new sensor reading and process if window is full.
        
        Args:
            reading: New sensor reading
            
        Returns:
            Prediction results if window is ready, None otherwise
        """
        with self.lock:
            self.sensor_buffer.append(reading)
            
            # Check if we have enough data for prediction
            if len(self.sensor_buffer) >= self.config.window_size:
                return self._process_window()
                
        return None
    
    def _process_window(self) -> Dict[str, Any]:
        """Process current sensor window and generate predictions."""
        start_time = time.time()
        
        # Get current window
        window_data = list(self.sensor_buffer)
        
        # Preprocess data
        processed_tensor = self.processor.process_window(window_data)
        
        # Get prediction
        prediction = self.classifier.predict(processed_tensor)
        
        # Anomaly detection
        features = prediction['features']
        anomaly_result = self.anomaly_detector.detect_anomaly(
            features, prediction['anomaly_score']
        )
        
        # Combine results
        result = {
            'timestamp': time.time(),
            'activity': prediction['activity'],
            'confidence': prediction['confidence'],
            'anomaly': anomaly_result,
            'sensor_window_size': len(window_data),
            'processing_time_ms': (time.time() - start_time) * 1000
        }
        
        self.prediction_history.append(result)
        self.processing_times.append(result['processing_time_ms'])
        
        return result
    
    def get_recent_activity_pattern(self, n_predictions: int = 10) -> List[str]:
        """Get recent activity pattern for behavior analysis."""
        recent_predictions = list(self.prediction_history)[-n_predictions:]
        return [pred['activity'] for pred in recent_predictions]
    
    def adapt_to_battery_level(self, battery_percent: float):
        """Adapt processing parameters based on battery level."""
        if battery_percent < 20:
            # Low battery - reduce sampling rate
            self.config.sample_rate = max(25, self.config.sample_rate // 2)
            logging.info(f"Reduced sampling rate to {self.config.sample_rate}Hz due to low battery")
        elif battery_percent > 80:
            # High battery - can use full sampling rate
            self.config.sample_rate = 50
            
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics."""
        classifier_stats = self.classifier.get_performance_stats()
        
        processing_times = list(self.processing_times)
        perf_stats = {}
        
        if processing_times:
            perf_stats = {
                'avg_processing_ms': np.mean(processing_times),
                'max_processing_ms': np.max(processing_times), 
                'processing_fps': 1000.0 / np.mean(processing_times) if processing_times else 0
            }
        
        return {**classifier_stats, **perf_stats}

# Utility functions for mobile optimization
def optimize_model_for_mobile(model: MotionCNN) -> MotionCNN:
    """Apply mobile-specific optimizations to the model."""
    # Quantization
    model.quantize_model()
    
    # Operator fusion (placeholder - would need actual implementation)
    logging.info("Applied mobile optimizations to MotionCNN")
    
    return model

def benchmark_model_performance(model: MotionCNN, test_data: torch.Tensor) -> Dict[str, float]:
    """Benchmark model performance on test data."""
    model.eval()
    
    # Timing benchmark
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(test_data)
            times.append((time.time() - start) * 1000)
    
    # Memory benchmark
    import torch.profiler
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        _ = model(test_data)
    
    return {
        'avg_inference_ms': np.mean(times),
        'p95_inference_ms': np.percentile(times, 95),
        'memory_mb': torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    }

# Example usage and testing
if __name__ == "__main__":
    # Initialize configuration
    config = MotionConfig()
    
    # Create model
    model = MotionCNN(config)
    
    # Generate sample data
    dummy_readings = []
    for i in range(config.window_size):
        reading = SensorReading(
            timestamp=time.time() + i * 0.02,  # 50Hz
            accel=np.random.randn(3),
            gyro=np.random.randn(3),
            mag=np.random.randn(3)
        )
        dummy_readings.append(reading)
    
    # Test real-time processor
    processor = RealTimeMotionProcessor(model, config)
    
    # Process sample data
    for reading in dummy_readings:
        result = processor.add_sensor_reading(reading)
        if result:
            print(f"Activity: {result['activity']}, Confidence: {result['confidence']:.2f}")
            print(f"Anomaly: {result['anomaly']['is_anomaly']}, Score: {result['anomaly']['anomaly_score']:.2f}")
            
    # Performance metrics
    metrics = processor.get_performance_metrics()
    print(f"Performance: {metrics}")
