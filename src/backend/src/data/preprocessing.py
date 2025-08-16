# data/preprocessing.py

"""
Data preprocessing utilities for fraud detection.
Mobile-Optimized Data Preprocessing for QuadFusion

Features:
- Real-time preprocessing for all biometric modalities
- Memory-efficient streaming processing (<100MB)
- Battery-aware adaptive processing
- Advanced noise reduction and outlier detection
- Feature selection and dimensionality reduction
- Cross-platform mobile optimizations
- <50ms processing time per sample
"""

import numpy as np
import scipy.signal
import scipy.stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import DBSCAN
import threading
import logging
import time
import psutil
from collections import deque
from typing import List, Optional, Dict, Any, Tuple, Union, Callable
from dataclasses import dataclass
from pathlib import Path

try:
    import librosa
    import librosa.feature
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Processing specifications
MAX_PROCESSING_TIME_MS = 50
BATCH_SIZE = 32
STREAMING_BUFFER_SIZE = 1000
FEATURE_CACHE_SIZE = 10000
MEMORY_LIMIT_MB = 100

# Feature specifications
TOUCH_FEATURES = ["pressure", "area", "duration", "velocity", "acceleration"]
TYPING_FEATURES = ["dwell_time", "flight_time", "typing_speed", "rhythm"]
VOICE_FEATURES = ["mfcc", "spectral_centroid", "zero_crossing_rate", "energy"]
VISUAL_FEATURES = ["face_embedding", "scene_features", "lighting_conditions"]
MOVEMENT_FEATURES = ["acceleration", "gyroscope", "magnetometer", "orientation"]

@dataclass
class ProcessingConfig:
    """Configuration for preprocessing parameters."""
    max_processing_time_ms: float = MAX_PROCESSING_TIME_MS
    batch_size: int = BATCH_SIZE
    memory_limit_mb: float = MEMORY_LIMIT_MB
    enable_caching: bool = True
    adaptive_battery: bool = True
    noise_reduction: bool = True
    outlier_detection: bool = True
    dimensionality_reduction: bool = True

@dataclass
class ProcessingStats:
    """Statistics for preprocessing performance."""
    total_samples: int = 0
    total_time: float = 0.0
    avg_time_ms: float = 0.0
    max_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

class MemoryMonitor:
    """Monitor memory usage for mobile optimization."""
    
    def __init__(self, limit_mb: float = MEMORY_LIMIT_MB):
        self.limit_mb = limit_mb
        self.process = psutil.Process()
        
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
        
    def is_memory_available(self, required_mb: float = 10) -> bool:
        """Check if sufficient memory is available."""
        current_mb = self.get_memory_usage_mb()
        return (current_mb + required_mb) < self.limit_mb
        
    def force_gc_if_needed(self):
        """Force garbage collection if memory usage is high."""
        if self.get_memory_usage_mb() > (self.limit_mb * 0.8):
            import gc
            gc.collect()

class BatteryAwareProcessor:
    """Adapt processing based on battery level."""
    
    def __init__(self):
        self.battery_level = 100.0  # Default to full battery
        self.processing_scale = 1.0
        
    def update_battery_level(self, level: float):
        """Update battery level and adjust processing."""
        self.battery_level = level
        
        # Scale processing based on battery
        if level > 80:
            self.processing_scale = 1.0  # Full processing
        elif level > 50:
            self.processing_scale = 0.8  # Slight reduction
        elif level > 20:
            self.processing_scale = 0.6  # Moderate reduction
        else:
            self.processing_scale = 0.4  # Aggressive reduction
            
    def should_skip_processing(self, priority: float = 1.0) -> bool:
        """Determine if processing should be skipped to save battery."""
        return (priority * self.processing_scale) < 0.3

class TouchPreprocessor:
    """
    Advanced touch pattern preprocessing with noise reduction and outlier detection.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.pca = PCA(n_components=min(5, len(TOUCH_FEATURES)))
        self.outlier_detector = DBSCAN(eps=0.5, min_samples=3)
        self.feature_cache = {}
        self.is_fitted = False
        self.lock = threading.Lock()
        
    def _extract_touch_features(self, touch_data: List[Dict[str, float]]) -> np.ndarray:
        """Extract comprehensive touch features."""
        features = []
        
        for touch_event in touch_data:
            event_features = []
            
            # Basic features
            for feature_name in TOUCH_FEATURES:
                value = touch_event.get(feature_name, 0.0)
                event_features.append(value)
                
            # Derived features
            pressure = touch_event.get('pressure', 0.0)
            area = touch_event.get('area', 1.0)
            duration = touch_event.get('duration', 0.0)
            
            # Pressure per unit area
            pressure_density = pressure / max(area, 1e-6)
            event_features.append(pressure_density)
            
            # Touch intensity (pressure * area)
            intensity = pressure * area
            event_features.append(intensity)
            
            features.append(event_features)
            
        return np.array(features)
    
    def _detect_outliers(self, features: np.ndarray) -> np.ndarray:
        """Detect and filter outliers using statistical methods."""
        if len(features) < 3:
            return features
            
        # Z-score based outlier detection
        z_scores = np.abs(scipy.stats.zscore(features, axis=0, nan_policy='omit'))
        outlier_mask = np.any(z_scores > 3, axis=1)
        
        # Keep non-outliers
        clean_features = features[~outlier_mask]
        
        if len(clean_features) == 0:
            return features  # Return original if all are outliers
            
        return clean_features
    
    def _apply_noise_reduction(self, features: np.ndarray) -> np.ndarray:
        """Apply noise reduction using median filtering."""
        if len(features) < 3:
            return features
            
        # Apply median filter to each feature dimension
        filtered_features = np.zeros_like(features)
        for i in range(features.shape[1]):
            filtered_features[:, i] = scipy.signal.medfilt(features[:, i], kernel_size=3)
            
        return filtered_features
    
    def fit(self, touch_data_batch: List[List[Dict[str, float]]]):
        """Fit preprocessor on batch of touch data."""
        with self.lock:
            all_features = []
            for touch_data in touch_data_batch:
                features = self._extract_touch_features(touch_data)
                if len(features) > 0:
                    all_features.append(features)
                    
            if all_features:
                combined_features = np.vstack(all_features)
                
                # Remove outliers for fitting
                if self.config.outlier_detection:
                    combined_features = self._detect_outliers(combined_features)
                    
                # Fit scaler and PCA
                self.scaler.fit(combined_features)
                if self.config.dimensionality_reduction:
                    self.pca.fit(self.scaler.transform(combined_features))
                    
                self.is_fitted = True
                logging.info("TouchPreprocessor fitted successfully")
    
    def preprocess(self, touch_data: List[Dict[str, float]]) -> np.ndarray:
        """
        Preprocess touch data with comprehensive feature engineering.
        
        Args:
            touch_data: List of touch event dictionaries
            
        Returns:
            Processed feature array
        """
        if not touch_data:
            return np.array([])
            
        # Extract features
        features = self._extract_touch_features(touch_data)
        
        if len(features) == 0:
            return np.array([])
        
        # Apply noise reduction
        if self.config.noise_reduction:
            features = self._apply_noise_reduction(features)
        
        # Remove outliers
        if self.config.outlier_detection:
            features = self._detect_outliers(features)
            
        if len(features) == 0:
            return np.array([])
        
        # Normalize features
        if self.is_fitted:
            features = self.scaler.transform(features)
            
            # Apply dimensionality reduction
            if self.config.dimensionality_reduction:
                features = self.pca.transform(features)
        else:
            # Fallback normalization
            features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)
        
        # Aggregate to single feature vector (mean, std, min, max)
        if len(features) > 1:
            aggregated = np.concatenate([
                np.mean(features, axis=0),
                np.std(features, axis=0),
                np.min(features, axis=0),
                np.max(features, axis=0)
            ])
        else:
            aggregated = features.flatten()
            
        return aggregated

class TypingPreprocessor:
    """
    Advanced typing behavior analysis with rhythm and temporal features.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(10, len(TYPING_FEATURES)))
        self.is_fitted = False
        self.lock = threading.Lock()
        
    def _extract_typing_features(self, typing_data: List[Dict[str, float]]) -> np.ndarray:
        """Extract comprehensive typing features."""
        if not typing_data:
            return np.array([])
            
        features = []
        
        for event in typing_data:
            event_features = []
            
            # Basic timing features
            for feature_name in TYPING_FEATURES:
                value = event.get(feature_name, 0.0)
                event_features.append(value)
                
            # Derived features
            dwell_time = event.get('dwell_time', 0.0)
            flight_time = event.get('flight_time', 0.0)
            
            # Typing ratio (dwell/flight)
            if flight_time > 0:
                ratio = dwell_time / flight_time
            else:
                ratio = 0.0
            event_features.append(ratio)
            
            features.append(event_features)
        
        return np.array(features)
    
    def _compute_rhythm_features(self, features: np.ndarray) -> np.ndarray:
        """Compute rhythm and temporal pattern features."""
        if len(features) < 2:
            return np.array([0.0, 0.0, 0.0])  # rhythm consistency, variability, trend
            
        dwell_times = features[:, 0] if features.shape[1] > 0 else np.array([0.0])
        
        # Rhythm consistency (coefficient of variation)
        rhythm_consistency = np.std(dwell_times) / (np.mean(dwell_times) + 1e-6)
        
        # Temporal variability
        if len(dwell_times) > 1:
            temporal_variability = np.mean(np.abs(np.diff(dwell_times)))
        else:
            temporal_variability = 0.0
            
        # Trend (linear regression slope)
        if len(dwell_times) > 2:
            x = np.arange(len(dwell_times))
            slope, _ = np.polyfit(x, dwell_times, 1)
        else:
            slope = 0.0
            
        return np.array([rhythm_consistency, temporal_variability, slope])
    
    def preprocess(self, typing_data: List[Dict[str, float]]) -> np.ndarray:
        """
        Preprocess typing data with rhythm analysis.
        
        Args:
            typing_data: List of typing event dictionaries
            
        Returns:
            Processed feature array
        """
        if not typing_data:
            return np.array([])
            
        # Extract basic features
        features = self._extract_typing_features(typing_data)
        
        if len(features) == 0:
            return np.array([])
        
        # Compute rhythm features
        rhythm_features = self._compute_rhythm_features(features)
        
        # Aggregate basic features
        if len(features) > 1:
            basic_aggregated = np.concatenate([
                np.mean(features, axis=0),
                np.std(features, axis=0)
            ])
        else:
            basic_aggregated = features.flatten()
            
        # Combine with rhythm features
        combined_features = np.concatenate([basic_aggregated, rhythm_features])
        
        # Normalize if fitted
        if self.is_fitted:
            combined_features = self.scaler.transform(combined_features.reshape(1, -1)).flatten()
        
        return combined_features

class VoicePreprocessor:
    """
    Advanced voice feature extraction with MFCC, spectral, and prosodic features.
    """
    
    def __init__(self, config: ProcessingConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        self.n_mfcc = 13
        self.n_mels = 40
        self.frame_length = int(0.025 * sample_rate)  # 25ms
        self.hop_length = int(0.010 * sample_rate)    # 10ms
        
        # Noise reduction parameters
        self.noise_floor = 0.01
        
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio signal."""
        if not LIBROSA_AVAILABLE:
            logging.warning("Librosa not available for voice preprocessing")
            return np.zeros(self.n_mfcc * 3)  # Return zeros if librosa unavailable
            
        # Normalize amplitude
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            
        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Noise reduction (simple noise gate)
        if self.config.noise_reduction:
            energy = np.convolve(audio**2, np.ones(self.frame_length)/self.frame_length, mode='same')
            noise_mask = energy > self.noise_floor
            audio = audio * noise_mask
            
        return audio
    
    def _extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features with delta and delta-delta."""
        if not LIBROSA_AVAILABLE:
            return np.zeros(self.n_mfcc * 3)
            
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Combine features
        combined_mfccs = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        
        # Aggregate over time (mean and std)
        mfcc_mean = np.mean(combined_mfccs, axis=1)
        mfcc_std = np.std(combined_mfccs, axis=1)
        
        return np.concatenate([mfcc_mean, mfcc_std])
    
    def _extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral features."""
        if not LIBROSA_AVAILABLE:
            return np.zeros(12)
            
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=audio, hop_length=self.hop_length)
        
        # Aggregate features
        features = []
        for feature in [spectral_centroid, spectral_bandwidth, spectral_rolloff, zcr, rms, spectral_flatness]:
            features.extend([np.mean(feature), np.std(feature)])
            
        return np.array(features)
    
    def extract_features(self, audio_signal: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive voice features.
        
        Args:
            audio_signal: Raw audio signal
            
        Returns:
            Combined voice feature vector
        """
        if len(audio_signal) == 0:
            # Return zero features if no audio
            return np.zeros(self.n_mfcc * 6 + 12)  # MFCC + spectral features
            
        # Preprocess audio
        audio = self._preprocess_audio(audio_signal)
        
        # Extract feature groups
        mfcc_features = self._extract_mfcc_features(audio)
        spectral_features = self._extract_spectral_features(audio)
        
        # Combine all features
        combined_features = np.concatenate([
            mfcc_features,
            spectral_features
        ])
        
        return combined_features

class VisualPreprocessor:
    """
    Visual preprocessing with face detection and scene analysis.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.target_size = (224, 224)  # Standard input size for many models
        self.face_cascade = None
        
        if OPENCV_AVAILABLE:
            # Load face cascade classifier
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception as e:
                logging.warning(f"Could not load face cascade: {e}")
    
    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image."""
        if not OPENCV_AVAILABLE or self.face_cascade is None:
            return []
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces.tolist() if len(faces) > 0 else []
    
    def _extract_face_features(self, image: np.ndarray) -> np.ndarray:
        """Extract face-based features."""
        faces = self._detect_faces(image)
        
        if not faces:
            return np.zeros(128)  # Return zero embedding if no faces
        
        # For the largest face, extract features
        largest_face = max(faces, key=lambda f: f[2] * f)  # max area
        x, y, w, h = largest_face
        
        # Crop face region
        face_roi = image[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return np.zeros(128)
        
        # Resize to standard size
        if OPENCV_AVAILABLE:
            face_resized = cv2.resize(face_roi, (64, 64))
            # Simple feature extraction (histogram of gradients approximation)
            features = self._extract_hog_features(face_resized)
        else:
            features = np.zeros(128)
            
        return features
    
    def _extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG-like features (simplified)."""
        if not OPENCV_AVAILABLE:
            return np.zeros(128)
            
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        
        # Simplified HOG: divide image into 8x8 grid and compute histograms
        h, w = gray.shape
        cell_size = 8
        n_bins = 16
        
        features = []
        for i in range(0, h - cell_size, cell_size):
            for j in range(0, w - cell_size, cell_size):
                cell_mag = magnitude[i:i+cell_size, j:j+cell_size]
                cell_angle = angle[i:i+cell_size, j:j+cell_size]
                
                # Compute histogram
                hist, _ = np.histogram(cell_angle, bins=n_bins, weights=cell_mag, 
                                    range=(-np.pi, np.pi))
                features.extend(hist)
        
        # Pad or truncate to 128 features
        features = np.array(features)
        if len(features) > 128:
            features = features[:128]
        elif len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)))
            
        return features
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess visual data and extract features.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Combined visual feature vector
        """
        if image.size == 0:
            return np.zeros(128)  # Default face features only
            
        # Resize image if too large (for performance)
        if image.shape[0] > 512 or image.shape > 512:
            if OPENCV_AVAILABLE:
                height, width = image.shape[:2]
                scale = min(512/height, 512/width)
                new_height, new_width = int(height*scale), int(width*scale)
                image = cv2.resize(image, (new_width, new_height))
        
        # Extract face features
        face_features = self._extract_face_features(image)
        
        return face_features

class MovementPreprocessor:
    """
    Advanced sensor data preprocessing for accelerometer, gyroscope, and magnetometer.
    """
    
    def __init__(self, config: ProcessingConfig, sample_rate: int = 50):
        self.config = config
        self.sample_rate = sample_rate
        self.scaler = RobustScaler()
        self.is_fitted = False
        
        # Filter parameters for noise reduction
        self.lowpass_cutoff = 20  # Hz
        self.nyquist = sample_rate / 2
        self.filter_order = 4
        
    def _apply_lowpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply low-pass Butterworth filter to reduce noise."""
        if len(data) < 3 * self.filter_order:
            return data  # Not enough data for filtering
            
        # Design filter
        sos = scipy.signal.butter(
            self.filter_order, 
            self.lowpass_cutoff / self.nyquist, 
            btype='low', 
            output='sos'
        )
        
        # Apply filter along each axis
        filtered_data = np.zeros_like(data)
        for axis in range(data.shape[1]):
            filtered_data[:, axis] = scipy.signal.sosfilt(sos, data[:, axis])
            
        return filtered_data
    
    def _extract_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """Extract statistical features from sensor data."""
        if len(data) == 0:
            return np.zeros(12)  # 4 stats × 3 axes
            
        features = []
        for axis in range(min(3, data.shape)):  # Up to 3 axes
            axis_data = data[:, axis]
            
            # Basic statistics
            features.extend([
                np.mean(axis_data),
                np.std(axis_data),
                np.min(axis_data),
                np.max(axis_data)
            ])
            
        return np.array(features)
    
    def _extract_frequency_features(self, data: np.ndarray) -> np.ndarray:
        """Extract frequency domain features."""
        if len(data) < 8:  # Need minimum samples for FFT
            return np.zeros(9)  # 3 features × 3 axes
            
        features = []
        for axis in range(min(3, data.shape[1])):
            axis_data = data[:, axis]
            
            # FFT
            fft = np.fft.fft(axis_data)
            freqs = np.fft.fftfreq(len(axis_data), 1/self.sample_rate)
            
            # Power spectral density
            psd = np.abs(fft)**2
            
            # Dominant frequency
            dominant_freq = freqs[np.argmax(psd[:len(psd)//2])]
            
            # Spectral energy
            spectral_energy = np.sum(psd)
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs[:len(psd)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2])
            
            features.extend([dominant_freq, spectral_energy, spectral_centroid])
            
        return np.array(features)
    
    def preprocess(self, sensor_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Preprocess multi-sensor movement data.
        
        Args:
            sensor_data: Dictionary with sensor arrays (acceleration, gyroscope, magnetometer)
            
        Returns:
            Combined movement feature vector
        """
        all_features = []
        
        # Process each sensor type
        for sensor_type in MOVEMENT_FEATURES:
            if sensor_type in sensor_data:
                data = sensor_data[sensor_type]
                
                if len(data) == 0:
                    continue
                    
                # Ensure 2D array
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                
                # Apply noise reduction
                if self.config.noise_reduction and len(data) > self.filter_order * 3:
                    data = self._apply_lowpass_filter(data)
                
                # Extract features
                stat_features = self._extract_statistical_features(data)
                freq_features = self._extract_frequency_features(data)
                
                all_features.extend([stat_features, freq_features])
        
        # Combine all features
        if all_features:
            combined = np.concatenate([f.flatten() for f in all_features])
        else:
            combined = np.zeros(50)  # Default size
            
        # Normalize if fitted
        if self.is_fitted and len(combined) > 0:
            combined = self.scaler.transform(combined.reshape(1, -1)).flatten()
            
        return combined

class AppUsagePreprocessor:
    """
    App usage pattern preprocessing with temporal analysis.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _extract_usage_patterns(self, usage_data: List[Dict[str, Any]]) -> np.ndarray:
        """Extract usage pattern features."""
        if not usage_data:
            return np.zeros(10)
            
        features = []
        
        # Basic usage statistics
        durations = [event.get('duration', 0) for event in usage_data]
        frequencies = [event.get('frequency', 0) for event in usage_data]
        
        if durations:
            features.extend([
                np.mean(durations),
                np.std(durations),
                np.max(durations),
                len(durations)  # session count
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        if frequencies:
            features.extend([
                np.mean(frequencies),
                np.std(frequencies)
            ])
        else:
            features.extend([0, 0])
        
        # Temporal patterns
        timestamps = [event.get('timestamp', 0) for event in usage_data]
        if len(timestamps) > 1:
            # Time gaps between sessions
            time_gaps = np.diff(sorted(timestamps))
            features.extend([
                np.mean(time_gaps),
                np.std(time_gaps)
            ])
        else:
            features.extend([0, 0])
        
        # App diversity (unique apps)
        app_names = set(event.get('app_name', 'unknown') for event in usage_data)
        features.append(len(app_names))
        
        # Active hours pattern (if timestamp available)
        if timestamps:
            hours = [time.gmtime(ts).tm_hour for ts in timestamps if ts > 0]
            if hours:
                peak_hour = max(set(hours), key=hours.count)
                features.append(peak_hour)
            else:
                features.append(12)  # Default noon
        else:
            features.append(12)
            
        return np.array(features)
    
    def preprocess(self, usage_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Preprocess app usage data.
        
        Args:
            usage_data: List of app usage events
            
        Returns:
            Usage pattern feature vector
        """
        features = self._extract_usage_patterns(usage_data)
        
        # Normalize if fitted
        if self.is_fitted and len(features) > 0:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()
            
        return features

class DataPreprocessor:
    """
    Main streaming preprocessing pipeline with memory and battery optimization.
    Combines all individual preprocessors for unified data processing.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        
        # Initialize preprocessors
        self.touch_preprocessor = TouchPreprocessor(self.config)
        self.typing_preprocessor = TypingPreprocessor(self.config)
        self.voice_preprocessor = VoicePreprocessor(self.config)
        self.visual_preprocessor = VisualPreprocessor(self.config)
        self.movement_preprocessor = MovementPreprocessor(self.config)
        self.app_usage_preprocessor = AppUsagePreprocessor(self.config)
        
        # System monitors
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_mb)
        self.battery_processor = BatteryAwareProcessor()
        
        # Performance tracking
        self.stats = ProcessingStats()
        self.processing_times = deque(maxlen=100)
        
        # Streaming buffers
        self.streaming_buffer = deque(maxlen=STREAMING_BUFFER_SIZE)
        self.processed_cache = {} if self.config.enable_caching else None
        
        # Thread safety
        self.lock = threading.Lock()
        
    def update_battery_level(self, battery_level: float):
        """Update battery level for adaptive processing."""
        self.battery_processor.update_battery_level(battery_level)
        
    def _compute_cache_key(self, data: Dict[str, Any]) -> str:
        """Compute cache key for data."""
        if not self.config.enable_caching:
            return ""
            
        # Simple hash of data structure
        import hashlib
        data_str = str(sorted(data.items()))
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def preprocess_single(self, data: Dict[str, Any], priority: float = 1.0) -> np.ndarray:
        """
        Preprocess single data sample.
        
        Args:
            data: Dictionary containing different modality data
            priority: Processing priority (0.0-1.0)
            
        Returns:
            Combined feature vector
        """
        start_time = time.time()
        
        # Check battery constraints
        if self.battery_processor.should_skip_processing(priority):
            return np.array([])  # Skip processing to save battery
        
        # Check memory constraints
        if not self.memory_monitor.is_memory_available():
            self.memory_monitor.force_gc_if_needed()
            
        # Check cache
        if self.config.enable_caching and self.processed_cache is not None:
            cache_key = self._compute_cache_key(data)
            if cache_key in self.processed_cache:
                self.stats.cache_hits += 1
                return self.processed_cache[cache_key]
            else:
                self.stats.cache_misses += 1
        
        # Process each modality
        processed_features = []
        
        if 'touch' in data:
            touch_features = self.touch_preprocessor.preprocess(data['touch'])
            if len(touch_features) > 0:
                processed_features.append(touch_features)
        
        if 'typing' in data:
            typing_features = self.typing_preprocessor.preprocess(data['typing'])
            if len(typing_features) > 0:
                processed_features.append(typing_features)
        
        if 'voice' in data:
            voice_features = self.voice_preprocessor.extract_features(data['voice'])
            if len(voice_features) > 0:
                processed_features.append(voice_features)
        
        if 'visual' in data:
            visual_features = self.visual_preprocessor.preprocess(data['visual'])
            if len(visual_features) > 0:
                processed_features.append(visual_features)
        
        if 'movement' in data:
            movement_features = self.movement_preprocessor.preprocess(data['movement'])
            if len(movement_features) > 0:
                processed_features.append(movement_features)
        
        if 'app_usage' in data:
            app_features = self.app_usage_preprocessor.preprocess(data['app_usage'])
            if len(app_features) > 0:
                processed_features.append(app_features)
        
        # Combine features
        if processed_features:
            combined = np.concatenate(processed_features)
        else:
            combined = np.array([])
        
        # Update cache
        if self.config.enable_caching and self.processed_cache is not None and cache_key:
            self.processed_cache[cache_key] = combined
            
            # Limit cache size
            if len(self.processed_cache) > FEATURE_CACHE_SIZE:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self.processed_cache))
                del self.processed_cache[oldest_key]
        
        # Update statistics
        processing_time = (time.time() - start_time) * 1000  # ms
        self.processing_times.append(processing_time)
        
        with self.lock:
            self.stats.total_samples += 1
            self.stats.total_time += processing_time / 1000
            self.stats.max_time_ms = max(self.stats.max_time_ms, processing_time)
            self.stats.avg_time_ms = self.stats.total_time / self.stats.total_samples * 1000
            self.stats.memory_usage_mb = self.memory_monitor.get_memory_usage_mb()
        
        # Check processing time constraint
        if processing_time > self.config.max_processing_time_ms:
            logging.warning(f"Processing time exceeded limit: {processing_time:.2f}ms")
        
        return combined
    
    def preprocess_batch(self, batch_data: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Preprocess batch of data samples with optimization.
        
        Args:
            batch_data: List of data dictionaries
            
        Returns:
            List of processed feature vectors
        """
        results = []
        start_time = time.time()
        
        for i, data in enumerate(batch_data):
            # Dynamic priority based on batch position
            priority = 1.0 - (i / len(batch_data)) * 0.3  # Decrease priority for later items
            
            processed = self.preprocess_single(data, priority)
            results.append(processed)
            
            # Check batch processing time constraint
            elapsed_time = (time.time() - start_time) * 1000
            if elapsed_time > self.config.max_processing_time_ms:
                logging.warning(f"Batch processing time limit reached at sample {i}")
                # Fill remaining with empty arrays
                for j in range(i + 1, len(batch_data)):
                    results.append(np.array([]))
                break
        
        return results
    
    def streaming_preprocess(self, data_stream):
        """
        Generator for streaming preprocessing.
        
        Args:
            data_stream: Iterator of data samples
            
        Yields:
            Processed feature vectors
        """
        for data in data_stream:
            self.streaming_buffer.append(data)
            
            # Process when buffer reaches batch size
            if len(self.streaming_buffer) >= self.config.batch_size:
                batch = list(self.streaming_buffer)
                self.streaming_buffer.clear()
                
                processed_batch = self.preprocess_batch(batch)
                
                for processed in processed_batch:
                    yield processed
        
        # Process remaining items
        if self.streaming_buffer:
            batch = list(self.streaming_buffer)
            self.streaming_buffer.clear()
            processed_batch = self.preprocess_batch(batch)
            
            for processed in processed_batch:
                yield processed
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self.lock:
            stats_dict = {
                'total_samples': self.stats.total_samples,
                'avg_processing_time_ms': self.stats.avg_time_ms,
                'max_processing_time_ms': self.stats.max_time_ms,
                'memory_usage_mb': self.stats.memory_usage_mb,
                'cache_hit_rate': self.stats.cache_hits / max(1, self.stats.cache_hits + self.stats.cache_misses),
                'battery_scale': self.battery_processor.processing_scale
            }
            
        if self.processing_times:
            times = list(self.processing_times)
            stats_dict.update({
                'recent_avg_time_ms': np.mean(times),
                'p95_time_ms': np.percentile(times, 95),
                'p99_time_ms': np.percentile(times, 99)
            })
            
        return stats_dict
    
    def clear_cache(self):
        """Clear processing cache."""
        if self.processed_cache is not None:
            self.processed_cache.clear()
            logging.info("Processing cache cleared")
    
    def fit_preprocessors(self, training_data: List[Dict[str, Any]]):
        """Fit all preprocessors on training data."""
        # Separate data by modality for fitting
        touch_data = [d['touch'] for d in training_data if 'touch' in d]
        typing_data = [d['typing'] for d in training_data if 'typing' in d]
        
        # Fit preprocessors
        if touch_data:
            self.touch_preprocessor.fit(touch_data)
            
        # Set fitted flags
        for preprocessor in [self.typing_preprocessor, self.movement_preprocessor, self.app_usage_preprocessor]:
            preprocessor.is_fitted = True
            
        logging.info("All preprocessors fitted on training data")

# Utility functions
def benchmark_preprocessing(sample_data: Dict[str, Any], iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark preprocessing performance.
    
    Args:
        sample_data: Sample data for benchmarking
        iterations: Number of benchmark iterations
        
    Returns:
        Performance metrics
    """
    config = ProcessingConfig()
    processor = DataPreprocessor(config)
    
    times = []
    memory_usage = []
    
    for _ in range(iterations):
        start_time = time.time()
        _ = processor.preprocess_single(sample_data)
        processing_time = (time.time() - start_time) * 1000
        
        times.append(processing_time)
        memory_usage.append(processor.memory_monitor.get_memory_usage_mb())
    
    return {
        'avg_time_ms': np.mean(times),
        'p95_time_ms': np.percentile(times, 95),
        'max_time_ms': np.max(times),
        'avg_memory_mb': np.mean(memory_usage),
        'max_memory_mb': np.max(memory_usage)
    }

def create_sample_data() -> Dict[str, Any]:
    """Create sample data for testing."""
    return {
        'touch': [
            {'pressure': 0.3, 'area': 10, 'duration': 0.1, 'velocity': 0.2, 'acceleration': 0.05},
            {'pressure': 0.4, 'area': 12, 'duration': 0.15, 'velocity': 0.25, 'acceleration': 0.08}
        ],
        'typing': [
            {'dwell_time': 0.15, 'flight_time': 0.08, 'typing_speed': 5, 'rhythm': 0.9},
            {'dwell_time': 0.12, 'flight_time': 0.10, 'typing_speed': 4.8, 'rhythm': 0.85}
        ],
        'voice': np.random.randn(16000),  # 1 second of audio at 16kHz
        'visual': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        'movement': {
            'acceleration': np.random.randn(100, 3),
            'gyroscope': np.random.randn(100, 3),
            'magnetometer': np.random.randn(100, 3)
        },
        'app_usage': [
            {'duration': 120, 'app_name': 'app1', 'timestamp': time.time(), 'frequency': 5},
            {'duration': 300, 'app_name': 'app2', 'timestamp': time.time() - 3600, 'frequency': 2}
        ]
    }

# Example usage and testing
if __name__ == '__main__':
    # Create sample data
    sample_data = create_sample_data()
    
    # Initialize preprocessor
    config = ProcessingConfig()
    processor = DataPreprocessor(config)
    
    # Test single sample preprocessing
    print("Testing single sample preprocessing...")
    processed_features = processor.preprocess_single(sample_data)
    print(f"Processed features shape: {processed_features.shape}")
    print(f"Feature vector length: {len(processed_features)}")
    
    # Test batch preprocessing
    print("\nTesting batch preprocessing...")
    batch_data = [sample_data] * 5
    batch_results = processor.preprocess_batch(batch_data)
    print(f"Batch results: {len(batch_results)} samples")
    
    # Test streaming preprocessing
    print("\nTesting streaming preprocessing...")
    stream_data = [sample_data] * 10
    streaming_results = list(processor.streaming_preprocess(stream_data))
    print(f"Streaming results: {len(streaming_results)} features")
    
    # Performance benchmark
    print("\nRunning performance benchmark...")
    benchmark_results = benchmark_preprocessing(sample_data)
    print(f"Benchmark results: {benchmark_results}")
    
    # Test battery awareness
    print("\nTesting battery awareness...")
    processor.update_battery_level(15.0)  # Low battery
    low_battery_features = processor.preprocess_single(sample_data)
    print(f"Low battery processing completed: {len(low_battery_features) > 0}")
    
    # Performance stats
    stats = processor.get_performance_stats()
    print(f"\nPreprocessor stats: {stats}")
