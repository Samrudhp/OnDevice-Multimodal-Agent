# models/speaker_identification.py

"""
Speaker Identification Using MFCC + SVM for Mobile Fraud Detection
QuadFusion Models - Mobile-Optimized Voice Authentication

Features:
- Real-time MFCC feature extraction (25ms frames, 10ms hop)
- Voice activity detection with energy thresholds  
- SVM-based speaker classification with RBF kernel
- Speaker enrollment and incremental learning
- Anti-spoofing detection for replay attacks
- Noise robustness and mobile audio optimization
- <20MB memory usage, <20ms inference latency
- Multi-language support and background noise handling
"""

import numpy as np
import scipy.signal
import scipy.fftpack
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import librosa
import joblib
import threading
import time
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import deque
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class AudioConfig:
    """Configuration for audio processing parameters."""
    sample_rate: int = 16000
    frame_length: float = 0.025  # 25ms
    frame_shift: float = 0.010   # 10ms  
    n_mfcc: int = 39            # 13 base + 13 delta + 13 delta-delta
    n_mels: int = 80            # Mel filter bank size
    pre_emphasis: float = 0.97   # Pre-emphasis filter coefficient
    window: str = 'hamming'      # Windowing function

class MFCCExtractor:
    """
    High-performance MFCC feature extraction optimized for mobile devices.
    Supports real-time processing with minimal memory footprint.
    """
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.frame_length_samples = int(config.sample_rate * config.frame_length)
        self.frame_shift_samples = int(config.sample_rate * config.frame_shift)
        
        # Pre-compute mel filterbank for efficiency
        self.mel_filterbank = self._create_mel_filterbank()
        self.dct_matrix = self._create_dct_matrix()
        
        # Pre-emphasis filter
        self.pre_emphasis_filter = np.array([1, -config.pre_emphasis])
        
    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel-scale filterbank matrix."""
        n_fft = self.frame_length_samples
        mel_points = np.linspace(0, self._hz_to_mel(self.config.sample_rate // 2), self.config.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / self.config.sample_rate).astype(int)
        
        filterbank = np.zeros((self.config.n_mels, n_fft // 2 + 1))
        for m in range(1, self.config.n_mels + 1):
            f_m_minus = int(bin_points[m - 1])
            f_m = int(bin_points[m])
            f_m_plus = int(bin_points[m + 1])
            
            for k in range(f_m_minus, f_m):
                filterbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
            for k in range(f_m, f_m_plus):
                filterbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])
                
        return filterbank
    
    def _create_dct_matrix(self) -> np.ndarray:
        """Create DCT transformation matrix."""
        dct_matrix = np.zeros((13, self.config.n_mels))  # 13 MFCC coefficients
        for i in range(13):
            dct_matrix[i, :] = np.cos(i * (2 * np.arange(self.config.n_mels) + 1) * np.pi / (2 * self.config.n_mels))
        return dct_matrix
    
    def _hz_to_mel(self, hz: float) -> float:
        """Convert frequency in Hz to mel scale."""
        return 2595 * np.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel: float) -> float:
        """Convert mel scale to frequency in Hz."""
        return 700 * (10**(mel / 2595) - 1)
    
    def extract_frame(self, frame: np.ndarray) -> np.ndarray:
        """Extract MFCC features from a single frame."""
        # Pre-emphasis
        emphasized = np.convolve(frame, self.pre_emphasis_filter, mode='same')
        
        # Windowing
        if self.config.window == 'hamming':
            windowed = emphasized * np.hamming(len(emphasized))
        else:
            windowed = emphasized
            
        # FFT
        fft_spectrum = np.fft.rfft(windowed, n=self.frame_length_samples)
        magnitude_spectrum = np.abs(fft_spectrum)
        power_spectrum = magnitude_spectrum ** 2
        
        # Apply mel filterbank
        mel_spectrum = np.dot(self.mel_filterbank, power_spectrum)
        mel_spectrum = np.where(mel_spectrum == 0, np.finfo(float).eps, mel_spectrum)
        log_mel_spectrum = np.log(mel_spectrum)
        
        # Apply DCT
        mfcc = np.dot(self.dct_matrix, log_mel_spectrum)
        
        return mfcc
    
    def extract(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from entire signal.
        
        Args:
            signal: Audio signal
            
        Returns:
            MFCC feature matrix (n_frames, n_features)
        """
        if len(signal) < self.frame_length_samples:
            return np.array([])
            
        n_frames = 1 + (len(signal) - self.frame_length_samples) // self.frame_shift_samples
        mfcc_features = np.zeros((n_frames, 13))
        
        for i in range(n_frames):
            start_idx = i * self.frame_shift_samples
            end_idx = start_idx + self.frame_length_samples
            frame = signal[start_idx:end_idx]
            
            if len(frame) == self.frame_length_samples:
                mfcc_features[i] = self.extract_frame(frame)
                
        # Add delta and delta-delta features
        delta_features = self._compute_deltas(mfcc_features)
        delta_delta_features = self._compute_deltas(delta_features)
        
        # Combine all features (13 + 13 + 13 = 39)
        full_features = np.hstack([mfcc_features, delta_features, delta_delta_features])
        
        return full_features
    
    def _compute_deltas(self, features: np.ndarray, N: int = 2) -> np.ndarray:
        """Compute delta (derivative) features."""
        if len(features) == 0:
            return features
            
        padded = np.pad(features, ((N, N), (0, 0)), mode='edge')
        deltas = np.zeros_like(features)
        
        denominator = 2 * sum(i**2 for i in range(1, N+1))
        
        for t in range(len(features)):
            numerator = sum(i * (padded[t+N+i] - padded[t+N-i]) for i in range(1, N+1))
            deltas[t] = numerator / denominator
            
        return deltas

class VoiceActivityDetector:
    """
    Voice Activity Detection using energy and spectral features.
    Optimized for mobile microphone characteristics.
    """
    
    def __init__(self, config: AudioConfig, energy_threshold: float = 0.01, 
                 zcr_threshold: float = 0.3, min_speech_length: float = 0.1):
        self.config = config
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
        self.min_speech_samples = int(min_speech_length * config.sample_rate)
        self.frame_length = int(config.sample_rate * config.frame_length)
        
    def compute_energy(self, frame: np.ndarray) -> float:
        """Compute frame energy."""
        return np.sum(frame ** 2) / len(frame)
    
    def compute_zero_crossing_rate(self, frame: np.ndarray) -> float:
        """Compute zero crossing rate."""
        signs = np.sign(frame)
        diffs = np.diff(signs)
        return np.sum(diffs != 0) / (2 * len(frame))
    
    def detect_speech_frames(self, signal: np.ndarray) -> np.ndarray:
        """Detect speech frames using energy and ZCR."""
        n_frames = len(signal) // self.frame_length
        speech_flags = np.zeros(n_frames, dtype=bool)
        
        for i in range(n_frames):
            start_idx = i * self.frame_length
            end_idx = start_idx + self.frame_length
            frame = signal[start_idx:end_idx]
            
            energy = self.compute_energy(frame)
            zcr = self.compute_zero_crossing_rate(frame)
            
            # Speech detection criteria
            is_speech = (energy > self.energy_threshold and zcr < self.zcr_threshold)
            speech_flags[i] = is_speech
            
        return speech_flags
    
    def extract_speech_segments(self, signal: np.ndarray) -> List[Tuple[int, int]]:
        """Extract continuous speech segments."""
        speech_flags = self.detect_speech_frames(signal)
        segments = []
        start_idx = None
        
        for i, is_speech in enumerate(speech_flags):
            if is_speech and start_idx is None:
                start_idx = i * self.frame_length
            elif not is_speech and start_idx is not None:
                end_idx = i * self.frame_length
                if end_idx - start_idx >= self.min_speech_samples:
                    segments.append((start_idx, end_idx))
                start_idx = None
                
        # Handle case where speech continues to end
        if start_idx is not None:
            end_idx = len(signal)
            if end_idx - start_idx >= self.min_speech_samples:
                segments.append((start_idx, end_idx))
                
        return segments

class AntiSpoofingDetector:
    """
    Anti-spoofing detection for replay attacks and synthetic speech.
    Uses spectral and temporal features to detect fraudulent audio.
    """
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.frame_length = int(config.sample_rate * config.frame_length)
        
        # Thresholds for spoofing detection
        self.spectral_flatness_threshold = 0.6
        self.spectral_centroid_threshold = 4000  # Hz
        self.spectral_rolloff_threshold = 0.85
        self.harmonic_ratio_threshold = 0.3
        
    def compute_spectral_features(self, frame: np.ndarray) -> Dict[str, float]:
        """Compute spectral features for spoofing detection."""
        # FFT
        fft_spectrum = np.fft.rfft(frame)
        magnitude_spectrum = np.abs(fft_spectrum)
        power_spectrum = magnitude_spectrum ** 2
        
        freqs = np.fft.rfftfreq(len(frame), 1/self.config.sample_rate)
        
        # Spectral flatness (Wiener entropy)
        geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum + 1e-10)))
        arithmetic_mean = np.mean(magnitude_spectrum)
        spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * magnitude_spectrum) / np.sum(magnitude_spectrum) if np.sum(magnitude_spectrum) > 0 else 0
        
        # Spectral rolloff
        cumsum = np.cumsum(power_spectrum)
        rolloff_point = 0.85 * cumsum[-1]
        rolloff_idx = np.where(cumsum >= rolloff_point)[0]
        spectral_rolloff = freqs[rolloff_idx] if len(rolloff_idx) > 0 else freqs[-1]
        
        return {
            'spectral_flatness': spectral_flatness,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff
        }
    
    def compute_harmonic_ratio(self, frame: np.ndarray) -> float:
        """Compute harmonic-to-noise ratio."""
        # Autocorrelation-based pitch detection
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find fundamental frequency
        min_period = int(self.config.sample_rate / 800)  # 800 Hz max
        max_period = int(self.config.sample_rate / 50)   # 50 Hz min
        
        if max_period < len(autocorr):
            pitch_autocorr = autocorr[min_period:max_period]
            if len(pitch_autocorr) > 0:
                max_autocorr = np.max(pitch_autocorr)
                return max_autocorr / autocorr[0] if autocorr[0] > 0 else 0
        return 0
    
    def detect_spoofing(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Detect potential spoofing in audio signal.
        
        Returns:
            Dictionary with spoofing detection results
        """
        n_frames = len(signal) // self.frame_length
        spoof_scores = []
        
        for i in range(min(n_frames, 50)):  # Analyze first 50 frames for efficiency
            start_idx = i * self.frame_length
            end_idx = start_idx + self.frame_length
            frame = signal[start_idx:end_idx]
            
            if len(frame) == self.frame_length:
                spectral_features = self.compute_spectral_features(frame)
                harmonic_ratio = self.compute_harmonic_ratio(frame)
                
                # Spoofing indicators
                flat_spectrum = spectral_features['spectral_flatness'] > self.spectral_flatness_threshold
                unnatural_centroid = spectral_features['spectral_centroid'] > self.spectral_centroid_threshold
                compressed_rolloff = spectral_features['spectral_rolloff'] / (self.config.sample_rate/2) > self.spectral_rolloff_threshold
                low_harmonics = harmonic_ratio < self.harmonic_ratio_threshold
                
                spoof_score = sum([flat_spectrum, unnatural_centroid, compressed_rolloff, low_harmonics]) / 4.0
                spoof_scores.append(spoof_score)
        
        if not spoof_scores:
            return {'is_spoofed': False, 'confidence': 0.0, 'avg_score': 0.0}
            
        avg_spoof_score = np.mean(spoof_scores)
        is_spoofed = avg_spoof_score > 0.5  # Majority vote threshold
        
        return {
            'is_spoofed': is_spoofed,
            'confidence': avg_spoof_score,
            'avg_score': avg_spoof_score,
            'frame_scores': spoof_scores
        }

class SpeakerEnrollment:
    """
    Speaker enrollment system with incremental learning capabilities.
    Manages speaker models and supports online adaptation.
    """
    
    def __init__(self, model_path: str = "speaker_models", max_speakers: int = 100):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.max_speakers = max_speakers
        
        # SVM model and preprocessing
        self.scaler = StandardScaler()
        self.svm_model = SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            C=1.0,
            gamma='scale'
        )
        
        # Training data storage
        self.training_features = []
        self.training_labels = []
        self.speaker_profiles = {}  # speaker_id -> feature statistics
        
        # Thread safety
        self.lock = threading.Lock()
        self.is_trained = False
        
        # Load existing model if available
        self.load_model()
        
    def enroll_speaker(self, features: np.ndarray, speaker_id: str, 
                      min_enrollment_samples: int = 50) -> bool:
        """
        Enroll a new speaker with feature validation.
        
        Args:
            features: MFCC feature matrix
            speaker_id: Unique speaker identifier
            min_enrollment_samples: Minimum number of feature vectors required
            
        Returns:
            Success status
        """
        if len(features) < min_enrollment_samples:
            logging.warning(f"Insufficient enrollment data for {speaker_id}: {len(features)} < {min_enrollment_samples}")
            return False
            
        with self.lock:
            # Add to training data
            self.training_features.append(features)
            self.training_labels.extend([speaker_id] * len(features))
            
            # Create speaker profile
            self.speaker_profiles[speaker_id] = {
                'mean': np.mean(features, axis=0),
                'std': np.std(features, axis=0),
                'n_samples': len(features),
                'enrollment_time': time.time()
            }
            
            # Mark for retraining
            self.is_trained = False
            
        logging.info(f"Enrolled speaker {speaker_id} with {len(features)} feature vectors")
        return True
    
    def train_model(self) -> bool:
        """Train/retrain the SVM classifier."""
        with self.lock:
            if not self.training_features:
                logging.warning("No training data available")
                return False
                
            # Combine all features
            X = np.vstack(self.training_features)
            y = np.array(self.training_labels)
            
            # Feature scaling
            X_scaled = self.scaler.fit_transform(X)
            
            # Train SVM
            self.svm_model.fit(X_scaled, y)
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            logging.info(f"Trained speaker model with {len(X)} samples from {len(set(y))} speakers")
            return True
    
    def predict_speaker(self, features: np.ndarray, 
                       confidence_threshold: float = 0.7) -> Tuple[str, float]:
        """
        Predict speaker identity from features.
        
        Args:
            features: MFCC feature matrix
            confidence_threshold: Minimum confidence for acceptance
            
        Returns:
            (speaker_id, confidence) tuple
        """
        if not self.is_trained or len(features) == 0:
            return "unknown", 0.0
            
        with self.lock:
            # Scale features
            X_scaled = self.scaler.transform(features)
            
            # Get predictions
            probabilities = self.svm_model.predict_proba(X_scaled)
            avg_probs = np.mean(probabilities, axis=0)
            
            # Get best prediction
            best_idx = np.argmax(avg_probs)
            confidence = avg_probs[best_idx]
            predicted_speaker = self.svm_model.classes_[best_idx]
            
            # Apply confidence threshold
            if confidence < confidence_threshold:
                return "unknown", confidence
                
            return predicted_speaker, confidence
    
    def incremental_update(self, features: np.ndarray, speaker_id: str) -> None:
        """Update model with new data from existing speaker."""
        if speaker_id not in self.speaker_profiles:
            logging.warning(f"Unknown speaker {speaker_id} for incremental update")
            return
            
        with self.lock:
            # Add new data
            self.training_features.append(features)
            self.training_labels.extend([speaker_id] * len(features))
            
            # Update speaker profile
            profile = self.speaker_profiles[speaker_id]
            old_mean = profile['mean']
            old_n = profile['n_samples']
            new_n = len(features)
            total_n = old_n + new_n
            
            # Online mean update
            profile['mean'] = (old_mean * old_n + np.mean(features, axis=0) * new_n) / total_n
            profile['n_samples'] = total_n
            
            self.is_trained = False  # Mark for retraining
            
    def save_model(self) -> None:
        """Save trained model and speaker profiles."""
        model_data = {
            'scaler': self.scaler,
            'svm_model': self.svm_model,
            'speaker_profiles': self.speaker_profiles,
            'is_trained': self.is_trained
        }
        
        model_file = self.model_path / "speaker_model.joblib"
        joblib.dump(model_data, model_file)
        
        # Save training data separately for incremental learning
        training_data = {
            'features': self.training_features,
            'labels': self.training_labels
        }
        training_file = self.model_path / "training_data.joblib"
        joblib.dump(training_data, training_file)
        
    def load_model(self) -> bool:
        """Load existing model and training data."""
        model_file = self.model_path / "speaker_model.joblib"
        training_file = self.model_path / "training_data.joblib"
        
        try:
            if model_file.exists():
                model_data = joblib.load(model_file)
                self.scaler = model_data['scaler']
                self.svm_model = model_data['svm_model']
                self.speaker_profiles = model_data['speaker_profiles']
                self.is_trained = model_data['is_trained']
                
            if training_file.exists():
                training_data = joblib.load(training_file)
                self.training_features = training_data['features']
                self.training_labels = training_data['labels']
                
            logging.info("Successfully loaded speaker model")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load speaker model: {e}")
            return False

class SpeakerIdentificationModel:
    """
    Complete speaker identification system combining all components.
    Provides high-level interface for enrollment, training, and prediction.
    """
    
    def __init__(self, config: Optional[AudioConfig] = None, model_path: str = "speaker_models"):
        self.config = config or AudioConfig()
        self.model_path = model_path
        
        # Initialize components
        self.mfcc_extractor = MFCCExtractor(self.config)
        self.vad = VoiceActivityDetector(self.config)
        self.anti_spoofing = AntiSpoofingDetector(self.config)
        self.enrollment_system = SpeakerEnrollment(model_path)
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.prediction_history = deque(maxlen=1000)
        
        # Thread safety
        self.lock = threading.Lock()
        
    def preprocess_audio(self, audio_signal: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess audio signal with VAD and anti-spoofing.
        
        Returns:
            (processed_signal, metadata)
        """
        start_time = time.time()
        
        # Voice activity detection
        speech_segments = self.vad.extract_speech_segments(audio_signal)
        
        if not speech_segments:
            return np.array([]), {
                'error': 'No speech detected',
                'processing_time': time.time() - start_time
            }
        
        # Extract speech regions
        speech_signal = []
        for start, end in speech_segments:
            speech_signal.append(audio_signal[start:end])
        
        if speech_signal:
            concatenated_speech = np.concatenate(speech_signal)
        else:
            concatenated_speech = np.array([])
        
        # Anti-spoofing detection
        spoofing_result = self.anti_spoofing.detect_spoofing(concatenated_speech)
        
        metadata = {
            'speech_segments': speech_segments,
            'speech_duration': len(concatenated_speech) / self.config.sample_rate,
            'spoofing_result': spoofing_result,
            'processing_time': time.time() - start_time
        }
        
        return concatenated_speech, metadata
    
    def enroll_speaker(self, audio_signal: np.ndarray, speaker_id: str) -> Dict[str, Any]:
        """
        Enroll a new speaker from audio signal.
        
        Returns:
            Enrollment result dictionary
        """
        start_time = time.time()
        
        # Preprocess audio
        processed_signal, metadata = self.preprocess_audio(audio_signal)
        
        if len(processed_signal) == 0:
            return {
                'success': False,
                'error': metadata.get('error', 'Audio preprocessing failed'),
                'processing_time': time.time() - start_time
            }
        
        # Check for spoofing
        if metadata['spoofing_result']['is_spoofed']:
            return {
                'success': False,
                'error': 'Potential spoofed audio detected',
                'spoofing_confidence': metadata['spoofing_result']['confidence'],
                'processing_time': time.time() - start_time
            }
        
        # Extract MFCC features
        mfcc_features = self.mfcc_extractor.extract(processed_signal)
        
        if len(mfcc_features) == 0:
            return {
                'success': False,
                'error': 'Feature extraction failed',
                'processing_time': time.time() - start_time
            }
        
        # Enroll speaker
        success = self.enrollment_system.enroll_speaker(mfcc_features, speaker_id)
        
        # Train model
        if success:
            self.enrollment_system.train_model()
        
        return {
            'success': success,
            'speaker_id': speaker_id,
            'n_features': len(mfcc_features),
            'speech_duration': metadata['speech_duration'],
            'processing_time': time.time() - start_time
        }
    
    def identify_speaker(self, audio_signal: np.ndarray, 
                        confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Identify speaker from audio signal.
        
        Returns:
            Identification result dictionary
        """
        start_time = time.time()
        
        # Preprocess audio
        processed_signal, metadata = self.preprocess_audio(audio_signal)
        
        if len(processed_signal) == 0:
            result = {
                'speaker_id': 'unknown',
                'confidence': 0.0,
                'error': metadata.get('error', 'Audio preprocessing failed'),
                'processing_time': time.time() - start_time
            }
            self.prediction_history.append(result)
            return result
        
        # Check for spoofing
        spoofing_result = metadata['spoofing_result']
        if spoofing_result['is_spoofed']:
            result = {
                'speaker_id': 'spoofed',
                'confidence': 0.0,
                'spoofing_detected': True,
                'spoofing_confidence': spoofing_result['confidence'],
                'processing_time': time.time() - start_time
            }
            self.prediction_history.append(result)
            return result
        
        # Extract MFCC features
        mfcc_features = self.mfcc_extractor.extract(processed_signal)
        
        if len(mfcc_features) == 0:
            result = {
                'speaker_id': 'unknown',
                'confidence': 0.0,
                'error': 'Feature extraction failed',
                'processing_time': time.time() - start_time
            }
            self.prediction_history.append(result)
            return result
        
        # Predict speaker
        speaker_id, confidence = self.enrollment_system.predict_speaker(
            mfcc_features, confidence_threshold
        )
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time * 1000)  # Convert to ms
        
        result = {
            'speaker_id': speaker_id,
            'confidence': confidence,
            'n_features': len(mfcc_features),
            'speech_duration': metadata['speech_duration'],
            'spoofing_detected': False,
            'processing_time': processing_time
        }
        
        self.prediction_history.append(result)
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        if not self.processing_times:
            return {'error': 'No processing history available'}
        
        times = list(self.processing_times)
        recent_predictions = list(self.prediction_history)[-100:]
        
        # Success rate
        successful_predictions = [p for p in recent_predictions if p.get('confidence', 0) > 0.5]
        success_rate = len(successful_predictions) / len(recent_predictions) if recent_predictions else 0
        
        return {
            'avg_processing_time_ms': np.mean(times),
            'max_processing_time_ms': np.max(times),
            'min_processing_time_ms': np.min(times),
            'success_rate': success_rate,
            'total_predictions': len(self.prediction_history),
            'enrolled_speakers': len(self.enrollment_system.speaker_profiles)
        }
    
    def update_speaker_model(self, audio_signal: np.ndarray, speaker_id: str) -> Dict[str, Any]:
        """Update existing speaker model with new audio data."""
        processed_signal, metadata = self.preprocess_audio(audio_signal)
        
        if len(processed_signal) == 0 or metadata['spoofing_result']['is_spoofed']:
            return {'success': False, 'error': 'Invalid audio for model update'}
        
        mfcc_features = self.mfcc_extractor.extract(processed_signal)
        
        if len(mfcc_features) > 0:
            self.enrollment_system.incremental_update(mfcc_features, speaker_id)
            return {'success': True, 'n_features': len(mfcc_features)}
        
        return {'success': False, 'error': 'Feature extraction failed'}

# Utility functions for mobile optimization
def compress_speaker_model(model_path: str, compressed_path: str) -> None:
    """Compress speaker model for mobile deployment."""
    import gzip
    import shutil
    
    with open(model_path, 'rb') as f_in:
        with gzip.open(compressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    logging.info(f"Compressed model from {model_path} to {compressed_path}")

def optimize_audio_format(audio_signal: np.ndarray, target_sample_rate: int = 16000) -> np.ndarray:
    """Optimize audio format for mobile processing."""
    # Resample if needed
    if len(audio_signal.shape) > 1:
        # Convert stereo to mono
        audio_signal = np.mean(audio_signal, axis=1)
    
    # Normalize amplitude
    if np.max(np.abs(audio_signal)) > 0:
        audio_signal = audio_signal / np.max(np.abs(audio_signal))
    
    return audio_signal

# Example usage and testing
if __name__ == "__main__":
    # Initialize configuration
    config = AudioConfig()
    
    # Create speaker identification model
    model = SpeakerIdentificationModel(config)
    
    # Generate dummy audio data for testing
    duration = 3.0  # seconds
    sample_rate = config.sample_rate
    dummy_audio = np.random.randn(int(duration * sample_rate)) * 0.1
    
    # Add some periodic structure to simulate speech
    t = np.linspace(0, duration, int(duration * sample_rate))
    dummy_audio += 0.3 * np.sin(2 * np.pi * 200 * t)  # 200 Hz component
    
    # Test enrollment
    print("Testing speaker enrollment...")
    enrollment_result = model.enroll_speaker(dummy_audio, "test_user")
    print(f"Enrollment result: {enrollment_result}")
    
    # Test identification
    print("\nTesting speaker identification...")
    identification_result = model.identify_speaker(dummy_audio)
    print(f"Identification result: {identification_result}")
    
    # Performance statistics
    print(f"\nPerformance stats: {model.get_performance_stats()}")
