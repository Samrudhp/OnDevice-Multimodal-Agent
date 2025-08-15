# training/train_voice_model.py
"""
Voice Model Trainer for QuadFusion
Mobile-optimized speaker identification and voice authentication.
"""

import numpy as np
import librosa
import logging
import time
import psutil
import gc
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

from .dataset_loaders import VoiceDataLoader

logger = logging.getLogger(__name__)

class VoiceFeatureExtractor(BaseEstimator, TransformerMixin):
    """Comprehensive voice feature extraction for speaker identification."""
    
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=2048, hop_length=512,
                 include_delta=True, include_spectral=True, include_prosodic=True):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.include_delta = include_delta
        self.include_spectral = include_spectral
        self.include_prosodic = include_prosodic
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """Extract comprehensive voice features."""
        try:
            if isinstance(X, list):
                return np.array([self._extract_features(x) for x in X])
            else:
                return self._extract_features(X)
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return X
    
    def _extract_features(self, audio_data):
        """Extract features from single audio sample."""
        try:
            features = []
            
            # MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio_data, 
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            mfcc_stats = self._calculate_statistics(mfcc)
            features.extend(mfcc_stats)
            
            # Delta and Delta-Delta features
            if self.include_delta:
                delta_mfcc = librosa.feature.delta(mfcc)
                delta2_mfcc = librosa.feature.delta(mfcc, order=2)
                features.extend(self._calculate_statistics(delta_mfcc))
                features.extend(self._calculate_statistics(delta2_mfcc))
            
            # Spectral features
            if self.include_spectral:
                spectral_features = self._extract_spectral_features(audio_data)
                features.extend(spectral_features)
            
            # Prosodic features
            if self.include_prosodic:
                prosodic_features = self._extract_prosodic_features(audio_data)
                features.extend(prosodic_features)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Feature extraction failed for sample: {e}")
            return np.zeros(self._get_feature_dimension())
    
    def _extract_spectral_features(self, audio_data):
        """Extract spectral features."""
        features = []
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
        features.extend(self._calculate_statistics(spectral_centroid))
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)
        features.extend(self._calculate_statistics(spectral_bandwidth))
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
        features.extend(self._calculate_statistics(spectral_rolloff))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        features.extend(self._calculate_statistics(zcr))
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
        features.extend(self._calculate_statistics(chroma))
        
        return features
    
    def _extract_prosodic_features(self, audio_data):
        """Extract prosodic features."""
        features = []
        
        # Fundamental frequency (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7')
        )
        
        # Remove NaN values
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            features.extend([
                np.mean(f0_clean),
                np.std(f0_clean),
                np.min(f0_clean),
                np.max(f0_clean),
                np.percentile(f0_clean, 25),
                np.percentile(f0_clean, 75)
            ])
        else:
            features.extend([0.0] * 6)
        
        # Voicing statistics
        voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
        features.append(voiced_ratio)
        
        # Energy features
        energy = librosa.feature.rms(y=audio_data)[0]
        features.extend(self._calculate_statistics(energy.reshape(1, -1)))
        
        return features
    
    def _calculate_statistics(self, feature_matrix):
        """Calculate statistical features from feature matrix."""
        stats = []
        for i in range(feature_matrix.shape[0]):
            feature_row = feature_matrix[i, :]
            stats.extend([
                np.mean(feature_row),
                np.std(feature_row),
                np.min(feature_row),
                np.max(feature_row),
                np.percentile(feature_row, 25),
                np.percentile(feature_row, 75),
                np.median(feature_row)
            ])
        return stats
    
    def _get_feature_dimension(self):
        """Calculate total feature dimension."""
        # MFCC stats: n_mfcc * 7 statistics
        mfcc_dim = self.n_mfcc * 7
        
        # Delta features
        delta_dim = mfcc_dim * 2 if self.include_delta else 0
        
        # Spectral features: 5 feature types * 7 statistics + chroma (12 * 7)
        spectral_dim = (5 * 7 + 12 * 7) if self.include_spectral else 0
        
        # Prosodic features: F0 (6) + voicing (1) + energy (7)
        prosodic_dim = 14 if self.include_prosodic else 0
        
        return mfcc_dim + delta_dim + spectral_dim + prosodic_dim

class VoiceActivityDetector:
    """Voice Activity Detection for preprocessing."""
    
    def __init__(self, sample_rate=16000, frame_length=2048, hop_length=512, 
                 energy_threshold=0.01, zcr_threshold=0.1):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
    
    def detect_voice_activity(self, audio_data):
        """Detect voice activity in audio signal."""
        try:
            # Calculate frame-wise energy
            energy = librosa.feature.rms(
                y=audio_data,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )[0]
            
            # Calculate zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                audio_data,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )[0]
            
            # Voice activity detection
            energy_threshold = np.percentile(energy, 30)  # Adaptive threshold
            voice_frames = (energy > energy_threshold) & (zcr < self.zcr_threshold)
            
            return voice_frames
            
        except Exception as e:
            logger.error(f"VAD failed: {e}")
            return np.ones(len(audio_data) // self.hop_length, dtype=bool)
    
    def remove_silence(self, audio_data):
        """Remove silent parts from audio."""
        try:
            voice_frames = self.detect_voice_activity(audio_data)
            
            # Convert frame indices to sample indices
            voice_samples = []
            for i, is_voice in enumerate(voice_frames):
                if is_voice:
                    start_sample = i * self.hop_length
                    end_sample = min(start_sample + self.hop_length, len(audio_data))
                    voice_samples.extend(audio_data[start_sample:end_sample])
            
            return np.array(voice_samples) if voice_samples else audio_data
            
        except Exception as e:
            logger.error(f"Silence removal failed: {e}")
            return audio_data

class VoiceModelTrainer:
    """Mobile-optimized voice model trainer for speaker identification."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_extractor = None
        self.vad = None
        self.scaler = None
        
        # Audio processing parameters
        self.sample_rate = config.get('sample_rate', 16000)
        self.n_mfcc = config.get('n_mfcc', 13)
        self.max_features = config.get('max_features', 100)  # Reduced for mobile
        
        # Model parameters
        self.model_type = config.get('model_type', 'svm')  # svm, random_forest, ensemble
        self.kernel = config.get('kernel', 'linear')  # For SVM
        self.n_estimators = config.get('n_estimators', 50)  # For Random Forest
        
        # Mobile optimization
        self.memory_limit_mb = config.get('memory_limit_mb', 100)
        self.quantization_enabled = config.get('quantization_enabled', True)
        
        # Privacy settings
        self.differential_privacy = config.get('differential_privacy', False)
        self.epsilon = config.get('epsilon', 1.0)
    
    def train(self, data_path: str, output_path: str) -> Dict[str, Any]:
        """
        Train voice model with mobile optimization.
        
        Args:
            data_path: Path to training data
            output_path: Path to save trained model
            
        Returns:
            Training metrics and model info
        """
        try:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.info("Starting voice model training...")
            
            # Load and preprocess data
            audio_data, labels = self._load_and_preprocess_data(data_path)
            
            # Initialize components
            self.feature_extractor = VoiceFeatureExtractor(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                include_delta=True,
                include_spectral=True,
                include_prosodic=True
            )
            
            self.vad = VoiceActivityDetector(sample_rate=self.sample_rate)
            
            # Build training pipeline
            pipeline = self._build_pipeline()
            
            # Train model
            metrics = self._train_model(pipeline, audio_data, labels)
            
            # Validate mobile requirements
            self._validate_mobile_requirements(pipeline)
            
            # Save model
            self._save_model(pipeline, output_path)
            
            # Calculate training statistics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            training_stats = {
                'training_time_seconds': end_time - start_time,
                'memory_usage_mb': end_memory - start_memory,
                'model_size_mb': self._get_model_size(output_path),
                'num_features': self.max_features,
                'sample_rate': self.sample_rate,
                **metrics
            }
            
            logger.info(f"Training completed. Time: {training_stats['training_time_seconds']:.2f}s")
            logger.info(f"Memory usage: {training_stats['memory_usage_mb']:.2f}MB")
            
            return training_stats
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            gc.collect()
    
    def _load_and_preprocess_data(self, data_path: str) -> Tuple[List[np.ndarray], List[int]]:
        """Load and preprocess voice data."""
        try:
            loader = VoiceDataLoader("voice_data", data_path, self.config)
            data = loader.load_data()
            
            if data is None or len(data) == 0:
                raise ValueError("No data loaded")
            
            audio_data = []
            labels = []
            
            for sample in data:
                if isinstance(sample, tuple) and len(sample) >= 2:
                    audio, label = sample[0], sample[1]
                    
                    # Apply VAD
                    if self.vad:
                        audio = self.vad.remove_silence(audio)
                    
                    # Apply differential privacy if enabled
                    if self.differential_privacy:
                        audio = self._apply_differential_privacy(audio)
                    
                    # Normalize audio
                    if len(audio) > 0:
                        audio = librosa.util.normalize(audio)
                        audio_data.append(audio)
                        labels.append(int(label))
            
            logger.info(f"Loaded {len(audio_data)} voice samples")
            return audio_data, labels
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def _build_pipeline(self) -> Pipeline:
        """Build mobile-optimized training pipeline."""
        try:
            pipeline_steps = [
                ('feature_extractor', self.feature_extractor),
                ('scaler', StandardScaler()),
                ('variance_selector', VarianceThreshold(threshold=0.01)),
                ('feature_selector', SelectKBest(f_classif, k=min(self.max_features, 100)))
            ]
            
            # Add model based on configuration
            if self.model_type == 'svm':
                pipeline_steps.append((
                    'classifier', 
                    SVC(
                        kernel=self.kernel,
                        probability=True,
                        random_state=42,
                        cache_size=100  # Reduced for mobile
                    )
                ))
            elif self.model_type == 'random_forest':
                pipeline_steps.append((
                    'classifier',
                    RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=10,  # Limited for mobile
                        random_state=42,
                        n_jobs=1  # Single thread for mobile
                    )
                ))
            elif self.model_type == 'ensemble':
                svm_model = SVC(kernel=self.kernel, probability=True, random_state=42)
                rf_model = RandomForestClassifier(
                    n_estimators=self.n_estimators // 2, 
                    max_depth=8,
                    random_state=42,
                    n_jobs=1
                )
                pipeline_steps.append((
                    'classifier',
                    VotingClassifier(
                        estimators=[('svm', svm_model), ('rf', rf_model)],
                        voting='soft'
                    )
                ))
            
            return Pipeline(pipeline_steps)
            
        except Exception as e:
            logger.error(f"Pipeline creation failed: {e}")
            raise
    
    def _train_model(self, pipeline: Pipeline, audio_data: List[np.ndarray], 
                    labels: List[int]) -> Dict[str, Any]:
        """Train the voice model."""
        try:
            logger.info("Extracting features and training model...")
            
            # Extract features
            features = []
            valid_labels = []
            
            for i, audio in enumerate(audio_data):
                try:
                    feature_vector = self.feature_extractor.transform([audio])[0]
                    if not np.any(np.isnan(feature_vector)) and not np.any(np.isinf(feature_vector)):
                        features.append(feature_vector)
                        valid_labels.append(labels[i])
                except Exception as e:
                    logger.warning(f"Feature extraction failed for sample {i}: {e}")
            
            if len(features) == 0:
                raise ValueError("No valid features extracted")
            
            features = np.array(features)
            valid_labels = np.array(valid_labels)
            
            # Train pipeline
            pipeline.fit(features, valid_labels)
            
            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, features, valid_labels, 
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='accuracy'
            )
            
            # Predictions for training metrics
            predictions = pipeline.predict(features)
            probabilities = pipeline.predict_proba(features)
            
            # Calculate metrics
            accuracy = accuracy_score(valid_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                valid_labels, predictions, average='weighted'
            )
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'num_samples': len(features),
                'num_classes': len(np.unique(valid_labels))
            }
            
            logger.info(f"Training accuracy: {accuracy:.3f}")
            logger.info(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def _validate_mobile_requirements(self, pipeline: Pipeline):
        """Validate model meets mobile requirements."""
        try:
            # Memory usage check
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            if current_memory > self.memory_limit_mb:
                logger.warning(f"Memory usage {current_memory:.2f}MB exceeds limit")
            
            # Feature count check
            n_features = pipeline.named_steps['feature_selector'].k
            if n_features > self.max_features:
                logger.warning(f"Model uses {n_features} features, may be too complex")
            
            # Model complexity check
            classifier = pipeline.named_steps['classifier']
            if hasattr(classifier, 'n_estimators'):
                if classifier.n_estimators > self.n_estimators:
                    logger.warning(f"Model has {classifier.n_estimators} estimators")
                    
            logger.info("Mobile requirements validation passed")
            
        except Exception as e:
            logger.error(f"Mobile validation failed: {e}")
    
    def _save_model(self, pipeline: Pipeline, output_path: str):
        """Save trained model in multiple formats."""
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save sklearn model
            joblib.dump(pipeline, output_path / "voice_model.pkl")
            
            # Convert to ONNX for mobile deployment
            try:
                initial_type = [('float_input', FloatTensorType([None, self.max_features]))]
                onnx_model = convert_sklearn(
                    pipeline,
                    initial_types=initial_type,
                    target_opset=11
                )
                
                with open(output_path / "voice_model.onnx", "wb") as f:
                    f.write(onnx_model.SerializeToString())
                    
                logger.info("ONNX model saved successfully")
                
            except Exception as e:
                logger.warning(f"ONNX conversion failed: {e}")
            
            # Save model metadata
            metadata = {
                'model_type': self.model_type,
                'agent_type': 'voice_command',
                'sample_rate': self.sample_rate,
                'n_mfcc': self.n_mfcc,
                'max_features': self.max_features,
                'training_config': self.config
            }
            
            with open(output_path / "model_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            raise
    
    def _get_model_size(self, model_path: str) -> float:
        """Get model size in MB."""
        try:
            model_path = Path(model_path)
            if model_path.exists():
                total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                return total_size / 1024 / 1024
            return 0.0
        except Exception:
            return 0.0
    
    def _apply_differential_privacy(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply differential privacy to audio data."""
        try:
            if not self.differential_privacy:
                return audio_data
                
            # Add Laplace noise for differential privacy
            sensitivity = np.ptp(audio_data)  # Range of audio data
            noise_scale = sensitivity / self.epsilon
            noise = np.random.laplace(0, noise_scale, audio_data.shape)
            
            return audio_data + noise
            
        except Exception as e:
            logger.error(f"Differential privacy application failed: {e}")
            return audio_data

def train_voice_model(config_path: str, data_path: str, output_path: str) -> Dict[str, Any]:
    """
    Convenience function for training voice model.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to training data
        output_path: Path to save model
        
    Returns:
        Training metrics
    """
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        trainer = VoiceModelTrainer(config)
        return trainer.train(data_path, output_path)
        
    except Exception as e:
        logger.error(f"Voice model training failed: {e}")
        raise