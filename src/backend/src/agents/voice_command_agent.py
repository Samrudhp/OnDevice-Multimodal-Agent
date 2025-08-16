# agents/voice_command_agent.py
"""
Voice Command Agent for speech recognition and speaker identification.
Uses Tiny Whisper for speech-to-text and MFCC + SVM for speaker identification.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import time
import hashlib

from .base_agent import BaseAgent, AgentResult, RiskLevel

class VoiceCommandAgent(BaseAgent):
    """
    Agent for voice command analysis and speaker identification.
    Combines speech content analysis with speaker verification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("VoiceCommandAgent", config)
        
        # Audio processing parameters
        self.sample_rate = config.get('sample_rate', 16000)
        self.n_mfcc = config.get('n_mfcc', 13)
        self.n_fft = config.get('n_fft', 2048)
        self.hop_length = config.get('hop_length', 512)
        self.max_audio_duration = config.get('max_audio_duration', 10.0)  # seconds
        
        # Speaker identification parameters
        self.speaker_threshold = config.get('speaker_threshold', 0.7)
        self.min_audio_length = config.get('min_audio_length', 1.0)  # seconds
        self.voice_activity_threshold = config.get('voice_activity_threshold', 0.01)
        
        # Model components
        self.speaker_classifier = None
        self.mfcc_scaler = StandardScaler()
        self.enrolled_speaker_embeddings = []
        self.speaker_id = None
        
        # Speech processing components (simplified - would integrate with TinyWhisper)
        self.speech_recognizer = None
        self.command_patterns = config.get('command_patterns', [])
        
        # Voice analysis buffers
        self.recent_audio_features = []
        self.speaker_scores = []
    
    def capture_data(self, sensor_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Capture and preprocess audio data.
        
        Expected sensor_data format:
        {
            'audio_data': np.ndarray,  # Raw audio samples
            'sample_rate': int,
            'duration': float,
            'command_text': str  # Optional: transcribed text from TinyWhisper
        }
        
        Returns:
            Combined feature vector: [speaker_features, speech_features]
        """
        try:
            audio_data = sensor_data.get('audio_data')
            if audio_data is None:
                return None
            
            sample_rate = sensor_data.get('sample_rate', self.sample_rate)
            command_text = sensor_data.get('command_text', '')
            
            # Validate audio
            if len(audio_data) == 0:
                return None
            
            # Convert to target sample rate if necessary
            if sample_rate != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
            
            # Validate audio duration
            duration = len(audio_data) / self.sample_rate
            if duration < self.min_audio_length or duration > self.max_audio_duration:
                return None
            
            # Check for voice activity
            if np.max(np.abs(audio_data)) < self.voice_activity_threshold:
                return None  # No significant audio detected
            
            # Extract speaker identification features
            speaker_features = self._extract_speaker_features(audio_data)
            
            # Extract speech content features
            speech_features = self._extract_speech_features(audio_data, command_text)
            
            if speaker_features is None or speech_features is None:
                return None
            
            # Combine features
            combined_features = np.concatenate([speaker_features, speech_features])
            
            return combined_features
            
        except Exception as e:
            print(f"Error capturing voice data: {e}")
            return None
    
    def _extract_speaker_features(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract MFCC features for speaker identification.
        
        Args:
            audio_data: Raw audio samples
            
        Returns:
            Speaker identification features
        """
        try:
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Calculate statistical features from MFCCs
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            
            # Additional spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            # Combine all speaker features
            speaker_features = np.concatenate([
                mfcc_mean,
                mfcc_std,
                mfcc_delta_mean,
                [np.mean(spectral_centroids)],
                [np.std(spectral_centroids)],
                [np.mean(spectral_rolloff)],
                [np.std(spectral_rolloff)],
                [np.mean(zero_crossing_rate)],
                [np.std(zero_crossing_rate)]
            ])
            
            return speaker_features
            
        except Exception as e:
            print(f"Error extracting speaker features: {e}")
            return None
    
    def _extract_speech_features(self, audio_data: np.ndarray, command_text: str = '') -> Optional[np.ndarray]:
        """
        Extract speech content features.
        
        Args:
            audio_data: Raw audio samples
            command_text: Transcribed text (if available)
            
        Returns:
            Speech content features
        """
        try:
            # Audio-based speech features
            duration = len(audio_data) / self.sample_rate
            
            # Speaking rate (approximated by zero crossing rate)
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            speaking_rate = np.mean(zcr) * self.sample_rate / 2  # Rough approximation
            
            # Energy and pitch features
            rms_energy = librosa.feature.rms(y=audio_data)[0]
            avg_energy = np.mean(rms_energy)
            energy_variance = np.var(rms_energy)
            
            # Pitch tracking (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                avg_pitch = np.mean(pitch_values)
                pitch_variance = np.var(pitch_values)
            else:
                avg_pitch = 0.0
                pitch_variance = 0.0
            
            # Text-based features (if command_text is available)
            if command_text:
                # Command length and complexity
                word_count = len(command_text.split())
                char_count = len(command_text)
                avg_word_length = char_count / max(word_count, 1)
                
                # Command pattern matching
                pattern_matches = sum(1 for pattern in self.command_patterns 
                                    if pattern.lower() in command_text.lower())
                pattern_match_ratio = pattern_matches / max(len(self.command_patterns), 1)
                
            else:
                word_count = 0
                char_count = 0
                avg_word_length = 0
                pattern_match_ratio = 0
            
            # Combine speech features
            speech_features = np.array([
                duration,
                speaking_rate,
                avg_energy,
                energy_variance,
                avg_pitch,
                pitch_variance,
                word_count,
                char_count,
                avg_word_length,
                pattern_match_ratio
            ])
            
            return speech_features
            
        except Exception as e:
            print(f"Error extracting speech features: {e}")
            return None
    
    def train_initial(self, training_data: List[np.ndarray]) -> bool:
        """
        Train initial speaker identification model.
        
        Args:
            training_data: List of combined feature vectors from enrolled speaker
            
        Returns:
            True if training successful
        """
        try:
            if len(training_data) < 20:
                print("Insufficient training data for VoiceCommandAgent")
                return False
            
            # Convert to numpy array
            X = np.array(training_data)
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Split features back into speaker and speech components
            # Assuming speaker features are first n_speaker_features
            n_speaker_features = self.n_mfcc * 3 + 5  # MFCCs + spectral features
            
            speaker_features = X[:, :n_speaker_features]
            
            # Train scaler on speaker features
            self.mfcc_scaler.fit(speaker_features)
            speaker_features_scaled = self.mfcc_scaler.transform(speaker_features)
            
            # Create speaker embedding (average of all training samples)
            enrolled_embedding = np.mean(speaker_features_scaled, axis=0)
            self.enrolled_speaker_embeddings = [enrolled_embedding]
            
            # Train one-class SVM for speaker verification
            self.speaker_classifier = SVC(
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            )
            
            # Create labels (all training data is from enrolled speaker)
            y = np.ones(len(training_data))  # All positive samples
            
            # Add some synthetic negative samples for better boundary estimation
            synthetic_negatives = self._generate_synthetic_negatives(speaker_features_scaled, n_samples=len(training_data)//4)
            
            if len(synthetic_negatives) > 0:
                X_combined = np.vstack([speaker_features_scaled, synthetic_negatives])
                y_combined = np.hstack([y, np.zeros(len(synthetic_negatives))])
            else:
                X_combined = speaker_features_scaled
                y_combined = y
            
            self.speaker_classifier.fit(X_combined, y_combined)
            
            # Generate speaker ID
            speaker_data_hash = hashlib.md5(str(enrolled_embedding).encode()).hexdigest()[:8]
            self.speaker_id = f"speaker_{speaker_data_hash}"
            
            self.is_trained = True
            print(f"VoiceCommandAgent trained on {len(training_data)} samples for speaker {self.speaker_id}")
            return True
            
        except Exception as e:
            print(f"Error training VoiceCommandAgent: {e}")
            return False
    
    def _generate_synthetic_negatives(self, positive_samples: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate synthetic negative samples for better decision boundary.
        
        Args:
            positive_samples: Training samples from enrolled speaker
            n_samples: Number of synthetic samples to generate
            
        Returns:
            Synthetic negative samples
        """
        try:
            if n_samples <= 0:
                return np.array([])
            
            # Calculate statistics of positive samples
            mean = np.mean(positive_samples, axis=0)
            std = np.std(positive_samples, axis=0)
            
            # Generate samples from shifted distributions
            negatives = []
            
            for _ in range(n_samples):
                # Shift mean by 2-4 standard deviations
                shift_factor = np.random.uniform(2.0, 4.0, size=mean.shape)
                shift_direction = np.random.choice([-1, 1], size=mean.shape)
                shifted_mean = mean + shift_factor * std * shift_direction
                
                # Add noise
                noise = np.random.normal(0, std * 0.5, size=mean.shape)
                synthetic_sample = shifted_mean + noise
                
                negatives.append(synthetic_sample)
            
            return np.array(negatives)
            
        except Exception as e:
            print(f"Error generating synthetic negatives: {e}")
            return np.array([])
    
    def predict(self, features: np.ndarray) -> AgentResult:
        """
        Predict anomaly score based on speaker verification and speech analysis.
        
        Args:
            features: Combined feature vector [speaker_features, speech_features]
            
        Returns:
            AgentResult with speaker verification score
        """
        start_time = time.time()
        
        try:
            if not self.is_trained or self.speaker_classifier is None:
                return AgentResult(
                    agent_name=self.agent_name,
                    anomaly_score=0.0,
                    risk_level=RiskLevel.LOW,
                    confidence=0.0,
                    features_used=['untrained'],
                    processing_time_ms=0,
                    metadata={'error': 'Model not trained'}
                )
            
            # Split features
            n_speaker_features = self.n_mfcc * 3 + 5
            
            if len(features) < n_speaker_features + 10:  # Expected total feature size
                return AgentResult(
                    agent_name=self.agent_name,
                    anomaly_score=0.0,
                    risk_level=RiskLevel.LOW,
                    confidence=0.0,
                    features_used=['invalid_features'],
                    processing_time_ms=0,
                    metadata={'error': f'Invalid feature size: {len(features)}'}
                )
            
            speaker_features = features[:n_speaker_features]
            speech_features = features[n_speaker_features:]
            
            # Scale speaker features
            speaker_features_scaled = self.mfcc_scaler.transform(speaker_features.reshape(1, -1))[0]
            
            # Speaker verification using SVM
            speaker_probability = self.speaker_classifier.predict_proba(speaker_features_scaled.reshape(1, -1))[0]
            
            # Get probability of being the enrolled speaker (class 1)
            if len(speaker_probability) > 1:
                speaker_match_prob = speaker_probability[1]  # Probability of positive class
            else:
                speaker_match_prob = speaker_probability[0]
            
            # Calculate similarity to enrolled speaker embedding
            if self.enrolled_speaker_embeddings:
                enrolled_embedding = self.enrolled_speaker_embeddings[0]
                similarity = cosine_similarity(
                    speaker_features_scaled.reshape(1, -1),
                    enrolled_embedding.reshape(1, -1)
                )[0][0]
                
                # Combine SVM probability and cosine similarity
                speaker_score = (speaker_match_prob + (similarity + 1) / 2) / 2  # Average of both scores
            else:
                speaker_score = speaker_match_prob
            
            # Anomaly score is inverse of speaker match (1.0 = not the enrolled speaker)
            speaker_anomaly = 1.0 - speaker_score
            
            # Analyze speech content for additional anomalies
            speech_anomaly = self._analyze_speech_content(speech_features)
            
            # Combine speaker and speech anomaly scores
            combined_anomaly = (speaker_anomaly * 0.7 + speech_anomaly * 0.3)  # Weight speaker verification more
            
            # Calculate confidence
            confidence = abs(speaker_score - 0.5) * 2  # Higher confidence when score is far from 0.5
            
            # Determine risk level
            risk_level = self.get_risk_level_from_score(combined_anomaly)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Extract speech characteristics for metadata
            duration = speech_features[0] if len(speech_features) > 0 else 0
            speaking_rate = speech_features[1] if len(speech_features) > 1 else 0
            avg_energy = speech_features[2] if len(speech_features) > 2 else 0
            avg_pitch = speech_features[4] if len(speech_features) > 4 else 0
            
            metadata = {
                'speaker_match_probability': float(speaker_match_prob),
                'speaker_anomaly_score': float(speaker_anomaly),
                'speech_anomaly_score': float(speech_anomaly),
                'enrolled_speaker_id': self.speaker_id,
                'audio_duration': float(duration),
                'speaking_rate': float(speaking_rate),
                'avg_energy': float(avg_energy),
                'avg_pitch': float(avg_pitch),
                'model_type': 'mfcc_svm_speaker_verification'
            }
            
            if self.enrolled_speaker_embeddings:
                metadata['cosine_similarity'] = float(similarity)
            
            feature_names = [
                'mfcc_features', 'spectral_features', 'energy_features',
                'pitch_features', 'speech_timing', 'command_patterns'
            ]
            
            # Store speaker score for incremental learning
            self.speaker_scores.append(speaker_score)
            if len(self.speaker_scores) > 1000:
                self.speaker_scores = self.speaker_scores[-1000:]
            
            return AgentResult(
                agent_name=self.agent_name,
                anomaly_score=combined_anomaly,
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
    
    def _analyze_speech_content(self, speech_features: np.ndarray) -> float:
        """
        Analyze speech content for anomalies.
        
        Args:
            speech_features: Speech-related features
            
        Returns:
            Speech anomaly score (0-1)
        """
        try:
            if len(speech_features) < 10:
                return 0.0
            
            duration, speaking_rate, avg_energy, energy_var, avg_pitch, pitch_var, \
            word_count, char_count, avg_word_length, pattern_match_ratio = speech_features
            
            anomaly_indicators = []
            
            # Check for unusual duration
            if duration < 0.5 or duration > 8.0:
                anomaly_indicators.append(0.3)
            
            # Check for unusual speaking rate
            if speaking_rate > 200 or speaking_rate < 50:  # Very fast or very slow
                anomaly_indicators.append(0.2)
            
            # Check for unusual energy patterns
            if avg_energy < 0.001 or avg_energy > 0.5:  # Too quiet or too loud
                anomaly_indicators.append(0.2)
            
            # Check for unusual pitch patterns
            if avg_pitch > 500 or (avg_pitch > 0 and avg_pitch < 50):  # Unusual pitch range
                anomaly_indicators.append(0.2)
            
            # Check command pattern matching (low matching could indicate spoofing)
            if pattern_match_ratio < 0.1 and word_count > 0:  # Commands don't match expected patterns
                anomaly_indicators.append(0.3)
            
            # Calculate overall speech anomaly
            if anomaly_indicators:
                speech_anomaly = min(1.0, sum(anomaly_indicators))
            else:
                speech_anomaly = 0.0
            
            return speech_anomaly
            
        except Exception:
            return 0.0
    
    def incremental_update(self, new_data: List[np.ndarray], 
                          is_anomaly: List[bool] = None) -> bool:
        """
        Update speaker model with new verified voice samples.
        
        Args:
            new_data: List of new feature vectors
            is_anomaly: Optional labels for supervised updates
            
        Returns:
            True if update successful
        """
        try:
            if not self.is_trained or len(new_data) < 3:
                return False
            
            # Filter to only normal/verified speaker data
            if is_anomaly is not None:
                normal_data = [data for data, anomaly in zip(new_data, is_anomaly) if not anomaly]
            else:
                # Use speaker score to filter likely genuine samples
                normal_data = []
                for features in new_data:
                    result = self.predict(features)
                    if result.anomaly_score < 0.3:  # High confidence it's the enrolled speaker
                        normal_data.append(features)
            
            if len(normal_data) < 2:
                return False
            
            # Extract speaker features from normal data
            n_speaker_features = self.n_mfcc * 3 + 5
            new_speaker_features = [data[:n_speaker_features] for data in normal_data]
            
            X_new = np.array(new_speaker_features)
            X_new = np.nan_to_num(X_new, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Scale new features
            X_new_scaled = self.mfcc_scaler.transform(X_new)
            
            # Update enrolled speaker embedding with exponential moving average
            new_embedding = np.mean(X_new_scaled, axis=0)
            alpha = 0.1  # Learning rate
            
            if self.enrolled_speaker_embeddings:
                old_embedding = self.enrolled_speaker_embeddings[0]
                updated_embedding = (1 - alpha) * old_embedding + alpha * new_embedding
                self.enrolled_speaker_embeddings[0] = updated_embedding
            else:
                self.enrolled_speaker_embeddings = [new_embedding]
            
            # Optionally retrain SVM with new data (simplified approach)
            # For production, you might want more sophisticated continual learning
            
            print(f"VoiceCommandAgent incrementally updated with {len(normal_data)} samples")
            return True
            
        except Exception as e:
            print(f"Error in incremental update for VoiceCommandAgent: {e}")
            return False
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        try:
            model_data = {
                'speaker_classifier': self.speaker_classifier,
                'mfcc_scaler': self.mfcc_scaler,
                'enrolled_speaker_embeddings': self.enrolled_speaker_embeddings,
                'speaker_id': self.speaker_id,
                'is_trained': self.is_trained,
                'baseline_data': self.baseline_data,
                'speaker_scores': self.speaker_scores,
                'config': self.config
            }
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            print(f"Error saving VoiceCommandAgent model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            model_data = joblib.load(filepath)
            self.speaker_classifier = model_data['speaker_classifier']
            self.mfcc_scaler = model_data['mfcc_scaler']
            self.enrolled_speaker_embeddings = model_data['enrolled_speaker_embeddings']
            self.speaker_id = model_data['speaker_id']
            self.is_trained = model_data['is_trained']
            self.baseline_data = model_data.get('baseline_data', [])
            self.speaker_scores = model_data.get('speaker_scores', [])
            self.config.update(model_data.get('config', {}))
            return True
        except Exception as e:
            print(f"Error loading VoiceCommandAgent model: {e}")
            return False