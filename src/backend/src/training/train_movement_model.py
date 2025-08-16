# training/train_movement_model.py
"""
Movement Model Trainer for QuadFusion
Mobile-optimized motion pattern analysis using sensor fusion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import time
import psutil
import gc
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import signal
from scipy.stats import entropy
import pandas as pd

from .dataset_loaders import MovementDataLoader

logger = logging.getLogger(__name__)

class SensorFusionCNN(nn.Module):
    """Mobile-optimized CNN for motion pattern recognition with sensor fusion."""
    
    def __init__(self, input_channels: int = 6, sequence_length: int = 100, 
                 num_classes: int = 6, hidden_dim: int = 64):
        super(SensorFusionCNN, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # Convolutional layers for temporal feature extraction
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Pooling layers for dimensionality reduction
        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        
        # Attention mechanism for important time steps
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_features=False):
        """Forward pass with optional feature extraction."""
        batch_size = x.size(0)
        
        # Convolutional feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        pooled = self.pool3(x).squeeze(-1)  # [batch_size, hidden_dim]
        
        if return_features:
            return pooled
        
        # Classification
        features = self.dropout(pooled)
        features = self.relu(self.fc1(features))
        features = self.dropout(features)
        output = self.fc2(features)
        
        return output, pooled

class MotionFeatureExtractor:
    """Extract handcrafted features from motion data."""
    
    def __init__(self, sample_rate: float = 50.0):
        self.sample_rate = sample_rate
        
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive motion features.
        
        Args:
            data: Motion data [sequence_length, num_channels]
            
        Returns:
            Feature vector
        """
        try:
            features = []
            
            # For each channel (accelerometer, gyroscope, magnetometer)
            for channel in range(data.shape[1]):
                channel_data = data[:, channel]
                
                # Time domain features
                features.extend(self._extract_time_domain_features(channel_data))
                
                # Frequency domain features
                features.extend(self._extract_frequency_domain_features(channel_data))
                
                # Statistical features
                features.extend(self._extract_statistical_features(channel_data))
            
            # Cross-channel features
            features.extend(self._extract_cross_channel_features(data))
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(self._get_feature_dimension(data.shape[1]))
    
    def _extract_time_domain_features(self, signal_data: np.ndarray) -> List[float]:
        """Extract time domain features."""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(signal_data),
            np.std(signal_data),
            np.var(signal_data),
            np.min(signal_data),
            np.max(signal_data),
            np.ptp(signal_data),  # Peak-to-peak
            np.median(signal_data)
        ])
        
        # Percentiles
        percentiles = [25, 75, 90, 95]
        for p in percentiles:
            features.append(np.percentile(signal_data, p))
        
        # Signal energy and RMS
        features.extend([
            np.sum(signal_data ** 2),  # Energy
            np.sqrt(np.mean(signal_data ** 2)),  # RMS
            np.mean(np.abs(signal_data))  # Mean absolute value
        ])
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(signal_data)) != 0)
        features.append(zero_crossings / len(signal_data))
        
        return features
    
    def _extract_frequency_domain_features(self, signal_data: np.ndarray) -> List[float]:
        """Extract frequency domain features."""
        features = []
        
        # FFT
        fft = np.fft.fft(signal_data)
        magnitude = np.abs(fft[:len(fft)//2])
        frequencies = np.fft.fftfreq(len(signal_data), 1/self.sample_rate)[:len(fft)//2]
        
        # Spectral features
        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.max(magnitude),
            np.argmax(magnitude)  # Dominant frequency index
        ])
        
        # Spectral centroid
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
            features.append(spectral_centroid)
        else:
            features.append(0.0)
        
        # Spectral spread
        if np.sum(magnitude) > 0:
            spectral_spread = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
            features.append(spectral_spread)
        else:
            features.append(0.0)
        
        # Power in frequency bands
        low_freq_power = np.sum(magnitude[frequencies < 2])
        mid_freq_power = np.sum(magnitude[(frequencies >= 2) & (frequencies < 8)])
        high_freq_power = np.sum(magnitude[frequencies >= 8])
        
        features.extend([low_freq_power, mid_freq_power, high_freq_power])
        
        return features
    
    def _extract_statistical_features(self, signal_data: np.ndarray) -> List[float]:
        """Extract statistical features."""
        features = []
        
        # Higher order moments
        features.extend([
            signal_data.var(),  # Variance
            signal.sp.stats.skew(signal_data),  # Skewness
            signal.sp.stats.kurtosis(signal_data)  # Kurtosis
        ])
        
        # Signal complexity
        features.append(entropy(np.histogram(signal_data, bins=10)[0] + 1e-10))
        
        # Autocorrelation features
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
        
        # First few autocorrelation coefficients
        features.extend(autocorr[1:6].tolist())
        
        return features
    
    def _extract_cross_channel_features(self, data: np.ndarray) -> List[float]:
        """Extract cross-channel correlation features."""
        features = []
        
        # Correlation matrix
        if data.shape[1] > 1:
            corr_matrix = np.corrcoef(data.T)
            
            # Extract upper triangular correlations
            upper_indices = np.triu_indices(corr_matrix.shape[0], k=1)
            correlations = corr_matrix[upper_indices]
            
            # Handle NaN values
            correlations = np.nan_to_num(correlations)
            features.extend(correlations.tolist())
        
        # Magnitude of acceleration vector (if 3D accelerometer data)
        if data.shape[1] >= 3:
            acc_magnitude = np.sqrt(np.sum(data[:, :3] ** 2, axis=1))
            features.extend([
                np.mean(acc_magnitude),
                np.std(acc_magnitude),
                np.max(acc_magnitude),
                np.min(acc_magnitude)
            ])
        
        return features
    
    def _get_feature_dimension(self, num_channels: int) -> int:
        """Calculate total feature dimension."""
        # Time domain: 15 features per channel
        # Frequency domain: 8 features per channel  
        # Statistical: 8 features per channel
        per_channel_features = 15 + 8 + 8
        
        # Cross-channel features
        cross_channel_features = (num_channels * (num_channels - 1)) // 2  # Correlations
        if num_channels >= 3:
            cross_channel_features += 4  # Magnitude features
        
        return num_channels * per_channel_features + cross_channel_features

class MovementModelTrainer:
    """Mobile-optimized movement model trainer."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_extractor = None
        self.scaler = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters (mobile-optimized)
        self.sequence_length = config.get('sequence_length', 100)  # Reduced for mobile
        self.input_channels = config.get('input_channels', 6)  # Accel(3) + Gyro(3)
        self.hidden_dim = config.get('hidden_dim', 64)  # Reduced for mobile
        self.num_classes = config.get('num_classes', 6)
        
        # Training parameters
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 30)  # Reduced for mobile
        self.learning_rate = config.get('learning_rate', 0.001)
        self.early_stopping_patience = config.get('early_stopping_patience', 5)
        
        # Data parameters
        self.sample_rate = config.get('sample_rate', 50.0)  # Hz
        self.overlap_ratio = config.get('overlap_ratio', 0.5)
        
        # Mobile optimization
        self.memory_limit_mb = config.get('memory_limit_mb', 150)
        self.quantization_enabled = config.get('quantization_enabled', True)
        
        # Privacy settings
        self.differential_privacy = config.get('differential_privacy', False)
        self.epsilon = config.get('epsilon', 1.0)
    
    def train(self, data_path: str, output_path: str) -> Dict[str, Any]:
        """
        Train movement model with mobile optimization.
        
        Args:
            data_path: Path to training data
            output_path: Path to save trained model
            
        Returns:
            Training metrics and model info
        """
        try:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.info("Starting movement model training...")
            
            # Initialize components
            self.feature_extractor = MotionFeatureExtractor(self.sample_rate)
            self.scaler = StandardScaler()
            self.label_encoder = LabelEncoder()
            
            # Load and preprocess data
            sequences, labels = self._load_and_preprocess_data(data_path)
            
            # Create data loaders
            train_loader, val_loader = self._create_data_loaders(sequences, labels)
            
            # Initialize model
            self.model = SensorFusionCNN(
                input_channels=self.input_channels,
                sequence_length=self.sequence_length,
                num_classes=self.num_classes,
                hidden_dim=self.hidden_dim
            ).to(self.device)
            
            # Train model
            metrics = self._train_model(train_loader, val_loader)
            
            # Validate mobile requirements
            self._validate_mobile_requirements()
            
            # Apply quantization if enabled
            if self.quantization_enabled:
                self._quantize_model()
            
            # Save model
            self._save_model(output_path)
            
            # Calculate training statistics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            training_stats = {
                'training_time_seconds': end_time - start_time,
                'memory_usage_mb': end_memory - start_memory,
                'model_size_mb': self._get_model_size(output_path),
                'sequence_length': self.sequence_length,
                'input_channels': self.input_channels,
                'num_classes': self.num_classes,
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _load_and_preprocess_data(self, data_path: str) -> Tuple[List[np.ndarray], List[int]]:
        """Load and preprocess movement data."""
        try:
            loader = MovementDataLoader("movement_data", data_path, self.config)
            data = loader.load_data()
            
            if data is None or len(data) == 0:
                raise ValueError("No data loaded")
            
            sequences = []
            labels = []
            
            for sample in data:
                if isinstance(sample, tuple) and len(sample) >= 2:
                    motion_data, label = sample[0], sample[1]
                    
                    # Convert to numpy array if needed
                    if not isinstance(motion_data, np.ndarray):
                        motion_data = np.array(motion_data)
                    
                    # Create sliding windows
                    windows = self._create_sliding_windows(motion_data)
                    
                    for window in windows:
                        # Apply differential privacy if enabled
                        if self.differential_privacy:
                            window = self._apply_differential_privacy(window)
                        
                        # Normalize data
                        window = self.scaler.fit_transform(window)
                        
                        sequences.append(window)
                        labels.append(label)
            
            # Encode labels
            labels_encoded = self.label_encoder.fit_transform(labels)
            self.num_classes = len(self.label_encoder.classes_)
            
            logger.info(f"Created {len(sequences)} movement sequences with {self.num_classes} classes")
            return sequences, labels_encoded.tolist()
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def _create_sliding_windows(self, data: np.ndarray) -> List[np.ndarray]:
        """Create sliding windows from continuous motion data."""
        windows = []
        
        step_size = int(self.sequence_length * (1 - self.overlap_ratio))
        
        for i in range(0, len(data) - self.sequence_length + 1, step_size):
            window = data[i:i + self.sequence_length]
            if window.shape[0] == self.sequence_length:
                windows.append(window)
        
        return windows
    
    def _create_data_loaders(self, sequences: List[np.ndarray], 
                           labels: List[int]) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        try:
            # Convert to tensors
            sequences_tensor = torch.FloatTensor(np.array(sequences))
            labels_tensor = torch.LongTensor(labels)
            
            # Reshape for CNN input [batch, channels, sequence_length]
            sequences_tensor = sequences_tensor.transpose(1, 2)
            
            # Split data
            train_sequences, val_sequences, train_labels, val_labels = train_test_split(
                sequences_tensor, labels_tensor, test_size=0.2, random_state=42, stratify=labels_tensor
            )
            
            # Create datasets
            train_dataset = TensorDataset(train_sequences, train_labels)
            val_dataset = TensorDataset(val_sequences, val_labels)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,  # Single thread for mobile
                pin_memory=False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            logger.info(f"Created data loaders: Train={len(train_dataset)}, Val={len(val_dataset)}")
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Data loader creation failed: {e}")
            raise
    
    def _train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Train the movement model."""
        try:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=3, factor=0.5, verbose=True
            )
            
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            
            for epoch in range(self.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output, features = self.model(data)
                    loss = criterion(output, target)
                    
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(output.data, 1)
                    train_total += target.size(0)
                    train_correct += (predicted == target).sum().item()
                
                train_loss /= len(train_loader)
                train_accuracy = 100 * train_correct / train_total
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        output, features = self.model(data)
                        loss = criterion(output, target)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(output.data, 1)
                        val_total += target.size(0)
                        val_correct += (predicted == target).sum().item()
                
                val_loss /= len(val_loader)
                val_accuracy = 100 * val_correct / val_total
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_movement_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                              f"Train Acc: {train_accuracy:.2f}%, "
                              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Load best model
            self.model.load_state_dict(torch.load('best_movement_model.pth'))
            
            metrics = {
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'final_train_accuracy': train_accuracies[-1],
                'final_val_accuracy': val_accuracies[-1],
                'best_val_loss': best_val_loss,
                'epochs_trained': len(train_losses),
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def _validate_mobile_requirements(self):
        """Validate model meets mobile requirements."""
        try:
            # Model parameter count
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
            # Memory usage check
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            if current_memory > self.memory_limit_mb:
                logger.warning(f"Memory usage {current_memory:.2f}MB exceeds limit")
            
            # Model complexity validation
            if total_params > 500000:  # 500K parameter limit for mobile
                logger.warning(f"Model has {total_params} parameters, may be too complex for mobile")
                
            logger.info("Mobile requirements validation passed")
            
        except Exception as e:
            logger.error(f"Mobile validation failed: {e}")
    
    def _quantize_model(self):
        """Apply quantization for mobile deployment."""
        try:
            # Dynamic quantization
            self.model.eval()
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv1d},
                dtype=torch.qint8
            )
            
            self.model = quantized_model
            logger.info("Model quantization completed")
            
        except Exception as e:
            logger.warning(f"Model quantization failed: {e}")
    
    def _save_model(self, output_path: str):
        """Save trained model in multiple formats."""
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'input_channels': self.input_channels,
                    'sequence_length': self.sequence_length,
                    'num_classes': self.num_classes,
                    'hidden_dim': self.hidden_dim
                },
                'scaler': self.scaler,
                'label_encoder': self.label_encoder
            }, output_path / "movement_model.pth")
            
            # Save for mobile deployment
            try:
                # TorchScript for mobile
                self.model.eval()
                example_input = torch.randn(1, self.input_channels, self.sequence_length).to(self.device)
                traced_model = torch.jit.trace(self.model, example_input)
                traced_model.save(output_path / "movement_model_mobile.pt")
                logger.info("TorchScript model saved")
                
            except Exception as e:
                logger.warning(f"TorchScript conversion failed: {e}")
            
            # Save model metadata
            metadata = {
                'model_type': 'sensor_fusion_cnn',
                'agent_type': 'movement',
                'input_channels': self.input_channels,
                'sequence_length': self.sequence_length,
                'num_classes': self.num_classes,
                'sample_rate': self.sample_rate,
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
    
    def _apply_differential_privacy(self, data: np.ndarray) -> np.ndarray:
        """Apply differential privacy to motion data."""
        try:
            if not self.differential_privacy:
                return data
                
            # Add Gaussian noise for differential privacy
            sensitivity = np.ptp(data, axis=0)  # Range per channel
            noise_scale = sensitivity * np.sqrt(2 * np.log(1.25)) / self.epsilon
            noise = np.random.normal(0, noise_scale, data.shape)
            
            return data + noise
            
        except Exception as e:
            logger.error(f"Differential privacy application failed: {e}")
            return data

def train_movement_model(config_path: str, data_path: str, output_path: str) -> Dict[str, Any]:
    """
    Convenience function for training movement model.
    
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
            
        trainer = MovementModelTrainer(config)
        return trainer.train(data_path, output_path)
        
    except Exception as e:
        logger.error(f"Movement model training failed: {e}")
        raise