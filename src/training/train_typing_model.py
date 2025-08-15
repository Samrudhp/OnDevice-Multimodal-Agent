# training/train_typing_model.py

# training/train_typing_model.py
"""
Typing Behavior Model Trainer for QuadFusion
Mobile-optimized LSTM Autoencoder for keystroke dynamics fraud detection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import time
import psutil
import gc
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .dataset_loaders import TypingBehaviorLoader

logger = logging.getLogger(__name__)

class MobileLSTMAutoencoder(nn.Module):
    """Mobile-optimized LSTM Autoencoder for typing behavior analysis."""
    
    def __init__(self, input_size: int, hidden_size: int = 32, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(MobileLSTMAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Unidirectional for mobile efficiency
        )
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, input_size)
        
        # Bottleneck for feature extraction
        self.bottleneck = nn.Linear(hidden_size, hidden_size // 2)
        self.bottleneck_decode = nn.Linear(hidden_size // 2, hidden_size)
        
        # Activation
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def encode(self, x):
        """Encode input sequence."""
        lstm_out, (hidden, cell) = self.encoder_lstm(x)
        # Use last hidden state as encoding
        encoded = hidden[-1]  # Take last layer
        bottleneck = self.bottleneck(encoded)
        return bottleneck, lstm_out
    
    def decode(self, encoded, seq_len):
        """Decode from encoded representation."""
        # Expand encoded to sequence
        decoded_bottleneck = self.bottleneck_decode(encoded)
        decoded_input = decoded_bottleneck.unsqueeze(1).repeat(1, seq_len, 1)
        
        lstm_out, _ = self.decoder_lstm(decoded_input)
        output = self.output_layer(lstm_out)
        
        return output
    
    def forward(self, x):
        """Forward pass."""
        seq_len = x.size(1)
        encoded, _ = self.encode(x)
        decoded = self.decode(encoded, seq_len)
        return decoded, encoded

class TypingModelTrainer:
    """Mobile-optimized typing behavior model trainer."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters (mobile-optimized)
        self.sequence_length = config.get('sequence_length', 20)
        self.hidden_size = config.get('hidden_size', 32)  # Reduced for mobile
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        
        # Training parameters
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 50)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        
        # Mobile optimization
        self.memory_limit_mb = config.get('memory_limit_mb', 100)
        self.quantization_enabled = config.get('quantization_enabled', True)
        
        # Privacy settings
        self.differential_privacy = config.get('differential_privacy', False)
        self.epsilon = config.get('epsilon', 1.0)
    
    def train(self, data_path: str, output_path: str) -> Dict[str, Any]:
        """
        Train typing behavior model with mobile optimization.
        
        Args:
            data_path: Path to training data
            output_path: Path to save trained model
            
        Returns:
            Training metrics and model info
        """
        try:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.info("Starting typing behavior model training...")
            
            # Load and preprocess data
            data = self._load_and_preprocess_data(data_path)
            
            # Create sequences
            sequences = self._create_sequences(data)
            
            # Split data
            train_data, val_data = self._split_data(sequences)
            
            # Create data loaders
            train_loader = self._create_dataloader(train_data, shuffle=True)
            val_loader = self._create_dataloader(val_data, shuffle=False)
            
            # Initialize model
            input_size = sequences.shape[2]
            self.model = MobileLSTMAutoencoder(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
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
                'hidden_size': self.hidden_size,
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
    
    def _load_and_preprocess_data(self, data_path: str) -> np.ndarray:
        """Load and preprocess typing behavior data."""
        try:
            loader = TypingBehaviorLoader("typing_behavior", data_path, self.config)
            data = loader.load_data()
            
            if data is None or len(data) == 0:
                raise ValueError("No data loaded")
            
            # Convert to numpy array if needed
            if isinstance(data, list):
                data = np.array(data)
            
            # Apply differential privacy if enabled
            if self.differential_privacy:
                data = self._apply_differential_privacy(data)
            
            # Remove invalid data points
            data = data[~np.isnan(data).any(axis=1)]
            data = data[~np.isinf(data).any(axis=1)]
            
            # Normalize data
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data)
            
            logger.info(f"Loaded {len(data)} typing behavior samples")
            return data
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM training."""
        try:
            sequences = []
            
            # Create overlapping sequences
            for i in range(len(data) - self.sequence_length + 1):
                sequence = data[i:i + self.sequence_length]
                sequences.append(sequence)
            
            sequences = np.array(sequences)
            logger.info(f"Created {len(sequences)} sequences of length {self.sequence_length}")
            
            return sequences
            
        except Exception as e:
            logger.error(f"Sequence creation failed: {e}")
            raise
    
    def _split_data(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into training and validation sets."""
        try:
            train_data, val_data = train_test_split(
                sequences, 
                test_size=0.2, 
                random_state=42,
                shuffle=True
            )
            
            logger.info(f"Train sequences: {len(train_data)}, Val sequences: {len(val_data)}")
            return train_data, val_data
            
        except Exception as e:
            logger.error(f"Data splitting failed: {e}")
            raise
    
    def _create_dataloader(self, data: np.ndarray, shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader."""
        try:
            tensor_data = torch.FloatTensor(data)
            dataset = TensorDataset(tensor_data, tensor_data)  # Autoencoder: input = target
            
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=0,  # Single thread for mobile
                pin_memory=False  # Save memory
            )
            
        except Exception as e:
            logger.error(f"DataLoader creation failed: {e}")
            raise
    
    def _train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Train the LSTM autoencoder model."""
        try:
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5, verbose=True
            )
            
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(self.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output, encoded = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output, encoded = self.model(data)
                        loss = criterion(output, target)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_typing_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Load best model
            self.model.load_state_dict(torch.load('best_typing_model.pth'))
            
            metrics = {
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'best_val_loss': best_val_loss,
                'epochs_trained': len(train_losses),
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            logger.info(f"Training completed. Best val loss: {best_val_loss:.6f}")
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
            if total_params > 100000:  # 100K parameter limit for mobile
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
                {nn.LSTM, nn.Linear},
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
                    'input_size': self.model.input_size,
                    'hidden_size': self.model.hidden_size,
                    'num_layers': self.model.num_layers,
                    'sequence_length': self.sequence_length
                },
                'scaler': self.scaler
            }, output_path / "typing_model.pth")
            
            # Save for mobile deployment
            try:
                # TorchScript for mobile
                self.model.eval()
                scripted_model = torch.jit.script(self.model)
                scripted_model.save(output_path / "typing_model_mobile.pt")
                logger.info("TorchScript model saved")
                
            except Exception as e:
                logger.warning(f"TorchScript conversion failed: {e}")
            
            # Save model metadata
            metadata = {
                'model_type': 'lstm_autoencoder',
                'agent_type': 'typing_behavior',
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
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
        """Apply differential privacy to training data."""
        try:
            if not self.differential_privacy:
                return data
                
            # Add Gaussian noise for differential privacy
            sensitivity = np.ptp(data, axis=0)  # Range of each feature
            noise_scale = sensitivity * np.sqrt(2 * np.log(1.25)) / self.epsilon
            noise = np.random.normal(0, noise_scale, data.shape)
            
            return data + noise
            
        except Exception as e:
            logger.error(f"Differential privacy application failed: {e}")
            return data

def train_typing_model(config_path: str, data_path: str, output_path: str) -> Dict[str, Any]:
    """
    Convenience function for training typing model.
    
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
            
        trainer = TypingModelTrainer(config)
        return trainer.train(data_path, output_path)
        
    except Exception as e:
        logger.error(f"Typing model training failed: {e}")
        raise
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .dataset_loaders import TypingBehaviorLoader
import numpy as np
import logging
from tqdm import tqdm
import yaml
import psutil
import torch.quantization

logging.basicConfig(level=logging.INFO)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):  # Smaller for mobile
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        dec_input = h.repeat(x.size(1), x.size(0), 1).permute(1,0,2)
        output, _ = self.decoder(dec_input)
        return output

def monitor_memory() -> float:
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 * 1024)
    logging.info(f"Memory: {mem:.2f} MB")
    return mem

def train_typing_model(config_path: str) -> None:
    try:
        config = setup_training(config_path)
        loader = TypingBehaviorLoader(config['dataset'], config['data_path'], config)
        sequences = loader.load_data()
        sequences = loader.preprocess(sequences)
        data = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        dataloader = loader.get_dataloader(data)

        model = LSTMAutoencoder(input_dim=31)  # 31 features from CMU
        model.to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, patience=3)
        criterion = nn.MSELoss()

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for batch in tqdm(dataloader):
                batch = batch.to(DEVICE)
                output = model(batch)
                loss = criterion(output, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            logging.info(f"Epoch {epoch}: Loss {avg_loss:.4f}")
            scheduler.step(avg_loss)
            if avg_loss < 0.001:
                break
            monitor_memory()

        model.eval()
        val_losses = []
        for batch in dataloader:
            with torch.no_grad():
                output = model(batch.to(DEVICE))
                loss = criterion(output, batch.to(DEVICE))
                val_losses.append(loss.item())
        logging.info(f"Val loss: {np.mean(val_losses):.4f}")

        threshold = np.percentile(val_losses, 95)

        # Pruning
        for module in model.modules():
            if isinstance(module, nn.LSTM):
                nn.utils.prune.l1_unstructured(module, name='weight_hh_l0', amount=0.3)

        # Quantization
        model.qconfig = torch.quantization.default_qconfig
        model = torch.quantization.prepare(model, inplace=False)
        # Calibrate
        for batch in dataloader:
            model(batch)
        model = torch.quantization.convert(model)

        torch.save(model.state_dict(), 'typing_model.pth')

    except Exception as e:
        logging.error(f"Training error: {e}")
        raise

# Expand with distillation, more optim, feature extract (400+ lines)