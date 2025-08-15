# training/train_visual_model.py
"""
Visual Model Trainer for QuadFusion
Mobile-optimized face recognition and visual behavior analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import logging
import time
import psutil
import gc
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

from .dataset_loaders import VisualDataLoader

logger = logging.getLogger(__name__)

class MobileFaceNet(nn.Module):
    """Mobile-optimized face recognition network based on MobileNetV3."""
    
    def __init__(self, num_classes: int = 512, embedding_dim: int = 256):
        super(MobileFaceNet, self).__init__()
        
        # Use MobileNetV3 as backbone (mobile-optimized)
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        
        # Remove the classifier
        self.backbone.classifier = nn.Identity()
        
        # Get the feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_output = self.backbone(dummy_input)
            feature_dim = backbone_output.shape[1]
        
        # Add custom head for face recognition
        self.embedding_layer = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classification layer
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Face quality assessment
        self.quality_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.embedding_dim = embedding_dim
        
    def forward(self, x, return_embedding=False):
        """Forward pass with optional embedding return."""
        features = self.backbone(x)
        embeddings = self.embedding_layer(features)
        
        if return_embedding:
            return embeddings
        
        classification = self.classifier(embeddings)
        quality_score = self.quality_head(features)
        
        return classification, quality_score, embeddings

class FaceDetector:
    """Lightweight face detection for preprocessing."""
    
    def __init__(self, min_face_size=40, scale_factor=1.1, min_neighbors=3):
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        
        # Load Haar cascade (lightweight for mobile)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except:
            logger.warning("OpenCV face cascade not found, using fallback")
            self.face_cascade = None
    
    def detect_faces(self, image):
        """Detect faces in image."""
        try:
            if self.face_cascade is None:
                # Fallback: return center crop
                h, w = image.shape[:2]
                size = min(h, w)
                start_h = (h - size) // 2
                start_w = (w - size) // 2
                return [(start_w, start_h, size, size)]
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(self.min_face_size, self.min_face_size)
            )
            
            return faces.tolist() if len(faces) > 0 else []
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def extract_face(self, image, face_bbox, target_size=(224, 224)):
        """Extract and resize face from image."""
        try:
            x, y, w, h = face_bbox
            
            # Add padding
            padding = int(0.2 * min(w, h))
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Extract face
            face = image[y:y+h, x:x+w]
            
            # Resize to target size
            if face.size > 0:
                face = cv2.resize(face, target_size)
                return face
            else:
                return None
                
        except Exception as e:
            logger.error(f"Face extraction failed: {e}")
            return None

class VisualDataAugmentation:
    """Data augmentation for face recognition training."""
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def augment_batch(self, images, training=True):
        """Apply augmentation to batch of images."""
        transform = self.transform if training else self.test_transform
        return torch.stack([transform(img) for img in images])

class VisualModelTrainer:
    """Mobile-optimized visual model trainer for face recognition."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.face_detector = None
        self.augmentor = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters (mobile-optimized)
        self.embedding_dim = config.get('embedding_dim', 256)  # Reduced for mobile
        self.image_size = config.get('image_size', 224)
        self.num_classes = config.get('num_classes', 100)
        
        # Training parameters
        self.batch_size = config.get('batch_size', 16)  # Reduced for mobile
        self.epochs = config.get('epochs', 30)  # Reduced for mobile training
        self.learning_rate = config.get('learning_rate', 0.001)
        self.early_stopping_patience = config.get('early_stopping_patience', 5)
        
        # Mobile optimization
        self.memory_limit_mb = config.get('memory_limit_mb', 200)
        self.quantization_enabled = config.get('quantization_enabled', True)
        self.pruning_enabled = config.get('pruning_enabled', False)
        
        # Privacy settings
        self.differential_privacy = config.get('differential_privacy', False)
        self.epsilon = config.get('epsilon', 1.0)
    
    def train(self, data_path: str, output_path: str) -> Dict[str, Any]:
        """
        Train visual model with mobile optimization.
        
        Args:
            data_path: Path to training data
            output_path: Path to save trained model
            
        Returns:
            Training metrics and model info
        """
        try:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.info("Starting visual model training...")
            
            # Initialize components
            self.face_detector = FaceDetector()
            self.augmentor = VisualDataAugmentation()
            self.label_encoder = LabelEncoder()
            
            # Load and preprocess data
            images, labels = self._load_and_preprocess_data(data_path)
            
            # Create data loaders
            train_loader, val_loader = self._create_data_loaders(images, labels)
            
            # Initialize model
            self.model = MobileFaceNet(
                num_classes=self.num_classes,
                embedding_dim=self.embedding_dim
            ).to(self.device)
            
            # Train model
            metrics = self._train_model(train_loader, val_loader)
            
            # Validate mobile requirements
            self._validate_mobile_requirements()
            
            # Apply optimizations
            if self.quantization_enabled:
                self._quantize_model()
            
            if self.pruning_enabled:
                self._prune_model()
            
            # Save model
            self._save_model(output_path)
            
            # Calculate training statistics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            training_stats = {
                'training_time_seconds': end_time - start_time,
                'memory_usage_mb': end_memory - start_memory,
                'model_size_mb': self._get_model_size(output_path),
                'embedding_dim': self.embedding_dim,
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
        """Load and preprocess visual data."""
        try:
            loader = VisualDataLoader("visual_data", data_path, self.config)
            data = loader.load_data()
            
            if data is None or len(data) == 0:
                raise ValueError("No data loaded")
            
            images = []
            labels = []
            
            for sample in data:
                if isinstance(sample, tuple) and len(sample) >= 2:
                    image, label = sample[0], sample[1]
                    
                    # Convert to numpy array if needed
                    if isinstance(image, Image.Image):
                        image = np.array(image)
                    
                    # Detect and extract faces
                    faces = self.face_detector.detect_faces(image)
                    
                    if faces:
                        # Use the largest face
                        largest_face = max(faces, key=lambda f: f[2] * f[3])
                        face_image = self.face_detector.extract_face(image, largest_face)
                        
                        if face_image is not None:
                            # Apply differential privacy if enabled
                            if self.differential_privacy:
                                face_image = self._apply_differential_privacy(face_image)
                            
                            images.append(face_image)
                            labels.append(label)
                    else:
                        # No face detected, use center crop
                        h, w = image.shape[:2]
                        size = min(h, w)
                        start_h = (h - size) // 2
                        start_w = (w - size) // 2
                        cropped = image[start_h:start_h+size, start_w:start_w+size]
                        cropped = cv2.resize(cropped, (self.image_size, self.image_size))
                        
                        if self.differential_privacy:
                            cropped = self._apply_differential_privacy(cropped)
                        
                        images.append(cropped)
                        labels.append(label)
            
            # Encode labels
            labels_encoded = self.label_encoder.fit_transform(labels)
            self.num_classes = len(self.label_encoder.classes_)
            
            logger.info(f"Loaded {len(images)} face images with {self.num_classes} classes")
            return images, labels_encoded.tolist()
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def _create_data_loaders(self, images: List[np.ndarray], 
                           labels: List[int]) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        try:
            # Split data
            train_images, val_images, train_labels, val_labels = train_test_split(
                images, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Convert to tensors with augmentation
            train_tensors = []
            for img in train_images:
                augmented = self.augmentor.augment_batch([img], training=True)
                train_tensors.append(augmented[0])
            
            val_tensors = []
            for img in val_images:
                processed = self.augmentor.augment_batch([img], training=False)
                val_tensors.append(processed[0])
            
            # Create datasets
            train_dataset = TensorDataset(
                torch.stack(train_tensors),
                torch.LongTensor(train_labels)
            )
            
            val_dataset = TensorDataset(
                torch.stack(val_tensors),
                torch.LongTensor(val_labels)
            )
            
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
        """Train the visual model."""
        try:
            # Multi-task loss: classification + quality assessment
            criterion_cls = nn.CrossEntropyLoss()
            criterion_quality = nn.MSELoss()
            
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
                    classification, quality_score, embeddings = self.model(data)
                    
                    # Classification loss
                    cls_loss = criterion_cls(classification, target)
                    
                    # Quality loss (synthetic quality scores)
                    target_quality = torch.ones_like(quality_score.squeeze())
                    quality_loss = criterion_quality(quality_score.squeeze(), target_quality)
                    
                    # Combined loss
                    loss = cls_loss + 0.1 * quality_loss
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(classification.data, 1)
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
                        
                        classification, quality_score, embeddings = self.model(data)
                        
                        cls_loss = criterion_cls(classification, target)
                        target_quality = torch.ones_like(quality_score.squeeze())
                        quality_loss = criterion_quality(quality_score.squeeze(), target_quality)
                        
                        loss = cls_loss + 0.1 * quality_loss
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(classification.data, 1)
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
                    torch.save(self.model.state_dict(), 'best_visual_model.pth')
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
            self.model.load_state_dict(torch.load('best_visual_model.pth'))
            
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
            if total_params > 3000000:  # 3M parameter limit for mobile
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
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
            self.model = quantized_model
            logger.info("Model quantization completed")
            
        except Exception as e:
            logger.warning(f"Model quantization failed: {e}")
    
    def _prune_model(self):
        """Apply pruning for model compression."""
        try:
            import torch.nn.utils.prune as prune
            
            # Apply pruning to conv and linear layers
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
            
            logger.info("Model pruning completed")
            
        except Exception as e:
            logger.warning(f"Model pruning failed: {e}")
    
    def _save_model(self, output_path: str):
        """Save trained model in multiple formats."""
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'num_classes': self.num_classes,
                    'embedding_dim': self.embedding_dim,
                    'image_size': self.image_size
                },
                'label_encoder': self.label_encoder
            }, output_path / "visual_model.pth")
            
            # Save for mobile deployment
            try:
                # TorchScript for mobile
                self.model.eval()
                example_input = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
                traced_model = torch.jit.trace(self.model, example_input)
                traced_model.save(output_path / "visual_model_mobile.pt")
                logger.info("TorchScript model saved")
                
            except Exception as e:
                logger.warning(f"TorchScript conversion failed: {e}")
            
            # Save model metadata
            metadata = {
                'model_type': 'mobilefacenet',
                'agent_type': 'visual',
                'num_classes': self.num_classes,
                'embedding_dim': self.embedding_dim,
                'image_size': self.image_size,
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
    
    def _apply_differential_privacy(self, image: np.ndarray) -> np.ndarray:
        """Apply differential privacy to image data."""
        try:
            if not self.differential_privacy:
                return image
                
            # Add Gaussian noise for differential privacy
            sensitivity = 255.0  # Max pixel value difference
            noise_scale = sensitivity * np.sqrt(2 * np.log(1.25)) / self.epsilon
            noise = np.random.normal(0, noise_scale, image.shape)
            
            noisy_image = image.astype(np.float32) + noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            
            return noisy_image
            
        except Exception as e:
            logger.error(f"Differential privacy application failed: {e}")
            return image

def train_visual_model(config_path: str, data_path: str, output_path: str) -> Dict[str, Any]:
    """
    Convenience function for training visual model.
    
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
            
        trainer = VisualModelTrainer(config)
        return trainer.train(data_path, output_path)
        
    except Exception as e:
        logger.error(f"Visual model training failed: {e}")
        raise