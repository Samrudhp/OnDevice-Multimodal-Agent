# agents/visual_agent.py
"""
Visual Agent for face and scene recognition using CLIP-Tiny.
Performs similarity matching against stored embeddings for user verification.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import cv2
from PIL import Image
import os
try:
    import torch
    import torchvision.transforms as transforms
    _TORCH_AVAILABLE = True
except Exception:
    transforms = None
    _TORCH_AVAILABLE = False
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import joblib
import time
import base64
import io

from .base_agent import BaseAgent, AgentResult, RiskLevel

class VisualAgent(BaseAgent):
    """
    Agent for visual authentication using face and scene recognition.
    Uses CLIP-Tiny model for embedding generation and similarity matching.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("VisualAgent", config)
        
        # Image processing parameters
        self.image_size = config.get('image_size', 224)
        self.max_faces = config.get('max_faces', 5)
        self.face_confidence_threshold = config.get('face_confidence_threshold', 0.7)
        self.scene_similarity_threshold = config.get('scene_similarity_threshold', 0.6)
        
        # Model parameters
        self.embedding_dim = config.get('embedding_dim', 512)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        
        # Initialize models (simplified - would use actual CLIP-Tiny in production)
        self.clip_model = None
        self.face_detector = None
        self.embedding_scaler = StandardScaler()
        
        # Stored embeddings for enrolled user
        self.enrolled_face_embeddings = []
        self.enrolled_scene_embeddings = []
        self.user_id = None
        
        # Image preprocessing (only if torchvision available)
        if _TORCH_AVAILABLE and transforms is not None:
            self.image_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = None
        
        # Analysis buffers
        self.recent_similarities = []
        self.face_detection_history = []
    
    def capture_data(self, sensor_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Capture and preprocess visual data.
        
        Expected sensor_data format:
        {
            'image_data': np.ndarray,  # RGB image array (H, W, 3)
            'timestamp': float,
            'camera_type': str,  # 'front' or 'rear'
            'lighting_condition': str,  # optional
            'image_base64': str  # alternative to image_data
        }
        
        Returns:
            Combined feature vector: [face_embedding, scene_embedding, metadata_features]
        """
        try:
            # Get image data
            image_data = sensor_data.get('image_data')
            image_base64 = sensor_data.get('image_base64')
            
            if image_data is None and image_base64 is None:
                return None
            
            # Convert base64 to image if needed
            if image_data is None and image_base64:
                image_data = self._base64_to_image(image_base64)
                if image_data is None:
                    return None
            
            # Validate image
            if image_data is None or len(image_data.shape) != 3 or image_data.shape[2] != 3:
                return None
            
            # Convert to PIL Image for processing
            if image_data.dtype != np.uint8:
                image_data = (image_data * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image_data)
            
            # Extract features
            face_features = self._extract_face_features(image_data)
            scene_features = self._extract_scene_features(pil_image)
            metadata_features = self._extract_metadata_features(sensor_data, image_data)
            
            if face_features is None or scene_features is None:
                return None
            
            # Combine all features
            combined_features = np.concatenate([
                face_features,
                scene_features,
                metadata_features
            ])
            
            return combined_features
            
        except Exception as e:
            print(f"Error capturing visual data: {e}")
            return None
    
    def _base64_to_image(self, base64_string: str) -> Optional[np.ndarray]:
        """Convert base64 string to numpy image array"""
        try:
            image_bytes = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_bytes))
            image_rgb = image.convert('RGB')
            return np.array(image_rgb)
        except Exception as e:
            print(f"Error converting base64 to image: {e}")
            return None
    
    def _extract_face_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face features using face detection and embedding generation.
        
        Args:
            image: RGB image array
            
        Returns:
            Face embedding features or zeros if no face detected
        """
        try:
            # Initialize face detector if not already done (using OpenCV for demo)
            if self.face_detector is None:
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                # No face detected - return zeros
                self.face_detection_history.append(False)
                return np.zeros(self.embedding_dim)
            
            self.face_detection_history.append(True)
            
            # Keep only recent detection history
            if len(self.face_detection_history) > 100:
                self.face_detection_history = self.face_detection_history[-100:]
            
            # Get the largest face (assumed to be the primary subject)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Extract face region with some padding
            padding = int(min(w, h) * 0.2)
            face_region = image[
                max(0, y-padding):min(image.shape[0], y+h+padding),
                max(0, x-padding):min(image.shape[1], x+w+padding)
            ]
            
            if face_region.size == 0:
                return np.zeros(self.embedding_dim)
            
            # Generate face embedding (simplified - would use actual CLIP model)
            face_embedding = self._generate_clip_embedding(face_region, is_face=True)
            
            return face_embedding
            
        except Exception as e:
            print(f"Error extracting face features: {e}")
            return np.zeros(self.embedding_dim)
    
    def _extract_scene_features(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Extract scene features using CLIP-Tiny.
        
        Args:
            image: PIL Image
            
        Returns:
            Scene embedding features
        """
        try:
            # Generate scene embedding (simplified - would use actual CLIP model)
            image_array = np.array(image)
            scene_embedding = self._generate_clip_embedding(image_array, is_face=False)
            
            return scene_embedding
            
        except Exception as e:
            print(f"Error extracting scene features: {e}")
            return np.zeros(self.embedding_dim)
    
    def _generate_clip_embedding(self, image_data: np.ndarray, is_face: bool = False) -> np.ndarray:
        """
        Generate CLIP-style embedding for image data.
        This is a simplified version - in production, would use actual CLIP-Tiny model.
        
        Args:
            image_data: Image array
            is_face: Whether this is a face region or full scene
            
        Returns:
            Embedding vector
        """
        try:
            # Simplified feature extraction using traditional computer vision
            # In production, this would be replaced with actual CLIP-Tiny inference
            
            # Resize image
            if len(image_data.shape) == 3:
                resized = cv2.resize(image_data, (self.image_size, self.image_size))
            else:
                resized = cv2.resize(image_data, (self.image_size, self.image_size))
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            
            # Extract basic visual features as a proxy for CLIP embeddings
            features = []
            
            # Color histogram features
            for i in range(3):  # RGB channels
                hist = cv2.calcHist([resized], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
            
            # Texture features using Local Binary Patterns
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            
            # Simple texture analysis
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Statistical features
            features.extend([
                np.mean(gray), np.std(gray),
                np.mean(sobel_x), np.std(sobel_x),
                np.mean(sobel_y), np.std(sobel_y)
            ])
            
            # If face, add specific face-related features
            if is_face:
                # Eye region analysis (simplified)
                h, w = gray.shape
                eye_region = gray[int(h*0.2):int(h*0.5), int(w*0.2):int(w*0.8)]
                features.extend([
                    np.mean(eye_region), np.std(eye_region)
                ])
                
                # Mouth region analysis
                mouth_region = gray[int(h*0.6):int(h*0.9), int(w*0.3):int(w*0.7)]
                features.extend([
                    np.mean(mouth_region), np.std(mouth_region)
                ])
            
            # Pad or truncate to target embedding dimension
            features_array = np.array(features)
            if len(features_array) > self.embedding_dim:
                embedding = features_array[:self.embedding_dim]
            else:
                embedding = np.zeros(self.embedding_dim)
                embedding[:len(features_array)] = features_array
            
            # Normalize embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def _extract_metadata_features(self, sensor_data: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Extract metadata features from image and sensor data.
        
        Args:
            sensor_data: Original sensor data
            image: Image array
            
        Returns:
            Metadata feature vector
        """
        try:
            # Image quality metrics
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Face detection confidence
            face_detected = len(self.face_detection_history) > 0 and self.face_detection_history[-1]
            face_detection_rate = sum(self.face_detection_history[-20:]) / min(20, len(self.face_detection_history)) if self.face_detection_history else 0
            
            # Camera type encoding
            camera_type = sensor_data.get('camera_type', 'unknown')
            is_front_camera = 1.0 if camera_type == 'front' else 0.0
            
            # Lighting condition encoding
            lighting = sensor_data.get('lighting_condition', 'normal')
            lighting_score = {
                'low': 0.2, 'normal': 1.0, 'bright': 0.6, 'artificial': 0.8
            }.get(lighting, 0.5)
            
            metadata_features = np.array([
                brightness / 255.0,  # Normalize brightness
                contrast / 128.0,    # Normalize contrast  
                laplacian_var / 1000.0,  # Normalize blur metric
                float(face_detected),
                face_detection_rate,
                is_front_camera,
                lighting_score
            ])
            
            return metadata_features
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata features: {e}")
            # Return default metadata features if extraction fails
            return np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5])
    
    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """
        Encode image to base64 string for storage.
        
        Args:
            image: Image array
            
        Returns:
            Base64 encoded string
        """
        try:
            # Convert to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
            
            return base64.b64encode(image_bytes).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Error encoding image to base64: {e}")
            return ""
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        return {
            'agent_name': self.agent_name,
            'is_enrolled': len(self.enrolled_face_embeddings) > 0,
            'num_face_embeddings': len(self.enrolled_face_embeddings),
            'num_scene_embeddings': len(self.enrolled_scene_embeddings),
            'processing_times': getattr(self, 'inference_times', [])[-100:],
            'face_detection_history': self.face_detection_history[-100:],
            'last_analysis': getattr(self, 'last_update_time', None)
        }

    # -------------------------
    # Implement abstract methods
    # -------------------------
    def train_initial(self, training_data: List[np.ndarray]) -> bool:
        """Initial training: store embeddings from provided training data."""
        try:
            self.enrolled_face_embeddings = []
            self.enrolled_scene_embeddings = []

            for sample in training_data:
                # Expecting combined vector: [face(emb_dim), scene(emb_dim), metadata(7)]
                if sample is None or len(sample) < (2 * self.embedding_dim):
                    continue
                face = sample[:self.embedding_dim]
                scene = sample[self.embedding_dim:2*self.embedding_dim]
                self.enrolled_face_embeddings.append(face / (np.linalg.norm(face) + 1e-8))
                self.enrolled_scene_embeddings.append(scene / (np.linalg.norm(scene) + 1e-8))

            if len(self.enrolled_face_embeddings) == 0 and len(self.enrolled_scene_embeddings) == 0:
                self.is_trained = False
                return False

            self.is_trained = True
            return True
        except Exception as e:
            print(f"VisualAgent.train_initial error: {e}")
            return False

    def predict(self, features: np.ndarray) -> AgentResult:
        """Make a prediction from a feature vector and return an AgentResult."""
        start = time.time()
        try:
            if features is None or len(features) < (2 * self.embedding_dim):
                # Cannot predict without full features
                anomaly_score = 0.5
                risk = self.get_risk_level_from_score(anomaly_score)
                return AgentResult(
                    agent_name=self.agent_name,
                    anomaly_score=anomaly_score,
                    risk_level=risk,
                    confidence=0.5,
                    features_used=['visual_features'],
                    processing_time_ms=0.0,
                    metadata={}
                )

            face = features[:self.embedding_dim]
            scene = features[self.embedding_dim:2*self.embedding_dim]

            # Normalize
            face = face / (np.linalg.norm(face) + 1e-8)
            scene = scene / (np.linalg.norm(scene) + 1e-8)

            # Compute similarity if enrolled embeddings exist
            face_sim = 0.0
            scene_sim = 0.0

            if len(self.enrolled_face_embeddings) > 0:
                sims = cosine_similarity([face], np.vstack(self.enrolled_face_embeddings))
                face_sim = float(np.max(sims))

            if len(self.enrolled_scene_embeddings) > 0:
                sims2 = cosine_similarity([scene], np.vstack(self.enrolled_scene_embeddings))
                scene_sim = float(np.max(sims2))

            # Weighted combination
            combined_sim = 0.6 * face_sim + 0.4 * scene_sim
            anomaly_score = float(max(0.0, min(1.0, 1.0 - combined_sim)))
            risk = self.get_risk_level_from_score(anomaly_score)
            confidence = float(combined_sim)

            processing_time = (time.time() - start) * 1000.0

            return AgentResult(
                agent_name=self.agent_name,
                anomaly_score=anomaly_score,
                risk_level=risk,
                confidence=confidence,
                features_used=['face_embedding', 'scene_embedding'],
                processing_time_ms=processing_time,
                metadata={'face_similarity': face_sim, 'scene_similarity': scene_sim}
            )

        except Exception as e:
            print(f"VisualAgent.predict error: {e}")
            processing_time = (time.time() - start) * 1000.0
            return AgentResult(
                agent_name=self.agent_name,
                anomaly_score=1.0,
                risk_level=self.get_risk_level_from_score(1.0),
                confidence=0.0,
                features_used=[],
                processing_time_ms=processing_time,
                metadata={'error': str(e)}
            )

    def incremental_update(self, new_data: List[np.ndarray], is_anomaly: List[bool] = None) -> bool:
        """Update enrolled embeddings / baseline with new data."""
        try:
            for sample in new_data:
                if sample is None or len(sample) < (2 * self.embedding_dim):
                    continue
                face = sample[:self.embedding_dim]
                scene = sample[self.embedding_dim:2*self.embedding_dim]
                self.enrolled_face_embeddings.append(face / (np.linalg.norm(face) + 1e-8))
                self.enrolled_scene_embeddings.append(scene / (np.linalg.norm(scene) + 1e-8))

            # Keep only recent embeddings up to a cap
            max_emb = self.config.get('max_embeds', 200)
            if len(self.enrolled_face_embeddings) > max_emb:
                self.enrolled_face_embeddings = self.enrolled_face_embeddings[-max_emb:]
            if len(self.enrolled_scene_embeddings) > max_emb:
                self.enrolled_scene_embeddings = self.enrolled_scene_embeddings[-max_emb:]

            self.is_trained = True if (len(self.enrolled_face_embeddings) + len(self.enrolled_scene_embeddings)) > 0 else False
            return True
        except Exception as e:
            print(f"VisualAgent.incremental_update error: {e}")
            return False

    def save_model(self, filepath: str) -> bool:
        """Save embeddings and config to disk using joblib."""
        try:
            data = {
                'enrolled_face_embeddings': np.vstack(self.enrolled_face_embeddings) if len(self.enrolled_face_embeddings) > 0 else np.empty((0, self.embedding_dim)),
                'enrolled_scene_embeddings': np.vstack(self.enrolled_scene_embeddings) if len(self.enrolled_scene_embeddings) > 0 else np.empty((0, self.embedding_dim)),
                'config': self.config
            }
            joblib.dump(data, filepath)
            return True
        except Exception as e:
            print(f"VisualAgent.save_model error: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load embeddings and config from disk using joblib."""
        try:
            if not filepath or not os.path.exists(filepath):
                return False
            data = joblib.load(filepath)
            face = data.get('enrolled_face_embeddings')
            scene = data.get('enrolled_scene_embeddings')
            if face is not None and face.size > 0:
                self.enrolled_face_embeddings = [row / (np.linalg.norm(row) + 1e-8) for row in face]
            if scene is not None and scene.size > 0:
                self.enrolled_scene_embeddings = [row / (np.linalg.norm(row) + 1e-8) for row in scene]
            self.config.update(data.get('config', {}))
            self.is_trained = True if (len(self.enrolled_face_embeddings) + len(self.enrolled_scene_embeddings)) > 0 else False
            return True
        except Exception as e:
            print(f"VisualAgent.load_model error: {e}")
            return False