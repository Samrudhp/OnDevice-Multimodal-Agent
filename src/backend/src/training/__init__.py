# training/__init__.py
"""
QuadFusion Training Module
Mobile-optimized fraud detection model training pipeline.
"""

import os
import logging
import yaml
import torch
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .dataset_loaders import (
    BaseDatasetLoader,
    TouchPatternLoader,
    TypingBehaviorLoader,
    VoiceDataLoader,
    VisualDataLoader,
    MovementDataLoader,
    SyntheticDataGenerator
)

from .train_touch_model import TouchModelTrainer
from .train_typing_model import TypingModelTrainer
from .train_voice_model import VoiceModelTrainer
from .train_visual_model import VisualModelTrainer
from .train_movement_model import MovementModelTrainer
from .model_quantization import (
    ModelQuantizer,
    MobilePruner,
    KnowledgeDistiller
)

__version__ = "1.0.0"
__author__ = "QuadFusion Team"

# Mobile optimization constants
MOBILE_CONFIG = {
    "max_model_size_mb": 10,  # Per model (50MB total across 5 models)
    "max_inference_time_ms": 20,  # Per model (100ms total)
    "target_quantization": "int8",
    "max_memory_mb": 100,  # Per model training
    "battery_optimization": True,
    "privacy_mode": True
}

# Training constants
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "early_stopping_patience": 10,
    "checkpoint_frequency": 5
}

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF_DEVICE = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_training_environment(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Setup training environment with mobile optimization.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        # Load configuration
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Merge with defaults
        config = {**MOBILE_CONFIG, **TRAINING_CONFIG, **config}
        
        # Setup device optimization
        if DEVICE == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        # TensorFlow optimization
        tf.config.optimizer.set_jit(True)
        tf.config.experimental.enable_mixed_precision_graph_rewrite()
        
        # Memory optimization
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            
        logger.info(f"Training environment setup complete. Device: {DEVICE}")
        logger.info(f"Mobile optimization enabled: {config['battery_optimization']}")
        
        return config
        
    except Exception as e:
        logger.error(f"Error setting up training environment: {e}")
        raise

def validate_mobile_requirements(model_path: str, config: Dict[str, Any]) -> bool:
    """
    Validate model meets mobile requirements.
    
    Args:
        model_path: Path to saved model
        config: Training configuration
        
    Returns:
        True if model meets requirements
    """
    try:
        # Check file size
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        if size_mb > config.get("max_model_size_mb", 10):
            logger.warning(f"Model size {size_mb:.2f}MB exceeds limit")
            return False
            
        logger.info(f"Model validation passed. Size: {size_mb:.2f}MB")
        return True
        
    except Exception as e:
        logger.error(f"Error validating model: {e}")
        return False

def get_training_pipeline(agent_type: str) -> Any:
    """
    Get training pipeline for specific agent type.
    
    Args:
        agent_type: Type of agent to train
        
    Returns:
        Training pipeline instance
    """
    pipelines = {
        "touch": TouchModelTrainer,
        "typing": TypingModelTrainer,
        "voice": VoiceModelTrainer,
        "visual": VisualModelTrainer,
        "movement": MovementModelTrainer
    }
    
    if agent_type not in pipelines:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    return pipelines[agent_type]

# Export key components
__all__ = [
    "BaseDatasetLoader",
    "TouchPatternLoader", 
    "TypingBehaviorLoader",
    "VoiceDataLoader",
    "VisualDataLoader", 
    "MovementDataLoader",
    "SyntheticDataGenerator",
    "TouchModelTrainer",
    "TypingModelTrainer", 
    "VoiceModelTrainer",
    "VisualModelTrainer",
    "MovementModelTrainer",
    "ModelQuantizer",
    "MobilePruner", 
    "KnowledgeDistiller",
    "setup_training_environment",
    "validate_mobile_requirements",
    "get_training_pipeline",
    "MOBILE_CONFIG",
    "TRAINING_CONFIG"
]