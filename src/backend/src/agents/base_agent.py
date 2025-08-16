# agents/base_agent.py
"""
Abstract base class for all fraud detection agents.
Provides standard interface and common functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import json
import time
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class AgentResult:
    """Standardized result from any agent"""
    agent_name: str
    anomaly_score: float  # 0.0 (normal) to 1.0 (highly anomalous)
    risk_level: RiskLevel
    confidence: float  # 0.0 to 1.0
    features_used: List[str]
    processing_time_ms: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'agent_name': self.agent_name,
            'anomaly_score': self.anomaly_score,
            'risk_level': self.risk_level.value,
            'confidence': self.confidence,
            'features_used': self.features_used,
            'processing_time_ms': self.processing_time_ms,
            'metadata': self.metadata or {}
        }

class BaseAgent(ABC):
    """
    Abstract base class for all fraud detection agents.
    Provides common interface and functionality.
    """
    
    def __init__(self, agent_name: str, config: Dict[str, Any]):
        self.agent_name = agent_name
        self.config = config
        self.is_trained = False
        self.model = None
        self.baseline_data = []
        self.last_update_time = time.time()
        
        # Performance tracking
        self.inference_times = []
        self.accuracy_scores = []
        
        # Incremental learning parameters
        self.min_samples_for_update = config.get('min_samples_for_update', 50)
        self.update_frequency_hours = config.get('update_frequency_hours', 24)
        self.max_baseline_samples = config.get('max_baseline_samples', 1000)
        
    @abstractmethod
    def capture_data(self, sensor_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Capture and preprocess sensor data for this agent.
        
        Args:
            sensor_data: Raw sensor data dictionary
            
        Returns:
            Preprocessed feature vector or None if data invalid
        """
        pass
    
    @abstractmethod
    def train_initial(self, training_data: List[np.ndarray]) -> bool:
        """
        Initial training of the agent model.
        
        Args:
            training_data: List of preprocessed training samples
            
        Returns:
            True if training successful, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> AgentResult:
        """
        Make prediction on preprocessed features.
        
        Args:
            features: Preprocessed feature vector
            
        Returns:
            AgentResult with anomaly score and metadata
        """
        pass
    
    @abstractmethod
    def incremental_update(self, new_data: List[np.ndarray], 
                          is_anomaly: List[bool] = None) -> bool:
        """
        Update model with new data for continual learning.
        
        Args:
            new_data: List of new preprocessed samples
            is_anomaly: Optional labels for supervised updates
            
        Returns:
            True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        pass
    
    def should_update_model(self) -> bool:
        """
        Check if model should be updated based on time and data availability.
        """
        time_since_update = time.time() - self.last_update_time
        hours_since_update = time_since_update / 3600
        
        has_enough_data = len(self.baseline_data) >= self.min_samples_for_update
        time_for_update = hours_since_update >= self.update_frequency_hours
        
        return has_enough_data and time_for_update
    
    def add_baseline_data(self, features: np.ndarray, is_normal: bool = True):
        """
        Add new data to baseline for incremental learning.
        
        Args:
            features: Feature vector to add
            is_normal: Whether this is normal (non-anomalous) data
        """
        if is_normal:
            self.baseline_data.append(features.copy())
            
            # Maintain sliding window of baseline data
            if len(self.baseline_data) > self.max_baseline_samples:
                # Remove oldest 10% of data
                remove_count = len(self.baseline_data) // 10
                self.baseline_data = self.baseline_data[remove_count:]
    
    def get_risk_level_from_score(self, anomaly_score: float) -> RiskLevel:
        """
        Convert anomaly score to risk level based on thresholds.
        
        Args:
            anomaly_score: Score from 0.0 to 1.0
            
        Returns:
            Risk level enum
        """
        low_threshold = self.config.get('low_risk_threshold', 0.3)
        high_threshold = self.config.get('high_risk_threshold', 0.7)
        
        if anomaly_score < low_threshold:
            return RiskLevel.LOW
        elif anomaly_score < high_threshold:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
    
    def track_performance(self, processing_time: float, accuracy: float = None):
        """
        Track agent performance metrics.
        
        Args:
            processing_time: Time taken for inference in ms
            accuracy: Optional accuracy score if available
        """
        self.inference_times.append(processing_time)
        if accuracy is not None:
            self.accuracy_scores.append(accuracy)
            
        # Keep only recent performance data
        max_metrics = 1000
        if len(self.inference_times) > max_metrics:
            self.inference_times = self.inference_times[-max_metrics:]
        if len(self.accuracy_scores) > max_metrics:
            self.accuracy_scores = self.accuracy_scores[-max_metrics:]
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        stats = {
            'avg_inference_time_ms': np.mean(self.inference_times) if self.inference_times else 0,
            'max_inference_time_ms': np.max(self.inference_times) if self.inference_times else 0,
            'baseline_samples': len(self.baseline_data),
            'is_trained': self.is_trained
        }
        
        if self.accuracy_scores:
            stats['avg_accuracy'] = np.mean(self.accuracy_scores)
            stats['min_accuracy'] = np.min(self.accuracy_scores)
            
        return stats
    
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Optional[AgentResult]:
        """
        Main processing pipeline: capture data -> predict -> track performance.
        
        Args:
            sensor_data: Raw sensor data
            
        Returns:
            AgentResult or None if processing failed
        """
        if not self.is_trained:
            return None
            
        start_time = time.time()
        
        try:
            # Capture and preprocess data
            features = self.capture_data(sensor_data)
            if features is None:
                return None
            
            # Make prediction
            result = self.predict(features)
            
            # Track processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            result.processing_time_ms = processing_time
            
            # Add to baseline if this appears to be normal behavior
            if result.anomaly_score < self.config.get('normal_threshold', 0.3):
                self.add_baseline_data(features, is_normal=True)
            
            # Track performance
            self.track_performance(processing_time)
            
            # Check if model should be updated
            if self.should_update_model():
                self.incremental_update(self.baseline_data[-self.min_samples_for_update:])
                self.last_update_time = time.time()
            
            return result
            
        except Exception as e:
            print(f"Error in {self.agent_name}: {str(e)}")
            return None
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update agent configuration"""
        self.config.update(new_config)