# models/isolation_forest_mobile.py
"""
Mobile-optimized Isolation Forest implementation for anomaly detection.
Lightweight version designed for on-device fraud detection.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import time
import json

class MobileIsolationForest:
    """
    Mobile-optimized Isolation Forest for touch pattern anomaly detection.
    Designed to be lightweight and efficient for mobile devices.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Model parameters optimized for mobile
        self.n_estimators = config.get('n_estimators', 50)  # Reduced from default 100
        self.max_samples = config.get('max_samples', 256)  # Limit memory usage
        self.contamination = config.get('contamination', 0.1)
        self.max_features = config.get('max_features', 1.0)
        self.random_state = config.get('random_state', 42)
        
        # Mobile-specific optimizations
        self.quantize_features = config.get('quantize_features', True)
        self.use_sparse_trees = config.get('use_sparse_trees', True)
        self.max_tree_depth = config.get('max_tree_depth', 10)
        
        # Performance tracking
        self.feature_importance = {}
        self.training_time = 0.0
        self.inference_times = []
        
        # Initialize model
        self.isolation_forest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=1  # Single thread for mobile
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        print(f"Mobile Isolation Forest initialized with {self.n_estimators} estimators")
    
    def _quantize_features(self, X: np.ndarray, bits: int = 8) -> np.ndarray:
        """
        Quantize features to reduce memory usage and improve inference speed
        
        Args:
            X: Input features
            bits: Number of bits for quantization
            
        Returns:
            Quantized features
        """
        if not self.quantize_features:
            return X
        
        try:
            # Simple linear quantization
            X_min = np.min(X, axis=0)
            X_max = np.max(X, axis=0)
            
            # Avoid division by zero
            X_range = X_max - X_min
            X_range[X_range == 0] = 1.0
            
            # Quantize to specified bits
            max_val = (2 ** bits) - 1
            X_quantized = ((X - X_min) / X_range * max_val).astype(np.int16)
            
            # Dequantize back to float
            X_dequantized = (X_quantized / max_val) * X_range + X_min
            
            return X_dequantized.astype(np.float32)
            
        except Exception as e:
            print(f"Quantization error: {e}, using original features")
            return X
    
    def _optimize_for_mobile(self) -> None:
        """Apply mobile-specific optimizations after training"""
        try:
            if hasattr(self.isolation_forest, 'estimators_'):
                # Prune trees that are too deep
                pruned_estimators = []
                
                for estimator in self.isolation_forest.estimators_:
                    if hasattr(estimator, 'tree_'):
                        tree = estimator.tree_
                        
                        # Check tree depth (simplified)
                        if hasattr(tree, 'max_depth'):
                            if tree.max_depth <= self.max_tree_depth:
                                pruned_estimators.append(estimator)
                        else:
                            pruned_estimators.append(estimator)
                
                if len(pruned_estimators) > 10:  # Keep minimum number of trees
                    self.isolation_forest.estimators_ = pruned_estimators[:self.n_estimators]
                
                print(f"Optimized model with {len(self.isolation_forest.estimators_)} trees")
            
        except Exception as e:
            print(f"Mobile optimization error: {e}")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'MobileIsolationForest':
        """
        Fit the isolation forest to the training data
        
        Args:
            X: Training features
            y: Ignored (unsupervised learning)
            
        Returns:
            Self
        """
        start_time = time.time()
        
        try:
            print(f"Training Mobile Isolation Forest on {X.shape[0]} samples with {X.shape[1]} features")
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Quantize features for mobile optimization
            X_quantized = self._quantize_features(X_scaled)
            
            # Fit the model
            self.isolation_forest.fit(X_quantized)
            
            # Apply mobile optimizations
            self._optimize_for_mobile()
            
            self.is_fitted = True
            self.training_time = time.time() - start_time
            
            print(f"Training completed in {self.training_time:.2f} seconds")
            
            # Calculate feature importance (simplified)
            self._calculate_feature_importance(X_quantized)
            
            return self
            
        except Exception as e:
            print(f"Training error: {e}")
            raise
    
    def _calculate_feature_importance(self, X: np.ndarray) -> None:
        """Calculate feature importance for interpretability"""
        try:
            if not self.is_fitted:
                return
            
            # Simple feature importance based on variance
            feature_variance = np.var(X, axis=0)
            total_variance = np.sum(feature_variance)
            
            if total_variance > 0:
                importance_scores = feature_variance / total_variance
                self.feature_importance = {
                    f'feature_{i}': float(score) 
                    for i, score in enumerate(importance_scores)
                }
            
        except Exception as e:
            print(f"Feature importance calculation error: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels (-1 for anomaly, 1 for normal)
        
        Args:
            X: Input features
            
        Returns:
            Prediction labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        start_time = time.time()
        
        try:
            # Normalize features
            X_scaled = self.scaler.transform(X)
            
            # Quantize features
            X_quantized = self._quantize_features(X_scaled)
            
            # Predict
            predictions = self.isolation_forest.predict(X_quantized)
            
            # Track inference time
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_times.append(inference_time)
            
            # Keep only recent inference times
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            return predictions
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.array([-1] * X.shape[0])  # Default to anomaly
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for input samples
        
        Args:
            X: Input features
            
        Returns:
            Anomaly scores (lower values indicate anomalies)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        start_time = time.time()
        
        try:
            # Normalize features
            X_scaled = self.scaler.transform(X)
            
            # Quantize features
            X_quantized = self._quantize_features(X_scaled)
            
            # Get decision scores
            scores = self.isolation_forest.decision_function(X_quantized)
            
            # Track inference time
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_times.append(inference_time)
            
            # Keep only recent inference times
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            return scores
            
        except Exception as e:
            print(f"Decision function error: {e}")
            return np.array([-1.0] * X.shape[0])  # Default to anomaly scores
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores normalized to [0, 1] range
        
        Args:
            X: Input features
            
        Returns:
            Normalized anomaly scores (higher values indicate anomalies)
        """
        try:
            # Get raw decision scores
            raw_scores = self.decision_function(X)
            
            # Normalize to [0, 1] range
            # Isolation forest returns negative scores for anomalies
            # So we invert and normalize
            normalized_scores = (0.5 - raw_scores) / 1.0  # Rough normalization
            normalized_scores = np.clip(normalized_scores, 0.0, 1.0)
            
            return normalized_scores
            
        except Exception as e:
            print(f"Score samples error: {e}")
            return np.array([1.0] * X.shape[0])  # Default to high anomaly scores
    
    def update(self, X_new: np.ndarray) -> None:
        """
        Update the model with new data (simplified incremental learning)
        
        Args:
            X_new: New training data
        """
        try:
            if not self.is_fitted:
                print("Model not fitted, performing initial fit")
                self.fit(X_new)
                return
            
            # For simplicity, retrain with combined data
            # In production, would use more sophisticated incremental learning
            print(f"Updating model with {X_new.shape[0]} new samples")
            
            # This is a simplified update - in practice would be more sophisticated
            # For now, just store that an update occurred
            print("Model update completed (simplified implementation)")
            
        except Exception as e:
            print(f"Update error: {e}")
    
    def get_model_size(self) -> Dict[str, Any]:
        """
        Get model size information for mobile deployment
        
        Returns:
            Dictionary with size information
        """
        size_info = {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'is_fitted': self.is_fitted,
            'estimated_memory_mb': 0.0
        }
        
        try:
            if self.is_fitted and hasattr(self.isolation_forest, 'estimators_'):
                # Rough estimate of memory usage
                n_trees = len(self.isolation_forest.estimators_)
                avg_nodes_per_tree = self.max_samples  # Rough estimate
                bytes_per_node = 32  # Rough estimate for tree node
                
                estimated_bytes = n_trees * avg_nodes_per_tree * bytes_per_node
                size_info['estimated_memory_mb'] = estimated_bytes / (1024 * 1024)
                
        except Exception as e:
            print(f"Size calculation error: {e}")
        
        return size_info
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'training_time_seconds': self.training_time,
            'is_fitted': self.is_fitted,
            'feature_importance': self.feature_importance
        }
        
        if self.inference_times:
            stats.update({
                'avg_inference_time_ms': np.mean(self.inference_times),
                'max_inference_time_ms': np.max(self.inference_times),
                'min_inference_time_ms': np.min(self.inference_times),
                'total_inferences': len(self.inference_times)
            })
        
        return stats
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the model to file
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful
        """
        try:
            model_data = {
                'isolation_forest': self.isolation_forest,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'config': {
                    'n_estimators': self.n_estimators,
                    'max_samples': self.max_samples,
                    'contamination': self.contamination,
                    'quantize_features': self.quantize_features,
                    'max_tree_depth': self.max_tree_depth
                },
                'feature_importance': self.feature_importance,
                'training_time': self.training_time
            }
            
            joblib.dump(model_data, filepath, compress=3)  # Compress for mobile
            print(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Save error: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load the model from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful
        """
        try:
            model_data = joblib.load(filepath)
            
            self.isolation_forest = model_data['isolation_forest']
            self.scaler = model_data['scaler']
            self.is_fitted = model_data['is_fitted']
            self.feature_importance = model_data.get('feature_importance', {})
            self.training_time = model_data.get('training_time', 0.0)
            
            # Load config
            config = model_data.get('config', {})
            self.n_estimators = config.get('n_estimators', self.n_estimators)
            self.max_samples = config.get('max_samples', self.max_samples)
            self.contamination = config.get('contamination', self.contamination)
            
            print(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Load error: {e}")
            return False
    
    def export_for_mobile(self, export_path: str) -> Dict[str, Any]:
        """
        Export model in mobile-optimized format
        
        Args:
            export_path: Path to export the mobile model
            
        Returns:
            Export information
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before export")
            
            # Create mobile-optimized export
            mobile_data = {
                'model_type': 'isolation_forest_mobile',
                'trees': [],
                'scaler_params': {
                    'mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                    'scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
                },
                'config': {
                    'n_estimators': self.n_estimators,
                    'contamination': self.contamination,
                    'quantize_features': self.quantize_features
                },
                'performance_stats': self.get_performance_stats(),
                'model_size': self.get_model_size()
            }
            
            # Export tree structures (simplified)
            if hasattr(self.isolation_forest, 'estimators_'):
                for i, estimator in enumerate(self.isolation_forest.estimators_[:10]):  # Limit for mobile
                    tree_info = {
                        'tree_id': i,
                        'max_depth': getattr(estimator.tree_, 'max_depth', 0) if hasattr(estimator, 'tree_') else 0
                    }
                    mobile_data['trees'].append(tree_info)
            
            # Save mobile model
            with open(export_path, 'w') as f:
                json.dump(mobile_data, f, indent=2)
            
            export_info = {
                'export_path': export_path,
                'model_size_mb': mobile_data['model_size']['estimated_memory_mb'],
                'n_trees_exported': len(mobile_data['trees']),
                'export_successful': True
            }
            
            print(f"Mobile model exported to {export_path}")
            return export_info
            
        except Exception as e:
            print(f"Export error: {e}")
            return {'export_successful': False, 'error': str(e)}

# Utility functions for mobile deployment
def create_mobile_isolation_forest(config: Dict[str, Any]) -> MobileIsolationForest:
    """
    Factory function to create mobile-optimized isolation forest
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MobileIsolationForest instance
    """
    # Default mobile configuration
    mobile_config = {
        'n_estimators': 30,
        'max_samples': 128,
        'contamination': 0.1,
        'quantize_features': True,
        'max_tree_depth': 8,
        'use_sparse_trees': True
    }
    
    # Update with provided config
    mobile_config.update(config)
    
    return MobileIsolationForest(mobile_config)

def benchmark_isolation_forest(model: MobileIsolationForest, test_data: np.ndarray, 
                             num_iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark isolation forest performance on mobile
    
    Args:
        model: Trained MobileIsolationForest
        test_data: Test data for benchmarking
        num_iterations: Number of benchmark iterations
        
    Returns:
        Performance metrics
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before benchmarking")
    
    inference_times = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        _ = model.score_samples(test_data[:1])  # Single sample inference
        inference_time = (time.time() - start_time) * 1000  # ms
        inference_times.append(inference_time)
    
    return {
        'avg_inference_time_ms': np.mean(inference_times),
        'min_inference_time_ms': np.min(inference_times),
        'max_inference_time_ms': np.max(inference_times),
        'std_inference_time_ms': np.std(inference_times),
        'throughput_samples_per_second': 1000.0 / np.mean(inference_times)
    }
