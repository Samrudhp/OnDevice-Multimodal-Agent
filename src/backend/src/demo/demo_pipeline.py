# demo/demo_pipeline.py

"""
End-to-end demonstration pipeline for QuadFusion

Features:
- Complete fraud detection pipeline demonstration
- Real-time data processing workflow
- Multi-agent coordination example
- Decision fusion demonstration
- Performance monitoring and visualization
- Interactive demo interface
- Batch processing examples
- Streaming data processing
- Model loading and initialization
- Result visualization and interpretation
"""

import asyncio
import threading
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import logging
import psutil
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Generator
from dataclasses import dataclass
from pathlib import Path

# Import QuadFusion components
try:
    from demo.simulate_sensor_data import SensorDataSimulator, FraudulentBehaviorInjector
    from data.preprocessing import DataPreprocessor, ProcessingConfig
    from models.motion_cnn import MotionCNN, MotionConfig
    from models.speaker_identification import SpeakerIdentificationModel
    from models.tiny_llm import TinyLLM, TinyLLMConfig
    from data.encryption import KeyManager, DataEncryption
    QUADFUSION_AVAILABLE = True
except ImportError:
    QUADFUSION_AVAILABLE = False
    logging.warning("QuadFusion components not available - using stubs")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class DemoConfig:
    """Configuration for demo pipeline."""
    batch_size: int = 32
    streaming_buffer_size: int = 100
    processing_threads: int = 2
    enable_visualization: bool = True
    enable_encryption: bool = True
    simulation_speed: float = 1.0  # Real-time multiplier
    export_results: bool = True
    demo_duration_sec: int = 60

@dataclass
class ProcessingResult:
    """Result from processing a single sample."""
    timestamp: float
    sample_id: str
    motion_prediction: Dict[str, Any]
    voice_prediction: Dict[str, Any]
    fused_decision: Dict[str, Any]
    processing_time_ms: float
    is_anomaly: bool
    confidence_score: float

class PerformanceMonitor:
    """
    Comprehensive performance monitoring for demo pipeline.
    Tracks throughput, latency, memory, CPU, and battery usage.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.processed_samples = 0
        self.processing_times = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=100)
        self.cpu_usage = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        
        # Performance thresholds
        self.latency_threshold_ms = 100
        self.memory_threshold_mb = 500
        self.cpu_threshold_percent = 80
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start background performance monitoring."""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        logging.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logging.info("Performance monitoring stopped")
        
    def _monitor_system(self):
        """Background system monitoring loop."""
        while self.monitoring_active:
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.used / 1024 / 1024)  # MB
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu_percent)
                
                # Throughput calculation
                current_time = time.time()
                elapsed = current_time - self.start_time
                throughput = self.processed_samples / elapsed if elapsed > 0 else 0
                self.throughput_history.append(throughput)
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                
    def record_sample(self, processing_time_ms: float):
        """Record processing time for a sample."""
        self.processed_samples += 1
        self.processing_times.append(processing_time_ms)
        
        # Alert on performance issues
        if processing_time_ms > self.latency_threshold_ms:
            logging.warning(f"High latency detected: {processing_time_ms:.2f}ms")
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        metrics = {
            'runtime_seconds': elapsed,
            'total_samples': self.processed_samples,
            'throughput_sps': self.processed_samples / elapsed if elapsed > 0 else 0
        }
        
        # Processing time stats
        if self.processing_times:
            metrics.update({
                'avg_latency_ms': np.mean(self.processing_times),
                'p95_latency_ms': np.percentile(self.processing_times, 95),
                'p99_latency_ms': np.percentile(self.processing_times, 99),
                'max_latency_ms': np.max(self.processing_times),
                'min_latency_ms': np.min(self.processing_times)
            })
        
        # System resource stats
        if self.memory_usage:
            metrics.update({
                'avg_memory_mb': np.mean(self.memory_usage),
                'max_memory_mb': np.max(self.memory_usage),
                'current_memory_mb': self.memory_usage[-1] if self.memory_usage else 0
            })
            
        if self.cpu_usage:
            metrics.update({
                'avg_cpu_percent': np.mean(self.cpu_usage),
                'max_cpu_percent': np.max(self.cpu_usage),
                'current_cpu_percent': self.cpu_usage[-1] if self.cpu_usage else 0
            })
            
        return metrics
    
    def get_alerts(self) -> List[str]:
        """Get performance alerts."""
        alerts = []
        
        if self.processing_times:
            avg_latency = np.mean(self.processing_times)
            if avg_latency > self.latency_threshold_ms:
                alerts.append(f"High average latency: {avg_latency:.2f}ms")
                
        if self.memory_usage:
            current_memory = self.memory_usage[-1]
            if current_memory > self.memory_threshold_mb:
                alerts.append(f"High memory usage: {current_memory:.2f}MB")
                
        if self.cpu_usage:
            current_cpu = self.cpu_usage[-1]
            if current_cpu > self.cpu_threshold_percent:
                alerts.append(f"High CPU usage: {current_cpu:.1f}%")
                
        return alerts

class ResultVisualizer:
    """
    Advanced visualization for demo results using matplotlib and Plotly.
    """
    
    def __init__(self, enable_interactive: bool = True):
        self.enable_interactive = enable_interactive
        self.results_history = []
        
    def add_result(self, result: ProcessingResult):
        """Add processing result to history."""
        self.results_history.append(result)
        
    def plot_performance_metrics(self, metrics: Dict[str, Any]):
        """Plot performance metrics over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('QuadFusion Demo Performance Metrics', fontsize=16)
        
        # Throughput
        axes[0, 0].plot(metrics.get('throughput_history', []))
        axes[0, 0].set_title('Throughput (samples/sec)')
        axes[0, 0].set_ylabel('Samples/sec')
        axes[0, 0].grid(True)
        
        # Latency distribution
        if 'processing_times' in metrics:
            axes[0, 1].hist(metrics['processing_times'], bins=30, alpha=0.7)
            axes[0, 1].set_title('Latency Distribution')
            axes[0, 1].set_xlabel('Processing Time (ms)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)
        
        # Memory usage
        if 'memory_history' in metrics:
            axes[1, 0].plot(metrics['memory_history'])
            axes[1, 0].set_title('Memory Usage')
            axes[1, 0].set_ylabel('Memory (MB)')
            axes[1, 0].grid(True)
        
        # CPU usage
        if 'cpu_history' in metrics:
            axes[1, 1].plot(metrics['cpu_history'])
            axes[1, 1].set_title('CPU Usage')
            axes[1, 1].set_ylabel('CPU (%)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_detection_results(self):
        """Plot anomaly detection results over time."""
        if not self.results_history:
            return
            
        # Extract data
        timestamps = [r.timestamp for r in self.results_history]
        confidence_scores = [r.confidence_score for r in self.results_history]
        anomalies = [r.is_anomaly for r in self.results_history]
        
        plt.figure(figsize=(12, 6))
        
        # Plot confidence scores
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, confidence_scores, 'b-', alpha=0.7, label='Confidence Score')
        
        # Highlight anomalies
        anomaly_times = [t for t, a in zip(timestamps, anomalies) if a]
        anomaly_scores = [s for s, a in zip(confidence_scores, anomalies) if a]
        plt.scatter(anomaly_times, anomaly_scores, color='red', s=50, label='Anomalies', zorder=5)
        
        plt.title('Fraud Detection Results Over Time')
        plt.ylabel('Confidence Score')
        plt.legend()
        plt.grid(True)
        
        # Plot anomaly timeline
        plt.subplot(2, 1, 2)
        anomaly_binary = [1 if a else 0 for a in anomalies]
        plt.fill_between(timestamps, anomaly_binary, alpha=0.3, color='red')
        plt.title('Anomaly Detection Timeline')
        plt.xlabel('Time')
        plt.ylabel('Anomaly Detected')
        plt.ylim(-0.1, 1.1)
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def create_confusion_matrix(self, true_labels: List[bool], predicted_labels: List[bool]):
        """Create and display confusion matrix."""
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(true_labels, predicted_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fraud'], 
                   yticklabels=['Normal', 'Fraud'])
        plt.title('Fraud Detection Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        # Print detailed metrics
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_labels, 
                                  target_names=['Normal', 'Fraud']))
        
    def export_results(self, filename: str = "demo_results.json"):
        """Export visualization results to file."""
        export_data = {
            'total_samples': len(self.results_history),
            'anomaly_count': sum(1 for r in self.results_history if r.is_anomaly),
            'avg_confidence': np.mean([r.confidence_score for r in self.results_history]),
            'results': [
                {
                    'timestamp': r.timestamp,
                    'is_anomaly': r.is_anomaly,
                    'confidence': r.confidence_score,
                    'processing_time_ms': r.processing_time_ms
                }
                for r in self.results_history
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        logging.info(f"Results exported to {filename}")

class AgentCoordinator:
    """
    Coordinates multi-agent processing: MotionCNN, Speaker Identification, TinyLLM decision fusion.
    Handles model loading, coordination, and result aggregation.
    """
    
    def __init__(self, config: DemoConfig):
        self.config = config
        
        # Initialize models (with fallbacks if not available)
        self.motion_model = None
        self.speaker_model = None
        self.llm_model = None
        self.preprocessor = None
        
        self._initialize_models()
        
        # Encryption for sensitive data
        if config.enable_encryption:
            self.key_manager = KeyManager() if QUADFUSION_AVAILABLE else None
            self.data_encryption = DataEncryption(self.key_manager) if self.key_manager else None
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()
        
        # Results storage
        self.results_buffer = deque(maxlen=config.streaming_buffer_size)
        self.processing_lock = threading.Lock()
        
    def _initialize_models(self):
        """Initialize all ML models."""
        try:
            if QUADFUSION_AVAILABLE:
                # Initialize preprocessing
                preprocessing_config = ProcessingConfig()
                self.preprocessor = DataPreprocessor(preprocessing_config)
                
                # Initialize motion model
                motion_config = MotionConfig()
                self.motion_model = MotionCNN(motion_config)
                
                # Initialize speaker model
                self.speaker_model = SpeakerIdentificationModel()
                
                # Initialize LLM
                llm_config = TinyLLMConfig()
                self.llm_model = TinyLLM(llm_config)
                
                logging.info("All models initialized successfully")
            else:
                logging.warning("Using model stubs - QuadFusion components not available")
                
        except Exception as e:
            logging.error(f"Model initialization failed: {e}")
            
    def process_sample(self, sample: Dict[str, Any], sample_id: str = None) -> ProcessingResult:
        """
        Process a single sensor data sample through the complete pipeline.
        
        Args:
            sample: Dictionary containing sensor data
            sample_id: Optional identifier for the sample
            
        Returns:
            ProcessingResult with predictions and metadata
        """
        start_time = time.time()
        timestamp = start_time
        
        if sample_id is None:
            sample_id = f"sample_{int(timestamp * 1000)}"
        
        try:
            # Preprocessing
            if self.preprocessor:
                features = self.preprocessor.preprocess_single(sample)
            else:
                # Stub preprocessing
                features = np.random.rand(50)
            
            # Motion prediction
            if self.motion_model and 'movement' in sample:
                motion_pred = self._predict_motion(sample['movement'])
            else:
                motion_pred = {
                    'activity': 'walking',
                    'confidence': 0.85 + np.random.rand() * 0.15,
                    'anomaly_score': np.random.rand() * 0.3
                }
            
            # Voice prediction
            if self.speaker_model and 'voice' in sample:
                voice_pred = self._predict_speaker(sample['voice'])
            else:
                voice_pred = {
                    'speaker_id': 'user_1',
                    'confidence': 0.90 + np.random.rand() * 0.10,
                    'spoofing_detected': np.random.rand() < 0.05
                }
            
            # Decision fusion
            if self.llm_model:
                fused_decision = self._fuse_decisions(motion_pred, voice_pred)
            else:
                # Simple fusion stub
                overall_confidence = (motion_pred['confidence'] + voice_pred['confidence']) / 2
                is_anomaly = (motion_pred['anomaly_score'] > 0.7 or 
                            voice_pred.get('spoofing_detected', False))
                
                fused_decision = {
                    'decision': 'fraud' if is_anomaly else 'legitimate',
                    'confidence': overall_confidence,
                    'risk_score': motion_pred['anomaly_score']
                }
            
            # Create result
            processing_time = (time.time() - start_time) * 1000  # ms
            
            result = ProcessingResult(
                timestamp=timestamp,
                sample_id=sample_id,
                motion_prediction=motion_pred,
                voice_prediction=voice_pred,
                fused_decision=fused_decision,
                processing_time_ms=processing_time,
                is_anomaly=fused_decision['decision'] == 'fraud',
                confidence_score=fused_decision['confidence']
            )
            
            # Record performance
            self.perf_monitor.record_sample(processing_time)
            
            # Store result
            with self.processing_lock:
                self.results_buffer.append(result)
            
            return result
            
        except Exception as e:
            logging.error(f"Processing failed for sample {sample_id}: {e}")
            
            # Return error result
            return ProcessingResult(
                timestamp=timestamp,
                sample_id=sample_id,
                motion_prediction={'error': str(e)},
                voice_prediction={'error': str(e)},
                fused_decision={'decision': 'error', 'confidence': 0.0, 'risk_score': 0.0},
                processing_time_ms=(time.time() - start_time) * 1000,
                is_anomaly=False,
                confidence_score=0.0
            )
    
    def _predict_motion(self, movement_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Predict motion patterns using MotionCNN."""
        # Stub implementation
        activities = ['walking', 'running', 'sitting', 'standing', 'climbing']
        activity = np.random.choice(activities)
        confidence = 0.8 + np.random.rand() * 0.2
        anomaly_score = np.random.rand() * 0.5
        
        return {
            'activity': activity,
            'confidence': confidence,
            'anomaly_score': anomaly_score,
            'features_extracted': True
        }
    
    def _predict_speaker(self, voice_data: np.ndarray) -> Dict[str, Any]:
        """Predict speaker identity and detect spoofing."""
        # Stub implementation
        speakers = ['user_1', 'user_2', 'user_3', 'unknown']
        speaker = np.random.choice(speakers)
        confidence = 0.85 + np.random.rand() * 0.15
        spoofing_detected = np.random.rand() < 0.1  # 10% chance
        
        return {
            'speaker_id': speaker,
            'confidence': confidence,
            'spoofing_detected': spoofing_detected,
            'voice_quality': 'good' if confidence > 0.8 else 'poor'
        }
    
    def _fuse_decisions(self, motion_pred: Dict[str, Any], voice_pred: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse decisions from multiple modalities."""
        # Weighted fusion
        motion_weight = 0.4
        voice_weight = 0.6
        
        overall_confidence = (motion_weight * motion_pred['confidence'] + 
                            voice_weight * voice_pred['confidence'])
        
        # Anomaly detection logic
        is_fraud = (motion_pred['anomaly_score'] > 0.7 or 
                   voice_pred.get('spoofing_detected', False) or
                   overall_confidence < 0.5)
        
        risk_score = max(motion_pred['anomaly_score'], 
                        0.8 if voice_pred.get('spoofing_detected', False) else 0.2)
        
        return {
            'decision': 'fraud' if is_fraud else 'legitimate',
            'confidence': overall_confidence,
            'risk_score': risk_score,
            'contributing_factors': {
                'motion_anomaly': motion_pred['anomaly_score'] > 0.7,
                'voice_spoofing': voice_pred.get('spoofing_detected', False),
                'low_confidence': overall_confidence < 0.5
            }
        }
    
    def run_realtime_processing(self, sample_stream: Generator[Dict[str, Any], None, None]) -> List[ProcessingResult]:
        """
        Process streaming sensor data in real-time.
        
        Args:
            sample_stream: Generator of sensor data samples
            
        Returns:
            List of processing results
        """
        results = []
        sample_count = 0
        
        logging.info("Starting real-time processing...")
        self.perf_monitor.start_monitoring()
        
        try:
            for sample in sample_stream:
                result = self.process_sample(sample, f"realtime_{sample_count}")
                results.append(result)
                sample_count += 1
                
                # Simulate real-time processing delay
                if self.config.simulation_speed < 1.0:
                    time.sleep((1.0 - self.config.simulation_speed) * 0.02)  # Up to 20ms delay
                
                # Log progress periodically
                if sample_count % 50 == 0:
                    logging.info(f"Processed {sample_count} samples")
                    
        except KeyboardInterrupt:
            logging.info("Real-time processing interrupted by user")
        except Exception as e:
            logging.error(f"Real-time processing error: {e}")
        finally:
            self.perf_monitor.stop_monitoring()
            logging.info(f"Real-time processing completed. Total samples: {len(results)}")
        
        return results
    
    def run_batch_processing(self, dataset: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """
        Process batch dataset efficiently.
        
        Args:
            dataset: List of sensor data samples
            
        Returns:
            List of processing results
        """
        results = []
        batch_size = self.config.batch_size
        
        logging.info(f"Starting batch processing of {len(dataset)} samples...")
        self.perf_monitor.start_monitoring()
        
        try:
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                batch_results = []
                
                # Process batch
                for j, sample in enumerate(batch):
                    result = self.process_sample(sample, f"batch_{i + j}")
                    batch_results.append(result)
                
                results.extend(batch_results)
                
                # Log progress
                progress = (i + len(batch)) / len(dataset) * 100
                logging.info(f"Batch processing progress: {progress:.1f}%")
                
        except Exception as e:
            logging.error(f"Batch processing error: {e}")
        finally:
            self.perf_monitor.stop_monitoring()
            logging.info(f"Batch processing completed. Total samples: {len(results)}")
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return self.perf_monitor.get_metrics()
    
    def get_performance_alerts(self) -> List[str]:
        """Get performance alerts."""
        return self.perf_monitor.get_alerts()

class DemoPipeline:
    """
    Main demo pipeline orchestrating all components.
    Provides high-level interface for running different demo scenarios.
    """
    
    def __init__(self, config: Optional[DemoConfig] = None):
        self.config = config or DemoConfig()
        
        # Initialize components
        self.agent_coordinator = AgentCoordinator(self.config)
        self.visualizer = ResultVisualizer(self.config.enable_visualization)
        
        # Data simulation
        self.data_simulator = None
        self.fraud_injector = None
        
        self._initialize_simulators()
        
        # Demo state
        self.demo_results = []
        self.demo_start_time = None
        
    def _initialize_simulators(self):
        """Initialize data simulators."""
        try:
            if QUADFUSION_AVAILABLE:
                self.data_simulator = SensorDataSimulator()
                self.fraud_injector = FraudulentBehaviorInjector()
            else:
                logging.warning("Using simulation stubs")
        except Exception as e:
            logging.error(f"Simulator initialization failed: {e}")
    
    def run_normal_behavior_demo(self, duration_sec: int = 60) -> Dict[str, Any]:
        """
        Run demo with normal user behavior patterns.
        
        Args:
            duration_sec: Duration of demo in seconds
            
        Returns:
            Demo results and metrics
        """
        logging.info(f"Starting normal behavior demo ({duration_sec}s)")
        self.demo_start_time = time.time()
        
        # Generate normal behavior data
        if self.data_simulator:
            sample_stream = self.data_simulator.generate_streaming_data(duration_secs=duration_sec)
        else:
            # Stub data generation
            sample_stream = self._generate_stub_samples(duration_sec)
        
        # Process data
        results = self.agent_coordinator.run_realtime_processing(sample_stream)
        self.demo_results = results
        
        # Update visualizer
        for result in results:
            self.visualizer.add_result(result)
        
        # Generate summary
        summary = self._generate_demo_summary(results, "Normal Behavior")
        
        if self.config.enable_visualization:
            self.visualizer.plot_detection_results()
            
        return summary
    
    def run_fraud_detection_demo(self, fraud_rate: float = 0.1, duration_sec: int = 60) -> Dict[str, Any]:
        """
        Run demo with injected fraudulent behavior.
        
        Args:
            fraud_rate: Proportion of samples containing fraud (0.0 to 1.0)
            duration_sec: Duration of demo in seconds
            
        Returns:
            Demo results and metrics
        """
        logging.info(f"Starting fraud detection demo ({duration_sec}s, {fraud_rate*100}% fraud rate)")
        self.demo_start_time = time.time()
        
        # Generate data with fraud injection
        if self.data_simulator and self.fraud_injector:
            normal_stream = self.data_simulator.generate_streaming_data(duration_secs=duration_sec)
            sample_stream = self.fraud_injector.inject_fraud(normal_stream, fraud_rate)
        else:
            # Stub data with simulated fraud
            sample_stream = self._generate_stub_samples_with_fraud(duration_sec, fraud_rate)
        
        # Process data
        results = self.agent_coordinator.run_realtime_processing(sample_stream)
        self.demo_results = results
        
        # Update visualizer
        for result in results:
            self.visualizer.add_result(result)
        
        # Generate summary
        summary = self._generate_demo_summary(results, "Fraud Detection")
        summary['fraud_rate'] = fraud_rate
        summary['fraud_detected'] = sum(1 for r in results if r.is_anomaly)
        summary['fraud_detection_rate'] = summary['fraud_detected'] / len(results) if results else 0
        
        if self.config.enable_visualization:
            self.visualizer.plot_detection_results()
            
        return summary
    
    def run_stress_test_demo(self, load_multiplier: float = 2.0, duration_sec: int = 30) -> Dict[str, Any]:
        """
        Run stress test with high data load.
        
        Args:
            load_multiplier: Multiplier for data generation rate
            duration_sec: Duration of demo in seconds
            
        Returns:
            Demo results and metrics
        """
        logging.info(f"Starting stress test demo ({load_multiplier}x load, {duration_sec}s)")
        self.demo_start_time = time.time()
        
        # Generate high-load data
        sample_stream = self._generate_stress_test_samples(duration_sec, load_multiplier)
        
        # Process data
        results = self.agent_coordinator.run_realtime_processing(sample_stream)
        self.demo_results = results
        
        # Generate summary
        summary = self._generate_demo_summary(results, "Stress Test")
        summary['load_multiplier'] = load_multiplier
        
        # Check for performance issues
        alerts = self.agent_coordinator.get_performance_alerts()
        summary['performance_alerts'] = alerts
        
        if self.config.enable_visualization:
            metrics = self.agent_coordinator.get_performance_metrics()
            self.visualizer.plot_performance_metrics(metrics)
            
        return summary
    
    def run_battery_constrained_demo(self, battery_level: float = 20.0, duration_sec: int = 60) -> Dict[str, Any]:
        """
        Run demo simulating low battery conditions.
        
        Args:
            battery_level: Simulated battery level (0-100)
            duration_sec: Duration of demo in seconds
            
        Returns:
            Demo results and metrics
        """
        logging.info(f"Starting battery constrained demo ({battery_level}% battery, {duration_sec}s)")
        self.demo_start_time = time.time()
        
        # Configure for battery saving
        if hasattr(self.agent_coordinator.preprocessor, 'update_battery_level'):
            self.agent_coordinator.preprocessor.update_battery_level(battery_level)
        
        # Generate normal data
        if self.data_simulator:
            sample_stream = self.data_simulator.generate_streaming_data(duration_secs=duration_sec)
        else:
            sample_stream = self._generate_stub_samples(duration_sec)
        
        # Process with battery constraints
        results = self.agent_coordinator.run_realtime_processing(sample_stream)
        self.demo_results = results
        
        # Generate summary
        summary = self._generate_demo_summary(results, "Battery Constrained")
        summary['battery_level'] = battery_level
        
        return summary
    
    def _generate_stub_samples(self, duration_sec: int) -> Generator[Dict[str, Any], None, None]:
        """Generate stub sensor data samples."""
        samples_per_sec = 20  # 20 Hz
        total_samples = duration_sec * samples_per_sec
        
        for i in range(total_samples):
            yield {
                'timestamp': time.time(),
                'movement': {
                    'acceleration': np.random.randn(3),
                    'gyroscope': np.random.randn(3),
                    'magnetometer': np.random.randn(3)
                },
                'voice': np.random.randn(1600),  # 0.1 second of audio
                'touch': [{'pressure': np.random.rand(), 'area': np.random.rand() * 20}],
                'sample_id': f'stub_{i}'
            }
            time.sleep(1.0 / samples_per_sec)
    
    def _generate_stub_samples_with_fraud(self, duration_sec: int, fraud_rate: float) -> Generator[Dict[str, Any], None, None]:
        """Generate stub data with simulated fraud."""
        samples_per_sec = 20
        total_samples = duration_sec * samples_per_sec
        
        for i in range(total_samples):
            is_fraud = np.random.rand() < fraud_rate
            
            sample = {
                'timestamp': time.time(),
                'movement': {
                    'acceleration': np.random.randn(3) * (2.0 if is_fraud else 1.0),
                    'gyroscope': np.random.randn(3) * (1.5 if is_fraud else 1.0),
                    'magnetometer': np.random.randn(3)
                },
                'voice': np.random.randn(1600) * (0.5 if is_fraud else 1.0),
                'touch': [{'pressure': np.random.rand() * (0.3 if is_fraud else 1.0), 
                          'area': np.random.rand() * 20}],
                'sample_id': f'stub_{"fraud" if is_fraud else "normal"}_{i}',
                'is_fraud': is_fraud  # Ground truth for evaluation
            }
            
            yield sample
            time.sleep(1.0 / samples_per_sec)
    
    def _generate_stress_test_samples(self, duration_sec: int, load_multiplier: float) -> Generator[Dict[str, Any], None, None]:
        """Generate high-load samples for stress testing."""
        samples_per_sec = int(20 * load_multiplier)
        total_samples = duration_sec * samples_per_sec
        
        for i in range(total_samples):
            # Larger, more complex samples
            yield {
                'timestamp': time.time(),
                'movement': {
                    'acceleration': np.random.randn(100, 3),  # Larger arrays
                    'gyroscope': np.random.randn(100, 3),
                    'magnetometer': np.random.randn(100, 3)
                },
                'voice': np.random.randn(8000),  # 0.5 second of audio
                'touch': [{'pressure': np.random.rand(), 'area': np.random.rand() * 20} 
                         for _ in range(10)],  # Multiple touch points
                'sample_id': f'stress_{i}'
            }
            time.sleep(1.0 / samples_per_sec)
    
    def _generate_demo_summary(self, results: List[ProcessingResult], demo_type: str) -> Dict[str, Any]:
        """Generate comprehensive demo summary."""
        if not results:
            return {'error': 'No results to summarize'}
        
        # Basic stats
        total_samples = len(results)
        anomalies_detected = sum(1 for r in results if r.is_anomaly)
        avg_confidence = np.mean([r.confidence_score for r in results])
        
        # Performance stats
        processing_times = [r.processing_time_ms for r in results]
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        
        # Time range
        start_time = min(r.timestamp for r in results)
        end_time = max(r.timestamp for r in results)
        duration = end_time - start_time
        
        summary = {
            'demo_type': demo_type,
            'total_samples': total_samples,
            'anomalies_detected': anomalies_detected,
            'anomaly_rate': anomalies_detected / total_samples,
            'avg_confidence_score': avg_confidence,
            'avg_processing_time_ms': avg_processing_time,
            'max_processing_time_ms': max_processing_time,
            'demo_duration_sec': duration,
            'throughput_sps': total_samples / duration if duration > 0 else 0,
            'performance_metrics': self.agent_coordinator.get_performance_metrics()
        }
        
        return summary
    
    def export_demo_results(self, filename: Optional[str] = None) -> str:
        """Export demo results to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"quadfusion_demo_results_{timestamp}.json"
        
        # Prepare export data
        export_data = {
            'demo_config': {
                'batch_size': self.config.batch_size,
                'enable_visualization': self.config.enable_visualization,
                'enable_encryption': self.config.enable_encryption
            },
            'results': [
                {
                    'timestamp': r.timestamp,
                    'sample_id': r.sample_id,
                    'is_anomaly': r.is_anomaly,
                    'confidence_score': r.confidence_score,
                    'processing_time_ms': r.processing_time_ms,
                    'motion_prediction': r.motion_prediction,
                    'voice_prediction': r.voice_prediction,
                    'fused_decision': r.fused_decision
                }
                for r in self.demo_results
            ],
            'performance_metrics': self.agent_coordinator.get_performance_metrics()
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logging.info(f"Demo results exported to {filename}")
        return filename

class InteractiveDemoUI:
    """
    Interactive command-line interface for running demos.
    Provides menu-driven access to different demo scenarios.
    """
    
    def __init__(self):
        self.pipeline = DemoPipeline()
        self.running = True
        
    def display_menu(self):
        """Display main demo menu."""
        print("\n" + "="*50)
        print("     QuadFusion Demo Pipeline")
        print("="*50)
        print("1. Normal Behavior Demo")
        print("2. Fraud Detection Demo")  
        print("3. Stress Test Demo")
        print("4. Battery Constrained Demo")
        print("5. View Performance Metrics")
        print("6. Export Results")
        print("7. Configuration")
        print("8. Exit")
        print("="*50)
        
    def run_interactive_demo(self):
        """Run interactive demo interface."""
        print("Welcome to QuadFusion Interactive Demo!")
        
        while self.running:
            self.display_menu()
            
            try:
                choice = input("\nSelect option (1-8): ").strip()
                
                if choice == '1':
                    self._run_normal_demo()
                elif choice == '2':
                    self._run_fraud_demo()
                elif choice == '3':
                    self._run_stress_demo()
                elif choice == '4':
                    self._run_battery_demo()
                elif choice == '5':
                    self._show_metrics()
                elif choice == '6':
                    self._export_results()
                elif choice == '7':
                    self._configure_demo()
                elif choice == '8':
                    print("Exiting demo. Goodbye!")
                    self.running = False
                else:
                    print("Invalid choice. Please select 1-8.")
                    
            except KeyboardInterrupt:
                print("\nDemo interrupted. Exiting...")
                self.running = False
            except Exception as e:
                print(f"Error: {e}")
    
    def _run_normal_demo(self):
        """Run normal behavior demo with user input."""
        print("\n--- Normal Behavior Demo ---")
        duration = self._get_duration_input("Duration in seconds (default 30): ", 30)
        
        print(f"Running normal behavior demo for {duration} seconds...")
        summary = self.pipeline.run_normal_behavior_demo(duration)
        self._display_summary(summary)
    
    def _run_fraud_demo(self):
        """Run fraud detection demo with user input."""
        print("\n--- Fraud Detection Demo ---")
        duration = self._get_duration_input("Duration in seconds (default 30): ", 30)
        fraud_rate = self._get_float_input("Fraud rate 0.0-1.0 (default 0.1): ", 0.1, 0.0, 1.0)
        
        print(f"Running fraud detection demo for {duration} seconds with {fraud_rate*100}% fraud rate...")
        summary = self.pipeline.run_fraud_detection_demo(fraud_rate, duration)
        self._display_summary(summary)
    
    def _run_stress_demo(self):
        """Run stress test demo with user input."""
        print("\n--- Stress Test Demo ---")
        duration = self._get_duration_input("Duration in seconds (default 20): ", 20)
        load = self._get_float_input("Load multiplier (default 2.0): ", 2.0, 1.0, 10.0)
        
        print(f"Running stress test for {duration} seconds with {load}x load...")
        summary = self.pipeline.run_stress_test_demo(load, duration)
        self._display_summary(summary)
    
    def _run_battery_demo(self):
        """Run battery constrained demo with user input."""
        print("\n--- Battery Constrained Demo ---")
        duration = self._get_duration_input("Duration in seconds (default 30): ", 30)
        battery = self._get_float_input("Battery level % (default 20): ", 20.0, 1.0, 100.0)
        
        print(f"Running battery demo for {duration} seconds at {battery}% battery...")
        summary = self.pipeline.run_battery_constrained_demo(battery, duration)
        self._display_summary(summary)
    
    def _show_metrics(self):
        """Display current performance metrics."""
        print("\n--- Performance Metrics ---")
        metrics = self.pipeline.agent_coordinator.get_performance_metrics()
        
        if not metrics:
            print("No metrics available. Run a demo first.")
            return
            
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # Show alerts
        alerts = self.pipeline.agent_coordinator.get_performance_alerts()
        if alerts:
            print("\nPerformance Alerts:")
            for alert in alerts:
                print(f"⚠️  {alert}")
    
    def _export_results(self):
        """Export demo results."""
        print("\n--- Export Results ---")
        if not self.pipeline.demo_results:
            print("No results to export. Run a demo first.")
            return
            
        filename = input("Filename (default: auto-generated): ").strip()
        filename = filename if filename else None
        
        exported_file = self.pipeline.export_demo_results(filename)
        print(f"Results exported to: {exported_file}")
    
    def _configure_demo(self):
        """Configure demo settings."""
        print("\n--- Demo Configuration ---")
        print(f"Current batch size: {self.pipeline.config.batch_size}")
        print(f"Visualization enabled: {self.pipeline.config.enable_visualization}")
        print(f"Encryption enabled: {self.pipeline.config.enable_encryption}")
        
        # Allow basic configuration changes
        new_batch_size = self._get_int_input("New batch size (1-128): ", 
                                           self.pipeline.config.batch_size, 1, 128)
        self.pipeline.config.batch_size = new_batch_size
        print("Configuration updated!")
    
    def _display_summary(self, summary: Dict[str, Any]):
        """Display demo summary in a formatted way."""
        print("\n" + "="*40)
        print("         DEMO SUMMARY")
        print("="*40)
        
        for key, value in summary.items():
            if key == 'performance_metrics':
                continue  # Skip nested dict
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("="*40)
    
    def _get_duration_input(self, prompt: str, default: int) -> int:
        """Get duration input with validation."""
        return self._get_int_input(prompt, default, 5, 300)
    
    def _get_int_input(self, prompt: str, default: int, min_val: int, max_val: int) -> int:
        """Get integer input with validation."""
        while True:
            try:
                value = input(prompt).strip()
                if not value:
                    return default
                value = int(value)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Value must be between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid integer")
    
    def _get_float_input(self, prompt: str, default: float, min_val: float, max_val: float) -> float:
        """Get float input with validation."""
        while True:
            try:
                value = input(prompt).strip()
                if not value:
                    return default
                value = float(value)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Value must be between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid number")

# Example usage and testing
def run_demo_examples():
    """Run example demo scenarios."""
    print("Running QuadFusion Demo Examples...")
    
    # Initialize demo pipeline
    config = DemoConfig(
        demo_duration_sec=10,  # Short demo for testing
        enable_visualization=False,  # Disable for automated testing
        batch_size=16
    )
    
    pipeline = DemoPipeline(config)
    
    # Run different demo scenarios
    print("\n1. Normal Behavior Demo:")
    normal_summary = pipeline.run_normal_behavior_demo(duration_sec=10)
    print(f"Processed {normal_summary.get('total_samples', 0)} samples")
    
    print("\n2. Fraud Detection Demo:")
    fraud_summary = pipeline.run_fraud_detection_demo(fraud_rate=0.2, duration_sec=10)
    print(f"Detected {fraud_summary.get('anomalies_detected', 0)} anomalies")
    
    print("\n3. Stress Test Demo:")
    stress_summary = pipeline.run_stress_test_demo(load_multiplier=1.5, duration_sec=5)
    print(f"Average latency: {stress_summary.get('avg_processing_time_ms', 0):.2f}ms")
    
    # Export results
    if pipeline.demo_results:
        exported_file = pipeline.export_demo_results()
        print(f"\nResults exported to: {exported_file}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        # Run interactive demo
        ui = InteractiveDemoUI()
        ui.run_interactive_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == '--examples':
        # Run example scenarios
        run_demo_examples()
    else:
        # Default: run interactive demo
        print("Starting QuadFusion Demo Pipeline...")
        print("Use --examples for automated demo or --interactive for menu")
        ui = InteractiveDemoUI()
        ui.run_interactive_demo()
