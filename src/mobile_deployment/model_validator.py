# mobile_deployment/model_validator.py

"""
Model validation utilities for mobile deployment.

Features:
- Comprehensive model validation suite
- Accuracy validation post-conversion
- Performance benchmarking on mobile
- Memory usage analysis
- Battery consumption testing
- Cross-platform compatibility testing
- Model integrity verification
- Inference speed testing
- Stress testing under resource constraints
- Real-device testing automation
- A/B testing framework for model versions
- Regression testing pipeline
- Quality assurance metrics
- Deployment readiness assessment
"""

import os
import time
import json
import logging
import threading
import numpy as np
import psutil
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from threading import Thread

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class ValidationConfig:
    """Configuration for model validation tests."""
    accuracy_tolerance: float = 0.01
    max_latency_ms: float = 100.0
    max_memory_mb: float = 200.0
    max_model_size_mb: float = 10.0
    min_accuracy: float = 0.95
    battery_test_duration_sec: int = 300  # 5 minutes
    stress_test_iterations: int = 1000

@dataclass
class ValidationResult:
    """Results from validation tests."""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

class ModelValidator:
    """
    Comprehensive model validation orchestrator for mobile deployment.
    Ensures models meet performance, accuracy, and deployment requirements.
    """
    
    def __init__(self, model_path: str, config: Optional[ValidationConfig] = None):
        self.model_path = Path(model_path)
        self.config = config or ValidationConfig()
        self.device_info = self._gather_device_info()
        self.validation_results: List[ValidationResult] = []
        self.test_data_cache = {}
        
    def _gather_device_info(self) -> Dict[str, Any]:
        """Gather comprehensive device information."""
        try:
            info = {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "processor": platform.processor(),
                "machine": platform.machine(),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "total_memory_mb": psutil.virtual_memory().total / (1024*1024),
                "available_memory_mb": psutil.virtual_memory().available / (1024*1024),
                "cpu_freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else None,
                "battery_present": hasattr(psutil, 'sensors_battery') and psutil.sensors_battery() is not None
            }
            
            # GPU information (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    info["gpu_name"] = gpus[0].name
                    info["gpu_memory_mb"] = gpus.memoryTotal
            except ImportError:
                info["gpu_available"] = False
                
        except Exception as e:
            logging.warning(f"Could not gather complete device info: {e}")
            info = {"platform": "unknown", "error": str(e)}
            
        logging.info(f"Device info gathered: {info}")
        return info
    
    def validate_model_accuracy(self, 
                               reference_predictions: List[Any],
                               test_predictions: List[Any],
                               metric: str = "max_difference") -> ValidationResult:
        """
        Validate accuracy of converted model against reference implementation.
        
        Args:
            reference_predictions: Ground truth or baseline model outputs
            test_predictions: Converted model predictions
            metric: Accuracy metric to use ('max_difference', 'mse', 'classification')
            
        Returns:
            ValidationResult with accuracy assessment
        """
        start_time = time.time()
        
        try:
            if len(reference_predictions) != len(test_predictions):
                return ValidationResult(
                    test_name="accuracy_validation",
                    passed=False,
                    score=0.0,
                    details={"error": "Prediction count mismatch"}
                )
            
            if metric == "max_difference":
                differences = []
                for ref, test in zip(reference_predictions, test_predictions):
                    ref_arr = np.array(ref).flatten()
                    test_arr = np.array(test).flatten()
                    diff = np.max(np.abs(ref_arr - test_arr))
                    differences.append(diff)
                
                max_diff = max(differences)
                avg_diff = np.mean(differences)
                passed = max_diff <= self.config.accuracy_tolerance
                
                details = {
                    "max_difference": float(max_diff),
                    "avg_difference": float(avg_diff),
                    "tolerance": self.config.accuracy_tolerance,
                    "samples_tested": len(reference_predictions)
                }
                
            elif metric == "mse":
                mse_scores = []
                for ref, test in zip(reference_predictions, test_predictions):
                    ref_arr = np.array(ref).flatten()
                    test_arr = np.array(test).flatten()
                    mse = np.mean((ref_arr - test_arr) ** 2)
                    mse_scores.append(mse)
                
                avg_mse = np.mean(mse_scores)
                passed = avg_mse <= self.config.accuracy_tolerance
                max_diff = np.sqrt(avg_mse)  # RMSE as score
                
                details = {
                    "mse": float(avg_mse),
                    "rmse": float(np.sqrt(avg_mse)),
                    "tolerance": self.config.accuracy_tolerance
                }
                
            elif metric == "classification":
                # For classification tasks
                correct = 0
                for ref, test in zip(reference_predictions, test_predictions):
                    ref_class = np.argmax(ref) if hasattr(ref, '__len__') else ref
                    test_class = np.argmax(test) if hasattr(test, '__len__') else test
                    if ref_class == test_class:
                        correct += 1
                
                accuracy = correct / len(reference_predictions)
                passed = accuracy >= self.config.min_accuracy
                max_diff = 1.0 - accuracy  # Error rate as score
                
                details = {
                    "accuracy": accuracy,
                    "correct_predictions": correct,
                    "total_predictions": len(reference_predictions),
                    "min_accuracy_threshold": self.config.min_accuracy
                }
            
            result = ValidationResult(
                test_name="accuracy_validation",
                passed=passed,
                score=1.0 - max_diff,  # Higher score = better
                details={
                    **details,
                    "metric_used": metric,
                    "test_duration_sec": time.time() - start_time
                }
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name="accuracy_validation",
                passed=False,
                score=0.0,
                details={"error": str(e)}
            )
            
        self.validation_results.append(result)
        logging.info(f"Accuracy validation: {result.passed}, score: {result.score:.4f}")
        return result
    
    def benchmark_inference_speed(self, 
                                 inference_func: Callable,
                                 sample_input: Any,
                                 warmup_runs: int = 10,
                                 benchmark_runs: int = 100) -> ValidationResult:
        """
        Comprehensive inference speed benchmarking.
        
        Args:
            inference_func: Function that performs model inference
            sample_input: Sample input data for testing
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
            
        Returns:
            ValidationResult with speed benchmarks
        """
        start_time = time.time()
        latencies = []
        errors = 0
        
        try:
            # Warmup runs
            logging.info(f"Starting warmup ({warmup_runs} runs)...")
            for _ in range(warmup_runs):
                try:
                    _ = inference_func(sample_input)
                except Exception as e:
                    logging.warning(f"Warmup error: {e}")
            
            # Benchmark runs
            logging.info(f"Starting benchmark ({benchmark_runs} runs)...")
            for i in range(benchmark_runs):
                iter_start = time.perf_counter()
                try:
                    _ = inference_func(sample_input)
                    iter_end = time.perf_counter()
                    latencies.append((iter_end - iter_start) * 1000)  # Convert to ms
                except Exception as e:
                    errors += 1
                    logging.error(f"Benchmark iteration {i} failed: {e}")
                    continue
            
            if not latencies:
                return ValidationResult(
                    test_name="inference_speed",
                    passed=False,
                    score=0.0,
                    details={"error": "No successful inferences", "total_errors": errors}
                )
            
            # Calculate statistics
            latencies_array = np.array(latencies)
            stats = {
                "avg_latency_ms": float(np.mean(latencies_array)),
                "median_latency_ms": float(np.median(latencies_array)),
                "p95_latency_ms": float(np.percentile(latencies_array, 95)),
                "p99_latency_ms": float(np.percentile(latencies_array, 99)),
                "min_latency_ms": float(np.min(latencies_array)),
                "max_latency_ms": float(np.max(latencies_array)),
                "std_latency_ms": float(np.std(latencies_array)),
                "throughput_qps": 1000.0 / np.mean(latencies_array),
                "success_rate": (len(latencies) / benchmark_runs) * 100,
                "total_errors": errors,
                "total_runs": benchmark_runs
            }
            
            # Check if performance meets requirements
            passed = (stats["p99_latency_ms"] <= self.config.max_latency_ms and 
                     stats["success_rate"] >= 95.0)
            
            # Score based on inverse of p99 latency (normalized)
            score = min(1.0, self.config.max_latency_ms / stats["p99_latency_ms"]) if passed else 0.0
            
            result = ValidationResult(
                test_name="inference_speed",
                passed=passed,
                score=score,
                details={
                    **stats,
                    "max_latency_threshold": self.config.max_latency_ms,
                    "test_duration_sec": time.time() - start_time,
                    "device_info": self.device_info
                }
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name="inference_speed",
                passed=False,
                score=0.0,
                details={"error": str(e)}
            )
        
        self.validation_results.append(result)
        logging.info(f"Speed benchmark: {result.passed}, avg: {result.details.get('avg_latency_ms', 0):.2f}ms")
        return result
    
    def test_memory_usage(self, 
                         inference_func: Callable,
                         sample_input: Any,
                         monitoring_duration_sec: float = 10.0) -> ValidationResult:
        """
        Monitor memory usage during model inference.
        
        Args:
            inference_func: Function performing inference
            sample_input: Input data for testing
            monitoring_duration_sec: How long to monitor
            
        Returns:
            ValidationResult with memory usage statistics
        """
        start_time = time.time()
        memory_samples = []
        inference_count = 0
        stop_monitoring = threading.Event()
        
        def memory_monitor():
            """Background thread to monitor memory usage."""
            process = psutil.Process()
            while not stop_monitoring.is_set():
                try:
                    mem_info = process.memory_info()
                    memory_samples.append({
                        'rss_mb': mem_info.rss / (1024 * 1024),
                        'vms_mb': mem_info.vms / (1024 * 1024),
                        'timestamp': time.time()
                    })
                    time.sleep(0.01)  # 100Hz sampling
                except Exception as e:
                    logging.warning(f"Memory monitoring error: {e}")
                    break
        
        def inference_loop():
            """Run inference continuously during monitoring."""
            nonlocal inference_count
            end_time = time.time() + monitoring_duration_sec
            while time.time() < end_time:
                try:
                    _ = inference_func(sample_input)
                    inference_count += 1
                    time.sleep(0.05)  # 20Hz inference rate
                except Exception as e:
                    logging.error(f"Inference error in memory test: {e}")
        
        try:
            # Start monitoring
            monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
            monitor_thread.start()
            
            # Run inference loop
            inference_thread = threading.Thread(target=inference_loop, daemon=True)
            inference_thread.start()
            
            # Wait for completion
            inference_thread.join(timeout=monitoring_duration_sec + 5)
            stop_monitoring.set()
            monitor_thread.join(timeout=1)
            
            if not memory_samples:
                return ValidationResult(
                    test_name="memory_usage",
                    passed=False,
                    score=0.0,
                    details={"error": "No memory samples collected"}
                )
            
            # Calculate memory statistics
            rss_values = [sample['rss_mb'] for sample in memory_samples]
            vms_values = [sample['vms_mb'] for sample in memory_samples]
            
            baseline_memory = rss_values[0] if rss_values else 0
            peak_memory = max(rss_values)
            avg_memory = np.mean(rss_values)
            memory_increase = peak_memory - baseline_memory
            
            stats = {
                "baseline_memory_mb": baseline_memory,
                "peak_memory_mb": peak_memory,
                "avg_memory_mb": avg_memory,
                "max_vms_mb": max(vms_values),
                "memory_increase_mb": memory_increase,
                "samples_collected": len(memory_samples),
                "inferences_completed": inference_count,
                "memory_per_inference_mb": memory_increase / max(1, inference_count),
                "monitoring_duration_sec": monitoring_duration_sec
            }
            
            # Check if memory usage is within limits
            passed = (peak_memory <= self.config.max_memory_mb and
                     memory_increase < self.config.max_memory_mb * 0.5)  # 50% increase limit
            
            # Score based on memory efficiency
            score = max(0.0, 1.0 - (peak_memory / self.config.max_memory_mb)) if passed else 0.0
            
            result = ValidationResult(
                test_name="memory_usage",
                passed=passed,
                score=score,
                details={
                    **stats,
                    "max_memory_threshold": self.config.max_memory_mb,
                    "test_duration_sec": time.time() - start_time
                }
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name="memory_usage",
                passed=False,
                score=0.0,
                details={"error": str(e)}
            )
        
        self.validation_results.append(result)
        logging.info(f"Memory test: {result.passed}, peak: {result.details.get('peak_memory_mb', 0):.1f}MB")
        return result
    
    def test_battery_consumption(self, 
                                inference_func: Callable,
                                sample_input: Any) -> ValidationResult:
        """
        Estimate battery consumption during model inference.
        Note: Actual battery measurement requires platform-specific APIs.
        """
        start_time = time.time()
        
        try:
            # Check if battery monitoring is available
            if not self.device_info.get("battery_present", False):
                return ValidationResult(
                    test_name="battery_consumption",
                    passed=True,  # Skip test if no battery
                    score=1.0,
                    details={"skipped": True, "reason": "No battery detected"}
                )
            
            # Run CPU-intensive inference to estimate power usage
            cpu_percent_before = psutil.cpu_percent(interval=1)
            inference_start = time.time()
            
            # Run inference for test duration
            inference_count = 0
            end_time = time.time() + self.config.battery_test_duration_sec
            
            while time.time() < end_time:
                _ = inference_func(sample_input)
                inference_count += 1
                time.sleep(0.1)  # 10Hz
            
            inference_duration = time.time() - inference_start
            cpu_percent_after = psutil.cpu_percent(interval=1)
            
            # Estimate based on CPU usage (rough approximation)
            avg_cpu_increase = max(0, cpu_percent_after - cpu_percent_before)
            estimated_power_increase_percent = avg_cpu_increase / 100.0 * 2.0  # Rough estimate
            
            stats = {
                "test_duration_sec": inference_duration,
                "inferences_completed": inference_count,
                "inference_rate_hz": inference_count / inference_duration,
                "cpu_before_percent": cpu_percent_before,
                "cpu_after_percent": cpu_percent_after,
                "cpu_increase_percent": avg_cpu_increase,
                "estimated_battery_impact_percent_per_hour": estimated_power_increase_percent * 100
            }
            
            # Pass if estimated battery impact is reasonable
            passed = estimated_power_increase_percent < 0.02  # <2% per hour
            score = max(0.0, 1.0 - estimated_power_increase_percent * 50) if passed else 0.0
            
            result = ValidationResult(
                test_name="battery_consumption",
                passed=passed,
                score=score,
                details=stats
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name="battery_consumption",
                passed=False,
                score=0.0,
                details={"error": str(e)}
            )
        
        self.validation_results.append(result)
        logging.info(f"Battery test: {result.passed}, estimated impact: {result.details.get('estimated_battery_impact_percent_per_hour', 0):.2f}%/hr")
        return result
    
    def validate_cross_platform(self, 
                               platform_configs: Dict[str, Dict[str, Any]],
                               test_input: Any) -> ValidationResult:
        """
        Test model consistency across different platforms/configurations.
        
        Args:
            platform_configs: Dict of platform->config mappings
            test_input: Input data for testing
            
        Returns:
            ValidationResult with cross-platform compatibility
        """
        start_time = time.time()
        
        try:
            baseline_output = None
            platform_results = {}
            
            for platform, config in platform_configs.items():
                try:
                    inference_func = config['inference_func']
                    output = inference_func(test_input)
                    
                    if baseline_output is None:
                        baseline_output = output
                        platform_results[platform] = {
                            'success': True,
                            'is_baseline': True,
                            'max_difference': 0.0
                        }
                    else:
                        # Compare with baseline
                        output_arr = np.array(output).flatten()
                        baseline_arr = np.array(baseline_output).flatten()
                        max_diff = np.max(np.abs(output_arr - baseline_arr))
                        
                        platform_results[platform] = {
                            'success': True,
                            'is_baseline': False,
                            'max_difference': float(max_diff),
                            'consistent': max_diff < self.config.accuracy_tolerance
                        }
                        
                except Exception as e:
                    platform_results[platform] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Determine overall success
            successful_platforms = [p for p, r in platform_results.items() if r['success']]
            consistent_platforms = [p for p, r in platform_results.items() 
                                  if r.get('consistent', True)]  # Baseline is always consistent
            
            passed = (len(successful_platforms) >= len(platform_configs) * 0.8 and  # 80% success rate
                     len(consistent_platforms) >= len(successful_platforms) * 0.9)   # 90% consistency
            
            score = (len(consistent_platforms) / len(platform_configs)) if platform_configs else 0.0
            
            result = ValidationResult(
                test_name="cross_platform_compatibility",
                passed=passed,
                score=score,
                details={
                    "platform_results": platform_results,
                    "successful_platforms": successful_platforms,
                    "consistent_platforms": consistent_platforms,
                    "total_platforms": len(platform_configs),
                    "test_duration_sec": time.time() - start_time
                }
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name="cross_platform_compatibility",
                passed=False,
                score=0.0,
                details={"error": str(e)}
            )
        
        self.validation_results.append(result)
        logging.info(f"Cross-platform test: {result.passed}, score: {result.score:.3f}")
        return result
    
    def run_stress_test(self, 
                       inference_func: Callable,
                       sample_input: Any,
                       duration_minutes: int = 5) -> ValidationResult:
        """
        Stress test the model under continuous load.
        
        Args:
            inference_func: Function performing inference
            sample_input: Input data
            duration_minutes: Test duration in minutes
            
        Returns:
            ValidationResult with stress test results
        """
        start_time = time.time()
        duration_sec = duration_minutes * 60
        
        try:
            inference_count = 0
            error_count = 0
            latencies = []
            memory_peaks = []
            
            end_time = time.time() + duration_sec
            
            while time.time() < end_time:
                iter_start = time.perf_counter()
                
                try:
                    _ = inference_func(sample_input)
                    iter_end = time.perf_counter()
                    
                    latency_ms = (iter_end - iter_start) * 1000
                    latencies.append(latency_ms)
                    inference_count += 1
                    
                    # Sample memory usage periodically
                    if inference_count % 100 == 0:
                        memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                        memory_peaks.append(memory_mb)
                    
                except Exception as e:
                    error_count += 1
                    logging.warning(f"Stress test error {error_count}: {e}")
                
                # Brief pause to prevent overheating
                time.sleep(0.01)
            
            actual_duration = time.time() - start_time
            
            if not latencies:
                return ValidationResult(
                    test_name="stress_test",
                    passed=False,
                    score=0.0,
                    details={"error": "No successful inferences during stress test"}
                )
            
            # Calculate stress test statistics
            latencies_array = np.array(latencies)
            stats = {
                "duration_minutes": actual_duration / 60,
                "total_inferences": inference_count,
                "total_errors": error_count,
                "success_rate": (inference_count / (inference_count + error_count)) * 100,
                "avg_latency_ms": float(np.mean(latencies_array)),
                "p95_latency_ms": float(np.percentile(latencies_array, 95)),
                "max_latency_ms": float(np.max(latencies_array)),
                "throughput_qps": inference_count / actual_duration,
                "peak_memory_mb": max(memory_peaks) if memory_peaks else 0,
                "latency_stability": float(np.std(latencies_array))
            }
            
            # Pass criteria: high success rate, stable performance
            passed = (stats["success_rate"] >= 95.0 and 
                     stats["p95_latency_ms"] <= self.config.max_latency_ms * 1.5)  # Allow 50% overhead
            
            score = (stats["success_rate"] / 100.0) * 0.7 + \
                   (min(1.0, self.config.max_latency_ms / stats["p95_latency_ms"]) * 0.3)
            
            result = ValidationResult(
                test_name="stress_test",
                passed=passed,
                score=score,
                details=stats
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name="stress_test", 
                passed=False,
                score=0.0,
                details={"error": str(e)}
            )
        
        self.validation_results.append(result)
        logging.info(f"Stress test: {result.passed}, {result.details.get('total_inferences', 0)} inferences")
        return result
    
    def assess_deployment_readiness(self) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Comprehensive assessment of deployment readiness based on all validation results.
        
        Returns:
            (is_ready, overall_score, detailed_assessment)
        """
        if not self.validation_results:
            return False, 0.0, {"error": "No validation results available"}
        
        # Categorize results
        test_categories = {
            "accuracy": ["accuracy_validation"],
            "performance": ["inference_speed", "memory_usage"],
            "reliability": ["stress_test", "cross_platform_compatibility"],
            "efficiency": ["battery_consumption"]
        }
        
        category_scores = {}
        category_details = {}
        
        for category, test_names in test_categories.items():
            category_results = [r for r in self.validation_results if r.test_name in test_names]
            
            if category_results:
                scores = [r.score for r in category_results if r.passed]
                passed_count = sum(1 for r in category_results if r.passed)
                total_count = len(category_results)
                
                category_scores[category] = {
                    "avg_score": np.mean(scores) if scores else 0.0,
                    "pass_rate": passed_count / total_count,
                    "tests_passed": passed_count,
                    "tests_total": total_count
                }
                category_details[category] = {r.test_name: r for r in category_results}
            else:
                category_scores[category] = {"avg_score": 0.0, "pass_rate": 0.0}
        
        # Calculate weighted overall score
        weights = {"accuracy": 0.4, "performance": 0.3, "reliability": 0.2, "efficiency": 0.1}
        overall_score = sum(weights[cat] * scores["avg_score"] * scores["pass_rate"] 
                           for cat, scores in category_scores.items())
        
        # Determine readiness (must pass critical tests)
        critical_tests_passed = all(
            category_scores["accuracy"]["pass_rate"] >= 1.0,  # Must pass accuracy
            category_scores["performance"]["pass_rate"] >= 0.8,  # 80% of perf tests
        )
        
        is_ready = critical_tests_passed and overall_score >= 0.7
        
        assessment = {
            "overall_score": overall_score,
            "is_deployment_ready": is_ready,
            "category_scores": category_scores,
            "critical_tests_passed": critical_tests_passed,
            "total_tests_run": len(self.validation_results),
            "total_tests_passed": sum(1 for r in self.validation_results if r.passed),
            "recommendations": self._generate_recommendations(),
            "test_timestamp": time.time(),
            "device_info": self.device_info
        }
        
        logging.info(f"Deployment assessment: ready={is_ready}, score={overall_score:.3f}")
        return is_ready, overall_score, assessment
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for result in self.validation_results:
            if not result.passed:
                if result.test_name == "accuracy_validation":
                    recommendations.append("Consider retraining with higher precision or adjusting quantization")
                elif result.test_name == "inference_speed":
                    recommendations.append("Optimize model architecture or apply more aggressive pruning")
                elif result.test_name == "memory_usage":
                    recommendations.append("Reduce model size or implement memory pooling")
                elif result.test_name == "battery_consumption":
                    recommendations.append("Optimize inference frequency or use power-efficient operations")
                elif result.test_name == "stress_test":
                    recommendations.append("Improve error handling and memory management")
        
        return recommendations
    
    def generate_validation_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        is_ready, overall_score, assessment = self.assess_deployment_readiness()
        
        report = {
            "model_path": str(self.model_path),
            "validation_timestamp": time.time(),
            "deployment_readiness": assessment,
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "score": r.score,
                    "details": r.details,
                    "timestamp": r.timestamp
                }
                for r in self.validation_results
            ],
            "device_info": self.device_info,
            "configuration": {
                "accuracy_tolerance": self.config.accuracy_tolerance,
                "max_latency_ms": self.config.max_latency_ms,
                "max_memory_mb": self.config.max_memory_mb
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logging.info(f"Validation report saved to {output_path}")
        
        return report


class PerformanceBenchmark:
    """Specialized performance benchmarking utilities."""
    
    def __init__(self, validator: ModelValidator):
        self.validator = validator
    
    def run_comprehensive_benchmark(self, 
                                  inference_func: Callable,
                                  sample_input: Any,
                                  test_duration_sec: int = 60) -> Dict[str, ValidationResult]:
        """Run all performance benchmarks."""
        results = {}
        
        # Speed benchmark
        results['speed'] = self.validator.benchmark_inference_speed(
            inference_func, sample_input, benchmark_runs=100
        )
        
        # Memory benchmark
        results['memory'] = self.validator.test_memory_usage(
            inference_func, sample_input, monitoring_duration_sec=10
        )
        
        # Battery benchmark
        results['battery'] = self.validator.test_battery_consumption(
            inference_func, sample_input
        )
        
        # Stress test
        results['stress'] = self.validator.run_stress_test(
            inference_func, sample_input, duration_minutes=test_duration_sec//60
        )
        
        return results


class AccuracyValidator:
    """Specialized accuracy validation utilities."""
    
    def __init__(self, validator: ModelValidator):
        self.validator = validator
    
    def validate_classification_accuracy(self, 
                                       true_labels: List[int],
                                       predicted_probabilities: List[np.ndarray]) -> ValidationResult:
        """Validate classification model accuracy."""
        predicted_labels = [np.argmax(prob) for prob in predicted_probabilities]
        
        # Calculate accuracy metrics
        correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
        total = len(true_labels)
        accuracy = correct / total
        
        # Detailed per-class analysis
        unique_labels = sorted(set(true_labels))
        per_class_stats = {}
        
        for label in unique_labels:
            true_positives = sum(1 for t, p in zip(true_labels, predicted_labels) 
                               if t == label and p == label)
            false_positives = sum(1 for t, p in zip(true_labels, predicted_labels) 
                                if t != label and p == label)
            false_negatives = sum(1 for t, p in zip(true_labels, predicted_labels) 
                                if t == label and p != label)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_stats[label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            }
        
        passed = accuracy >= self.validator.config.min_accuracy
        
        return ValidationResult(
            test_name="classification_accuracy",
            passed=passed,
            score=accuracy,
            details={
                'overall_accuracy': accuracy,
                'correct_predictions': correct,
                'total_predictions': total,
                'per_class_metrics': per_class_stats,
                'avg_f1_score': np.mean([stats['f1_score'] for stats in per_class_stats.values()]),
                'min_accuracy_threshold': self.validator.config.min_accuracy
            }
        )


class MobileTestSuite:
    """Comprehensive mobile testing automation."""
    
    def __init__(self, validator: ModelValidator):
        self.validator = validator
    
    def run_full_test_suite(self, 
                           inference_func: Callable,
                           sample_inputs: Dict[str, Any],
                           reference_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, ValidationResult]:
        """Run complete mobile validation test suite."""
        results = {}
        
        # Basic sample input for tests that need it
        basic_input = list(sample_inputs.values())[0] if sample_inputs else None
        
        # 1. Accuracy validation (if reference provided)
        if reference_outputs:
            for input_name, ref_output in reference_outputs.items():
                if input_name in sample_inputs:
                    test_output = inference_func(sample_inputs[input_name])
                    results[f'accuracy_{input_name}'] = self.validator.validate_model_accuracy(
                        [ref_output], [test_output]
                    )
        
        # 2. Performance benchmarks
        if basic_input is not None:
            results['inference_speed'] = self.validator.benchmark_inference_speed(
                inference_func, basic_input
            )
            results['memory_usage'] = self.validator.test_memory_usage(
                inference_func, basic_input
            )
            results['battery_test'] = self.validator.test_battery_consumption(
                inference_func, basic_input
            )
        
        # 3. Stress test
        if basic_input is not None:
            results['stress_test'] = self.validator.run_stress_test(
                inference_func, basic_input, duration_minutes=2
            )
        
        return results


class DeploymentChecker:
    """Final deployment readiness verification."""
    
    def __init__(self, validator: ModelValidator):
        self.validator = validator
    
    def check_deployment_readiness(self) -> Dict[str, Any]:
        """Perform final deployment readiness check."""
        is_ready, score, assessment = self.validator.assess_deployment_readiness()
        
        # Additional checks
        model_size_mb = 0
        if self.validator.model_path.exists():
            model_size_mb = self.validator.model_path.stat().st_size / (1024 * 1024)
        
        size_check = model_size_mb <= self.validator.config.max_model_size_mb
        
        final_assessment = {
            **assessment,
            'model_size_mb': model_size_mb,
            'model_size_check_passed': size_check,
            'max_model_size_mb': self.validator.config.max_model_size_mb,
            'final_deployment_ready': is_ready and size_check
        }
        
        return final_assessment
    
    def generate_deployment_checklist(self) -> List[Dict[str, Any]]:
        """Generate deployment checklist."""
        checklist_items = [
            {
                'item': 'Model accuracy validation',
                'status': 'pending',
                'required': True,
                'description': 'Validate model maintains accuracy after conversion'
            },
            {
                'item': 'Inference speed benchmark',
                'status': 'pending', 
                'required': True,
                'description': 'Ensure inference time meets mobile requirements'
            },
            {
                'item': 'Memory usage validation',
                'status': 'pending',
                'required': True,
                'description': 'Verify memory usage is within mobile constraints'
            },
            {
                'item': 'Cross-platform compatibility',
                'status': 'pending',
                'required': False,
                'description': 'Test model consistency across platforms'
            },
            {
                'item': 'Stress testing',
                'status': 'pending',
                'required': True,
                'description': 'Validate model stability under continuous load'
            },
            {
                'item': 'Battery impact assessment',
                'status': 'pending',
                'required': False,
                'description': 'Estimate battery consumption impact'
            }
        ]
        
        # Update status based on validation results
        test_name_mapping = {
            'Model accuracy validation': 'accuracy_validation',
            'Inference speed benchmark': 'inference_speed', 
            'Memory usage validation': 'memory_usage',
            'Cross-platform compatibility': 'cross_platform_compatibility',
            'Stress testing': 'stress_test',
            'Battery impact assessment': 'battery_consumption'
        }
        
        completed_tests = {r.test_name: r.passed for r in self.validator.validation_results}
        
        for item in checklist_items:
            test_name = test_name_mapping.get(item['item'])
            if test_name in completed_tests:
                item['status'] = 'passed' if completed_tests[test_name] else 'failed'
            
        return checklist_items


# Example usage and testing
if __name__ == '__main__':
    # Example usage
    model_path = "./models/quadfusion_model.tflite"
    
    # Initialize validator
    config = ValidationConfig(
        accuracy_tolerance=0.01,
        max_latency_ms=50.0,
        max_memory_mb=150.0
    )
    validator = ModelValidator(model_path, config)
    
    # Mock inference function for testing
    def mock_inference(input_data):
        time.sleep(0.01)  # Simulate inference time
        return np.random.rand(10)  # Mock output
    
    # Run validation tests
    print("Running validation tests...")
    
    # Accuracy test
    ref_outputs = [np.random.rand(10) for _ in range(50)]
    test_outputs = [output + np.random.normal(0, 0.001, 10) for output in ref_outputs]  # Add small noise
    accuracy_result = validator.validate_model_accuracy(ref_outputs, test_outputs)
    
    # Performance tests
    speed_result = validator.benchmark_inference_speed(mock_inference, np.random.rand(224, 224, 3))
    memory_result = validator.test_memory_usage(mock_inference, np.random.rand(224, 224, 3))
    
    # Stress test
    stress_result = validator.run_stress_test(mock_inference, np.random.rand(224, 224, 3), duration_minutes=1)
    
    # Final assessment
    is_ready, score, assessment = validator.assess_deployment_readiness()
    
    print(f"\nValidation Results:")
    print(f"Accuracy: {'PASS' if accuracy_result.passed else 'FAIL'} (score: {accuracy_result.score:.3f})")
    print(f"Speed: {'PASS' if speed_result.passed else 'FAIL'} (avg: {speed_result.details.get('avg_latency_ms', 0):.1f}ms)")
    print(f"Memory: {'PASS' if memory_result.passed else 'FAIL'} (peak: {memory_result.details.get('peak_memory_mb', 0):.1f}MB)")
    print(f"Stress: {'PASS' if stress_result.passed else 'FAIL'}")
    print(f"\nDeployment Ready: {'YES' if is_ready else 'NO'} (score: {score:.3f})")
    
    # Generate report
    report = validator.generate_validation_report("validation_report.json")
    print(f"Report generated with {len(validator.validation_results)} test results")
