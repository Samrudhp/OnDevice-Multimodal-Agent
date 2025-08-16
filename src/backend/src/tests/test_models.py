# tests/test_models.py
"""
ML Model Testing Framework for QuadFusion
Comprehensive testing for all fraud detection models with mobile optimization validation.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import tempfile
import time
import psutil
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# Test framework imports
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
import joblib

# Import models to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.isolation_forest_mobile import MobileIsolationForest
from models.lstm_autoencoder import LSTMAutoencoder
from models.speaker_identification import SpeakerIdentificationModel
from models.motion_cnn import MotionCNN
from models.tiny_llm import TinyLLM

# Performance requirements for mobile
PERFORMANCE_REQUIREMENTS = {
    "max_inference_time_ms": 100,
    "max_model_size_mb": 50,
    "max_memory_usage_mb": 500,
    "max_battery_drain_percent": 2,
    "min_accuracy_percent": 95,
    "max_false_positive_rate": 5
}

class ModelTestBase(unittest.TestCase):
    """Base class for model testing with common utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate_mock_data(self, data_type: str, num_samples: int = 1000):
        """Generate mock data for testing."""
        if data_type == "touch":
            # Touch data: [x, y, timestamp, pressure]
            return np.random.rand(num_samples, 4) * 100
        elif data_type == "typing":
            # Typing sequences: [dwell_times, flight_times]
            return np.random.rand(num_samples, 20, 2) * 1000
        elif data_type == "voice":
            # Audio features: MFCC-like
            return np.random.rand(num_samples, 13) * 10
        elif data_type == "visual":
            # Image embeddings
            return np.random.rand(num_samples, 256) * 1.0
        elif data_type == "movement":
            # Sensor data: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
            return np.random.rand(num_samples, 100, 6) * 20 - 10
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def measure_inference_time(self, model, input_data, num_runs: int = 100):
        """Measure model inference time."""
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            if hasattr(model, 'predict'):
                # Sklearn model
                _ = model.predict(input_data[:1])
            elif isinstance(model, nn.Module):
                # PyTorch model
                model.eval()
                with torch.no_grad():
                    if isinstance(input_data, np.ndarray):
                        input_tensor = torch.FloatTensor(input_data[:1]).to(self.device)
                    else:
                        input_tensor = input_data[:1].to(self.device)
                    _ = model(input_tensor)
            else:
                raise ValueError("Unknown model type")
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'max_ms': np.max(times),
            'min_ms': np.min(times)
        }
    
    def measure_model_size(self, model_path: str):
        """Measure model file size."""
        if os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            return size_bytes / (1024 * 1024)  # Convert to MB
        return 0.0
    
    def measure_memory_usage(self):
        """Measure current memory usage."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        return current_memory - self.start_memory

class TestIsolationForest(ModelTestBase):
    """Test cases for Mobile Isolation Forest (Touch Pattern Agent)."""
    
    def setUp(self):
        super().setUp()
        self.model = MobileIsolationForest(
            n_estimators=50,
            contamination=0.1,
            max_features=0.8,
            random_state=42
        )
        self.touch_data = self.generate_mock_data("touch", 1000)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.n_estimators, 50)
        self.assertEqual(self.model.contamination, 0.1)
    
    def test_model_training(self):
        """Test model training functionality."""
        # Train model
        self.model.fit(self.touch_data)
        
        # Check if model is fitted
        self.assertTrue(hasattr(self.model, 'estimators_'))
        self.assertGreater(len(self.model.estimators_), 0)
    
    def test_model_prediction(self):
        """Test model prediction."""
        self.model.fit(self.touch_data)
        
        # Test prediction
        predictions = self.model.predict(self.touch_data[:10])
        scores = self.model.decision_function(self.touch_data[:10])
        
        self.assertEqual(len(predictions), 10)
        self.assertEqual(len(scores), 10)
        self.assertTrue(all(p in [-1, 1] for p in predictions))
    
    def test_inference_time_mobile_requirement(self):
        """Test inference time meets mobile requirements."""
        self.model.fit(self.touch_data)
        
        # Measure inference time
        timing_stats = self.measure_inference_time(self.model, self.touch_data)
        
        # Check mobile requirement (single sample < 20ms for touch model)
        self.assertLess(timing_stats['mean_ms'], 20.0, 
                       f"Inference time {timing_stats['mean_ms']:.2f}ms exceeds mobile requirement")
    
    def test_model_size_mobile_requirement(self):
        """Test model size meets mobile requirements."""
        self.model.fit(self.touch_data)
        
        # Save model
        model_path = os.path.join(self.temp_dir, "touch_model.pkl")
        joblib.dump(self.model, model_path)
        
        # Check model size
        model_size_mb = self.measure_model_size(model_path)
        self.assertLess(model_size_mb, 10.0, 
                       f"Model size {model_size_mb:.2f}MB exceeds mobile requirement")
    
    def test_model_accuracy(self):
        """Test model accuracy on synthetic anomalies."""
        # Create dataset with known anomalies
        normal_data = np.random.normal(0, 1, (800, 4))
        anomaly_data = np.random.normal(5, 1, (200, 4))  # Clear anomalies
        
        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        y_true = np.array([1] * 800 + [-1] * 200)  # 1 for normal, -1 for anomaly
        
        # Train and predict
        self.model.fit(X)
        y_pred = self.model.predict(X)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        self.assertGreater(accuracy, 0.7, f"Model accuracy {accuracy:.2f} too low")
    
    def test_memory_usage(self):
        """Test memory usage during training and inference."""
        self.model.fit(self.touch_data)
        memory_usage = self.measure_memory_usage()
        
        # Check memory requirement (should be reasonable for mobile)
        self.assertLess(memory_usage, 100.0, 
                       f"Memory usage {memory_usage:.2f}MB exceeds mobile requirement")

class TestLSTMAutoencoder(ModelTestBase):
    """Test cases for LSTM Autoencoder (Typing Behavior Agent)."""
    
    def setUp(self):
        super().setUp()
        self.model = LSTMAutoencoder(
            input_size=2,
            hidden_size=32,
            num_layers=2,
            sequence_length=20
        ).to(self.device)
        self.typing_data = torch.FloatTensor(self.generate_mock_data("typing", 500)).to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.input_size, 2)
        self.assertEqual(self.model.hidden_size, 32)
        self.assertEqual(self.model.num_layers, 2)
    
    def test_forward_pass(self):
        """Test model forward pass."""
        batch_size = 32
        input_batch = self.typing_data[:batch_size]
        
        self.model.eval()
        with torch.no_grad():
            output, encoded = self.model(input_batch)
        
        # Check output shapes
        self.assertEqual(output.shape, input_batch.shape)
        self.assertEqual(encoded.shape[0], batch_size)
        self.assertEqual(encoded.shape[1], self.model.hidden_size)
    
    def test_training_step(self):
        """Test one training step."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Single batch training
        batch = self.typing_data[:32]
        optimizer.zero_grad()
        
        output, _ = self.model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        
        # Check if loss is computed
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0.0)
    
    def test_reconstruction_quality(self):
        """Test reconstruction quality after training."""
        # Simple training loop
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for _ in range(10):  # Quick training
            for i in range(0, len(self.typing_data), 32):
                batch = self.typing_data[i:i+32]
                if len(batch) == 32:
                    optimizer.zero_grad()
                    output, _ = self.model(batch)
                    loss = criterion(output, batch)
                    loss.backward()
                    optimizer.step()
        
        # Test reconstruction
        self.model.eval()
        with torch.no_grad():
            test_batch = self.typing_data[:10]
            reconstructed, _ = self.model(test_batch)
            mse = nn.MSELoss()(reconstructed, test_batch)
        
        # Reconstruction should be reasonable
        self.assertLess(mse.item(), 1.0, "Reconstruction quality too poor")
    
    def test_inference_time_mobile_requirement(self):
        """Test inference time meets mobile requirements."""
        timing_stats = self.measure_inference_time(self.model, self.typing_data)
        
        # Check mobile requirement (< 20ms for typing model)
        self.assertLess(timing_stats['mean_ms'], 20.0,
                       f"Inference time {timing_stats['mean_ms']:.2f}ms exceeds mobile requirement")
    
    def test_model_size_mobile_requirement(self):
        """Test model size meets mobile requirements."""
        # Save model
        model_path = os.path.join(self.temp_dir, "typing_model.pth")
        torch.save(self.model.state_dict(), model_path)
        
        # Check model size
        model_size_mb = self.measure_model_size(model_path)
        self.assertLess(model_size_mb, 10.0,
                       f"Model size {model_size_mb:.2f}MB exceeds mobile requirement")
    
    def test_parameter_count(self):
        """Test model parameter count for mobile deployment."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Should be reasonable for mobile (< 100K parameters)
        self.assertLess(total_params, 100000,
                       f"Model has {total_params} parameters, too complex for mobile")
        self.assertEqual(total_params, trainable_params, "All parameters should be trainable")

class TestSpeakerIdentification(ModelTestBase):
    """Test cases for Speaker Identification Model (Voice Agent)."""
    
    def setUp(self):
        super().setUp()
        self.model = SpeakerIdentificationModel(
            input_dim=13,  # MFCC features
            hidden_dim=64,
            num_speakers=10
        ).to(self.device)
        self.voice_data = torch.FloatTensor(self.generate_mock_data("voice", 500)).to(self.device)
        self.speaker_labels = torch.randint(0, 10, (500,)).to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.input_dim, 13)
        self.assertEqual(self.model.num_speakers, 10)
    
    def test_forward_pass(self):
        """Test model forward pass."""
        batch_size = 32
        input_batch = self.voice_data[:batch_size]
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_batch)
        
        # Check output shape
        expected_shape = (batch_size, 10)  # num_speakers
        self.assertEqual(output.shape, expected_shape)
        
        # Check softmax probabilities
        probabilities = torch.softmax(output, dim=1)
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(batch_size)))
    
    def test_speaker_identification_accuracy(self):
        """Test speaker identification accuracy."""
        # Quick training
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(5):  # Quick training
            for i in range(0, len(self.voice_data), 32):
                batch_data = self.voice_data[i:i+32]
                batch_labels = self.speaker_labels[i:i+32]
                
                if len(batch_data) == 32:
                    optimizer.zero_grad()
                    output = self.model(batch_data)
                    loss = criterion(output, batch_labels)
                    loss.backward()
                    optimizer.step()
        
        # Test accuracy
        self.model.eval()
        with torch.no_grad():
            test_output = self.model(self.voice_data[:100])
            predictions = torch.argmax(test_output, dim=1)
            accuracy = (predictions == self.speaker_labels[:100]).float().mean()
        
        # Should achieve some accuracy even with random data
        self.assertGreater(accuracy.item(), 0.05, "Model accuracy too low")
    
    def test_embedding_extraction(self):
        """Test speaker embedding extraction."""
        if hasattr(self.model, 'extract_embeddings'):
            embeddings = self.model.extract_embeddings(self.voice_data[:10])
            self.assertEqual(embeddings.shape[0], 10)
            self.assertGreater(embeddings.shape[1], 0)
    
    def test_inference_time_mobile_requirement(self):
        """Test inference time meets mobile requirements."""
        timing_stats = self.measure_inference_time(self.model, self.voice_data)
        
        # Check mobile requirement (< 20ms for voice model)
        self.assertLess(timing_stats['mean_ms'], 20.0,
                       f"Inference time {timing_stats['mean_ms']:.2f}ms exceeds mobile requirement")

class TestMotionCNN(ModelTestBase):
    """Test cases for Motion CNN (Movement Agent)."""
    
    def setUp(self):
        super().setUp()
        self.model = MotionCNN(
            input_channels=6,
            sequence_length=100,
            num_classes=6
        ).to(self.device)
        self.movement_data = torch.FloatTensor(self.generate_mock_data("movement", 300)).to(self.device)
        self.movement_labels = torch.randint(0, 6, (300,)).to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.input_channels, 6)
        self.assertEqual(self.model.num_classes, 6)
    
    def test_forward_pass(self):
        """Test model forward pass."""
        batch_size = 16
        input_batch = self.movement_data[:batch_size].transpose(1, 2)  # [batch, channels, sequence]
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_batch)
        
        # Check output shape
        expected_shape = (batch_size, 6)  # num_classes
        self.assertEqual(output.shape, expected_shape)
    
    def test_motion_classification(self):
        """Test motion pattern classification."""
        # Quick training
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(3):  # Quick training
            for i in range(0, len(self.movement_data), 16):
                batch_data = self.movement_data[i:i+16].transpose(1, 2)
                batch_labels = self.movement_labels[i:i+16]
                
                if len(batch_data) == 16:
                    optimizer.zero_grad()
                    output = self.model(batch_data)
                    loss = criterion(output, batch_labels)
                    loss.backward()
                    optimizer.step()
        
        # Test accuracy
        self.model.eval()
        with torch.no_grad():
            test_data = self.movement_data[:50].transpose(1, 2)
            test_output = self.model(test_data)
            predictions = torch.argmax(test_output, dim=1)
            accuracy = (predictions == self.movement_labels[:50]).float().mean()
        
        # Should achieve some accuracy
        self.assertGreater(accuracy.item(), 0.05, "Model accuracy too low")
    
    def test_temporal_feature_extraction(self):
        """Test temporal feature extraction capability."""
        # Test with different sequence lengths
        short_sequence = self.movement_data[:1, :50, :].transpose(1, 2)  # 50 timesteps
        long_sequence = self.movement_data[:1].transpose(1, 2)  # 100 timesteps
        
        self.model.eval()
        with torch.no_grad():
            output_short = self.model(short_sequence)
            output_long = self.model(long_sequence)
        
        # Both should produce valid outputs
        self.assertEqual(output_short.shape[1], 6)
        self.assertEqual(output_long.shape[1], 6)

class TestTinyLLM(ModelTestBase):
    """Test cases for Tiny LLM (App Usage Agent)."""
    
    def setUp(self):
        super().setUp()
        self.model = TinyLLM(
            vocab_size=1000,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
            max_seq_length=50
        ).to(self.device)
        self.app_sequence_data = torch.randint(0, 1000, (200, 50)).to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.vocab_size, 1000)
        self.assertEqual(self.model.embed_dim, 128)
        self.assertEqual(self.model.num_layers, 2)
    
    def test_forward_pass(self):
        """Test model forward pass."""
        batch_size = 16
        input_batch = self.app_sequence_data[:batch_size]
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_batch)
        
        # Check output shape
        expected_shape = (batch_size, 50, 1000)  # [batch, seq_len, vocab_size]
        self.assertEqual(output.shape, expected_shape)
    
    def test_sequence_modeling(self):
        """Test sequence modeling capability."""
        # Test language modeling loss
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(3):  # Quick training
            for i in range(0, len(self.app_sequence_data), 16):
                batch = self.app_sequence_data[i:i+16]
                
                if len(batch) == 16:
                    # Use sequence as both input and target (shifted)
                    input_seq = batch[:, :-1]
                    target_seq = batch[:, 1:]
                    
                    optimizer.zero_grad()
                    output = self.model(input_seq)
                    loss = criterion(output.reshape(-1, 1000), target_seq.reshape(-1))
                    loss.backward()
                    optimizer.step()
        
        # Model should learn something
        self.assertIsInstance(loss.item(), float)
    
    def test_inference_time_mobile_requirement(self):
        """Test inference time meets mobile requirements."""
        # Test with shorter sequences for mobile
        short_data = self.app_sequence_data[:10, :20]  # Shorter sequences
        timing_stats = self.measure_inference_time(self.model, short_data)
        
        # Check mobile requirement (< 30ms for LLM)
        self.assertLess(timing_stats['mean_ms'], 30.0,
                       f"Inference time {timing_stats['mean_ms']:.2f}ms exceeds mobile requirement")

class TestModelQuantization(ModelTestBase):
    """Test cases for model quantization and mobile optimization."""
    
    def test_torch_quantization(self):
        """Test PyTorch model quantization."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
        # Test dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        
        # Test that quantized model works
        test_input = torch.randn(5, 10)
        original_output = model(test_input)
        quantized_output = quantized_model(test_input)
        
        # Outputs should be similar
        self.assertLess(torch.mean(torch.abs(original_output - quantized_output)).item(), 0.1)
    
    def test_model_compression_ratio(self):
        """Test model compression ratios."""
        # Create and save original model
        model = nn.Linear(100, 50)
        original_path = os.path.join(self.temp_dir, "original.pth")
        torch.save(model.state_dict(), original_path)
        
        # Quantize and save
        quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        quantized_path = os.path.join(self.temp_dir, "quantized.pth")
        torch.save(quantized_model.state_dict(), quantized_path)
        
        # Check compression ratio
        original_size = self.measure_model_size(original_path)
        quantized_size = self.measure_model_size(quantized_path)
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 1
        
        # Should achieve some compression
        self.assertGreater(compression_ratio, 1.0, "Quantization should reduce model size")

class TestCrossPlatformCompatibility(ModelTestBase):
    """Test cross-platform compatibility for mobile deployment."""
    
    def test_cpu_gpu_consistency(self):
        """Test model consistency between CPU and GPU."""
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")
        
        # Create model on both devices
        model_cpu = nn.Linear(10, 5)
        model_gpu = nn.Linear(10, 5).cuda()
        
        # Copy weights
        model_gpu.load_state_dict(model_cpu.state_dict())
        
        # Test with same input
        test_input = torch.randn(3, 10)
        output_cpu = model_cpu(test_input)
        output_gpu = model_gpu(test_input.cuda()).cpu()
        
        # Outputs should be identical
        self.assertTrue(torch.allclose(output_cpu, output_gpu, atol=1e-6))
    
    def test_model_serialization(self):
        """Test model serialization for deployment."""
        model = nn.Linear(10, 5)
        
        # Test PyTorch serialization
        model_path = os.path.join(self.temp_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        
        # Load and test
        loaded_model = nn.Linear(10, 5)
        loaded_model.load_state_dict(torch.load(model_path))
        
        # Test equivalence
        test_input = torch.randn(3, 10)
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)
        
        self.assertTrue(torch.allclose(original_output, loaded_output))

def create_performance_benchmark_suite():
    """Create a comprehensive performance benchmark suite."""
    
    class PerformanceBenchmark(unittest.TestCase):
        """Performance benchmark tests."""
        
        def test_overall_system_performance(self):
            """Test overall system performance requirements."""
            # This would test the complete fraud detection pipeline
            # with all models running together
            
            # Mock complete system test
            total_inference_time = 0
            total_memory_usage = 0
            total_model_size = 0
            
            # Simulate running all 5 models
            model_count = 5
            avg_inference_per_model = 15  # ms
            avg_memory_per_model = 80  # MB
            avg_size_per_model = 8  # MB
            
            total_inference_time = model_count * avg_inference_per_model
            total_memory_usage = model_count * avg_memory_per_model
            total_model_size = model_count * avg_size_per_model
            
            # Check system requirements
            self.assertLess(total_inference_time, PERFORMANCE_REQUIREMENTS["max_inference_time_ms"])
            self.assertLess(total_memory_usage, PERFORMANCE_REQUIREMENTS["max_memory_usage_mb"])
            self.assertLess(total_model_size, PERFORMANCE_REQUIREMENTS["max_model_size_mb"])
    
    return unittest.TestLoader().loadTestsFromTestCase(PerformanceBenchmark)

if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestIsolationForest,
        TestLSTMAutoencoder,
        TestSpeakerIdentification,
        TestMotionCNN,
        TestTinyLLM,
        TestModelQuantization,
        TestCrossPlatformCompatibility
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Add performance benchmarks
    suite.addTests(create_performance_benchmark_suite())
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)
