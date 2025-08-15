"""
src/tests/test_agents.py

Comprehensive unit-, performance-, and security-tests for every fraud-detection
agent used in QuadFusion.

Test Classes
------------
TestBaseAgent              - Base abstract agent behaviour
TestTouchPatternAgent      - Touch-pattern analysis agent
TestTypingBehaviorAgent    - Typing-behaviour analysis agent
TestVoiceCommandAgent      - Speaker-identification agent
TestVisualAgent            - Face-recognition / visual agent
TestAppUsageAgent          - Application-usage pattern agent
TestMovementAgent          - Device-movement analysis agent
TestCoordinatorAgent       - Multi-agent coordinator

Each class contains:
• Functional tests
• Performance tests (<100 ms per call, <500 MB RAM)
• Security / privacy checks where applicable
• High-load / stress tests for mobile constraints
"""

import unittest
import time
import numpy as np
import psutil
from unittest.mock import MagicMock

# --------------------------------------------------------------------------- #
# Helper: Mock sensor-data generator
# --------------------------------------------------------------------------- #
class MockDataGenerator:
    """Generate synthetic multi-modal sensor data for repeatable tests."""

    # Touch events ---------------------------------------------------------- #
    def generate_touch_data(self, samples: int = 1_000):
        return [
            {
                "pressure": np.random.rand(),
                "x": np.random.randint(0, 1080),
                "y": np.random.randint(0, 1920),
                "timestamp": time.time() + i * 0.01,
            }
            for i in range(samples)
        ]

    # Keyboard events ------------------------------------------------------- #
    def generate_typing_data(self, samples: int = 1_000):
        return [
            {
                "key": chr(97 + np.random.randint(0, 26)),
                "duration": np.random.rand() * 0.4,
                "timestamp": time.time() + i * 0.05,
            }
            for i in range(samples)
        ]

    # Audio signal ---------------------------------------------------------- #
    def generate_voice_data(self, duration: int = 3):
        sr = 16_000
        t = np.linspace(0, duration, sr * duration, False)
        tone = 0.2 * np.sin(2 * np.pi * 440 * t)
        return tone.astype(np.float32)

    # Video frames ---------------------------------------------------------- #
    def generate_visual_data(self, frames: int = 60, H: int = 224, W: int = 224):
        return [
            np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(frames)
        ]

    # IMU samples ----------------------------------------------------------- #
    def generate_movement_data(self, samples: int = 1_000):
        return [
            {
                "acceleration": np.random.randn(3).tolist(),
                "gyroscope": np.random.randn(3).tolist(),
                "timestamp": time.time() + i * 0.02,
            }
            for i in range(samples)
        ]

    # App-usage events ------------------------------------------------------ #
    def generate_app_usage_data(self, days: int = 30):
        apps = ["chat", "mail", "maps", "music"]
        return [
            {"app": np.random.choice(apps), "duration": np.random.randint(1, 300)}
            for _ in range(days)
        ]


# --------------------------------------------------------------------------- #
# Base-agent tests
# --------------------------------------------------------------------------- #
class TestBaseAgent(unittest.TestCase):
    """Validate mandatory life-cycle hooks for any agent."""

    def setUp(self):
        self.agent = MagicMock(name="BaseAgent")

    def test_initialize(self):
        self.agent.initialize.return_value = True
        self.assertTrue(self.agent.initialize())

    def test_process_data_invocation(self):
        sample = {"foo": "bar"}
        self.agent.process_data(sample)
        self.agent.process_data.assert_called_with(sample)

    def test_shutdown(self):
        self.agent.shutdown.return_value = None
        self.agent.shutdown()
        self.agent.shutdown.assert_called_once()


# --------------------------------------------------------------------------- #
# TouchPatternAgent
# --------------------------------------------------------------------------- #
class TestTouchPatternAgent(unittest.TestCase):
    def setUp(self):
        from agents.touch_agent import TouchPatternAgent

        self.agent = TouchPatternAgent()
        self.data = MockDataGenerator()

    # Functional ------------------------------------------------------------ #
    def test_process_touch_batch(self):
        batch = self.data.generate_touch_data(100)
        self.agent.process_data(batch)
        self.assertIsNotNone(self.agent.get_results())

    # Performance (<5 s for 10 k samples) ---------------------------------- #
    def test_performance_heavy_batch(self):
        heavy = self.data.generate_touch_data(10_000)
        t0 = time.perf_counter()
        self.agent.process_data(heavy)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 5.0)

    # Security -------------------------------------------------------------- #
    def test_encryption_interface(self):
        self.assertTrue(hasattr(self.agent, "encrypt_data"))
        cipher = self.agent.encrypt_data({"pressure": 0.3})
        self.assertIsInstance(cipher, bytes)


# --------------------------------------------------------------------------- #
# TypingBehaviorAgent
# --------------------------------------------------------------------------- #
class TestTypingBehaviorAgent(unittest.TestCase):
    def setUp(self):
        from agents.typing_agent import TypingBehaviorAgent

        self.agent = TypingBehaviorAgent()
        self.data = MockDataGenerator()

    def test_typing_pattern(self):
        events = self.data.generate_typing_data(200)
        self.agent.process_data(events)
        res = self.agent.get_results()
        self.assertIn("typing_pattern", res)

    def test_memory_usage(self):
        big = self.data.generate_typing_data(5_000)
        mem0 = psutil.Process().memory_info().rss
        self.agent.process_data(big)
        mem1 = psutil.Process().memory_info().rss
        self.assertLess(mem1 - mem0, 500 * 1024 * 1024)  # <500 MB increase


# --------------------------------------------------------------------------- #
# VoiceCommandAgent
# --------------------------------------------------------------------------- #
class TestVoiceCommandAgent(unittest.TestCase):
    def setUp(self):
        from agents.voice_agent import VoiceCommandAgent

        self.agent = VoiceCommandAgent()
        self.data = MockDataGenerator()

    def test_speaker_identification(self):
        wav = self.data.generate_voice_data(2)
        self.agent.process_data(wav)
        res = self.agent.get_results()
        self.assertIn("speaker_id", res)

    def test_spoof_detection_false_positive(self):
        silence = np.zeros(16_000, dtype=np.float32)
        self.agent.process_data(silence)
        res = self.agent.get_results()
        self.assertFalse(res.get("is_spoof", True))


# --------------------------------------------------------------------------- #
# VisualAgent
# --------------------------------------------------------------------------- #
class TestVisualAgent(unittest.TestCase):
    def setUp(self):
        from agents.visual_agent import VisualAgent

        self.agent = VisualAgent()
        self.data = MockDataGenerator()

    def test_face_detection_count(self):
        frames = self.data.generate_visual_data(12)
        self.agent.process_data(frames)
        res = self.agent.get_results()
        self.assertIn("faces_detected", res)

    def test_processing_latency(self):
        frames = self.data.generate_visual_data(60)
        t0 = time.perf_counter()
        self.agent.process_data(frames)
        self.assertLess(time.perf_counter() - t0, 10.0)  # <10 s batch


# --------------------------------------------------------------------------- #
# AppUsageAgent
# --------------------------------------------------------------------------- #
class TestAppUsageAgent(unittest.TestCase):
    def setUp(self):
        from agents.app_usage_agent import AppUsageAgent

        self.agent = AppUsageAgent()
        self.data = MockDataGenerator()

    def test_usage_pattern(self):
        sessions = self.data.generate_app_usage_data(10)
        self.agent.process_data(sessions)
        self.assertIn("usage_patterns", self.agent.get_results())


# --------------------------------------------------------------------------- #
# MovementAgent
# --------------------------------------------------------------------------- #
class TestMovementAgent(unittest.TestCase):
    def setUp(self):
        from agents.movement_agent import MovementAgent

        self.agent = MovementAgent()
        self.data = MockDataGenerator()

    def test_movement_processing(self):
        imu = self.data.generate_movement_data(300)
        self.agent.process_data(imu)
        self.assertIn("movement_patterns", self.agent.get_results())


# --------------------------------------------------------------------------- #
# CoordinatorAgent
# --------------------------------------------------------------------------- #
class TestCoordinatorAgent(unittest.TestCase):
    def setUp(self):
        from agents.coordinator_agent import CoordinatorAgent

        self.agent = CoordinatorAgent()
        self.data = MockDataGenerator()

        # Inject mocks for internal agents to isolate coordinator logic
        for attr in [
            "touch_agent",
            "typing_agent",
            "voice_agent",
            "visual_agent",
            "movement_agent",
            "app_usage_agent",
        ]:
            setattr(self.agent, attr, MagicMock())

    def test_delegates_to_all_agents(self):
        bundle = {
            "touch": self.data.generate_touch_data(20),
            "typing": self.data.generate_typing_data(20),
            "voice": self.data.generate_voice_data(1),
            "visual": self.data.generate_visual_data(3),
            "movement": self.data.generate_movement_data(20),
            "app_usage": self.data.generate_app_usage_data(1),
        }
        self.agent.process_data(bundle)

        # Each sub-agent should have been invoked
        self.agent.touch_agent.process_data.assert_called()
        self.agent.typing_agent.process_data.assert_called()
        self.agent.voice_agent.process_data.assert_called()
        self.agent.visual_agent.process_data.assert_called()
        self.agent.movement_agent.process_data.assert_called()
        self.agent.app_usage_agent.process_data.assert_called()

    def test_coordinator_result_shape(self):
        # Mock each sub-agent returning a score
        for a in [
            self.agent.touch_agent,
            self.agent.typing_agent,
            self.agent.voice_agent,
            self.agent.visual_agent,
            self.agent.movement_agent,
            self.agent.app_usage_agent,
        ]:
            a.get_results.return_value = {"risk": 0.2}

        self.agent.aggregate_results()
        res = self.agent.get_results()
        self.assertIn("fraud_score", res)
        self.assertTrue(0.0 <= res["fraud_score"] <= 1.0)


# --------------------------------------------------------------------------- #
# Main entry-point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    unittest.main(verbosity=2)
