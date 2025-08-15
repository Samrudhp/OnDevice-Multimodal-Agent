# demo/simulate_sensor_data.py
"""
QuadFusion Sensor Data Simulation Module
=======================================

Provides realistic, multi-modal sensor data generation for
development, testing, benchmarking, and live demonstrations.

• Touch pattern simulation (swipes, taps, gestures)
• Typing behavior simulation (keystroke dynamics)
• Voice data simulation (speech & spoofing)
• Visual data simulation (face / scene)
• Motion data simulation (accelerometer, gyroscope, magnetometer)
• App-usage pattern simulation
• Fraudulent-behavior injection
• Real-time streaming & batch generation
• Export to CSV / JSON / Parquet
• Configurable user profiles & device characteristics
• Statistical validation utilities
"""

from __future__ import annotations

import os
import io
import cv2
import yaml
import json
import time
import math
import uuid
import wave
import queue
import enum
import copy
import random
import base64
import struct
import string
import shutil
import psutil
import librosa
import soundfile as sf
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Deque,
    Optional,
    Iterator,
    Generator,
)

from collections import deque
from threading import Thread, Event, Lock

# --------------------------------------------------------------------------- #
#  Global constants & defaults                                                #
# --------------------------------------------------------------------------- #

_DEFAULT_SAMPLE_RATE = 16_000               # Audio Hz
_DEFAULT_FRAME_SIZE = 512                  # Audio samples / frame
_DEFAULT_IMAGE_SIZE = (224, 224, 3)        # Visual frames
_DEFAULT_SENSOR_HZ  = 50                   # Motion sensor rate
_DEFAULT_TOUCH_HZ   = 25                   # Touch event rate
_DEFAULT_TYPING_HZ  = 12                   # Keystroke / sec

RNG = np.random.default_rng()

# --------------------------------------------------------------------------- #
#  Utility helpers                                                            #
# --------------------------------------------------------------------------- #

def _gauss(mu: float, sigma: float) -> float:
    return float(RNG.normal(mu, sigma))

def _randf(a: float = 0.0, b: float = 1.0) -> float:
    return float(RNG.uniform(a, b))

def _randint(a: int, b: int) -> int:
    return int(RNG.integers(a, b + 1))

def _now() -> float:
    return time.time()

def _choose(seq):
    return seq[_randint(0, len(seq) - 1)]

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _unit_vector() -> np.ndarray:
    v = RNG.normal(size=3)
    v /= np.linalg.norm(v) + 1e-8
    return v

# --------------------------------------------------------------------------- #
#  Base simulator data classes                                                #
# --------------------------------------------------------------------------- #

@dataclass
class UserProfile:
    """
    Represents characteristics that drive synthetic behaviour generation.
    """
    user_id: str
    handedness: str = field(default_factory=lambda: _choose(["left", "right"]))
    typing_speed_wpm: float = field(default_factory=lambda: _gauss(42, 7))
    touch_pressure_mu: float = field(default_factory=lambda: _gauss(0.45, 0.05))
    voice_pitch_hz: float = field(default_factory=lambda: _gauss(180, 20))
    walking_speed_mps: float = field(default_factory=lambda: _gauss(1.34, 0.15))
    device_model: str = field(default_factory=lambda: _choose(
        ["Pixel-7", "iPhone-14", "Galaxy-S23", "OnePlus-11"]))

    def to_dict(self):
        return asdict(self)

@dataclass
class SimulatorConfig:
    """
    Master configuration controlling all modality simulators.
    """
    audio_sr: int = _DEFAULT_SAMPLE_RATE
    motion_hz: int = _DEFAULT_SENSOR_HZ
    touch_hz: int = _DEFAULT_TOUCH_HZ
    typing_hz: int = _DEFAULT_TYPING_HZ
    image_size: Tuple[int, int, int] = _DEFAULT_IMAGE_SIZE
    streaming_buffer_size: int = 1_000
    max_users: int = 5
    fraud_chance: float = 0.05          # baseline fraud probability
    seed: Optional[int] = None

# --------------------------------------------------------------------------- #
#  Fraud scenario definitions                                                 #
# --------------------------------------------------------------------------- #

class FraudType(enum.Enum):
    NONE               = "none"
    DEVICE_THEFT       = "device_theft"
    SHOULDER_SURFING   = "shoulder_surfing"
    VOICE_SPOOF        = "voice_spoof"
    FACE_SPOOF         = "face_spoof"
    SENSOR_REPLAY      = "sensor_replay"


@dataclass
class FraudEvent:
    """Describes a fraudulent action injected into the data."""
    fraud_type: FraudType
    start_ts: float
    duration_s: float
    meta: Dict[str, Any] = field(default_factory=dict)

# --------------------------------------------------------------------------- #
#  Touch pattern simulator                                                    #
# --------------------------------------------------------------------------- #

class TouchPatternSimulator:
    """
    Generates swipe / tap / pinch gestures with realistic parameters.
    """

    def __init__(self, cfg: SimulatorConfig):
        self.cfg = cfg

    def _simulate_tap(self, profile: UserProfile) -> Dict[str, Any]:
        pressure = _gauss(profile.touch_pressure_mu, 0.05)
        area     = _gauss(12, 3)
        return dict(
            type="tap",
            pressure=_clamp(pressure, 0.1, 1.0),
            area=_clamp(area, 5, 30),
            duration=_gauss(0.08, 0.02),
            x=_randint(0, 1080),
            y=_randint(0, 2400),
            timestamp=_now()
        )

    def _simulate_swipe(self, profile: UserProfile) -> List[Dict[str, Any]]:
        start_x, start_y = _randint(0, 1080), _randint(200, 2200)
        length_px = _randint(200, 800)
        direction = _choose(["up", "down", "left", "right"])
        steps     = _randint(5, 15)
        events    = []
        dx = dy = 0
        if direction == "up":    dy = -length_px
        if direction == "down":  dy =  length_px
        if direction == "left":  dx = -length_px
        if direction == "right": dx =  length_px
        for i in range(steps):
            frac = (i + 1) / steps
            events.append(dict(
                type="swipe",
                progress=frac,
                pressure=_clamp(_gauss(profile.touch_pressure_mu, 0.07), 0.05, 1),
                area=_clamp(_gauss(14, 4), 4, 40),
                x=int(start_x + dx * frac),
                y=int(start_y + dy * frac),
                timestamp=_now() + i / self.cfg.touch_hz
            ))
        return events

    def generate(self, profile: UserProfile) -> List[Dict[str, Any]]:
        """
        Yield a list of touch events (a gesture) for a given user profile.
        """
        if _randf() < 0.6:
            return [self._simulate_tap(profile)]
        else:
            return self._simulate_swipe(profile)

# --------------------------------------------------------------------------- #
#  Typing behaviour simulator                                                 #
# --------------------------------------------------------------------------- #

class TypingBehaviorSimulator:
    """
    Simulates keystroke-level timing (dwell / flight) based on
    empirical keystroke dynamics studies.
    """

    _KEYS = list(string.ascii_lowercase) + list(" .,!?")

    def __init__(self, cfg: SimulatorConfig):
        self.cfg = cfg

    def _generate_sentence(self) -> str:
        words = [_choose(self._KEYS[:26]) for _ in range(_randint(3, 8))]
        return "".join(words)

    def generate_sequence(self, profile: UserProfile) -> List[Dict[str, Any]]:
        speed_cps = profile.typing_speed_wpm * 5 / 60  # char / sec
        dwell_mu  = 0.1
        flight_mu = 1 / speed_cps - dwell_mu
        if flight_mu < 0.03:
            flight_mu = 0.03
        text = self._generate_sentence()
        events = []
        ts = _now()
        for ch in text:
            dwell = _clamp(_gauss(dwell_mu, 0.015), 0.05, 0.25)
            flight = _clamp(_gauss(flight_mu, 0.02), 0.02, 0.6)
            events.append(dict(
                key=ch,
                dwell_time=dwell,
                flight_time=flight,
                timestamp=ts
            ))
            ts += dwell + flight
        return events

# --------------------------------------------------------------------------- #
#  Voice data simulator                                                       #
# --------------------------------------------------------------------------- #

class VoiceDataSimulator:
    """
    Generates synthetic speech snippets using basic DSP (sine-wave vowel
    approximations) plus background noise for demonstration.
    """

    def __init__(self, cfg: SimulatorConfig):
        self.cfg = cfg

    def _sine_wave(self, freq: float, dur: float) -> np.ndarray:
        sr = self.cfg.audio_sr
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        return 0.3 * np.sin(2 * np.pi * freq * t)

    def _add_noise(self, signal: np.ndarray, snr_db: float = 20) -> np.ndarray:
        sig_power = np.mean(signal ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = RNG.normal(scale=np.sqrt(noise_power), size=len(signal))
        return signal + noise

    def generate(self, profile: UserProfile, phrase: str = "hello") -> np.ndarray:
        dur = 0.3 + 0.05 * len(phrase)
        base = self._sine_wave(profile.voice_pitch_hz, dur)
        voiced = self._add_noise(base, snr_db=_randf(15, 25))
        return voiced.astype(np.float32)

    def spoof(self, profile: UserProfile) -> np.ndarray:
        """Generate voice spoof using pitch shifting / cloning cheat."""
        orig = self.generate(profile)
        # Simple spoof: invert waveform & add artifacts
        spoof = -orig * _randf(0.9, 1.1)
        spoof += self._sine_wave(profile.voice_pitch_hz * _randf(0.6, 1.4), len(orig)/self.cfg.audio_sr)
        return self._add_noise(spoof, snr_db=10)

# --------------------------------------------------------------------------- #
#  Visual data simulator                                                      #
# --------------------------------------------------------------------------- #

class VisualDataSimulator:
    """
    Produces synthetic face / scene images by procedural generation
    (solid color + noise placeholder) — suitable for pipeline testing
    without large media assets.
    """

    def __init__(self, cfg: SimulatorConfig):
        self.cfg = cfg

    def _solid_color(self, rgb: Tuple[int, int, int]) -> np.ndarray:
        h, w, c = self.cfg.image_size
        img = np.zeros((h, w, c), dtype=np.uint8)
        img[:] = rgb
        return img

    def _add_noise(self, img: np.ndarray, intensity: float = 0.05) -> np.ndarray:
        noise = RNG.normal(scale=255 * intensity, size=img.shape)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def generate_face(self, profile: UserProfile) -> np.ndarray:
        base_color = (_randint(60, 200), _randint(60, 200), _randint(60, 200))
        img = self._solid_color(base_color)
        return self._add_noise(img, intensity=0.08)

    def spoof_face(self, profile: UserProfile) -> np.ndarray:
        """Simulate face spoof (e.g., printed photo) by adding blur."""
        img = self.generate_face(profile)
        blurred = cv2.GaussianBlur(img, (15, 15), 0)
        return blurred

# --------------------------------------------------------------------------- #
#  Motion data simulator                                                      #
# --------------------------------------------------------------------------- #

class MotionDataSimulator:
    """
    Generates accelerometer, gyroscope, magnetometer data based on
    random walks approximating human movement.
    """

    def __init__(self, cfg: SimulatorConfig):
        self.cfg = cfg

    def _random_walk(self, n: int, step_sigma: float = 0.2) -> np.ndarray:
        steps = RNG.normal(scale=step_sigma, size=(n, 3))
        return np.cumsum(steps, axis=0)

    def generate_sequence(
        self,
        profile: UserProfile,
        seconds: float = 1.0,
        activity: str = "walking"
    ) -> Dict[str, np.ndarray]:
        n = int(self.cfg.motion_hz * seconds)
        # Base acceleration ~ gravity
        accel = np.tile(np.array([[0, 0, 9.81]]), (n, 1))
        gyro  = np.zeros((n, 3))
        mag   = np.tile(np.array([[0.3, 0, 0.1]]), (n, 1))
        # Add activity signature
        if activity == "walking":
            stride_hz = profile.walking_speed_mps / 0.75  # stride frequency
            t = np.arange(n) / self.cfg.motion_hz
            accel[:, 2] += 1.2 * np.sin(2 * np.pi * stride_hz * t)
            gyro += self._random_walk(n, 0.02)
        elif activity == "running":
            t = np.arange(n) / self.cfg.motion_hz
            accel[:, 2] += 3.0 * np.sin(2 * np.pi * 2.3 * t)
            gyro += self._random_walk(n, 0.04)
        elif activity == "sitting":
            accel += self._random_walk(n, 0.02)
        return dict(
            acceleration=accel.astype(np.float32),
            gyroscope=gyro.astype(np.float32),
            magnetometer=mag.astype(np.float32),
            timestamps=np.linspace(_now(), _now() + seconds, n)
        )

    def spoof_sequence(self, seq: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Replay attack: return previously recorded sequence with slight delay."""
        spoof = copy.deepcopy(seq)
        spoof["timestamps"] = spoof["timestamps"] + _randf(0.1, 0.5)
        return spoof

# --------------------------------------------------------------------------- #
#  App-usage pattern simulator                                                #
# --------------------------------------------------------------------------- #

class AppUsagePredefined(enum.Enum):
    SOCIAL = ["instagram", "twitter", "whatsapp"]
    FINANCE = ["banking", "stocks", "crypto"]
    WORK = ["slack", "gmail", "docs"]

class AppUsageSimulator:
    """
    Generates app-open / close events with durations & frequencies.
    """

    def __init__(self, cfg: SimulatorConfig):
        self.cfg = cfg

    def generate_session_log(
        self,
        profile: UserProfile,
        minutes: float = 5.0
    ) -> List[Dict[str, Any]]:
        total_seconds = int(minutes * 60)
        ts = _now()
        events = []
        while ts < _now() + total_seconds:
            app_group = _choose(list(AppUsagePredefined))
            app_name = _choose(app_group.value)
            duration = _randf(15, 300)
            events.append(dict(
                app_name=app_name,
                start_time=ts,
                duration=duration,
                frequency=_randint(1, 10)
            ))
            ts += duration + _randf(10, 120)
        return events

# --------------------------------------------------------------------------- #
#  Fraudulent behaviour injector                                              #
# --------------------------------------------------------------------------- #

class FraudulentBehaviorInjector:
    """
    Injects fraudulent patterns into otherwise benign data streams.
    """

    def __init__(self, cfg: SimulatorConfig):
        self.cfg = cfg

    def inject_touch_fraud(
        self,
        touch_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # Replace pressures with unusually high or low values
        for ev in touch_events:
            ev["pressure"] = _clamp(ev["pressure"] * _randf(2, 4), 0, 1)
        return touch_events

    def inject_typing_fraud(
        self,
        typing_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # Extremely fast typing -> potential bot
        for ev in typing_events:
            ev["flight_time"] *= _randf(0.05, 0.2)
            ev["dwell_time"] *= _randf(0.05, 0.2)
        return typing_events

    def inject_motion_fraud(
        self,
        motion_seq: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        return MotionDataSimulator(self.cfg).spoof_sequence(motion_seq)

    def inject_voice_fraud(
        self,
        voice_sig: np.ndarray
    ) -> np.ndarray:
        return voice_sig[::-1].copy() * _randf(0.9, 1.1)  # Simple reverse spoof

    def inject_visual_fraud(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        # Add obvious artifacts
        overlay = np.zeros_like(image)
        cv2.putText(
            overlay, "SPOOF", (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 8
        )
        return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# --------------------------------------------------------------------------- #
#  SensorDataSimulator orchestrating all modalities                           #
# --------------------------------------------------------------------------- #

class SensorDataSimulator:
    """
    Orchestrates modality-specific simulators to emit synchronized
    multi-modal samples either as a batch or streaming generator.
    """

    def __init__(self, cfg: Optional[SimulatorConfig] = None):
        self.cfg = cfg or SimulatorConfig()
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)
            random.seed(self.cfg.seed)

        # Create per-modality simulators
        self.touch_sim   = TouchPatternSimulator(self.cfg)
        self.type_sim    = TypingBehaviorSimulator(self.cfg)
        self.voice_sim   = VoiceDataSimulator(self.cfg)
        self.visual_sim  = VisualDataSimulator(self.cfg)
        self.motion_sim  = MotionDataSimulator(self.cfg)
        self.app_sim     = AppUsageSimulator(self.cfg)
        self.fraud_inj   = FraudulentBehaviorInjector(self.cfg)

        self.user_profiles: List[UserProfile] = [
            UserProfile(user_id=f"user_{i+1}") for i in range(self.cfg.max_users)
        ]

    # --------------------------------------------------------------------- #
    #  Single composite sample generation                                   #
    # --------------------------------------------------------------------- #

    def _maybe_inject_fraud(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Randomly inject fraud based on global probability."""
        if _randf() >= self.cfg.fraud_chance:
            return sample  # no fraud

        fraud_type = _choose(list(FraudType)[1:])  # exclude NONE
        sample["fraud_type"] = fraud_type.value

        if fraud_type == FraudType.DEVICE_THEFT:
            # Replace user profile with impostor
            impostor_profile = UserProfile(user_id="attacker")
            sample["motion"] = self.motion_sim.generate_sequence(impostor_profile, seconds=1.0, activity="running")
        elif fraud_type == FraudType.SHOULDER_SURFING:
            sample["typing"] = self.fraud_inj.inject_typing_fraud(sample["typing"])
        elif fraud_type == FraudType.VOICE_SPOOF:
            sample["voice"] = self.fraud_inj.inject_voice_fraud(sample["voice"])
        elif fraud_type == FraudType.FACE_SPOOF:
            sample["visual"] = self.fraud_inj.inject_visual_fraud(sample["visual"])
        elif fraud_type == FraudType.SENSOR_REPLAY:
            sample["motion"] = self.fraud_inj.inject_motion_fraud(sample["motion"])
        return sample

    def generate_sample(self, user: Optional[UserProfile] = None) -> Dict[str, Any]:
        """
        Produce one synchronized multi-modal sample.
        """
        profile = user or _choose(self.user_profiles)
        sample_id = str(uuid.uuid4())
        sample_ts = _now()

        sample = dict(
            sample_id=sample_id,
            user_id=profile.user_id,
            timestamp=sample_ts,
            touch=self.touch_sim.generate(profile),
            typing=self.type_sim.generate_sequence(profile),
            voice=self.voice_sim.generate(profile),
            visual=self.visual_sim.generate_face(profile),
            motion=self.motion_sim.generate_sequence(profile, seconds=1.0),
            app_usage=self.app_sim.generate_session_log(profile, minutes=0.2)
        )

        return self._maybe_inject_fraud(sample)

    # --------------------------------------------------------------------- #
    #  Batch / streaming interfaces                                         #
    # --------------------------------------------------------------------- #

    def generate_batch_data(
        self,
        num_samples: int = 100,
        users: Optional[List[UserProfile]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a list of synthetic samples (batch mode).
        """
        profiles = users or self.user_profiles
        return [self.generate_sample(_choose(profiles)) for _ in range(num_samples)]

    def generate_streaming_data(
        self,
        duration_secs: float = 30,
        fps: float = 20.0,
        users: Optional[List[UserProfile]] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield samples continuously to emulate live streaming.
        """
        profiles = users or self.user_profiles
        end_time = _now() + duration_secs
        interval = 1.0 / fps
        while _now() < end_time:
            yield self.generate_sample(_choose(profiles))
            time.sleep(interval)

    # --------------------------------------------------------------------- #
    #  Export helpers                                                       #
    # --------------------------------------------------------------------- #

    def _encode_image(self, img: np.ndarray) -> str:
        """Encode image array to base64 PNG string."""
        _, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf).decode()

    def _encode_audio(self, audio: np.ndarray) -> str:
        """Encode audio to base64 WAV string."""
        buf = io.BytesIO()
        sf.write(buf, audio, self.cfg.audio_sr, format="WAV")
        return base64.b64encode(buf.getvalue()).decode()

    def export_sample(
        self,
        sample: Dict[str, Any],
        fmt: str = "json",
        path: Optional[Path] = None
    ) -> str | bytes:
        """
        Export a single sample to JSON/CSV/Parquet (returns serialized data or
        writes to file if path given).
        """
        if fmt not in {"json", "csv", "parquet"}:
            raise ValueError("Unsupported format")

        # Flatten sample for tabular formats
        flat = dict(
            sample_id=sample["sample_id"],
            user_id=sample["user_id"],
            timestamp=sample["timestamp"],
            fraud_type=sample.get("fraud_type", "none"),
            touch_events=len(sample["touch"]),
            typing_events=len(sample["typing"]),
            motion_len=len(sample["motion"]["timestamps"]),
        )
        # Encode heavy modalities for JSON
        if fmt == "json":
            json_obj = copy.deepcopy(sample)
            json_obj["visual"] = self._encode_image(sample["visual"])
            json_obj["voice"]  = self._encode_audio(sample["voice"])
            serialized = json.dumps(json_obj, indent=2)
            if path:
                Path(path).write_text(serialized)
            return serialized

        # CSV / Parquet using pandas
        df = pd.DataFrame([flat])
        if fmt == "csv":
            if path:
                df.to_csv(path, index=False)
            return df.to_csv(index=False).encode()
        else:
            if path:
                df.to_parquet(path, index=False)
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            return buf.getvalue()

# --------------------------------------------------------------------------- #
#  Quick validation / test harness                                            #
# --------------------------------------------------------------------------- #

def _demo():
    cfg = SimulatorConfig(fraud_chance=0.15, max_users=3, seed=42)
    sim = SensorDataSimulator(cfg)

    # Batch generation demo
    batch = sim.generate_batch_data(5)
    print(f"Generated batch with {len(batch)} samples")
    sample0_json = sim.export_sample(batch[0], "json")
    print(f"Sample-0 JSON size: {len(sample0_json)} bytes")

    # Streaming demo (5 seconds)
    print("Streaming demo (5s)…")
    for sample in sim.generate_streaming_data(5, fps=5):
        print(f"➜ {sample['sample_id']}  fraud={sample.get('fraud_type','none')}")
    print("Done.")

if __name__ == "__main__":
    _demo()
