# demo/mobile_integration_example.py

"""
Mobile integration examples and guides for QuadFusion

Features:
- Complete mobile app integration guide
- Android native example (Java/Kotlin patterns)
- iOS native example (Swift/Objective-C patterns)
- React Native integration example
- Flutter integration example
- Model loading and initialization
- Sensor data collection examples
- Real-time processing integration
- UI/UX integration patterns
- Background processing examples
- Battery optimization examples
- Memory management patterns
- Privacy-compliant data handling
- Permission handling
"""

import threading
import time
import logging
import json
import os
from typing import Any, Dict, Optional, List, Callable
from dataclasses import dataclass
from pathlib import Path
import asyncio
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class MobileConfig:
    """Configuration for mobile integration."""
    platform: str = "unknown"
    model_path: str = "./models"
    batch_size: int = 16
    inference_interval_ms: int = 50  # 20 Hz
    enable_background_processing: bool = True
    enable_battery_optimization: bool = True
    max_memory_mb: float = 100
    cache_size: int = 100

@dataclass 
class PermissionRequest:
    """Mobile permission request structure."""
    permission_type: str
    required: bool
    rationale: str

class MobileIntegrationHelper:
    """
    Common helper utilities for mobile integration across platforms.
    Provides platform detection, configuration, and utility functions.
    """
    
    def __init__(self):
        self.platform = self.detect_platform()
        self.config = MobileConfig(platform=self.platform)
        self.required_permissions = self._get_required_permissions()
        
    def detect_platform(self) -> str:
        """Detect the current platform."""
        import platform
        system = platform.system().lower()
        
        # Platform detection logic
        if 'android' in system or os.getenv('ANDROID_ROOT'):
            return 'android'
        elif 'ios' in system or 'darwin' in system:
            # In real iOS app, check for iOS-specific APIs
            return 'ios'
        elif 'linux' in system:
            return 'linux'
        elif 'windows' in system:
            return 'windows'
        else:
            return 'unknown'
    
    def _get_required_permissions(self) -> List[PermissionRequest]:
        """Get list of required permissions for the platform."""
        permissions = []
        
        if self.platform == 'android':
            permissions.extend([
                PermissionRequest('CAMERA', True, 'Required for facial recognition'),
                PermissionRequest('RECORD_AUDIO', True, 'Required for voice authentication'),
                PermissionRequest('ACCESS_FINE_LOCATION', False, 'Optional for location-based security'),
                PermissionRequest('WRITE_EXTERNAL_STORAGE', False, 'For model caching')
            ])
        elif self.platform == 'ios':
            permissions.extend([
                PermissionRequest('NSCameraUsageDescription', True, 'Required for facial recognition'),
                PermissionRequest('NSMicrophoneUsageDescription', True, 'Required for voice authentication'),
                PermissionRequest('NSLocationWhenInUseUsageDescription', False, 'Optional location security')
            ])
            
        return permissions
    
    def log_platform_info(self):
        """Log platform and configuration information."""
        logging.info(f"Platform detected: {self.platform}")
        logging.info(f"Model path: {self.config.model_path}")
        logging.info(f"Batch size: {self.config.batch_size}")
        logging.info(f"Required permissions: {len(self.required_permissions)}")
    
    def check_device_capabilities(self) -> Dict[str, bool]:
        """Check device capabilities and hardware support."""
        capabilities = {
            'camera_available': True,  # Stub - would check actual camera
            'microphone_available': True,  # Stub - would check actual microphone
            'accelerometer_available': True,
            'gyroscope_available': True,
            'magnetometer_available': True,
            'sufficient_memory': True,  # Stub - would check actual memory
            'sufficient_storage': True
        }
        
        logging.info(f"Device capabilities: {capabilities}")
        return capabilities

class AndroidIntegration:
    """
    Android native integration example with Java/Kotlin patterns.
    Demonstrates TensorFlow Lite integration, sensor management, and background services.
    """
    
    def __init__(self, config: MobileConfig):
        self.config = config
        self.sensor_manager = SensorManager(config)
        self.model_manager = ModelManager(config)
        self.ui_callback: Optional[Callable] = None
        self.background_thread: Optional[threading.Thread] = None
        self.is_running = False
        
    def initialize(self) -> bool:
        """
        Initialize Android integration.
        Equivalent to onCreate() or onResume() in Android Activity.
        """
        logging.info("🤖 Android: Initializing QuadFusion integration")
        
        try:
            # Request permissions (simulated)
            if not self._request_permissions():
                logging.error("Android: Permission request failed")
                return False
            
            # Load TensorFlow Lite models
            if not self.model_manager.load_models_tflite():
                logging.error("Android: Model loading failed")
                return False
                
            # Initialize sensors
            if not self.sensor_manager.start():
                logging.error("Android: Sensor initialization failed")
                return False
            
            # Android-specific optimizations
            self._apply_android_optimizations()
            
            logging.info("✅ Android: Initialization successful")
            return True
            
        except Exception as e:
            logging.error(f"❌ Android: Initialization failed: {e}")
            return False
    
    def _request_permissions(self) -> bool:
        """Simulate Android permission request flow."""
        required_permissions = [
            'android.permission.CAMERA',
            'android.permission.RECORD_AUDIO',
            'android.permission.WAKE_LOCK'
        ]
        
        for permission in required_permissions:
            # In real Android app:
            # if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
            #     ActivityCompat.requestPermissions(this, new String[]{permission}, REQUEST_CODE);
            # }
            logging.info(f"Android: Requesting permission {permission}")
            
        return True  # Assume granted for demo
    
    def _apply_android_optimizations(self):
        """Apply Android-specific performance optimizations."""
        # Battery optimization
        if self.config.enable_battery_optimization:
            logging.info("Android: Applying battery optimizations")
            # Request battery optimization exemption
            # PowerManager.isIgnoringBatteryOptimizations()
        
        # Background processing optimization
        if self.config.enable_background_processing:
            logging.info("Android: Configuring background processing")
            # Use WorkManager for background tasks
            # Or JobScheduler for periodic tasks
    
    def start_background_service(self):
        """
        Start Android background service for continuous monitoring.
        Equivalent to startForegroundService() in Android.
        """
        if self.is_running:
            return
            
        self.is_running = True
        self.background_thread = threading.Thread(target=self._background_service_loop, daemon=True)
        self.background_thread.start()
        
        logging.info("🔄 Android: Background service started")
    
    def _background_service_loop(self):
        """Background service main loop."""
        logging.info("Android: Background service loop started")
        
        while self.is_running:
            try:
                # Collect sensor data
                sensor_data = self.sensor_manager.get_latest_data()
                
                if sensor_data:
                    # Run inference
                    results = self.model_manager.run_inference(sensor_data)
                    
                    # Update UI on main thread (simulated)
                    if self.ui_callback:
                        self.ui_callback(results)
                    
                    # Check for fraud
                    if results.get('fraud_score', 0) > 0.7:
                        self._handle_fraud_detection(results)
                
                # Sleep for inference interval
                time.sleep(self.config.inference_interval_ms / 1000)
                
            except Exception as e:
                logging.error(f"Android: Background service error: {e}")
                time.sleep(1)  # Error recovery delay
    
    def _handle_fraud_detection(self, results: Dict[str, Any]):
        """Handle fraud detection event."""
        logging.warning(f"🚨 Android: Fraud detected! Score: {results.get('fraud_score')}")
        
        # In real app:
        # - Show security alert dialog
        # - Lock the app
        # - Send notification to admin
        # - Log security event
        
        # Send local notification
        self._send_notification("Security Alert", "Unusual activity detected")
    
    def _send_notification(self, title: str, message: str):
        """Send Android notification."""
        logging.info(f"📱 Android: Notification - {title}: {message}")
        
        # In real Android app:
        # NotificationCompat.Builder builder = new NotificationCompat.Builder(context, CHANNEL_ID)
        #     .setSmallIcon(R.drawable.notification_icon)
        #     .setContentTitle(title)
        #     .setContentText(message)
        #     .setPriority(NotificationCompat.PRIORITY_HIGH);
        # NotificationManagerCompat.from(context).notify(notificationId, builder.build());
    
    def set_ui_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for UI updates."""
        self.ui_callback = callback
    
    def stop(self):
        """Stop background processing."""
        self.is_running = False
        if self.background_thread:
            self.background_thread.join(timeout=2)
        self.sensor_manager.stop()
        logging.info("🛑 Android: Integration stopped")

class iOSIntegration:
    """
    iOS native integration example with Swift/Objective-C patterns.
    Demonstrates Core ML integration, Core Motion, and background app refresh.
    """
    
    def __init__(self, config: MobileConfig):
        self.config = config
        self.sensor_manager = SensorManager(config)
        self.model_manager = ModelManager(config)
        self.ui_callback: Optional[Callable] = None
        self.background_task_id = None
        self.is_running = False
    
    def initialize(self) -> bool:
        """
        Initialize iOS integration.
        Equivalent to viewDidLoad or applicationDidFinishLaunching.
        """
        logging.info("🍎 iOS: Initializing QuadFusion integration")
        
        try:
            # Request permissions
            if not self._request_permissions():
                logging.error("iOS: Permission request failed")
                return False
            
            # Load Core ML models
            if not self.model_manager.load_models_coreml():
                logging.error("iOS: Core ML model loading failed")
                return False
            
            # Initialize Core Motion
            if not self.sensor_manager.start():
                logging.error("iOS: Core Motion initialization failed")
                return False
            
            # iOS-specific optimizations
            self._apply_ios_optimizations()
            
            logging.info("✅ iOS: Initialization successful")
            return True
            
        except Exception as e:
            logging.error(f"❌ iOS: Initialization failed: {e}")
            return False
    
    def _request_permissions(self) -> bool:
        """Request iOS permissions using Info.plist and runtime requests."""
        # In real iOS app, permissions are requested via:
        # - Info.plist entries (NSCameraUsageDescription, etc.)
        # - Runtime permission requests
        
        permissions = [
            'NSCameraUsageDescription',
            'NSMicrophoneUsageDescription',
            'NSMotionUsageDescription'
        ]
        
        for permission in permissions:
            logging.info(f"iOS: Requesting permission {permission}")
            # In Swift:
            # AVCaptureDevice.requestAccess(for: .video) { granted in ... }
            # AVAudioSession.sharedInstance().requestRecordPermission { granted in ... }
        
        return True  # Assume granted for demo
    
    def _apply_ios_optimizations(self):
        """Apply iOS-specific optimizations."""
        # Background app refresh
        logging.info("iOS: Configuring background app refresh")
        # UIApplication.shared.setMinimumBackgroundFetchInterval(UIApplication.backgroundFetchIntervalMinimum)
        
        # Battery optimization
        if self.config.enable_battery_optimization:
            logging.info("iOS: Applying battery optimizations")
            # ProcessInfo.processInfo.isLowPowerModeEnabled
    
    def start_background_task(self):
        """
        Start iOS background task.
        Uses BGTaskScheduler for background processing.
        """
        if self.is_running:
            return
            
        self.is_running = True
        
        # In real iOS app:
        # self.backgroundTaskID = UIApplication.shared.beginBackgroundTask {
        #     self.endBackgroundTask()
        # }
        
        self.background_thread = threading.Thread(target=self._background_task_loop, daemon=True)
        self.background_thread.start()
        
        logging.info("🔄 iOS: Background task started")
    
    def _background_task_loop(self):
        """iOS background processing loop."""
        logging.info("iOS: Background task loop started")
        
        while self.is_running:
            try:
                # Check if app is in background
                # In real app: UIApplication.shared.applicationState == .background
                
                # Collect sensor data
                sensor_data = self.sensor_manager.get_latest_data()
                
                if sensor_data:
                    # Run Core ML inference
                    results = self.model_manager.run_inference_coreml(sensor_data)
                    
                    # Update UI on main queue (simulated)
                    if self.ui_callback:
                        # DispatchQueue.main.async { self.updateUI(results) }
                        self.ui_callback(results)
                    
                    # Handle fraud detection
                    if results.get('fraud_score', 0) > 0.7:
                        self._handle_fraud_detection(results)
                
                time.sleep(self.config.inference_interval_ms / 1000)
                
            except Exception as e:
                logging.error(f"iOS: Background task error: {e}")
                time.sleep(1)
    
    def _handle_fraud_detection(self, results: Dict[str, Any]):
        """Handle fraud detection on iOS."""
        logging.warning(f"🚨 iOS: Fraud detected! Score: {results.get('fraud_score')}")
        
        # Send local notification
        self._send_local_notification("Security Alert", "Unusual activity detected")
        
        # In real app:
        # - Present security alert
        # - Lock app with Face ID/Touch ID
        # - Send notification to admin
    
    def _send_local_notification(self, title: str, message: str):
        """Send iOS local notification."""
        logging.info(f"📱 iOS: Local notification - {title}: {message}")
        
        # In real iOS app:
        # let content = UNMutableNotificationContent()
        # content.title = title
        # content.body = message
        # content.sound = UNNotificationSound.default
        # let request = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
        # UNUserNotificationCenter.current().add(request)
    
    def set_ui_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for UI updates."""
        self.ui_callback = callback
    
    def stop(self):
        """Stop background processing."""
        self.is_running = False
        if hasattr(self, 'background_thread'):
            self.background_thread.join(timeout=2)
        self.sensor_manager.stop()
        
        # End background task
        # UIApplication.shared.endBackgroundTask(self.backgroundTaskID)
        
        logging.info("🛑 iOS: Integration stopped")

class CrossPlatformIntegration:
    """
    Cross-platform integration for React Native, Flutter, and other frameworks.
    Provides unified API across platforms with platform-specific optimizations.
    """
    
    def __init__(self, config: MobileConfig, framework: str = "react_native"):
        self.config = config
        self.framework = framework  # 'react_native', 'flutter', 'xamarin'
        self.sensor_manager = SensorManager(config)
        self.model_manager = ModelManager(config)
        self.native_bridge = self._create_native_bridge()
        self.is_running = False
    
    def _create_native_bridge(self):
        """Create platform-specific native bridge."""
        if self.framework == "react_native":
            return ReactNativeBridge()
        elif self.framework == "flutter":
            return FlutterBridge()
        else:
            return GenericBridge()
    
    def initialize(self) -> bool:
        """Initialize cross-platform integration."""
        logging.info(f"🌐 {self.framework}: Initializing QuadFusion integration")
        
        try:
            # Initialize native bridge
            if not self.native_bridge.initialize():
                logging.error("Cross-platform: Native bridge initialization failed")
                return False
            
            # Load models using platform-appropriate method
            if not self.model_manager.load_models_cross_platform(self.config.platform):
                logging.error("Cross-platform: Model loading failed")
                return False
            
            # Initialize sensors
            if not self.sensor_manager.start():
                logging.error("Cross-platform: Sensor initialization failed")
                return False
            
            logging.info(f"✅ {self.framework}: Initialization successful")
            return True
            
        except Exception as e:
            logging.error(f"❌ {self.framework}: Initialization failed: {e}")
            return False
    
    def start_processing(self):
        """Start cross-platform processing."""
        if self.is_running:
            return
            
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logging.info(f"🔄 {self.framework}: Processing started")
    
    def _processing_loop(self):
        """Cross-platform processing loop."""
        while self.is_running:
            try:
                # Get sensor data
                sensor_data = self.sensor_manager.get_latest_data()
                
                if sensor_data:
                    # Run inference
                    results = self.model_manager.run_inference_cross_platform(sensor_data)
                    
                    # Send results to JavaScript/Dart side
                    self.native_bridge.send_results(results)
                    
                    # Handle fraud
                    if results.get('fraud_score', 0) > 0.7:
                        self.native_bridge.send_fraud_alert(results)
                
                time.sleep(self.config.inference_interval_ms / 1000)
                
            except Exception as e:
                logging.error(f"{self.framework}: Processing error: {e}")
                time.sleep(1)
    
    def stop(self):
        """Stop cross-platform processing."""
        self.is_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2)
        self.sensor_manager.stop()
        self.native_bridge.cleanup()
        logging.info(f"🛑 {self.framework}: Integration stopped")

class ReactNativeBridge:
    """React Native bridge for communication with JavaScript."""
    
    def initialize(self) -> bool:
        logging.info("React Native: Bridge initialized")
        # In real RN: NativeModules or TurboModules setup
        return True
    
    def send_results(self, results: Dict[str, Any]):
        """Send results to React Native side."""
        logging.info(f"React Native: Sending results to JS: {results}")
        # In real RN: emit event to JavaScript
        # DeviceEventEmitter.emit('QuadFusionResults', results)
    
    def send_fraud_alert(self, results: Dict[str, Any]):
        """Send fraud alert to React Native."""
        logging.warning(f"🚨 React Native: Fraud alert sent to JS")
        # DeviceEventEmitter.emit('QuadFusionFraudAlert', results)
    
    def cleanup(self):
        logging.info("React Native: Bridge cleaned up")

class FlutterBridge:
    """Flutter platform channel bridge."""
    
    def initialize(self) -> bool:
        logging.info("Flutter: Platform channel initialized")
        # In real Flutter: MethodChannel setup
        return True
    
    def send_results(self, results: Dict[str, Any]):
        """Send results to Flutter/Dart side."""
        logging.info(f"Flutter: Sending results to Dart: {results}")
        # In real Flutter: methodChannel.invokeMethod('updateResults', results)
    
    def send_fraud_alert(self, results: Dict[str, Any]):
        """Send fraud alert to Flutter."""
        logging.warning(f"🚨 Flutter: Fraud alert sent to Dart")
        # methodChannel.invokeMethod('fraudAlert', results)
    
    def cleanup(self):
        logging.info("Flutter: Platform channel cleaned up")

class GenericBridge:
    """Generic bridge for other frameworks."""
    
    def initialize(self) -> bool:
        logging.info("Generic: Bridge initialized")
        return True
    
    def send_results(self, results: Dict[str, Any]):
        logging.info(f"Generic: Results: {results}")
    
    def send_fraud_alert(self, results: Dict[str, Any]):
        logging.warning(f"🚨 Generic: Fraud alert")
    
    def cleanup(self):
        logging.info("Generic: Bridge cleaned up")

class SensorManager:
    """
    Unified sensor data manager for all platforms.
    Handles accelerometer, gyroscope, magnetometer, camera, and microphone.
    """
    
    def __init__(self, config: MobileConfig):
        self.config = config
        self.latest_data = {}
        self.collecting = False
        self.sensor_thread: Optional[threading.Thread] = None
        self.data_buffer = deque(maxlen=config.cache_size)
        self.lock = threading.Lock()
        
    def start(self) -> bool:
        """Start sensor data collection."""
        if self.collecting:
            return True
            
        try:
            self.collecting = True
            self.sensor_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self.sensor_thread.start()
            
            logging.info("📱 SensorManager: Data collection started")
            return True
            
        except Exception as e:
            logging.error(f"SensorManager: Start failed: {e}")
            return False
    
    def _collection_loop(self):
        """Main sensor collection loop."""
        while self.collecting:
            try:
                # Simulate sensor data collection
                timestamp = time.time()
                
                # Motion sensors (20Hz)
                motion_data = {
                    'accelerometer': self._get_accelerometer_data(),
                    'gyroscope': self._get_gyroscope_data(), 
                    'magnetometer': self._get_magnetometer_data(),
                    'timestamp': timestamp
                }
                
                # Touch data (when available)
                touch_data = self._get_touch_data()
                if touch_data:
                    motion_data['touch'] = touch_data
                
                # Audio data (when recording)
                audio_data = self._get_audio_data()
                if audio_data is not None:
                    motion_data['audio'] = audio_data
                
                # Store latest data
                with self.lock:
                    self.latest_data = motion_data
                    self.data_buffer.append(motion_data)
                
                # Sleep for sampling rate
                time.sleep(1.0 / 20)  # 20 Hz
                
            except Exception as e:
                logging.error(f"SensorManager: Collection error: {e}")
                time.sleep(0.1)  # Error recovery
    
    def _get_accelerometer_data(self) -> List[float]:
        """Get simulated accelerometer data."""
        # In real app: CoreMotion (iOS) or SensorManager (Android)
        import random
        return [
            random.gauss(0, 0.1),  # X
            random.gauss(0, 0.1),  # Y
            random.gauss(9.8, 0.2) # Z (gravity)
        ]
    
    def _get_gyroscope_data(self) -> List[float]:
        """Get simulated gyroscope data."""
        import random
        return [
            random.gauss(0, 0.05),  # X rotation
            random.gauss(0, 0.05),  # Y rotation  
            random.gauss(0, 0.05)   # Z rotation
        ]
    
    def _get_magnetometer_data(self) -> List[float]:
        """Get simulated magnetometer data."""
        import random
        return [
            random.gauss(0.3, 0.1),  # X magnetic
            random.gauss(0.0, 0.1),  # Y magnetic
            random.gauss(0.1, 0.1)   # Z magnetic
        ]
    
    def _get_touch_data(self) -> Optional[Dict[str, Any]]:
        """Get touch/tap data when available."""
        # Simulate occasional touch events
        import random
        if random.random() < 0.1:  # 10% chance
            return {
                'pressure': random.random(),
                'area': random.randint(10, 50),
                'x': random.randint(0, 1080),
                'y': random.randint(0, 1920)
            }
        return None
    
    def _get_audio_data(self) -> Optional[List[float]]:
        """Get audio data when recording."""
        # Simulate audio only occasionally
        import random
        if random.random() < 0.05:  # 5% chance
            # Simulate 0.1 second of audio at 16kHz
            return [random.gauss(0, 0.1) for _ in range(1600)]
        return None
    
    def get_latest_data(self) -> Dict[str, Any]:
        """Get the most recent sensor data."""
        with self.lock:
            return self.latest_data.copy()
    
    def get_buffered_data(self, n_samples: int = 10) -> List[Dict[str, Any]]:
        """Get last N samples from buffer."""
        with self.lock:
            return list(self.data_buffer)[-n_samples:]
    
    def stop(self):
        """Stop sensor data collection."""
        self.collecting = False
        if self.sensor_thread:
            self.sensor_thread.join(timeout=2)
        logging.info("📱 SensorManager: Data collection stopped")

class ModelManager:
    """
    Cross-platform model manager supporting TensorFlow Lite, Core ML, and ONNX.
    Handles model loading, caching, and inference optimization.
    """
    
    def __init__(self, config: MobileConfig):
        self.config = config
        self.loaded_models = {}
        self.model_cache = {}
        self.inference_lock = threading.Lock()
        
    def load_models_tflite(self) -> bool:
        """Load TensorFlow Lite models for Android."""
        try:
            logging.info("🤖 ModelManager: Loading TensorFlow Lite models")
            
            # In real Android app:
            # Interpreter motionInterpreter = new Interpreter(loadModelFile("motion_cnn.tflite"));
            # Interpreter speakerInterpreter = new Interpreter(loadModelFile("speaker_id.tflite"));
            
            self.loaded_models['motion_cnn'] = 'TFLite Motion CNN Model'
            self.loaded_models['speaker_id'] = 'TFLite Speaker ID Model'
            self.loaded_models['tiny_llm'] = 'TFLite Tiny LLM Model'
            
            logging.info("✅ ModelManager: TensorFlow Lite models loaded")
            return True
            
        except Exception as e:
            logging.error(f"❌ ModelManager: TFLite loading failed: {e}")
            return False
    
    def load_models_coreml(self) -> bool:
        """Load Core ML models for iOS."""
        try:
            logging.info("🍎 ModelManager: Loading Core ML models")
            
            # In real iOS app:
            # let motionModel = try MotionCNN(configuration: MLModelConfiguration())
            # let speakerModel = try SpeakerID(configuration: MLModelConfiguration())
            
            self.loaded_models['motion_cnn'] = 'Core ML Motion CNN Model'
            self.loaded_models['speaker_id'] = 'Core ML Speaker ID Model'
            self.loaded_models['tiny_llm'] = 'Core ML Tiny LLM Model'
            
            logging.info("✅ ModelManager: Core ML models loaded")
            return True
            
        except Exception as e:
            logging.error(f"❌ ModelManager: Core ML loading failed: {e}")
            return False
    
    def load_models_cross_platform(self, platform: str) -> bool:
        """Load models for cross-platform frameworks."""
        try:
            logging.info(f"🌐 ModelManager: Loading models for {platform}")
            
            if platform == 'android':
                return self.load_models_tflite()
            elif platform == 'ios':
                return self.load_models_coreml()
            else:
                # Generic loading for other platforms
                self.loaded_models['motion_cnn'] = 'Generic Motion CNN Model'
                self.loaded_models['speaker_id'] = 'Generic Speaker ID Model'
                self.loaded_models['tiny_llm'] = 'Generic Tiny LLM Model'
                return True
                
        except Exception as e:
            logging.error(f"❌ ModelManager: Cross-platform loading failed: {e}")
            return False
    
    def run_inference(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference using loaded models."""
        with self.inference_lock:
            start_time = time.time()
            
            try:
                # Simulate model inference
                results = {
                    'motion_prediction': {
                        'activity': 'walking',
                        'confidence': 0.92,
                        'anomaly_score': 0.15
                    },
                    'speaker_prediction': {
                        'speaker_id': 'user_123',
                        'confidence': 0.88,
                        'spoofing_detected': False
                    },
                    'fusion_result': {
                        'decision': 'legitimate',
                        'fraud_score': 0.12,
                        'risk_level': 'low'
                    },
                    'inference_time_ms': (time.time() - start_time) * 1000,
                    'timestamp': time.time()
                }
                
                # Cache result
                self.model_cache[sensor_data.get('timestamp', time.time())] = results
                
                return results
                
            except Exception as e:
                logging.error(f"ModelManager: Inference failed: {e}")
                return {'error': str(e), 'fraud_score': 0.0}
    
    def run_inference_coreml(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference using Core ML models."""
        # iOS-specific optimizations for Core ML
        logging.info("🍎 ModelManager: Running Core ML inference")
        return self.run_inference(sensor_data)
    
    def run_inference_cross_platform(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference for cross-platform frameworks."""
        logging.info("🌐 ModelManager: Running cross-platform inference")
        return self.run_inference(sensor_data)
    
    def clear_cache(self):
        """Clear model cache to free memory."""
        self.model_cache.clear()
        logging.info("ModelManager: Cache cleared")

# Integration examples and usage patterns

def create_android_example():
    """Example Android integration usage."""
    config = MobileConfig(
        platform='android',
        batch_size=16,
        inference_interval_ms=50,
        enable_background_processing=True
    )
    
    android_app = AndroidIntegration(config)
    
    # Initialize
    if android_app.initialize():
        # Set UI callback
        def update_ui(results):
            logging.info(f"UI Update: {results}")
        
        android_app.set_ui_callback(update_ui)
        
        # Start background service
        android_app.start_background_service()
        
        # Simulate running for 10 seconds
        time.sleep(10)
        
        # Stop
        android_app.stop()
    
    return android_app

def create_ios_example():
    """Example iOS integration usage."""
    config = MobileConfig(
        platform='ios',
        batch_size=8,  # Smaller batch for iOS
        inference_interval_ms=100,  # Lower frequency for battery
        enable_battery_optimization=True
    )
    
    ios_app = iOSIntegration(config)
    
    # Initialize
    if ios_app.initialize():
        # Set UI callback
        def update_ui(results):
            logging.info(f"iOS UI Update: {results}")
        
        ios_app.set_ui_callback(update_ui)
        
        # Start background task
        ios_app.start_background_task()
        
        # Simulate running for 10 seconds
        time.sleep(10)
        
        # Stop
        ios_app.stop()
    
    return ios_app

def create_react_native_example():
    """Example React Native integration usage."""
    config = MobileConfig(
        platform='android',  # Can be detected at runtime
        batch_size=16,
        inference_interval_ms=75
    )
    
    rn_app = CrossPlatformIntegration(config, framework='react_native')
    
    # Initialize
    if rn_app.initialize():
        # Start processing
        rn_app.start_processing()
        
        # Simulate running for 10 seconds
        time.sleep(10)
        
        # Stop
        rn_app.stop()
    
    return rn_app

def create_flutter_example():
    """Example Flutter integration usage."""
    config = MobileConfig(
        platform='ios',  # Can be detected at runtime
        batch_size=12,
        inference_interval_ms=80
    )
    
    flutter_app = CrossPlatformIntegration(config, framework='flutter')
    
    # Initialize
    if flutter_app.initialize():
        # Start processing
        flutter_app.start_processing()
        
        # Simulate running for 10 seconds
        time.sleep(10)
        
        # Stop
        flutter_app.stop()
    
    return flutter_app

# Example usage and testing
if __name__ == '__main__':
    # Initialize helper
    helper = MobileIntegrationHelper()
    helper.log_platform_info()
    
    # Check device capabilities
    capabilities = helper.check_device_capabilities()
    logging.info(f"Device capabilities: {capabilities}")
    
    # Run platform-specific examples
    logging.info("\n" + "="*50)
    logging.info("Running Android Example")
    logging.info("="*50)
    android_example = create_android_example()
    
    logging.info("\n" + "="*50)
    logging.info("Running iOS Example")
    logging.info("="*50)
    ios_example = create_ios_example()
    
    logging.info("\n" + "="*50)
    logging.info("Running React Native Example")
    logging.info("="*50)
    rn_example = create_react_native_example()
    
    logging.info("\n" + "="*50)
    logging.info("Running Flutter Example")
    logging.info("="*50)
    flutter_example = create_flutter_example()
    
    logging.info("\n✅ All mobile integration examples completed!")
