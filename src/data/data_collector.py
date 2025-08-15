# data/data_collector.py
"""
Sensor Data Collector for fraud detection system.
Handles collection and buffering of sensor data from various sources.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import json
from datetime import datetime

@dataclass
class SensorReading:
    """Represents a single sensor reading"""
    timestamp: float
    sensor_type: str
    data: Dict[str, Any]
    source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DataCollector:
    """
    Central data collector for all sensor inputs.
    Buffers and manages sensor data for fraud detection agents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Buffer configuration
        self.buffer_size = config.get('buffer_size', 10000)
        self.collection_rate_hz = config.get('collection_rate_hz', 50)
        self.auto_cleanup = config.get('auto_cleanup', True)
        self.cleanup_interval = config.get('cleanup_interval', 300)  # 5 minutes
        
        # Data retention
        self.retention_hours = config.get('retention_hours', 24)
        self.max_memory_mb = config.get('max_memory_mb', 100)
        
        # Data buffers for different sensor types
        self.sensor_buffers = {
            'accelerometer': deque(maxlen=self.buffer_size),
            'gyroscope': deque(maxlen=self.buffer_size),
            'magnetometer': deque(maxlen=self.buffer_size),
            'touch': deque(maxlen=self.buffer_size),
            'keyboard': deque(maxlen=self.buffer_size),
            'audio': deque(maxlen=self.buffer_size // 10),  # Smaller buffer for audio
            'camera': deque(maxlen=self.buffer_size // 20),  # Smaller buffer for images
            'app_usage': deque(maxlen=self.buffer_size),
            'system': deque(maxlen=self.buffer_size)
        }
        
        # Statistics tracking
        self.collection_stats = defaultdict(int)
        self.last_readings = {}
        
        # Threading for background operations
        self.collection_thread = None
        self.cleanup_thread = None
        self.is_collecting = False
        self.lock = threading.Lock()
        
        # Callbacks for real-time processing
        self.callbacks = defaultdict(list)
        
        print(f"DataCollector initialized with {self.buffer_size} buffer size")
    
    def start_collection(self) -> None:
        """Start background data collection"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        
        # Start cleanup thread
        if self.auto_cleanup:
            self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self.cleanup_thread.start()
        
        print("Data collection started")
    
    def stop_collection(self) -> None:
        """Stop background data collection"""
        self.is_collecting = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=1.0)
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=1.0)
        
        print("Data collection stopped")
    
    def add_sensor_reading(self, sensor_type: str, data: Dict[str, Any], 
                          timestamp: Optional[float] = None, source: str = "unknown") -> None:
        """
        Add a new sensor reading to the appropriate buffer
        
        Args:
            sensor_type: Type of sensor ('accelerometer', 'touch', etc.)
            data: Sensor data dictionary
            timestamp: Reading timestamp (current time if None)
            source: Source identifier for the reading
        """
        if timestamp is None:
            timestamp = time.time()
        
        try:
            with self.lock:
                # Create sensor reading
                reading = SensorReading(
                    timestamp=timestamp,
                    sensor_type=sensor_type,
                    data=data,
                    source=source
                )
                
                # Add to appropriate buffer
                if sensor_type in self.sensor_buffers:
                    self.sensor_buffers[sensor_type].append(reading)
                else:
                    # Create new buffer for unknown sensor type
                    self.sensor_buffers[sensor_type] = deque(maxlen=self.buffer_size)
                    self.sensor_buffers[sensor_type].append(reading)
                
                # Update statistics
                self.collection_stats[sensor_type] += 1
                self.collection_stats['total'] += 1
                self.last_readings[sensor_type] = timestamp
                
                # Trigger callbacks
                self._trigger_callbacks(sensor_type, reading)
        
        except Exception as e:
            print(f"Error adding sensor reading: {e}")
    
    def add_accelerometer_data(self, x: float, y: float, z: float, 
                             timestamp: Optional[float] = None) -> None:
        """Add accelerometer reading"""
        data = {'x': x, 'y': y, 'z': z, 'magnitude': np.sqrt(x*x + y*y + z*z)}
        self.add_sensor_reading('accelerometer', data, timestamp, 'accelerometer_sensor')
    
    def add_gyroscope_data(self, x: float, y: float, z: float, 
                          timestamp: Optional[float] = None) -> None:
        """Add gyroscope reading"""
        data = {'x': x, 'y': y, 'z': z, 'magnitude': np.sqrt(x*x + y*y + z*z)}
        self.add_sensor_reading('gyroscope', data, timestamp, 'gyroscope_sensor')
    
    def add_touch_event(self, x: float, y: float, pressure: float, action: str,
                       timestamp: Optional[float] = None) -> None:
        """Add touch event"""
        data = {
            'x': x, 'y': y, 'pressure': pressure, 'action': action,
            'touch_major': 0.0, 'touch_minor': 0.0  # Placeholder values
        }
        self.add_sensor_reading('touch', data, timestamp, 'touch_screen')
    
    def add_keystroke_event(self, key_code: int, action: str, pressure: float = 0.0,
                           timestamp: Optional[float] = None) -> None:
        """Add keystroke event"""
        data = {'key_code': key_code, 'action': action, 'pressure': pressure}
        self.add_sensor_reading('keyboard', data, timestamp, 'keyboard')
    
    def add_audio_data(self, audio_samples: np.ndarray, sample_rate: int,
                      timestamp: Optional[float] = None) -> None:
        """Add audio data"""
        data = {
            'samples': audio_samples.tolist() if len(audio_samples) < 1000 else audio_samples[:1000].tolist(),
            'sample_rate': sample_rate,
            'duration': len(audio_samples) / sample_rate,
            'rms_energy': float(np.sqrt(np.mean(audio_samples**2)))
        }
        self.add_sensor_reading('audio', data, timestamp, 'microphone')
    
    def add_image_data(self, image_data: np.ndarray, image_type: str = 'camera',
                      timestamp: Optional[float] = None) -> None:
        """Add image/camera data"""
        data = {
            'shape': image_data.shape,
            'type': image_type,
            'size_bytes': image_data.nbytes,
            'mean_intensity': float(np.mean(image_data))
        }
        # Note: Not storing full image data to save memory
        self.add_sensor_reading('camera', data, timestamp, 'camera')
    
    def add_app_usage_event(self, app_name: str, action: str, duration: float = 0.0,
                           timestamp: Optional[float] = None) -> None:
        """Add app usage event"""
        data = {'app_name': app_name, 'action': action, 'duration': duration}
        self.add_sensor_reading('app_usage', data, timestamp, 'system')
    
    def get_recent_data(self, sensor_type: str, time_window: float = 60.0) -> List[SensorReading]:
        """
        Get recent sensor data within time window
        
        Args:
            sensor_type: Type of sensor data to retrieve
            time_window: Time window in seconds
            
        Returns:
            List of recent sensor readings
        """
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        try:
            with self.lock:
                if sensor_type not in self.sensor_buffers:
                    return []
                
                recent_data = []
                for reading in self.sensor_buffers[sensor_type]:
                    if reading.timestamp >= cutoff_time:
                        recent_data.append(reading)
                
                return recent_data
        
        except Exception as e:
            print(f"Error retrieving recent data: {e}")
            return []
    
    def get_data_sequence(self, sensor_type: str, sequence_length: int) -> List[SensorReading]:
        """
        Get the most recent sequence of sensor data
        
        Args:
            sensor_type: Type of sensor data
            sequence_length: Number of readings to retrieve
            
        Returns:
            List of recent sensor readings
        """
        try:
            with self.lock:
                if sensor_type not in self.sensor_buffers:
                    return []
                
                buffer = self.sensor_buffers[sensor_type]
                start_idx = max(0, len(buffer) - sequence_length)
                return list(buffer)[start_idx:]
        
        except Exception as e:
            print(f"Error retrieving data sequence: {e}")
            return []
    
    def get_aggregated_data(self, sensor_type: str, time_window: float = 60.0) -> Dict[str, Any]:
        """
        Get aggregated statistics for sensor data
        
        Args:
            sensor_type: Type of sensor data
            time_window: Time window for aggregation
            
        Returns:
            Dictionary with aggregated statistics
        """
        recent_data = self.get_recent_data(sensor_type, time_window)
        
        if not recent_data:
            return {}
        
        try:
            # Extract numeric values for aggregation
            numeric_values = defaultdict(list)
            
            for reading in recent_data:
                for key, value in reading.data.items():
                    if isinstance(value, (int, float)):
                        numeric_values[key].append(value)
            
            # Calculate statistics
            aggregated = {
                'count': len(recent_data),
                'time_span': recent_data[-1].timestamp - recent_data[0].timestamp,
                'rate_hz': len(recent_data) / max(time_window, 1.0)
            }
            
            for key, values in numeric_values.items():
                if values:
                    aggregated[f'{key}_mean'] = float(np.mean(values))
                    aggregated[f'{key}_std'] = float(np.std(values))
                    aggregated[f'{key}_min'] = float(np.min(values))
                    aggregated[f'{key}_max'] = float(np.max(values))
            
            return aggregated
        
        except Exception as e:
            print(f"Error calculating aggregated data: {e}")
            return {'count': len(recent_data)}
    
    def register_callback(self, sensor_type: str, callback: Callable[[SensorReading], None]) -> None:
        """
        Register a callback for real-time data processing
        
        Args:
            sensor_type: Type of sensor to monitor
            callback: Function to call when new data arrives
        """
        self.callbacks[sensor_type].append(callback)
        print(f"Callback registered for {sensor_type}")
    
    def _trigger_callbacks(self, sensor_type: str, reading: SensorReading) -> None:
        """Trigger registered callbacks for sensor type"""
        try:
            for callback in self.callbacks[sensor_type]:
                try:
                    callback(reading)
                except Exception as e:
                    print(f"Callback error for {sensor_type}: {e}")
        except Exception as e:
            print(f"Error triggering callbacks: {e}")
    
    def _cleanup_worker(self) -> None:
        """Background worker for data cleanup"""
        while self.is_collecting:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_old_data()
                self._check_memory_usage()
            except Exception as e:
                print(f"Cleanup worker error: {e}")
    
    def _cleanup_old_data(self) -> None:
        """Remove old data beyond retention period"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (self.retention_hours * 3600)
            
            with self.lock:
                for sensor_type, buffer in self.sensor_buffers.items():
                    # Count items to remove
                    items_to_remove = 0
                    for reading in buffer:
                        if reading.timestamp < cutoff_time:
                            items_to_remove += 1
                        else:
                            break
                    
                    # Remove old items
                    for _ in range(items_to_remove):
                        if buffer:
                            buffer.popleft()
            
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    def _check_memory_usage(self) -> None:
        """Check and manage memory usage"""
        try:
            # Simple memory check based on buffer sizes
            total_items = sum(len(buffer) for buffer in self.sensor_buffers.values())
            
            # If too many items, reduce buffer sizes
            if total_items > self.buffer_size * len(self.sensor_buffers) * 0.8:
                with self.lock:
                    for buffer in self.sensor_buffers.values():
                        if len(buffer) > self.buffer_size // 2:
                            # Remove oldest half
                            items_to_remove = len(buffer) // 2
                            for _ in range(items_to_remove):
                                if buffer:
                                    buffer.popleft()
                
                print("Memory usage optimized - removed old data")
        
        except Exception as e:
            print(f"Memory check error: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get data collection statistics"""
        with self.lock:
            stats = {
                'total_readings': dict(self.collection_stats),
                'buffer_sizes': {sensor: len(buffer) for sensor, buffer in self.sensor_buffers.items()},
                'last_readings': dict(self.last_readings),
                'is_collecting': self.is_collecting
            }
            
            # Calculate rates
            current_time = time.time()
            for sensor_type, last_time in self.last_readings.items():
                if current_time - last_time < 3600:  # Within last hour
                    stats[f'{sensor_type}_active'] = True
                else:
                    stats[f'{sensor_type}_active'] = False
            
            return stats
    
    def export_data(self, sensor_types: Optional[List[str]] = None, 
                   time_window: Optional[float] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Export sensor data for analysis or storage
        
        Args:
            sensor_types: List of sensor types to export (all if None)
            time_window: Time window in seconds (all data if None)
            
        Returns:
            Dictionary of exported sensor data
        """
        try:
            if sensor_types is None:
                sensor_types = list(self.sensor_buffers.keys())
            
            exported_data = {}
            
            for sensor_type in sensor_types:
                if time_window is not None:
                    readings = self.get_recent_data(sensor_type, time_window)
                else:
                    with self.lock:
                        readings = list(self.sensor_buffers[sensor_type])
                
                exported_data[sensor_type] = [reading.to_dict() for reading in readings]
            
            return exported_data
        
        except Exception as e:
            print(f"Export error: {e}")
            return {}
    
    def clear_buffers(self, sensor_types: Optional[List[str]] = None) -> None:
        """
        Clear sensor data buffers
        
        Args:
            sensor_types: List of sensor types to clear (all if None)
        """
        try:
            with self.lock:
                if sensor_types is None:
                    sensor_types = list(self.sensor_buffers.keys())
                
                for sensor_type in sensor_types:
                    if sensor_type in self.sensor_buffers:
                        self.sensor_buffers[sensor_type].clear()
                        self.collection_stats[sensor_type] = 0
                
                print(f"Cleared buffers for: {sensor_types}")
        
        except Exception as e:
            print(f"Clear buffers error: {e}")
    
    def save_data(self, filepath: str, sensor_types: Optional[List[str]] = None,
                 time_window: Optional[float] = None) -> bool:
        """
        Save sensor data to file
        
        Args:
            filepath: Path to save data
            sensor_types: Sensor types to save
            time_window: Time window to save
            
        Returns:
            True if successful
        """
        try:
            data_to_save = {
                'export_timestamp': time.time(),
                'export_datetime': datetime.now().isoformat(),
                'collection_stats': self.get_collection_stats(),
                'sensor_data': self.export_data(sensor_types, time_window)
            }
            
            with open(filepath, 'w') as f:
                json.dump(data_to_save, f, indent=2, default=str)
            
            print(f"Data saved to {filepath}")
            return True
        
        except Exception as e:
            print(f"Save error: {e}")
            return False
    
    def load_data(self, filepath: str) -> bool:
        """
        Load sensor data from file
        
        Args:
            filepath: Path to load data from
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                saved_data = json.load(f)
            
            sensor_data = saved_data.get('sensor_data', {})
            
            with self.lock:
                for sensor_type, readings in sensor_data.items():
                    if sensor_type not in self.sensor_buffers:
                        self.sensor_buffers[sensor_type] = deque(maxlen=self.buffer_size)
                    
                    for reading_dict in readings:
                        reading = SensorReading(
                            timestamp=reading_dict['timestamp'],
                            sensor_type=reading_dict['sensor_type'],
                            data=reading_dict['data'],
                            source=reading_dict.get('source', 'loaded')
                        )
                        self.sensor_buffers[sensor_type].append(reading)
            
            print(f"Data loaded from {filepath}")
            return True
        
        except Exception as e:
            print(f"Load error: {e}")
            return False

# Utility functions for data collection
def create_data_collector(config: Optional[Dict[str, Any]] = None) -> DataCollector:
    """Create a data collector with default mobile configuration"""
    default_config = {
        'buffer_size': 5000,  # Smaller for mobile
        'collection_rate_hz': 50,
        'retention_hours': 12,  # Shorter retention
        'max_memory_mb': 50,    # Limited memory
        'auto_cleanup': True,
        'cleanup_interval': 180  # 3 minutes
    }
    
    if config:
        default_config.update(config)
    
    return DataCollector(default_config)
