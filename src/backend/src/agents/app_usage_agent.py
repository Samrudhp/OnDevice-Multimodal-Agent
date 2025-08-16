# agents/app_usage_agent.py
"""
App Usage Agent for analyzing application usage patterns.
Uses statistical baselines to detect anomalous app usage behavior.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, deque
import time
import json
from datetime import datetime, timedelta

from .base_agent import BaseAgent, AgentResult, RiskLevel

class AppUsageAgent(BaseAgent):
    """
    Agent for analyzing app usage patterns and detecting anomalies.
    Uses statistical analysis of app usage frequency, duration, and timing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("AppUsageAgent", config)
        
        # Usage analysis parameters
        self.history_window_days = config.get('history_window_days', 30)
        self.min_usage_sessions = config.get('min_usage_sessions', 10)
        self.anomaly_threshold = config.get('anomaly_threshold', 2.5)  # Z-score threshold
        
        # Time-based parameters
        self.time_bucket_hours = config.get('time_bucket_hours', 1)
        self.session_timeout_minutes = config.get('session_timeout_minutes', 5)
        
        # App usage data structures
        self.app_usage_history = defaultdict(list)  # app_name -> [(timestamp, duration), ...]
        self.daily_usage_patterns = defaultdict(lambda: defaultdict(list))  # app -> hour -> [durations]
        self.app_frequency = defaultdict(int)  # app -> count
        self.app_durations = defaultdict(list)  # app -> [durations]
        
        # Statistical models
        self.usage_stats = {}  # app -> {'mean_duration', 'std_duration', 'freq_mean', 'freq_std'}
        self.hourly_stats = {}  # app -> hour -> {'mean', 'std'}
        
        # Current session tracking
        self.current_sessions = {}  # app -> start_time
        self.last_activity_time = time.time()
        
        # Training state
        self.is_trained = False
        self.last_update_time = time.time()
        
        print(f"[{self.agent_name}] Initialized with {self.history_window_days}-day history window")
    
    def log_app_usage(self, app_name: str, action: str, timestamp: float = None) -> None:
        """
        Log app usage event
        
        Args:
            app_name: Name of the application
            action: 'open', 'close', or 'switch_to'
            timestamp: Event timestamp (current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        try:
            if action == 'open' or action == 'switch_to':
                # Start new session
                self.current_sessions[app_name] = timestamp
                self.last_activity_time = timestamp
                
            elif action == 'close':
                # End session and record duration
                if app_name in self.current_sessions:
                    start_time = self.current_sessions[app_name]
                    duration = timestamp - start_time
                    
                    if duration > 0:  # Valid session
                        self.app_usage_history[app_name].append((timestamp, duration))
                        self.app_frequency[app_name] += 1
                        self.app_durations[app_name].append(duration)
                        
                        # Log to hourly patterns
                        hour = datetime.fromtimestamp(timestamp).hour
                        self.daily_usage_patterns[app_name][hour].append(duration)
                    
                    del self.current_sessions[app_name]
                
                self.last_activity_time = timestamp
            
            # Clean old data periodically
            self._cleanup_old_data(timestamp)
            
        except Exception as e:
            print(f"[{self.agent_name}] Error logging usage: {e}")
    
    def _cleanup_old_data(self, current_time: float) -> None:
        """Remove usage data older than the history window"""
        cutoff_time = current_time - (self.history_window_days * 24 * 3600)
        
        for app_name in list(self.app_usage_history.keys()):
            # Filter old usage records
            self.app_usage_history[app_name] = [
                (ts, dur) for ts, dur in self.app_usage_history[app_name]
                if ts > cutoff_time
            ]
            
            # Remove empty entries
            if not self.app_usage_history[app_name]:
                del self.app_usage_history[app_name]
                if app_name in self.app_frequency:
                    del self.app_frequency[app_name]
                if app_name in self.app_durations:
                    del self.app_durations[app_name]
    
    def train_baseline(self) -> bool:
        """
        Train baseline usage patterns from historical data
        
        Returns:
            True if training successful
        """
        try:
            print(f"[{self.agent_name}] Training baseline patterns...")
            
            total_sessions = sum(len(sessions) for sessions in self.app_usage_history.values())
            
            if total_sessions < self.min_usage_sessions:
                print(f"[{self.agent_name}] Insufficient data: {total_sessions} sessions (need {self.min_usage_sessions})")
                return False
            
            # Calculate usage statistics for each app
            for app_name, usage_data in self.app_usage_history.items():
                if len(usage_data) < 3:  # Need minimum data points
                    continue
                
                durations = [dur for _, dur in usage_data]
                
                # Basic duration statistics
                mean_duration = np.mean(durations)
                std_duration = np.std(durations)
                
                # Frequency statistics (sessions per day)
                timestamps = [ts for ts, _ in usage_data]
                time_span_days = (max(timestamps) - min(timestamps)) / 86400
                freq_per_day = len(usage_data) / max(time_span_days, 1)
                
                self.usage_stats[app_name] = {
                    'mean_duration': mean_duration,
                    'std_duration': std_duration,
                    'freq_per_day': freq_per_day,
                    'total_sessions': len(usage_data)
                }
                
                # Hourly usage patterns
                hourly_data = defaultdict(list)
                for timestamp, duration in usage_data:
                    hour = datetime.fromtimestamp(timestamp).hour
                    hourly_data[hour].append(duration)
                
                self.hourly_stats[app_name] = {}
                for hour, durations in hourly_data.items():
                    if len(durations) >= 2:
                        self.hourly_stats[app_name][hour] = {
                            'mean': np.mean(durations),
                            'std': np.std(durations),
                            'count': len(durations)
                        }
            
            self.is_trained = True
            self.last_update_time = time.time()
            
            print(f"[{self.agent_name}] Baseline trained for {len(self.usage_stats)} apps")
            return True
            
        except Exception as e:
            print(f"[{self.agent_name}] Training error: {e}")
            return False
    
    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        """
        Analyze current app usage for anomalies
        
        Args:
            data: Dictionary containing usage data
            
        Returns:
            AgentResult with anomaly detection results
        """
        start_time = time.time()
        
        try:
            if not self.is_trained:
                return self._create_error_result("Agent not trained", start_time)
            
            # Extract usage data
            current_usage = data.get('current_usage', {})
            analysis_window = data.get('analysis_window_hours', 24)
            
            if not current_usage:
                return self._create_error_result("No usage data provided", start_time)
            
            # Analyze recent usage patterns
            current_time = time.time()
            window_start = current_time - (analysis_window * 3600)
            
            anomaly_scores = []
            feature_anomalies = []
            
            # Check each app's recent usage
            for app_name, recent_sessions in current_usage.items():
                if app_name not in self.usage_stats:
                    continue  # Unknown app - skip for now
                
                app_stats = self.usage_stats[app_name]
                app_anomalies = []
                
                # Analyze session durations
                if recent_sessions:
                    recent_durations = [s.get('duration', 0) for s in recent_sessions]
                    
                    for duration in recent_durations:
                        if app_stats['std_duration'] > 0:
                            z_score = abs(duration - app_stats['mean_duration']) / app_stats['std_duration']
                            if z_score > self.anomaly_threshold:
                                app_anomalies.append(f"unusual_duration_{app_name}")
                                anomaly_scores.append(min(z_score / self.anomaly_threshold, 1.0))
                    
                    # Analyze frequency
                    current_freq = len(recent_sessions) / (analysis_window / 24)  # sessions per day
                    expected_freq = app_stats['freq_per_day']
                    
                    if expected_freq > 0:
                        freq_ratio = current_freq / expected_freq
                        if freq_ratio > 3 or freq_ratio < 0.3:  # 3x more or 70% less
                            app_anomalies.append(f"unusual_frequency_{app_name}")
                            anomaly_scores.append(min(abs(1 - freq_ratio), 1.0))
                    
                    # Analyze timing patterns
                    for session in recent_sessions:
                        session_time = session.get('timestamp', current_time)
                        hour = datetime.fromtimestamp(session_time).hour
                        
                        if app_name in self.hourly_stats and hour in self.hourly_stats[app_name]:
                            hour_stats = self.hourly_stats[app_name][hour]
                            duration = session.get('duration', 0)
                            
                            if hour_stats['std'] > 0:
                                z_score = abs(duration - hour_stats['mean']) / hour_stats['std']
                                if z_score > self.anomaly_threshold:
                                    app_anomalies.append(f"unusual_timing_{app_name}")
                                    anomaly_scores.append(min(z_score / self.anomaly_threshold, 1.0))
                
                feature_anomalies.extend(app_anomalies)
            
            # Check for completely new apps
            for app_name in current_usage.keys():
                if app_name not in self.usage_stats:
                    feature_anomalies.append(f"new_app_{app_name}")
                    anomaly_scores.append(0.5)  # Moderate anomaly for new apps
            
            # Calculate overall anomaly score
            if anomaly_scores:
                overall_anomaly = np.mean(anomaly_scores)
            else:
                overall_anomaly = 0.0
            
            # Determine risk level
            if overall_anomaly > 0.7:
                risk_level = RiskLevel.HIGH
            elif overall_anomaly > 0.4:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Calculate confidence based on amount of historical data
            total_sessions = sum(stats['total_sessions'] for stats in self.usage_stats.values())
            confidence = min(0.9, total_sessions / (self.min_usage_sessions * 10))
            
            # Prepare features used
            features_used = ['app_duration', 'app_frequency', 'usage_timing', 'app_diversity']
            
            # Create metadata
            metadata = {
                'apps_analyzed': len(current_usage),
                'known_apps': len(self.usage_stats),
                'anomaly_details': feature_anomalies,
                'analysis_window_hours': analysis_window,
                'total_historical_sessions': total_sessions
            }
            
            processing_time = (time.time() - start_time) * 1000
            
            return AgentResult(
                agent_name=self.agent_name,
                anomaly_score=float(overall_anomaly),
                risk_level=risk_level,
                confidence=float(confidence),
                features_used=features_used,
                processing_time_ms=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            return self._create_error_result(f"Analysis error: {str(e)}", start_time)
    
    def update_model(self, new_data: Dict[str, Any]) -> bool:
        """
        Update the model with new usage data
        
        Args:
            new_data: Dictionary containing new usage events
            
        Returns:
            True if update successful
        """
        try:
            usage_events = new_data.get('usage_events', [])
            
            # Process new usage events
            for event in usage_events:
                app_name = event.get('app_name')
                action = event.get('action')
                timestamp = event.get('timestamp', time.time())
                
                if app_name and action:
                    self.log_app_usage(app_name, action, timestamp)
            
            # Retrain if enough new data
            if len(usage_events) > 10:
                return self.train_baseline()
            
            return True
            
        except Exception as e:
            print(f"[{self.agent_name}] Update error: {e}")
            return False
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of current usage patterns"""
        return {
            'total_apps_tracked': len(self.usage_stats),
            'total_sessions': sum(len(sessions) for sessions in self.app_usage_history.values()),
            'active_sessions': len(self.current_sessions),
            'most_used_apps': sorted(
                self.app_frequency.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            'is_trained': self.is_trained,
            'last_update_time': self.last_update_time
        }
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to file"""
        try:
            model_data = {
                'usage_stats': self.usage_stats,
                'hourly_stats': self.hourly_stats,
                'app_usage_history': dict(self.app_usage_history),
                'app_frequency': dict(self.app_frequency),
                'is_trained': self.is_trained,
                'last_update_time': self.last_update_time,
                'config': {
                    'history_window_days': self.history_window_days,
                    'min_usage_sessions': self.min_usage_sessions,
                    'anomaly_threshold': self.anomaly_threshold
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(model_data, f, default=str)
            
            print(f"[{self.agent_name}] Model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"[{self.agent_name}] Save error: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from file"""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.usage_stats = model_data['usage_stats']
            self.hourly_stats = model_data['hourly_stats']
            self.app_usage_history = defaultdict(list, model_data['app_usage_history'])
            self.app_frequency = defaultdict(int, model_data['app_frequency'])
            self.is_trained = model_data['is_trained']
            self.last_update_time = model_data['last_update_time']
            
            print(f"[{self.agent_name}] Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"[{self.agent_name}] Load error: {e}")
            return False
    
    def _create_error_result(self, error_message: str, start_time: float) -> AgentResult:
        """Create an error result"""
        processing_time = (time.time() - start_time) * 1000
        return AgentResult(
            agent_name=self.agent_name,
            anomaly_score=1.0,
            risk_level=RiskLevel.HIGH,
            confidence=0.0,
            features_used=[],
            processing_time_ms=processing_time,
            metadata={'error': error_message}
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model state"""
        return {
            'agent_name': self.agent_name,
            'is_trained': self.is_trained,
            'apps_tracked': len(self.usage_stats),
            'total_sessions': sum(len(sessions) for sessions in self.app_usage_history.values()),
            'last_update_time': self.last_update_time,
            'model_parameters': {
                'history_window_days': self.history_window_days,
                'min_usage_sessions': self.min_usage_sessions,
                'anomaly_threshold': self.anomaly_threshold
            }
        }
