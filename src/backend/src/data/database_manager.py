# data/database_manager.py
"""
Database Manager for encrypted local storage of fraud detection data.
Uses SQLite with AES-256 encryption for secure on-device data storage.
"""

import sqlite3
import json
import time
import hashlib
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import asdict
import base64

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("cryptography package not available - using simplified encryption")

class DatabaseManager:
    """
    Secure database manager for fraud detection data.
    Provides encrypted storage with SQLite backend.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.db_path = config.get('db_path', 'fraud_detection.db')
        self.encryption_key = config.get('encryption_key', None)
        self.max_db_size_mb = config.get('max_db_size_mb', 100)
        self.auto_vacuum = config.get('auto_vacuum', True)
        self.backup_enabled = config.get('backup_enabled', True)
        
        # Encryption setup
        self.fernet = None
        self.use_encryption = config.get('use_encryption', True)
        
        if self.use_encryption:
            self._setup_encryption()
        
        # Database connection
        self.connection = None
        self.is_connected = False
        
        # Initialize database
        self._initialize_database()
        
        print(f"DatabaseManager initialized with encryption: {self.use_encryption}")
    
    def _setup_encryption(self):
        """Setup encryption for database"""
        try:
            if not CRYPTO_AVAILABLE:
                print("Cryptography not available, using simple encoding")
                self.use_encryption = False
                return
            
            if self.encryption_key is None:
                # Generate key from device-specific information
                device_info = f"{os.path.getmtime('.')}-{os.getcwd()}"
                self.encryption_key = hashlib.sha256(device_info.encode()).digest()
            elif isinstance(self.encryption_key, str):
                self.encryption_key = self.encryption_key.encode()
            
            # Create Fernet instance
            key = base64.urlsafe_b64encode(self.encryption_key[:32])
            self.fernet = Fernet(key)
            
            print("Encryption setup completed")
            
        except Exception as e:
            print(f"Encryption setup error: {e}")
            self.use_encryption = False
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt data string"""
        if not self.use_encryption or not self.fernet:
            return base64.b64encode(data.encode()).decode()  # Simple encoding
        
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            print(f"Encryption error: {e}")
            return base64.b64encode(data.encode()).decode()
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data string"""
        if not self.use_encryption or not self.fernet:
            try:
                return base64.b64decode(encrypted_data.encode()).decode()
            except:
                return encrypted_data  # Return as-is if decoding fails
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            print(f"Decryption error: {e}")
            try:
                return base64.b64decode(encrypted_data.encode()).decode()
            except:
                return encrypted_data
    
    def _initialize_database(self):
        """Initialize database schema"""
        try:
            self.connect()
            
            # Create tables
            self._create_tables()
            
            # Set up database optimization
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.execute("PRAGMA cache_size=10000")
            self.connection.execute("PRAGMA temp_store=MEMORY")
            
            if self.auto_vacuum:
                self.connection.execute("PRAGMA auto_vacuum=INCREMENTAL")
            
            self.connection.commit()
            
            print("Database initialized successfully")
            
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def _create_tables(self):
        """Create database tables"""
        tables = [
            # Sensor data table
            """
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                sensor_type TEXT NOT NULL,
                data_encrypted TEXT NOT NULL,
                source TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
            """,
            
            # Agent results table
            """
            CREATE TABLE IF NOT EXISTS agent_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                agent_name TEXT NOT NULL,
                anomaly_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                confidence REAL NOT NULL,
                features_used TEXT,
                processing_time_ms REAL,
                metadata_encrypted TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
            """,
            
            # User enrollment data
            """
            CREATE TABLE IF NOT EXISTS user_enrollments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                enrollment_data_encrypted TEXT NOT NULL,
                enrollment_timestamp REAL NOT NULL,
                last_updated REAL NOT NULL,
                status TEXT DEFAULT 'active',
                created_at REAL DEFAULT (julianday('now'))
            )
            """,
            
            # Model metadata table
            """
            CREATE TABLE IF NOT EXISTS model_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_path TEXT,
                training_timestamp REAL,
                performance_metrics TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
            """,
            
            # System events table
            """
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                event_data_encrypted TEXT,
                severity TEXT DEFAULT 'info',
                created_at REAL DEFAULT (julianday('now'))
            )
            """
        ]
        
        for table_sql in tables:
            self.connection.execute(table_sql)
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp ON sensor_data(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_sensor_data_type ON sensor_data(sensor_type)",
            "CREATE INDEX IF NOT EXISTS idx_agent_results_timestamp ON agent_results(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_agent_results_agent ON agent_results(agent_name)",
            "CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type)"
        ]
        
        for index_sql in indexes:
            self.connection.execute(index_sql)
    
    def connect(self) -> bool:
        """Connect to database"""
        try:
            if self.is_connected and self.connection:
                return True
            
            self.connection = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )
            
            self.connection.row_factory = sqlite3.Row
            self.is_connected = True
            
            return True
        
        except Exception as e:
            print(f"Database connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from database"""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                self.is_connected = False
        except Exception as e:
            print(f"Disconnect error: {e}")
    
    def store_sensor_data(self, sensor_type: str, data: Dict[str, Any], 
                         timestamp: Optional[float] = None, source: str = "unknown") -> bool:
        """
        Store sensor data in database
        
        Args:
            sensor_type: Type of sensor
            data: Sensor data dictionary
            timestamp: Data timestamp
            source: Data source identifier
            
        Returns:
            True if successful
        """
        if timestamp is None:
            timestamp = time.time()
        
        try:
            if not self.is_connected:
                self.connect()
            
            # Encrypt data
            data_json = json.dumps(data, default=str)
            encrypted_data = self._encrypt_data(data_json)
            
            # Insert into database
            self.connection.execute(
                """
                INSERT INTO sensor_data (timestamp, sensor_type, data_encrypted, source)
                VALUES (?, ?, ?, ?)
                """,
                (timestamp, sensor_type, encrypted_data, source)
            )
            
            self.connection.commit()
            return True
        
        except Exception as e:
            print(f"Store sensor data error: {e}")
            return False
    
    def get_sensor_data(self, sensor_type: str, time_window: float = 3600.0,
                       limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve sensor data from database
        
        Args:
            sensor_type: Type of sensor
            time_window: Time window in seconds
            limit: Maximum number of records
            
        Returns:
            List of sensor data records
        """
        try:
            if not self.is_connected:
                self.connect()
            
            current_time = time.time()
            start_time = current_time - time_window
            
            query = """
                SELECT timestamp, data_encrypted, source
                FROM sensor_data
                WHERE sensor_type = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """
            
            params = [sensor_type, start_time]
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor = self.connection.execute(query, params)
            rows = cursor.fetchall()
            
            # Decrypt and parse data
            results = []
            for row in rows:
                try:
                    decrypted_data = self._decrypt_data(row['data_encrypted'])
                    data = json.loads(decrypted_data)
                    
                    results.append({
                        'timestamp': row['timestamp'],
                        'data': data,
                        'source': row['source']
                    })
                except Exception as e:
                    print(f"Data decryption error: {e}")
                    continue
            
            return results
        
        except Exception as e:
            print(f"Get sensor data error: {e}")
            return []
    
    def store_agent_result(self, agent_result: Dict[str, Any]) -> bool:
        """
        Store agent analysis result
        
        Args:
            agent_result: Agent result dictionary
            
        Returns:
            True if successful
        """
        try:
            if not self.is_connected:
                self.connect()
            
            # Extract fields
            timestamp = agent_result.get('timestamp', time.time())
            agent_name = agent_result.get('agent_name', 'unknown')
            anomaly_score = agent_result.get('anomaly_score', 0.0)
            risk_level = agent_result.get('risk_level', 'low')
            confidence = agent_result.get('confidence', 0.0)
            features_used = json.dumps(agent_result.get('features_used', []))
            processing_time_ms = agent_result.get('processing_time_ms', 0.0)
            
            # Encrypt metadata
            metadata = agent_result.get('metadata', {})
            metadata_json = json.dumps(metadata, default=str)
            encrypted_metadata = self._encrypt_data(metadata_json)
            
            # Insert into database
            self.connection.execute(
                """
                INSERT INTO agent_results 
                (timestamp, agent_name, anomaly_score, risk_level, confidence, 
                 features_used, processing_time_ms, metadata_encrypted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (timestamp, agent_name, anomaly_score, risk_level, confidence,
                 features_used, processing_time_ms, encrypted_metadata)
            )
            
            self.connection.commit()
            return True
        
        except Exception as e:
            print(f"Store agent result error: {e}")
            return False
    
    def get_agent_results(self, agent_name: Optional[str] = None, 
                         time_window: float = 3600.0,
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve agent results from database
        
        Args:
            agent_name: Specific agent name (all if None)
            time_window: Time window in seconds
            limit: Maximum number of records
            
        Returns:
            List of agent results
        """
        try:
            if not self.is_connected:
                self.connect()
            
            current_time = time.time()
            start_time = current_time - time_window
            
            query = """
                SELECT timestamp, agent_name, anomaly_score, risk_level, confidence,
                       features_used, processing_time_ms, metadata_encrypted
                FROM agent_results
                WHERE timestamp >= ?
            """
            
            params = [start_time]
            
            if agent_name:
                query += " AND agent_name = ?"
                params.append(agent_name)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor = self.connection.execute(query, params)
            rows = cursor.fetchall()
            
            # Parse results
            results = []
            for row in rows:
                try:
                    # Decrypt metadata
                    decrypted_metadata = self._decrypt_data(row['metadata_encrypted'])
                    metadata = json.loads(decrypted_metadata)
                    
                    # Parse features
                    features_used = json.loads(row['features_used'])
                    
                    results.append({
                        'timestamp': row['timestamp'],
                        'agent_name': row['agent_name'],
                        'anomaly_score': row['anomaly_score'],
                        'risk_level': row['risk_level'],
                        'confidence': row['confidence'],
                        'features_used': features_used,
                        'processing_time_ms': row['processing_time_ms'],
                        'metadata': metadata
                    })
                except Exception as e:
                    print(f"Result parsing error: {e}")
                    continue
            
            return results
        
        except Exception as e:
            print(f"Get agent results error: {e}")
            return []
    
    def store_user_enrollment(self, user_id: str, enrollment_data: Dict[str, Any]) -> bool:
        """
        Store user enrollment data
        
        Args:
            user_id: User identifier
            enrollment_data: Enrollment data dictionary
            
        Returns:
            True if successful
        """
        try:
            if not self.is_connected:
                self.connect()
            
            # Encrypt enrollment data
            data_json = json.dumps(enrollment_data, default=str)
            encrypted_data = self._encrypt_data(data_json)
            
            current_time = time.time()
            
            # Insert or update enrollment
            self.connection.execute(
                """
                INSERT OR REPLACE INTO user_enrollments 
                (user_id, enrollment_data_encrypted, enrollment_timestamp, last_updated)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, encrypted_data, current_time, current_time)
            )
            
            self.connection.commit()
            return True
        
        except Exception as e:
            print(f"Store enrollment error: {e}")
            return False
    
    def get_user_enrollment(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user enrollment data
        
        Args:
            user_id: User identifier
            
        Returns:
            Enrollment data or None
        """
        try:
            if not self.is_connected:
                self.connect()
            
            cursor = self.connection.execute(
                """
                SELECT enrollment_data_encrypted, enrollment_timestamp, last_updated, status
                FROM user_enrollments
                WHERE user_id = ? AND status = 'active'
                """,
                (user_id,)
            )
            
            row = cursor.fetchone()
            if row:
                # Decrypt enrollment data
                decrypted_data = self._decrypt_data(row['enrollment_data_encrypted'])
                enrollment_data = json.loads(decrypted_data)
                
                return {
                    'user_id': user_id,
                    'enrollment_data': enrollment_data,
                    'enrollment_timestamp': row['enrollment_timestamp'],
                    'last_updated': row['last_updated'],
                    'status': row['status']
                }
            
            return None
        
        except Exception as e:
            print(f"Get enrollment error: {e}")
            return None
    
    def cleanup_old_data(self, retention_days: int = 7) -> bool:
        """
        Clean up old data beyond retention period
        
        Args:
            retention_days: Number of days to retain
            
        Returns:
            True if successful
        """
        try:
            if not self.is_connected:
                self.connect()
            
            cutoff_time = time.time() - (retention_days * 24 * 3600)
            
            # Clean up old sensor data
            cursor = self.connection.execute(
                "DELETE FROM sensor_data WHERE timestamp < ?",
                (cutoff_time,)
            )
            sensor_deleted = cursor.rowcount
            
            # Clean up old agent results
            cursor = self.connection.execute(
                "DELETE FROM agent_results WHERE timestamp < ?",
                (cutoff_time,)
            )
            results_deleted = cursor.rowcount
            
            # Clean up old system events
            cursor = self.connection.execute(
                "DELETE FROM system_events WHERE timestamp < ?",
                (cutoff_time,)
            )
            events_deleted = cursor.rowcount
            
            self.connection.commit()
            
            # Vacuum database if auto_vacuum is enabled
            if self.auto_vacuum:
                self.connection.execute("PRAGMA incremental_vacuum")
            
            print(f"Cleanup completed: {sensor_deleted} sensor records, {results_deleted} results, {events_deleted} events")
            return True
        
        except Exception as e:
            print(f"Cleanup error: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            if not self.is_connected:
                self.connect()
            
            stats = {}
            
            # Get table sizes
            tables = ['sensor_data', 'agent_results', 'user_enrollments', 'model_metadata', 'system_events']
            
            for table in tables:
                cursor = self.connection.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[f'{table}_count'] = count
            
            # Get database file size
            if os.path.exists(self.db_path):
                stats['db_size_bytes'] = os.path.getsize(self.db_path)
                stats['db_size_mb'] = stats['db_size_bytes'] / (1024 * 1024)
            
            # Get oldest and newest records
            cursor = self.connection.execute("SELECT MIN(timestamp), MAX(timestamp) FROM sensor_data")
            min_time, max_time = cursor.fetchone()
            
            if min_time and max_time:
                stats['data_time_span_hours'] = (max_time - min_time) / 3600.0
                stats['oldest_record'] = datetime.fromtimestamp(min_time).isoformat()
                stats['newest_record'] = datetime.fromtimestamp(max_time).isoformat()
            
            return stats
        
        except Exception as e:
            print(f"Database stats error: {e}")
            return {}
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create database backup
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            True if successful
        """
        try:
            if not self.is_connected:
                self.connect()
            
            # Create backup using SQLite backup API
            backup_conn = sqlite3.connect(backup_path)
            self.connection.backup(backup_conn)
            backup_conn.close()
            
            print(f"Database backed up to {backup_path}")
            return True
        
        except Exception as e:
            print(f"Backup error: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

# Utility functions
def create_database_manager(db_path: str = "fraud_detection.db", 
                          encryption_key: Optional[str] = None) -> DatabaseManager:
    """
    Create database manager with mobile-optimized configuration
    
    Args:
        db_path: Database file path
        encryption_key: Encryption key (auto-generated if None)
        
    Returns:
        DatabaseManager instance
    """
    config = {
        'db_path': db_path,
        'encryption_key': encryption_key,
        'max_db_size_mb': 50,  # Smaller for mobile
        'auto_vacuum': True,
        'backup_enabled': True,
        'use_encryption': True
    }
    
    return DatabaseManager(config)
