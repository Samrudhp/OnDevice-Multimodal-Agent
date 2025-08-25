#!/usr/bin/env python3
"""
QuadFusion FastAPI Server
========================

REST API server for QuadFusion behavioral fraud detection system.
Provides endpoints for real-time and batch sensor data processing,
user enrollment, model management, and configuration.

Usage:
    # From the src/backend directory:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

API docs: http://localhost:8000/docs
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# QuadFusion imports (with fallbacks if not available)
try:
    from src.agents.base_agent import BaseAgent, AgentResult, RiskLevel
    from src.agents.touch_pattern_agent import TouchPatternAgent
    from src.agents.typing_behavior_agent import TypingBehaviorAgent
    from src.agents.voice_command_agent import VoiceCommandAgent
    from src.agents.visual_agent import VisualAgent
    from src.agents.movement_agent import MovementAgent
    from src.agents.app_usage_agent import AppUsageAgent
    from src.agents.coordinator_agent import CoordinatorAgent
    from src.demo.simulate_sensor_data import SensorDataSimulator
    from src.training.dataset_loaders import SyntheticDataGenerator
    QUADFUSION_AVAILABLE = True
except ImportError:
    QUADFUSION_AVAILABLE = False
    logging.warning("QuadFusion modules not available - using stubs")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="QuadFusion API",
    description="Behavioral fraud detection through multi-modal biometric analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class TouchEvent(BaseModel):
    timestamp: float
    x: float
    y: float
    pressure: float
    touch_major: float = 0.0
    touch_minor: float = 0.0
    action: str = Field(..., pattern="^(down|move|up)$")

class KeystrokeEvent(BaseModel):
    timestamp: float
    key_code: int
    action: str = Field(..., pattern="^(down|up)$")
    pressure: float = 0.0

class MotionData(BaseModel):
    accelerometer: List[float] = Field(..., min_items=3, max_items=3)
    gyroscope: List[float] = Field(..., min_items=3, max_items=3)
    magnetometer: List[float] = Field(..., min_items=3, max_items=3)
    timestamp: float

class AppUsageEvent(BaseModel):
    app_name: str
    action: str = Field(..., pattern="^(open|close|switch_to)$")
    timestamp: float

class SensorData(BaseModel):
    touch_events: List[TouchEvent] = []
    keystroke_events: List[KeystrokeEvent] = []
    motion_data: Optional[MotionData] = None
    audio_data: Optional[str] = None  # base64 encoded
    sample_rate: int = 16000
    audio_duration: float = 0.0
    image_data: Optional[str] = None  # base64 encoded
    camera_type: str = "unknown"
    app_usage: List[AppUsageEvent] = []

class BiometricEnrollment(BaseModel):
    voice_samples: List[str] = []  # base64 audio data
    face_images: List[str] = []    # base64 image data
    typing_samples: List[Dict[str, Any]] = []
    touch_samples: List[Dict[str, Any]] = []

class UserRegistrationRequest(BaseModel):
    user_id: str
    device_id: str
    biometric_enrollment: BiometricEnrollment

class RealtimeProcessingRequest(BaseModel):
    session_id: str
    timestamp: Optional[str] = None
    sensor_data: SensorData

class BatchProcessingRequest(BaseModel):
    session_id: str
    batch_data: List[SensorData]

class ModelRetrainRequest(BaseModel):
    model_type: str = Field(..., pattern="^(typing|touch|voice|visual|movement|all)$")
    training_data: List[Dict[str, Any]] = []
    config: Dict[str, Any] = {}

class ConfigurationUpdate(BaseModel):
    agent_weights: Optional[Dict[str, float]] = None
    risk_thresholds: Optional[Dict[str, float]] = None
    processing_config: Optional[Dict[str, Any]] = None

class AgentResultResponse(BaseModel):
    anomaly_score: float
    risk_level: str
    confidence: float
    features_analyzed: List[str]
    processing_time_ms: float
    metadata: Dict[str, Any]

class ProcessingResult(BaseModel):
    anomaly_score: float
    risk_level: str
    confidence: float
    processing_time_ms: float
    agent_results: Dict[str, AgentResultResponse]
    metadata: Dict[str, Any]

class EnrollmentResult(BaseModel):
    enrollment_id: str
    status: str
    models_trained: List[str]
    message: str

# ============================================================================
# Global State and Agent Management
# ============================================================================

class AgentManager:
    """Manages all fraud detection agents and their states"""
    
    def __init__(self):
        self.agents = {}
        self.coordinator = None
        self.synthetic_data_generator = None
        self.sessions = {}  # session_id -> user_data
        self.config = {
            "agent_weights": {
                "TouchPatternAgent": 0.2,
                "TypingBehaviorAgent": 0.15,
                "VoiceCommandAgent": 0.2,
                "VisualAgent": 0.25,
                "AppUsageAgent": 0.1,
                "MovementAgent": 0.1
            },
            "risk_thresholds": {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.8
            }
        }
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all fraud detection agents"""
        try:
            if QUADFUSION_AVAILABLE:
                # Initialize agents with default configs
                default_config = {"contamination": 0.1, "batch_size": 32}
                
                self.agents = {
                    "TouchPatternAgent": TouchPatternAgent(default_config),
                    "TypingBehaviorAgent": TypingBehaviorAgent(default_config),
                    "VoiceCommandAgent": VoiceCommandAgent(default_config),
                    "VisualAgent": VisualAgent(default_config),
                    "MovementAgent": MovementAgent(default_config),
                    "AppUsageAgent": AppUsageAgent(default_config)
                }
                
                # Initialize coordinator
                coordinator_config = {
                    "agent_weights": self.config["agent_weights"],
                    "risk_thresholds": self.config["risk_thresholds"]
                }
                self.coordinator = CoordinatorAgent(coordinator_config)
                
                # Initialize synthetic data generator
                self.synthetic_data_generator = SyntheticDataGenerator()
                
                logger.info("All agents initialized successfully")
            else:
                logger.warning("QuadFusion not available - using stub agents")
                self._initialize_stub_agents()
                
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            self._initialize_stub_agents()
    
    def _initialize_stub_agents(self):
        """Initialize stub agents for testing when QuadFusion is not available"""
        class StubAgent:
            def __init__(self, name):
                self.agent_name = name
                self.is_trained = False
            
            def process_sensor_data(self, sensor_data):
                return {
                    "agent_name": self.agent_name,
                    "anomaly_score": np.random.random() * 0.5,  # Random low score
                    "risk_level": "low",
                    "confidence": np.random.random() * 0.8 + 0.2,
                    "features_used": ["stub_feature"],
                    "processing_time_ms": np.random.random() * 50,
                    "metadata": {"stub": True}
                }
        
        self.agents = {
            "TouchPatternAgent": StubAgent("TouchPatternAgent"),
            "TypingBehaviorAgent": StubAgent("TypingBehaviorAgent"),
            "VoiceCommandAgent": StubAgent("VoiceCommandAgent"),
            "VisualAgent": StubAgent("VisualAgent"),
            "MovementAgent": StubAgent("MovementAgent"),
            "AppUsageAgent": StubAgent("AppUsageAgent")
        }
    
    def process_sensor_data(self, sensor_data: SensorData, session_id: str) -> ProcessingResult:
        """Process sensor data through all agents and coordinator"""
        start_time = time.time()
        
        # Convert pydantic model to dict format expected by agents
        sensor_dict = self._convert_sensor_data(sensor_data)
        
        # Process through individual agents
        agent_results = {}
        for agent_name, agent in self.agents.items():
            try:
                result = agent.process_sensor_data(sensor_dict)
                if result:
                    if hasattr(result, '__dict__'):
                        # Convert AgentResult object to dict
                        agent_results[agent_name] = {
                            "anomaly_score": result.anomaly_score,
                            "risk_level": result.risk_level.value if hasattr(result.risk_level, 'value') else str(result.risk_level),
                            "confidence": result.confidence,
                            "features_analyzed": result.features_used,
                            "processing_time_ms": result.processing_time_ms,
                            "metadata": result.metadata
                        }
                    else:
                        # Handle stub agents that return dicts
                        agent_results[agent_name] = result
                        agent_results[agent_name]["features_analyzed"] = result.get("features_used", [])
            except Exception as e:
                logger.error(f"Agent {agent_name} processing failed: {e}")
                # Add error result
                agent_results[agent_name] = {
                    "anomaly_score": 1.0,
                    "risk_level": "high",
                    "confidence": 0.0,
                    "features_analyzed": [],
                    "processing_time_ms": 0.0,
                    "metadata": {"error": str(e)}
                }
        
        # Coordinate results if coordinator available
        if self.coordinator and len(agent_results) > 0:
            try:
                coord_result = self.coordinator.analyze({"agent_results": agent_results})
                if hasattr(coord_result, '__dict__'):
                    final_result = {
                        "anomaly_score": coord_result.anomaly_score,
                        "risk_level": coord_result.risk_level.value if hasattr(coord_result.risk_level, 'value') else str(coord_result.risk_level),
                        "confidence": coord_result.confidence,
                        "processing_time_ms": (time.time() - start_time) * 1000,
                        "agent_results": {name: AgentResultResponse(**result) for name, result in agent_results.items()},
                        "metadata": coord_result.metadata
                    }
                else:
                    # Fallback coordination
                    final_result = self._simple_coordination(agent_results, start_time)
            except Exception as e:
                logger.error(f"Coordinator processing failed: {e}")
                final_result = self._simple_coordination(agent_results, start_time)
        else:
            final_result = self._simple_coordination(agent_results, start_time)
        
        return ProcessingResult(**final_result)
    
    def _convert_sensor_data(self, sensor_data: SensorData) -> Dict[str, Any]:
        """Convert Pydantic sensor data to format expected by agents"""
        result = {}
        
        # Touch events
        if sensor_data.touch_events:
            result["touch_events"] = [event.dict() for event in sensor_data.touch_events]
        
        # Keystroke events
        if sensor_data.keystroke_events:
            result["keystroke_events"] = [event.dict() for event in sensor_data.keystroke_events]
        
        # Motion data
        if sensor_data.motion_data:
            result["motion_sequence"] = {
                "accelerometer": sensor_data.motion_data.accelerometer,
                "gyroscope": sensor_data.motion_data.gyroscope,
                "magnetometer": sensor_data.motion_data.magnetometer,
                "timestamp": sensor_data.motion_data.timestamp
            }
        
        # Audio data
        if sensor_data.audio_data:
            try:
                # Decode base64 audio
                audio_bytes = base64.b64decode(sensor_data.audio_data)
                # Convert to numpy array (assuming WAV format)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                result["audio_data"] = audio_array
                result["sample_rate"] = sensor_data.sample_rate
            except Exception as e:
                logger.error(f"Audio decoding failed: {e}")
        
        # Image data
        if sensor_data.image_data:
            result["image_base64"] = sensor_data.image_data
            result["camera_type"] = sensor_data.camera_type
        
        # App usage
        if sensor_data.app_usage:
            result["current_usage"] = {
                event.app_name: [{"timestamp": event.timestamp, "action": event.action}]
                for event in sensor_data.app_usage
            }
        
        return result
    
    def _simple_coordination(self, agent_results: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Simple coordination when coordinator is not available"""
        if not agent_results:
            return {
                "anomaly_score": 0.0,
                "risk_level": "low",
                "confidence": 0.0,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "agent_results": {},
                "metadata": {"coordination": "simple_average", "agents_used": 0}
            }
        
        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0
        total_confidence = 0.0
        
        for agent_name, result in agent_results.items():
            weight = self.config["agent_weights"].get(agent_name, 0.1)
            total_score += result["anomaly_score"] * weight
            total_weight += weight
            total_confidence += result["confidence"]
        
        avg_score = total_score / max(total_weight, 0.01)
        avg_confidence = total_confidence / len(agent_results)
        
        # Determine risk level
        if avg_score >= self.config["risk_thresholds"]["high"]:
            risk_level = "high"
        elif avg_score >= self.config["risk_thresholds"]["medium"]:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "anomaly_score": avg_score,
            "risk_level": risk_level,
            "confidence": avg_confidence,
            "processing_time_ms": (time.time() - start_time) * 1000,
            "agent_results": {name: AgentResultResponse(**result) for name, result in agent_results.items()},
            "metadata": {
                "coordination": "simple_average",
                "agents_used": len(agent_results),
                "total_weight": total_weight
            }
        }
    
    def enroll_user(self, request: UserRegistrationRequest) -> EnrollmentResult:
        """Enroll a new user with biometric data"""
        enrollment_id = str(uuid.uuid4())
        
        try:
            # Store user session
            self.sessions[request.user_id] = {
                "device_id": request.device_id,
                "enrollment_id": enrollment_id,
                "enrolled_at": datetime.now().isoformat(),
                "models_trained": []
            }
            
            # Train agents with provided biometric data
            models_trained = []
            
            # TODO: Implement actual training with biometric data
            # This is a simplified version for demo purposes
            if request.biometric_enrollment.voice_samples:
                models_trained.append("voice")
            if request.biometric_enrollment.face_images:
                models_trained.append("visual")
            if request.biometric_enrollment.typing_samples:
                models_trained.append("typing")
            if request.biometric_enrollment.touch_samples:
                models_trained.append("touch")
            
            self.sessions[request.user_id]["models_trained"] = models_trained
            
            return EnrollmentResult(
                enrollment_id=enrollment_id,
                status="enrolled",
                models_trained=models_trained,
                message=f"Successfully enrolled user {request.user_id} with {len(models_trained)} models"
            )
            
        except Exception as e:
            logger.error(f"User enrollment failed: {e}")
            return EnrollmentResult(
                enrollment_id=enrollment_id,
                status="failed",
                models_trained=[],
                message=f"Enrollment failed: {str(e)}"
            )
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all trained models"""
        status = {}
        
        for agent_name, agent in self.agents.items():
            status[agent_name.lower().replace("agent", "")] = {
                "is_trained": getattr(agent, 'is_trained', False),
                "training_samples": 0,  # TODO: Get actual training sample count
                "last_updated": datetime.now().isoformat()
            }
        
        return {"models": status}
    
    def update_configuration(self, config: ConfigurationUpdate) -> Dict[str, Any]:
        """Update system configuration"""
        updated = {}
        
        if config.agent_weights:
            self.config["agent_weights"].update(config.agent_weights)
            updated["agent_weights"] = True
        
        if config.risk_thresholds:
            self.config["risk_thresholds"].update(config.risk_thresholds)
            updated["risk_thresholds"] = True
        
        if config.processing_config:
            # TODO: Update agent processing configurations
            updated["processing_config"] = True
        
        return {
            "status": "updated",
            "updated_fields": updated,
            "current_config": self.config
        }

# Initialize global agent manager
agent_manager = AgentManager()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.session_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.session_connections[session_id] = websocket

    def disconnect(self, websocket: WebSocket, session_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if session_id in self.session_connections:
            del self.session_connections[session_id]

    async def send_personal_message(self, message: dict, session_id: str):
        websocket = self.session_connections.get(session_id)
        if websocket:
            await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

websocket_manager = ConnectionManager()

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "QuadFusion API",
        "version": "1.0.0",
        "description": "Behavioral fraud detection through multi-modal biometric analysis",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "auth": "/api/v1/auth/*",
            "processing": "/api/v1/process/*",
            "models": "/api/v1/models/*",
            "config": "/api/v1/config",
            "stream": "/api/v1/stream/{session_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "quadfusion_available": QUADFUSION_AVAILABLE,
        "agents_initialized": len(agent_manager.agents),
        "active_sessions": len(agent_manager.sessions)
    }

# Authentication & User Management Endpoints
@app.post("/api/v1/auth/register", response_model=EnrollmentResult)
async def register_user(request: UserRegistrationRequest):
    """Register a new user with biometric enrollment"""
    try:
        result = agent_manager.enroll_user(request)
        return result
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/auth/verify")
async def verify_user(request: RealtimeProcessingRequest):
    """Verify user authenticity through behavioral analysis"""
    try:
        result = agent_manager.process_sensor_data(request.sensor_data, request.session_id)
        
        # Determine if user is authentic based on results
        is_authentic = result.anomaly_score < 0.5 and result.risk_level in ["low", "medium"]
        
        return {
            "verification_result": {
                "is_authentic": is_authentic,
                "confidence_score": result.confidence,
                "risk_level": result.risk_level,
                "agent_scores": {
                    name: details.anomaly_score 
                    for name, details in result.agent_results.items()
                },
                "anomaly_details": [
                    f"{name}: {details.anomaly_score:.3f}"
                    for name, details in result.agent_results.items()
                    if details.anomaly_score > 0.5
                ]
            }
        }
    except Exception as e:
        logger.error(f"User verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Real-time Processing Endpoints
@app.post("/api/v1/process/realtime", response_model=ProcessingResult)
async def process_realtime(request: RealtimeProcessingRequest):
    """Process sensor data in real-time"""
    try:
        result = agent_manager.process_sensor_data(request.sensor_data, request.session_id)
        
        # Send alert via WebSocket if high risk
        if result.risk_level == "high":
            await websocket_manager.send_personal_message({
                "type": "fraud_alert",
                "data": {
                    "alert_level": "high",
                    "anomaly_score": result.anomaly_score,
                    "triggered_agents": [
                        name for name, details in result.agent_results.items()
                        if details.anomaly_score > 0.7
                    ],
                    "recommended_action": "block_transaction"
                }
            }, request.session_id)
        
        return result
    except Exception as e:
        logger.error(f"Real-time processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/process/batch")
async def process_batch(request: BatchProcessingRequest):
    """Process multiple sensor data samples in batch"""
    try:
        results = []
        start_time = time.time()
        
        for sensor_data in request.batch_data:
            result = agent_manager.process_sensor_data(sensor_data, request.session_id)
            results.append(result)
        
        # Calculate summary statistics
        anomalies_detected = sum(1 for r in results if r.risk_level == "high")
        avg_processing_time = sum(r.processing_time_ms for r in results) / len(results)
        
        return {
            "batch_results": results,
            "summary": {
                "total_samples": len(results),
                "anomalies_detected": anomalies_detected,
                "avg_processing_time_ms": avg_processing_time,
                "performance_metrics": {
                    "total_processing_time_ms": (time.time() - start_time) * 1000,
                    "throughput_samples_per_second": len(results) / max((time.time() - start_time), 0.001)
                }
            }
        }
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model Management Endpoints
@app.get("/api/v1/models/status")
async def get_model_status():
    """Get status of all trained models"""
    try:
        return agent_manager.get_model_status()
    except Exception as e:
        logger.error(f"Getting model status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/models/retrain")
async def retrain_models(request: ModelRetrainRequest):
    """Retrain specific models with new data"""
    try:
        # TODO: Implement actual model retraining
        # This is a placeholder implementation
        
        return {
            "status": "retraining_started",
            "model_type": request.model_type,
            "training_samples": len(request.training_data),
            "estimated_completion_time": "5 minutes",
            "message": f"Retraining {request.model_type} models with {len(request.training_data)} samples"
        }
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration Endpoints
@app.get("/api/v1/config")
async def get_configuration():
    """Get current system configuration"""
    return agent_manager.config

@app.put("/api/v1/config")
async def update_configuration(config: ConfigurationUpdate):
    """Update system configuration"""
    try:
        result = agent_manager.update_configuration(config)
        return result
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket Endpoint for Real-time Streaming
@app.websocket("/api/v1/stream/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time data streaming"""
    await websocket_manager.connect(websocket, session_id)
    logger.info(f"WebSocket connected for session {session_id}")
    
    try:
        while True:
            # Receive sensor data
            data = await websocket.receive_json()
            
            if data.get("type") == "sensor_data":
                try:
                    # Parse sensor data
                    sensor_data = SensorData(**data["data"])
                    
                    # Process through agents
                    result = agent_manager.process_sensor_data(sensor_data, session_id)
                    
                    # Send result back
                    await websocket.send_json({
                        "type": "processing_result",
                        "data": result.dict()
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": str(e)}
                    })
                    
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, session_id)
        logger.info(f"WebSocket disconnected for session {session_id}")

# Utility Endpoints
@app.get("/api/v1/demo/generate-sample-data")
async def generate_sample_data():
    """Generate sample sensor data for testing"""
    try:
        if QUADFUSION_AVAILABLE and agent_manager.synthetic_data_generator:
            # Use real synthetic data generator
            sample = agent_manager.synthetic_data_generator.generate_touch(None)
            return {"sample_data": sample, "type": "real_synthetic"}
        else:
            # Generate simple stub data
            sample = {
                "touch_events": [
                    {
                        "timestamp": time.time(),
                        "x": 540.0,
                        "y": 960.0,
                        "pressure": 0.8,
                        "touch_major": 15.0,
                        "touch_minor": 12.0,
                        "action": "down"
                    }
                ],
                "keystroke_events": [],
                "motion_data": {
                    "accelerometer": [0.1, -0.05, 9.81],
                    "gyroscope": [0.01, 0.02, -0.01],
                    "magnetometer": [23.4, -12.1, 45.6],
                    "timestamp": time.time()
                }
            }
            return {"sample_data": sample, "type": "stub_synthetic"}
    except Exception as e:
        logger.error(f"Sample data generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/demo/stress-test")
async def stress_test(num_samples: int = 100):
    """Run a stress test with multiple samples"""
    try:
        results = []
        start_time = time.time()
        
        for i in range(num_samples):
            # Generate random sensor data
            sensor_data = SensorData(
                touch_events=[
                    TouchEvent(
                        timestamp=time.time(),
                        x=np.random.uniform(0, 1080),
                        y=np.random.uniform(0, 1920),
                        pressure=np.random.uniform(0.1, 1.0),
                        action="down"
                    )
                ],
                motion_data=MotionData(
                    accelerometer=[np.random.normal(0, 0.1) for _ in range(3)],
                    gyroscope=[np.random.normal(0, 0.05) for _ in range(3)],
                    magnetometer=[np.random.normal(0, 10) for _ in range(3)],
                    timestamp=time.time()
                )
            )
            
            result = agent_manager.process_sensor_data(sensor_data, f"stress_test_{i}")
            results.append({
                "sample_id": i,
                "anomaly_score": result.anomaly_score,
                "processing_time_ms": result.processing_time_ms
            })
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "stress_test_results": {
                "total_samples": num_samples,
                "total_time_ms": total_time,
                "avg_processing_time_ms": sum(r["processing_time_ms"] for r in results) / len(results),
                "throughput_samples_per_second": num_samples / (total_time / 1000),
                "anomalies_detected": sum(1 for r in results if r["anomaly_score"] > 0.5),
                "results": results[:10]  # Return first 10 results
            }
        }
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": {"code": "VALIDATION_ERROR", "message": str(exc)}}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}}
    )

# ============================================================================
# Server Startup
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QuadFusion FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    logger.info(f"Starting QuadFusion API server on {args.host}:{args.port}")
    logger.info(f"QuadFusion modules available: {QUADFUSION_AVAILABLE}")
    logger.info(f"API documentation available at http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )
