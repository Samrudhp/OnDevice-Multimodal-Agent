# QuadFusion Mobile API Specification

## System Overview
QuadFusion is a multi-modal behavioral fraud detection system that analyzes user interactions through:
- Touch patterns (swipes, taps, pressure)
- Typing behavior (keystroke dynamics)  
- Voice patterns (speaker identification + speech-to-text)
- Visual biometrics (face/scene recognition)
- Device motion (accelerometer, gyroscope, magnetometer)
- App usage patterns

## Core API Endpoints (To Be Implemented)

### 1. Authentication & User Management
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "user_id": "string",
  "device_id": "string", 
  "biometric_enrollment": {
    "voice_samples": ["base64_audio_data"],
    "face_images": ["base64_image_data"],
    "typing_samples": [{"keystrokes": [], "timings": []}],
    "touch_samples": [{"gestures": [], "pressure_data": []}]
  }
}

Response: 
{
  "enrollment_id": "string",
  "status": "enrolled|pending|failed",
  "models_trained": ["typing", "voice", "visual", "touch"]
}
```

```http
POST /api/v1/auth/verify
Content-Type: application/json

{
  "user_id": "string",
  "session_id": "string",
  "sensor_data": {
    "touch_events": [],
    "keystroke_events": [],
    "audio_data": "base64",
    "image_data": "base64", 
    "motion_data": {},
    "app_usage": []
  }
}

Response:
{
  "verification_result": {
    "is_authentic": boolean,
    "confidence_score": float,
    "risk_level": "low|medium|high",
    "agent_scores": {
      "typing": float,
      "touch": float, 
      "voice": float,
      "visual": float,
      "movement": float
    },
    "anomaly_details": []
  }
}
```

### 2. Real-time Sensor Data Processing
```http
POST /api/v1/process/realtime
Content-Type: application/json

{
  "session_id": "string",
  "timestamp": "ISO8601",
  "sensor_data": {
    "touch_events": [
      {
        "timestamp": float,
        "x": float,
        "y": float, 
        "pressure": float,
        "touch_major": float,
        "touch_minor": float,
        "action": "down|move|up"
      }
    ],
    "keystroke_events": [
      {
        "timestamp": float,
        "key_code": int,
        "action": "down|up",
        "pressure": float
      }
    ],
    "motion_data": {
      "accelerometer": [x, y, z],
      "gyroscope": [x, y, z],
      "magnetometer": [x, y, z],
      "timestamp": float
    },
    "audio_data": "base64_encoded_audio",
    "image_data": "base64_encoded_image",
    "app_usage": [
      {
        "app_name": "string",
        "action": "open|close|switch_to",
        "timestamp": float
      }
    ]
  }
}

Response:
{
  "processing_result": {
    "anomaly_score": float,
    "risk_level": "low|medium|high", 
    "confidence": float,
    "processing_time_ms": float,
    "agent_results": {
      "TouchPatternAgent": {
        "anomaly_score": float,
        "features_analyzed": ["swipe_speed", "pressure_variance", "tremor_score"],
        "metadata": {}
      },
      "TypingBehaviorAgent": {
        "anomaly_score": float, 
        "features_analyzed": ["dwell_times", "flight_times", "rhythm"],
        "metadata": {}
      },
      "VoiceCommandAgent": {
        "anomaly_score": float,
        "features_analyzed": ["speaker_match", "speech_patterns"],
        "metadata": {"speaker_id": "string"}
      },
      "VisualAgent": {
        "anomaly_score": float,
        "features_analyzed": ["face_match", "scene_consistency"],
        "metadata": {}
      },
      "MovementAgent": {
        "anomaly_score": float,
        "features_analyzed": ["activity_level", "motion_patterns"],
        "metadata": {}
      },
      "AppUsageAgent": {
        "anomaly_score": float,
        "features_analyzed": ["usage_frequency", "timing_patterns"],
        "metadata": {}
      }
    }
  }
}
```

### 3. Batch Processing
```http
POST /api/v1/process/batch
Content-Type: application/json

{
  "session_id": "string",
  "batch_data": [
    // Array of sensor_data objects (same structure as realtime)
  ]
}

Response:
{
  "batch_results": [
    // Array of processing_result objects
  ],
  "summary": {
    "total_samples": int,
    "anomalies_detected": int,
    "avg_processing_time_ms": float,
    "performance_metrics": {}
  }
}
```

### 4. Model Management
```http
GET /api/v1/models/status
Response:
{
  "models": {
    "typing": {
      "is_trained": boolean,
      "training_samples": int,
      "last_updated": "ISO8601"
    },
    "touch": {"is_trained": boolean, ...},
    "voice": {"is_trained": boolean, ...},
    "visual": {"is_trained": boolean, ...},
    "movement": {"is_trained": boolean, ...},
    "coordinator": {"is_trained": boolean, ...}
  }
}
```

```http
POST /api/v1/models/retrain
Content-Type: application/json

{
  "model_type": "typing|touch|voice|visual|movement|all",
  "training_data": [],
  "config": {}
}
```

### 5. Configuration & Settings
```http
GET /api/v1/config
Response:
{
  "agent_weights": {
    "TouchPatternAgent": float,
    "TypingBehaviorAgent": float,
    "VoiceCommandAgent": float,
    "VisualAgent": float,
    "MovementAgent": float,
    "AppUsageAgent": float
  },
  "risk_thresholds": {
    "low": float,
    "medium": float, 
    "high": float
  },
  "processing_config": {}
}
```

```http
PUT /api/v1/config
Content-Type: application/json
// Same structure as GET response
```

## Data Models

### SensorData Format
```json
{
  "touch_events": [
    {
      "timestamp": 1692182400.123,
      "x": 540.5,
      "y": 960.0,
      "pressure": 0.8,
      "touch_major": 15.2,
      "touch_minor": 12.1,
      "action": "down"
    }
  ],
  "keystroke_events": [
    {
      "timestamp": 1692182400.456,
      "key_code": 65,
      "action": "down",
      "pressure": 0.6
    }
  ],
  "motion_data": {
    "accelerometer": [0.12, -0.05, 9.81],
    "gyroscope": [0.01, 0.02, -0.01],
    "magnetometer": [23.4, -12.1, 45.6],
    "timestamp": 1692182400.789
  },
  "audio_data": "base64_encoded_wav_data",
  "sample_rate": 16000,
  "audio_duration": 2.5,
  "image_data": "base64_encoded_jpeg",
  "camera_type": "front|rear",
  "app_usage": [
    {
      "app_name": "banking_app",
      "action": "open",
      "timestamp": 1692182400.0
    }
  ]
}
```

### ProcessingResult Format
```json
{
  "anomaly_score": 0.75,
  "risk_level": "high",
  "confidence": 0.89,
  "processing_time_ms": 45.2,
  "features_used": ["dwell_times", "swipe_speed", "speaker_match"],
  "metadata": {
    "models_used": 6,
    "total_features_analyzed": 156,
    "performance_alerts": []
  }
}
```

## WebSocket Real-time Stream
```javascript
// WebSocket connection for real-time processing
ws://localhost:8000/api/v1/stream/{session_id}

// Client sends:
{
  "type": "sensor_data",
  "data": {
    // SensorData format
  }
}

// Server responds:
{
  "type": "processing_result", 
  "data": {
    // ProcessingResult format
  }
}

// Server sends alerts:
{
  "type": "fraud_alert",
  "data": {
    "alert_level": "high",
    "anomaly_score": 0.95,
    "triggered_agents": ["voice", "visual"],
    "recommended_action": "block_transaction"
  }
}
```

## Mobile App Integration Examples

### Android (Kotlin)
```kotlin
class QuadFusionSDK(private val apiBaseUrl: String) {
    
    suspend fun processRealtimeSensorData(sensorData: SensorData): ProcessingResult {
        return httpClient.post("$apiBaseUrl/api/v1/process/realtime") {
            contentType(ContentType.Application.Json)
            setBody(sensorData)
        }.body()
    }
    
    suspend fun enrollUser(userId: String, biometricData: BiometricEnrollment): EnrollmentResult {
        return httpClient.post("$apiBaseUrl/api/v1/auth/register") {
            contentType(ContentType.Application.Json) 
            setBody(mapOf(
                "user_id" to userId,
                "device_id" to getDeviceId(),
                "biometric_enrollment" to biometricData
            ))
        }.body()
    }
}
```

### iOS (Swift)
```swift
class QuadFusionSDK {
    private let apiBaseURL: String
    
    func processRealtimeSensorData(_ sensorData: SensorData) async throws -> ProcessingResult {
        let url = URL(string: "\(apiBaseURL)/api/v1/process/realtime")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(sensorData)
        
        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(ProcessingResult.self, from: data)
    }
}
```

### React Native
```javascript
class QuadFusionAPI {
  constructor(baseURL) {
    this.baseURL = baseURL;
  }
  
  async processRealtimeSensorData(sensorData) {
    const response = await fetch(`${this.baseURL}/api/v1/process/realtime`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sensor_data: sensorData })
    });
    return response.json();
  }
  
  async enrollUser(userId, biometricData) {
    const response = await fetch(`${this.baseURL}/api/v1/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        device_id: await this.getDeviceId(),
        biometric_enrollment: biometricData
      })
    });
    return response.json();
  }
}
```

### Flutter (Dart)
```dart
class QuadFusionAPI {
  final String baseURL;
  final Dio _dio = Dio();
  
  QuadFusionAPI(this.baseURL);
  
  Future<ProcessingResult> processRealtimeSensorData(SensorData sensorData) async {
    final response = await _dio.post(
      '$baseURL/api/v1/process/realtime',
      data: sensorData.toJson(),
    );
    return ProcessingResult.fromJson(response.data);
  }
  
  Future<EnrollmentResult> enrollUser(String userId, BiometricEnrollment biometricData) async {
    final response = await _dio.post(
      '$baseURL/api/v1/auth/register', 
      data: {
        'user_id': userId,
        'device_id': await getDeviceId(),
        'biometric_enrollment': biometricData.toJson(),
      },
    );
    return EnrollmentResult.fromJson(response.data);
  }
}
```

## Technical Requirements

### Performance Requirements
- Real-time processing: < 100ms latency
- Batch processing: < 50ms per sample  
- Memory usage: < 200MB on mobile devices
- Model sizes: < 50MB total for all agents
- Network: Works on 3G+ connections

### Security Requirements
- TLS 1.3 encryption for all API calls
- API key authentication + JWT tokens
- Biometric data encrypted at rest and in transit
- No raw biometric data stored on servers
- GDPR/CCPA compliant data handling

### Platform Support
- Android 8.0+ (API 26+)
- iOS 13.0+
- React Native 0.68+
- Flutter 3.0+

## Error Handling
```json
{
  "error": {
    "code": "PROCESSING_FAILED",
    "message": "Touch pattern analysis failed",
    "details": {
      "agent": "TouchPatternAgent",
      "reason": "Insufficient touch data"
    },
    "timestamp": "2025-08-16T10:30:00Z"
  }
}
```

## Rate Limits
- Real-time processing: 100 requests/minute per user
- Batch processing: 10 requests/minute per user
- Model retraining: 1 request/hour per user
- Enrollment: 5 requests/day per device

This specification provides everything needed to build a mobile app that integrates with QuadFusion for behavioral fraud detection.
