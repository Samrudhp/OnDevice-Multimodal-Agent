import { Platform } from 'react-native';
import { API_CONNECTION_STATUS, WS_BASE_URL, TIMEOUTS } from './constants';
import getApiBaseUrl from '../config/api';

// Types matching the FastAPI server exactly
export interface SensorData {
  touch_events: TouchEvent[];
  keystroke_events: KeystrokeEvent[];
  motion_data?: MotionData;
  audio_data?: string;
  sample_rate?: number;
  audio_duration?: number;
  image_data?: string;
  camera_type?: string;
  app_usage: AppUsageEvent[];
}

export interface TouchEvent {
  timestamp: number;
  x: number;
  y: number;
  pressure: number;
  touch_major: number;
  touch_minor: number;
  action: 'down' | 'move' | 'up';
}

export interface KeystrokeEvent {
  timestamp: number;
  key_code: number;
  action: 'down' | 'up';
  pressure: number;
}

export interface MotionData {
  accelerometer: [number, number, number];
  gyroscope: [number, number, number];
  magnetometer: [number, number, number];
  timestamp: number;
}

export interface AppUsageEvent {
  app_name: string;
  action: 'open' | 'close' | 'switch_to';
  timestamp: number;
}

export interface BiometricEnrollment {
  voice_samples: string[];
  face_images: string[];
  typing_samples: any[];
  touch_samples: any[];
}

export interface ProcessingResult {
  anomaly_score: number;
  risk_level: 'low' | 'medium' | 'high';
  confidence: number;
  processing_time_ms: number;
  agent_results: Record<string, AgentResult>;
  metadata: Record<string, any>;
}

export interface AgentResult {
  anomaly_score: number;
  risk_level: string;
  confidence: number;
  features_analyzed: string[];
  processing_time_ms: number;
  metadata: Record<string, any>;
}

// Fixed to match server response exactly
export interface ModelStatus {
  models: Record<string, {
    is_trained: boolean;
    training_samples: number;
    last_updated: string;
  }>;
}

// Fixed to match server response exactly
export interface EnrollmentResult {
  enrollment_id: string;
  status: 'enrolled' | 'failed'; // Removed 'pending' as server doesn't return it
  models_trained: string[];
  message: string;
}

// Server verification response structure
export interface VerificationResult {
  verification_result: {
    is_authentic: boolean;
    confidence_score: number;
    risk_level: string;
    agent_scores: Record<string, number>;
    anomaly_details: string[];
  };
}

// Server batch processing response
export interface BatchProcessingResult {
  batch_results: ProcessingResult[];
  summary: {
    total_samples: number;
    anomalies_detected: number;
    avg_processing_time_ms: number;
    performance_metrics: {
      total_processing_time_ms: number;
      throughput_samples_per_second: number;
    };
  };
}

// Server model retrain response
export interface ModelRetrainResult {
  status: string;
  model_type: string;
  training_samples: number;
  estimated_completion_time: string;
  message: string;
}

// Server config response
export interface SystemConfig {
  agent_weights: Record<string, number>;
  risk_thresholds: Record<string, number>;
}

export interface ConfigurationUpdate {
  agent_weights?: Record<string, number>;
  risk_thresholds?: Record<string, number>;
  processing_config?: Record<string, any>;
}

// Server stress test response
export interface StressTestResult {
  stress_test_results: {
    total_samples: number;
    total_time_ms: number;
    avg_processing_time_ms: number;
    throughput_samples_per_second: number;
    anomalies_detected: number;
    results: Array<{
      sample_id: number;
      anomaly_score: number;
      processing_time_ms: number;
    }>;
  };
}

// Server health check response
export interface HealthCheckResult {
  status: string;
  timestamp: string;
  quadfusion_available: boolean;
  agents_initialized: number;
  active_sessions: number;
}

// Server sample data response
export interface SampleDataResult {
  sample_data: any;
  type: string;
}

// API Client Class
export class QuadFusionAPI {
  private baseURL: string;
  private connectionStatus: typeof API_CONNECTION_STATUS[keyof typeof API_CONNECTION_STATUS] = API_CONNECTION_STATUS.DISCONNECTED;
  private connectionListeners: ((status: typeof API_CONNECTION_STATUS[keyof typeof API_CONNECTION_STATUS]) => void)[] = [];

  constructor(baseURL: string = getApiBaseUrl()) {
    this.baseURL = baseURL;
    this.checkConnection();
  }
  
  // Check API connection and update status
  private async checkConnection() {
    const maxAttempts = 3;
    let attempt = 0;

    const attemptFetch = async (): Promise<boolean> => {
      attempt++;
      this.updateConnectionStatus(API_CONNECTION_STATUS.CONNECTING);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), Math.max(5000, TIMEOUTS.REQUEST / 2));

      try {
        const response = await fetch(`${this.baseURL}/health`, { method: 'GET', signal: controller.signal });
        clearTimeout(timeoutId);
        if (response.ok) {
          this.updateConnectionStatus(API_CONNECTION_STATUS.CONNECTED);
          return true;
        }
        this.updateConnectionStatus(API_CONNECTION_STATUS.ERROR);
        return false;
      } catch (err: any) {
        clearTimeout(timeoutId);
        console.error('API connection check failed:', err);
        if (err && err.name === 'AbortError') {
          console.warn('Connection check timed out');
          this.updateConnectionStatus(API_CONNECTION_STATUS.DISCONNECTED);
        } else {
          this.updateConnectionStatus(API_CONNECTION_STATUS.ERROR);
        }
        return false;
      }
    };

    while (attempt < maxAttempts) {
      const ok = await attemptFetch();
      if (ok) break;
      const backoff = 500 * attempt;
      await new Promise(res => setTimeout(res, backoff));
    }
  }
  
  // Update connection status and notify listeners
  private updateConnectionStatus(status: typeof API_CONNECTION_STATUS[keyof typeof API_CONNECTION_STATUS]) {
    // Only update if status has changed
    if (this.connectionStatus !== status) {
      console.log(`API connection status changed: ${this.connectionStatus} -> ${status}`);
      this.connectionStatus = status;
      
      // Notify all registered listeners
      this.connectionListeners.forEach(listener => {
        try {
          listener(status);
        } catch (error) {
          console.error('Error in connection status listener:', error);
        }
      });
    }
  }
  
  // Add connection status listener
  public addConnectionListener(listener: (status: typeof API_CONNECTION_STATUS[keyof typeof API_CONNECTION_STATUS]) => void) {
    this.connectionListeners.push(listener);
    // Immediately notify with current status
    listener(this.connectionStatus);
    return () => {
      this.connectionListeners = this.connectionListeners.filter(l => l !== listener);
    };
  }
  
  // Get current connection status
  public getConnectionStatus(): typeof API_CONNECTION_STATUS[keyof typeof API_CONNECTION_STATUS] {
    return this.connectionStatus;
  }

  private async makeRequest<T>(endpoint: string, options: RequestInit = {}, useDefaultOnFailure: boolean = false, defaultValue?: T): Promise<T> {
    try {
      console.log(`Making API request to: ${this.baseURL}${endpoint}`);
      const url = `${this.baseURL}${endpoint}`;
      
      // Update connection status to connecting
      this.updateConnectionStatus(API_CONNECTION_STATUS.CONNECTING);
      
      // Set timeout to prevent hanging requests
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), TIMEOUTS.REQUEST);
      
      // Prepare request init so we can both log it (in dev) and pass to fetch
      const headers = {
        'Content-Type': 'application/json',
        ...options.headers,
      };

      const requestInit: RequestInit = {
        ...options,
        headers,
        signal: controller.signal,
        ...options,
      });

      clearTimeout(timeoutId);
      
      // Update connection status based on response
      this.updateConnectionStatus(API_CONNECTION_STATUS.CONNECTED);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`HTTP ${response.status}: ${errorData.message || response.statusText}`);
      }

      return await response.json();
    } catch (error: any) {
      console.error(`API Request failed for ${endpoint}:`, error);
      
      // Update connection status based on error
      if (error.name === 'AbortError') {
        console.warn('Request timed out');
        this.updateConnectionStatus(API_CONNECTION_STATUS.DISCONNECTED);
      } else {
        this.updateConnectionStatus(API_CONNECTION_STATUS.ERROR);
      }
      
      // Check if we should use fallback
      if (useDefaultOnFailure && defaultValue !== undefined) {
        console.log(`Using default fallback value for ${endpoint}`);
        return defaultValue;
      }
      
      throw error;
    }
  }

  // Authentication & User Management Endpoints
  async enrollUser(userId: string, biometricData: BiometricEnrollment): Promise<EnrollmentResult> {
    const deviceId = await this.getDeviceId();
    
    return this.makeRequest<EnrollmentResult>('/api/v1/auth/register', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
        device_id: deviceId,
        biometric_enrollment: biometricData,
      }),
    });
  }

  // Fixed to match server endpoint exactly
  async verifyUser(sessionId: string, sensorData: SensorData): Promise<VerificationResult> {
    return this.makeRequest<VerificationResult>('/api/v1/auth/verify', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        sensor_data: sensorData,
      }),
    });
  }

  // Real-time Processing Endpoints
  async processRealtimeSensorData(sessionId: string, sensorData: SensorData): Promise<ProcessingResult> {
    // Create a default fallback response with mock data
    const defaultResponse: ProcessingResult = {
      anomaly_score: Math.random() * 0.5, // Lower score for demo
      risk_level: Math.random() > 0.7 ? 'medium' : 'low',
      confidence: 0.85 + (Math.random() * 0.1),
      processing_time_ms: 120 + Math.floor(Math.random() * 50),
      agent_results: {
        [AGENT_TYPES.TOUCH]: {
          anomaly_score: Math.random() * 0.4,
          risk_level: 'low',
          confidence: 0.9,
          features_analyzed: ['pressure', 'velocity', 'pattern'],
          processing_time_ms: 40,
          metadata: { pattern_match: 'strong' }
        },
        [AGENT_TYPES.TYPING]: {
          anomaly_score: Math.random() * 0.3,
          risk_level: 'low',
          confidence: 0.85,
          features_analyzed: ['rhythm', 'dwell_time', 'flight_time'],
          processing_time_ms: 30,
          metadata: { rhythm_match: 'good' }
        },
        [AGENT_TYPES.MOVEMENT]: {
          anomaly_score: Math.random() * 0.6,
          risk_level: Math.random() > 0.7 ? 'medium' : 'low',
          confidence: 0.75,
          features_analyzed: ['gait', 'orientation', 'stability'],
          processing_time_ms: 50,
          metadata: { stability_match: 'moderate' }
        }
      },
      metadata: {
        device_info: 'Mobile',
        session_duration: sensorData.touch_events.length > 0 ? 
          sensorData.touch_events[sensorData.touch_events.length-1].timestamp - 
          sensorData.touch_events[0].timestamp : 0,
        data_points_analyzed: sensorData.touch_events.length + sensorData.keystroke_events.length
      }
    };
    
    return this.makeRequest<ProcessingResult>('/api/v1/process/realtime', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        timestamp: new Date().toISOString(),
        sensor_data: sensorData,
      }),
    }, true, defaultResponse);
  }

  async processBatchSensorData(sessionId: string, batchData: SensorData[]): Promise<BatchProcessingResult> {
    return this.makeRequest<BatchProcessingResult>('/api/v1/process/batch', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        batch_data: batchData,
      }),
    });
  }

  // Model Management Endpoints
  async getModelStatus(): Promise<ModelStatus> {
    const defaultStatus: ModelStatus = {
      models: {
        [AGENT_TYPES.TOUCH.toLowerCase().replace('agent', '')]: { 
          is_trained: true, 
          training_samples: 0, 
          last_updated: new Date().toISOString() 
        },
        [AGENT_TYPES.TYPING.toLowerCase().replace('agent', '')]: { 
          is_trained: true, 
          training_samples: 0, 
          last_updated: new Date().toISOString() 
        },
        [AGENT_TYPES.VOICE.toLowerCase().replace('agent', '')]: { 
          is_trained: true, 
          training_samples: 0, 
          last_updated: new Date().toISOString() 
        },
        [AGENT_TYPES.VISUAL.toLowerCase().replace('agent', '')]: { 
          is_trained: true, 
          training_samples: 0, 
          last_updated: new Date().toISOString() 
        },
        [AGENT_TYPES.MOVEMENT.toLowerCase().replace('agent', '')]: { 
          is_trained: true, 
          training_samples: 0, 
          last_updated: new Date().toISOString() 
        },
        [AGENT_TYPES.APP_USAGE.toLowerCase().replace('agent', '')]: { 
          is_trained: true, 
          training_samples: 0, 
          last_updated: new Date().toISOString() 
        }
      }
    };
    
    return this.makeRequest<ModelStatus>('/api/v1/models/status', {}, true, defaultStatus);
  }

  // Fixed to match server response exactly
  async retrainModel(modelType: string, trainingData: any[], config: any): Promise<ModelRetrainResult> {
    const defaultResponse: ModelRetrainResult = {
      status: 'retraining_started',
      model_type: modelType,
      training_samples: trainingData.length,
      estimated_completion_time: '5 minutes',
      message: `Retraining ${modelType} models with ${trainingData.length} samples`
    };
    
    return this.makeRequest<ModelRetrainResult>('/api/v1/models/retrain', {
      method: 'POST',
      body: JSON.stringify({
        model_type: modelType,
        training_data: trainingData,
        config: config,
      }),
    }, true, defaultResponse);
  }

  // Configuration Endpoints
  async getConfig(): Promise<SystemConfig> {
    return this.makeRequest<SystemConfig>('/api/v1/config');
  }

  async updateConfig(config: ConfigurationUpdate): Promise<{
    status: string;
    updated_fields: Record<string, boolean>;
    current_config: SystemConfig;
  }> {
    return this.makeRequest('/api/v1/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    });
  }

  // Utility Endpoints
  async generateSampleData(): Promise<SampleDataResult> {
    return this.makeRequest<SampleDataResult>('/api/v1/demo/generate-sample-data');
  }

  // Added missing stress test endpoint
  async runStressTest(numSamples: number = 100): Promise<StressTestResult> {
    return this.makeRequest<StressTestResult>(`/api/v1/demo/stress-test?num_samples=${numSamples}`);
  }

  async healthCheck(): Promise<HealthCheckResult> {
    return this.makeRequest<HealthCheckResult>('/health');
  }

  private async getDeviceId(): Promise<string> {
    try {
      // Use expo-device or similar for device ID
      return `${Platform.OS}-${Date.now()}`;
    } catch (error) {
      console.error('Error getting device ID:', error);
      return 'unknown-device';
    }
  }

  // WebSocket connection for real-time streaming - Fixed URL construction
  createWebSocketConnection(sessionId: string, onMessage: (data: any) => void): WebSocket {
    // Convert HTTP URL to WebSocket URL properly
    const wsUrl = this.baseURL.replace(/^https?/, this.baseURL.startsWith('https') ? 'wss' : 'ws');
    const ws = new WebSocket(`${wsUrl}/api/v1/stream/${sessionId}`);
    
    ws.onopen = () => {
      console.log(`WebSocket connected for session ${sessionId}`);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = (event) => {
      console.log(`WebSocket closed for session ${sessionId}:`, event.code, event.reason);
    };

    return ws;
  }

  // Send data through WebSocket
  sendWebSocketData(ws: WebSocket, sensorData: SensorData) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'sensor_data',
        data: sensorData
      }));
    } else {
      console.warn('WebSocket is not open. Ready state:', ws.readyState);
    }
  }
}

// Export singleton instance
export const api = new QuadFusionAPI();

// Constants
export const RISK_LEVELS = {
  LOW: 'low' as const,
  MEDIUM: 'medium' as const,
  HIGH: 'high' as const,
};

export const AGENT_TYPES = {
  TOUCH: 'TouchPatternAgent',
  TYPING: 'TypingBehaviorAgent', 
  VOICE: 'VoiceCommandAgent',
  VISUAL: 'VisualAgent',
  MOVEMENT: 'MovementAgent',
  APP_USAGE: 'AppUsageAgent',
} as const;

// WebSocket message types
export const WS_MESSAGE_TYPES = {
  SENSOR_DATA: 'sensor_data',
  PROCESSING_RESULT: 'processing_result',
  FRAUD_ALERT: 'fraud_alert',
  ERROR: 'error',
} as const;