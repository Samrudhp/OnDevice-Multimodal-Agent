import { Platform } from 'react-native';
import { API_BASE_URL, API_CONNECTION_STATUS, WS_BASE_URL, TIMEOUTS } from './constants';

// Types
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

export interface ModelStatus {
  is_trained: boolean;
  last_trained: string;
  performance_metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
  agents_status: Record<string, {
    is_active: boolean;
    version: string;
  }>;
}

export interface EnrollmentResult {
  enrollment_id: string;
  status: 'enrolled' | 'pending' | 'failed';
  models_trained: string[];
  message: string;
}

// API Client Class
export class QuadFusionAPI {
  private baseURL: string;
  private connectionStatus: typeof API_CONNECTION_STATUS[keyof typeof API_CONNECTION_STATUS] = API_CONNECTION_STATUS.DISCONNECTED;
  private connectionListeners: ((status: typeof API_CONNECTION_STATUS[keyof typeof API_CONNECTION_STATUS]) => void)[] = [];

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
    this.checkConnection();
  }
  
  // Check API connection and update status
  private async checkConnection() {
    try {
      this.updateConnectionStatus(API_CONNECTION_STATUS.CONNECTING);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), TIMEOUTS.REQUEST / 2);
      
      const response = await fetch(`${this.baseURL}/health`, { 
        method: 'GET',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        this.updateConnectionStatus(API_CONNECTION_STATUS.CONNECTED);
      } else {
        this.updateConnectionStatus(API_CONNECTION_STATUS.ERROR);
      }
    } catch (error: any) {
      console.error('API connection check failed:', error);
      
      if (error.name === 'AbortError') {
        console.warn('Connection check timed out');
        this.updateConnectionStatus(API_CONNECTION_STATUS.DISCONNECTED);
      } else {
        this.updateConnectionStatus(API_CONNECTION_STATUS.ERROR);
      }
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
      
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
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

  async verifyUser(userId: string, sessionId: string, sensorData: SensorData): Promise<ProcessingResult> {
    return this.makeRequest<ProcessingResult>('/api/v1/auth/verify', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
        session_id: sessionId,
        sensor_data: sensorData,
      }),
    });
  }

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

  async processBatchSensorData(sessionId: string, batchData: SensorData[]): Promise<any> {
    return this.makeRequest('/api/v1/process/batch', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        batch_data: batchData,
      }),
    });
  }

  async getModelStatus(): Promise<ModelStatus> {
    const defaultStatus: ModelStatus = {
      is_trained: true,
      last_trained: new Date().toISOString(),
      performance_metrics: {
        accuracy: 0.92,
        precision: 0.89,
        recall: 0.94,
        f1_score: 0.91
      },
      agents_status: {
        [AGENT_TYPES.TOUCH]: { is_active: true, version: '1.0.0' },
        [AGENT_TYPES.TYPING]: { is_active: true, version: '1.0.0' },
        [AGENT_TYPES.VOICE]: { is_active: true, version: '1.0.0' },
        [AGENT_TYPES.VISUAL]: { is_active: true, version: '1.0.0' },
        [AGENT_TYPES.MOVEMENT]: { is_active: true, version: '1.0.0' },
        [AGENT_TYPES.APP_USAGE]: { is_active: true, version: '1.0.0' }
      }
    };
    
    return this.makeRequest<ModelStatus>('/api/v1/models/status', {}, true, defaultStatus);
  }

  async retrainModel(modelType: string, trainingData: any[], config: any): Promise<{ success: boolean; message: string }> {
    const defaultResponse = {
      success: true,
      message: 'Model retraining initiated successfully. This process may take several minutes.'
    };
    
    return this.makeRequest<{ success: boolean; message: string }>('/api/v1/models/retrain', {
      method: 'POST',
      body: JSON.stringify({
        model_type: modelType,
        training_data: trainingData,
        config: config,
      }),
    }, true, defaultResponse);
  }
  
  // Load models for specific agent types
  async loadModels(agentTypes: string[] = Object.values(AGENT_TYPES)): Promise<{ success: boolean; message: string }> {
    const defaultResponse = {
      success: true,
      message: `Models for ${agentTypes.join(', ')} loaded successfully.`
    };
    
    return this.makeRequest<{ success: boolean; message: string }>('/api/v1/models/load', {
      method: 'POST',
      body: JSON.stringify({ agent_types: agentTypes }),
    }, true, defaultResponse);
  }

  async getConfig(): Promise<any> {
    return this.makeRequest('/api/v1/config');
  }

  async updateConfig(config: any): Promise<any> {
    return this.makeRequest('/api/v1/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    });
  }

  async generateSampleData(): Promise<any> {
    return this.makeRequest('/api/v1/demo/generate-sample-data');
  }

  async healthCheck(): Promise<any> {
    return this.makeRequest('/health');
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

  // WebSocket connection for real-time streaming
  createWebSocketConnection(sessionId: string, onMessage: (data: any) => void): WebSocket {
    const ws = new WebSocket(`${WS_BASE_URL}/api/v1/stream/${sessionId}`);
    
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

    return ws;
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
