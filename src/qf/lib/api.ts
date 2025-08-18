import { Platform } from 'react-native';
import { API_BASE_URL, WS_BASE_URL, TIMEOUTS } from './constants';

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

export interface EnrollmentResult {
  enrollment_id: string;
  status: 'enrolled' | 'pending' | 'failed';
  models_trained: string[];
  message: string;
}

// API Client Class
export class QuadFusionAPI {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  private async makeRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    try {
      const url = `${this.baseURL}${endpoint}`;
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`HTTP ${response.status}: ${errorData.message || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API Request failed:', error);
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
    return this.makeRequest<ProcessingResult>('/api/v1/process/realtime', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        timestamp: new Date().toISOString(),
        sensor_data: sensorData,
      }),
    });
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

  async getModelStatus(): Promise<any> {
    return this.makeRequest('/api/v1/models/status');
  }

  async retrainModel(modelType: string, trainingData: any[], config: any): Promise<any> {
    return this.makeRequest('/api/v1/models/retrain', {
      method: 'POST',
      body: JSON.stringify({
        model_type: modelType,
        training_data: trainingData,
        config: config,
      }),
    });
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
