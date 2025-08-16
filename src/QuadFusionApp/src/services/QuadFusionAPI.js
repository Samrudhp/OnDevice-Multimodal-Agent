import { API_BASE_URL } from '../utils/constants';
import { getDeviceId } from '../utils/deviceUtils';

class QuadFusionAPI {
  constructor(baseURL = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async makeRequest(endpoint, options = {}) {
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
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API Request failed:', error);
      throw error;
    }
  }

  async enrollUser(userId, biometricData) {
    const deviceId = await getDeviceId();
    
    return this.makeRequest('/api/v1/auth/register', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
        device_id: deviceId,
        biometric_enrollment: biometricData,
      }),
    });
  }

  async verifyUser(userId, sessionId, sensorData) {
    return this.makeRequest('/api/v1/auth/verify', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
        session_id: sessionId,
        sensor_data: sensorData,
      }),
    });
  }

  async processRealtimeSensorData(sessionId, sensorData) {
    return this.makeRequest('/api/v1/process/realtime', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        timestamp: new Date().toISOString(),
        sensor_data: sensorData,
      }),
    });
  }

  async processBatchSensorData(sessionId, batchData) {
    return this.makeRequest('/api/v1/process/batch', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        batch_data: batchData,
      }),
    });
  }

  async getModelStatus() {
    return this.makeRequest('/api/v1/models/status');
  }

  async retrainModel(modelType, trainingData, config) {
    return this.makeRequest('/api/v1/models/retrain', {
      method: 'POST',
      body: JSON.stringify({
        model_type: modelType,
        training_data: trainingData,
        config: config,
      }),
    });
  }

  async getConfig() {
    return this.makeRequest('/api/v1/config');
  }

  async updateConfig(config) {
    return this.makeRequest('/api/v1/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    });
  }
}

export default new QuadFusionAPI();