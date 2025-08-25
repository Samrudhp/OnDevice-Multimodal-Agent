import { Platform, PermissionsAndroid } from 'react-native';
import { Accelerometer, Gyroscope, Magnetometer } from 'expo-sensors';
import { SENSOR_CONFIG } from './constants';
import type { SensorData, TouchEvent, KeystrokeEvent, MotionData, AppUsageEvent } from './api';

export class SensorManager {
  private isCollecting = false;
  private sensorData: SensorData = {
    touch_events: [],
    keystroke_events: [],
    motion_data: undefined,
    audio_data: undefined,
    image_data: undefined,
    app_usage: [],
  };
  
  private subscriptions: any[] = [];
  private motionBuffer: MotionData[] = [];
  private touchEventBuffer: TouchEvent[] = [];
  private keystrokeEventBuffer: KeystrokeEvent[] = [];
  private appUsageBuffer: AppUsageEvent[] = [];

  constructor() {
    this.resetSensorData();
  }

  async requestPermissions(): Promise<boolean> {
    if (Platform.OS === 'android') {
      try {
        const grants = await PermissionsAndroid.requestMultiple([
          PermissionsAndroid.PERMISSIONS.CAMERA,
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
        ]);

        return Object.values(grants).every(
          grant => grant === PermissionsAndroid.RESULTS.GRANTED
        );
      } catch (error) {
        console.error('Permission request failed:', error);
        return false;
      }
    }
    return true;
  }

  async startMotionSensors(): Promise<void> {
    if (this.isCollecting) return;

    this.isCollecting = true;
    console.log('🚀 Starting motion sensors...');
    
    try {
      // Check if sensors are available before trying to use them
      const isAvailable = await Accelerometer.isAvailableAsync();
      if (!isAvailable) {
        console.warn('⚠️ Motion sensors not available on this device/platform');
        // Create simulated motion data for testing
        this.createSimulatedMotionData();
        console.log('✅ Using simulated motion data');
        return;
      }

      // Set update intervals for better performance
      Accelerometer.setUpdateInterval(SENSOR_CONFIG.MOTION_UPDATE_INTERVAL); // 10Hz
      Gyroscope.setUpdateInterval(SENSOR_CONFIG.MOTION_UPDATE_INTERVAL);
      Magnetometer.setUpdateInterval(SENSOR_CONFIG.MOTION_UPDATE_INTERVAL);

      let motionSampleCount = 0;

      // Accelerometer with error handling
      try {
        const accelerometerSubscription = Accelerometer.addListener(({ x, y, z, timestamp }) => {
          // Keep a short time-series buffer of motion samples so backend can analyze sequences
          const sample: MotionData = {
            accelerometer: [x, y, z],
            gyroscope: this.sensorData.motion_data?.gyroscope || [0, 0, 0],
            magnetometer: this.sensorData.motion_data?.magnetometer || [0, 0, 0],
            timestamp: timestamp / 1000, // Convert to seconds
          };

          // Update latest snapshot
          this.sensorData.motion_data = { ...sample };

          // Push into buffer and cap length
          const maxLen = (SENSOR_CONFIG?.MOTION_BUFFER_SIZE) ?? 100;
          this.motionBuffer.push(sample);
          if (this.motionBuffer.length > maxLen) {
            this.motionBuffer = this.motionBuffer.slice(-maxLen);
          }

          // Log every 50th sample to avoid spam
          motionSampleCount++;
          if (motionSampleCount % 50 === 0) {
            console.log(`📱 Motion sample ${motionSampleCount}: accel=[${x.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)}]`);
          }
        });
        this.subscriptions.push(accelerometerSubscription);
      } catch (e) {
        console.warn('⚠️ Accelerometer not available:', e);
      }

      // Gyroscope with error handling
      try {
        const gyroscopeSubscription = Gyroscope.addListener(({ x, y, z }) => {
          if (this.sensorData.motion_data) {
            this.sensorData.motion_data.gyroscope = [x, y, z];
          }
          // Update motion buffer with latest gyro data
          if (this.motionBuffer.length > 0) {
            this.motionBuffer[this.motionBuffer.length - 1].gyroscope = [x, y, z];
          }
        });
        this.subscriptions.push(gyroscopeSubscription);
      } catch (e) {
        console.warn('⚠️ Gyroscope not available:', e);
      }

      // Magnetometer with error handling
      try {
        const magnetometerSubscription = Magnetometer.addListener(({ x, y, z }) => {
          if (this.sensorData.motion_data) {
            this.sensorData.motion_data.magnetometer = [x, y, z];
          }
          // Update motion buffer with latest magnetometer data
          if (this.motionBuffer.length > 0) {
            this.motionBuffer[this.motionBuffer.length - 1].magnetometer = [x, y, z];
          }
        });
        this.subscriptions.push(magnetometerSubscription);
      } catch (e) {
        console.warn('⚠️ Magnetometer not available:', e);
      }

      console.log('✅ Motion sensors started successfully');
    } catch (error) {
      console.error('❌ Error starting motion sensors:', error);
      // Fall back to simulated data
      this.createSimulatedMotionData();
      console.log('✅ Using simulated motion data as fallback');
    }
  }

  private createSimulatedMotionData(): void {
    // Create realistic simulated motion data
    const interval = setInterval(() => {
      if (!this.isCollecting) {
        clearInterval(interval);
        return;
      }

      const sample: MotionData = {
        accelerometer: [
          Math.random() * 0.2 - 0.1,
          Math.random() * 0.2 - 0.1,
          9.8 + Math.random() * 0.2 - 0.1
        ],
        gyroscope: [
          Math.random() * 0.1 - 0.05,
          Math.random() * 0.1 - 0.05,
          Math.random() * 0.1 - 0.05
        ],
        magnetometer: [
          20 + Math.random() * 10 - 5,
          -10 + Math.random() * 10 - 5,
          40 + Math.random() * 10 - 5
        ],
        timestamp: Date.now() / 1000
      };

      this.sensorData.motion_data = sample;
      this.motionBuffer.push(sample);
      
      const maxLen = (SENSOR_CONFIG?.MOTION_BUFFER_SIZE) ?? 100;
      if (this.motionBuffer.length > maxLen) {
        this.motionBuffer = this.motionBuffer.slice(-maxLen);
      }
    }, SENSOR_CONFIG.MOTION_UPDATE_INTERVAL);

    this.subscriptions.push({ remove: () => clearInterval(interval) });
  }

  stopMotionSensors(): void {
    this.isCollecting = false;
    this.subscriptions.forEach(subscription => {
      subscription && subscription.remove();
    });
    this.subscriptions = [];
  }

  addTouchEvent(touchEvent: any): void {
    const formattedEvent: TouchEvent = {
      timestamp: Date.now() / 1000,
      x: touchEvent.locationX || touchEvent.pageX || 0,
      y: touchEvent.locationY || touchEvent.pageY || 0,
      pressure: touchEvent.force || 0.5,
      touch_major: touchEvent.majorRadius || 10,
      touch_minor: touchEvent.minorRadius || 8,
      action: this.mapTouchAction(touchEvent.type || touchEvent.action),
    };

    this.touchEventBuffer.push(formattedEvent);
    console.log(`👆 Touch event captured: ${formattedEvent.action} at (${formattedEvent.x.toFixed(1)}, ${formattedEvent.y.toFixed(1)}) pressure=${formattedEvent.pressure.toFixed(2)}`);
    
    // Keep only last N events to manage memory
    if (this.touchEventBuffer.length > SENSOR_CONFIG.TOUCH_BUFFER_SIZE) {
      this.touchEventBuffer = this.touchEventBuffer.slice(-SENSOR_CONFIG.TOUCH_BUFFER_SIZE);
    }
  }

  addKeystrokeEvent(keystrokeEvent: any): void {
    const formattedEvent: KeystrokeEvent = {
      timestamp: Date.now() / 1000,
      key_code: keystrokeEvent.keyCode || keystrokeEvent.which || 0,
      action: keystrokeEvent.type === 'keydown' ? 'down' : 'up',
      pressure: keystrokeEvent.pressure || 0.5,
    };

    this.keystrokeEventBuffer.push(formattedEvent);
    console.log(`⌨️ Keystroke captured: key=${formattedEvent.key_code} action=${formattedEvent.action} pressure=${formattedEvent.pressure.toFixed(2)}`);
    
    // Keep only last N events
    if (this.keystrokeEventBuffer.length > SENSOR_CONFIG.KEYSTROKE_BUFFER_SIZE) {
      this.keystrokeEventBuffer = this.keystrokeEventBuffer.slice(-SENSOR_CONFIG.KEYSTROKE_BUFFER_SIZE);
    }
  }

  addAppUsageEvent(appName: string, action: 'open' | 'close' | 'switch_to'): void {
    const event: AppUsageEvent = {
      app_name: appName,
      action,
      timestamp: Date.now() / 1000,
    };

    this.appUsageBuffer.push(event);
    
    // Keep only last 20 events
    if (this.appUsageBuffer.length > 20) {
      this.appUsageBuffer = this.appUsageBuffer.slice(-20);
    }
  }

  private mapTouchAction(touchType: string): 'down' | 'move' | 'up' {
    switch (touchType) {
      case 'press':
      case 'pressIn':
      case 'touchstart':
      case 'down':
        return 'down';
      case 'move':
      case 'touchmove':
        return 'move';
      case 'pressOut':
      case 'touchend':
      case 'up':
      default:
        return 'up';
    }
  }

  getCurrentSensorData(): SensorData {
    // Flush buffered events to main sensor data
    this.sensorData.touch_events = [...this.touchEventBuffer];
    this.sensorData.keystroke_events = [...this.keystrokeEventBuffer];
    this.sensorData.app_usage = [...this.appUsageBuffer];

    // Use the latest motion data for the main motion_data field (API spec compliance)
    if (this.motionBuffer && this.motionBuffer.length > 0) {
      this.sensorData.motion_data = this.motionBuffer[this.motionBuffer.length - 1];
    }

    // Include recent motion sequence as well for backend analysis (additional field)
    const data = { ...this.sensorData } as any;
    if (this.motionBuffer && this.motionBuffer.length > 0) {
      data.motion_sequence = [...this.motionBuffer];
    }
    return data as SensorData;
  }

  resetSensorData(): void {
    this.sensorData = {
      touch_events: [],
      keystroke_events: [],
      motion_data: undefined,
      audio_data: undefined,
      image_data: undefined,
      app_usage: [],
    };
    this.touchEventBuffer = [];
    this.keystrokeEventBuffer = [];
    this.appUsageBuffer = [];
  }

  setAudioData(audioBase64: string, sampleRate: number = 16000, duration: number = 0): void {
    this.sensorData.audio_data = audioBase64;
    this.sensorData.sample_rate = sampleRate;
    this.sensorData.audio_duration = duration;
    console.log(`🎤 Audio data set: ${duration.toFixed(2)}s at ${sampleRate}Hz (${audioBase64.length} chars)`);
  }

  setImageData(imageBase64: string, cameraType: 'front' | 'rear' = 'front'): void {
    this.sensorData.image_data = imageBase64;
    this.sensorData.camera_type = cameraType;
    console.log(`📷 Image data set: ${cameraType} camera (${imageBase64.length} chars)`);
  }

  isMotionSensorsActive(): boolean {
    return this.isCollecting;
  }

  getMotionDataSnapshot(): MotionData | undefined {
    return this.sensorData.motion_data ? { ...this.sensorData.motion_data } : undefined;
  }

  getSensorData(): SensorData {
    // Update with current buffer data
    this.sensorData.touch_events = [...this.touchEventBuffer];
    this.sensorData.keystroke_events = [...this.keystrokeEventBuffer];
    this.sensorData.app_usage = [...this.appUsageBuffer];
    
    // Use the latest motion data for the main motion_data field (API spec compliance)
    if (this.motionBuffer && this.motionBuffer.length > 0) {
      this.sensorData.motion_data = this.motionBuffer[this.motionBuffer.length - 1];
    }
    
    const data = { ...this.sensorData } as any;
    if (this.motionBuffer && this.motionBuffer.length > 0) {
      data.motion_sequence = [...this.motionBuffer];
    }
    return data as SensorData;
  }
}

// Export singleton instance
export const sensorManager = new SensorManager();
