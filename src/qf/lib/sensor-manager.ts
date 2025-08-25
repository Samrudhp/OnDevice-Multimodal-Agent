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
    
    try {
      // Set update intervals for better performance
      Accelerometer.setUpdateInterval(SENSOR_CONFIG.MOTION_UPDATE_INTERVAL); // 10Hz
      Gyroscope.setUpdateInterval(SENSOR_CONFIG.MOTION_UPDATE_INTERVAL);
      Magnetometer.setUpdateInterval(SENSOR_CONFIG.MOTION_UPDATE_INTERVAL);

      // Accelerometer
      const accelerometerSubscription = Accelerometer.addListener(({ x, y, z, timestamp }) => {
        if (!this.sensorData.motion_data) {
          this.sensorData.motion_data = {
            accelerometer: [0, 0, 0],
            gyroscope: [0, 0, 0],
            magnetometer: [0, 0, 0],
            timestamp: 0,
          };
        }
        this.sensorData.motion_data.accelerometer = [x, y, z];
        this.sensorData.motion_data.timestamp = timestamp / 1000; // Convert to seconds
      });

      // Gyroscope
      const gyroscopeSubscription = Gyroscope.addListener(({ x, y, z }) => {
        if (this.sensorData.motion_data) {
          this.sensorData.motion_data.gyroscope = [x, y, z];
        }
      });

      // Magnetometer
      const magnetometerSubscription = Magnetometer.addListener(({ x, y, z }) => {
        if (this.sensorData.motion_data) {
          this.sensorData.motion_data.magnetometer = [x, y, z];
        }
      });

      this.subscriptions.push(
        accelerometerSubscription,
        gyroscopeSubscription,
        magnetometerSubscription
      );
    } catch (error) {
      console.error('Error starting motion sensors:', error);
      this.stopMotionSensors();
    }
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

    return { ...this.sensorData };
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
  }

  setImageData(imageBase64: string, cameraType: 'front' | 'rear' = 'front'): void {
    this.sensorData.image_data = imageBase64;
    this.sensorData.camera_type = cameraType;
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
    
    return { ...this.sensorData };
  }
}

// Export singleton instance
export const sensorManager = new SensorManager();
