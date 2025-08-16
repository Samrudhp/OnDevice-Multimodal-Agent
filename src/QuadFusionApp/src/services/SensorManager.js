import { accelerometer, gyroscope, magnetometer } from 'react-native-sensors';
import { PermissionsAndroid, Platform } from 'react-native';

class SensorManager {
  constructor() {
    this.isCollecting = false;
    this.sensorData = {
      touch_events: [],
      keystroke_events: [],
      motion_data: {},
      audio_data: null,
      image_data: null,
      app_usage: [],
    };
    
    this.subscriptions = [];
  }

  async requestPermissions() {
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

  startMotionSensors() {
    if (this.isCollecting) return;

    this.isCollecting = true;
    
    // Accelerometer
    const accelerometerSubscription = accelerometer.subscribe(({ x, y, z, timestamp }) => {
      this.sensorData.motion_data.accelerometer = [x, y, z];
      this.sensorData.motion_data.timestamp = timestamp / 1000; // Convert to seconds
    });

    // Gyroscope
    const gyroscopeSubscription = gyroscope.subscribe(({ x, y, z, timestamp }) => {
      this.sensorData.motion_data.gyroscope = [x, y, z];
    });

    // Magnetometer
    const magnetometerSubscription = magnetometer.subscribe(({ x, y, z, timestamp }) => {
      this.sensorData.motion_data.magnetometer = [x, y, z];
    });

    this.subscriptions.push(
      accelerometerSubscription,
      gyroscopeSubscription,
      magnetometerSubscription
    );
  }

  stopMotionSensors() {
    this.isCollecting = false;
    this.subscriptions.forEach(subscription => {
      subscription.unsubscribe();
    });
    this.subscriptions = [];
  }

  addTouchEvent(touchEvent) {
    const formattedEvent = {
      timestamp: Date.now() / 1000,
      x: touchEvent.locationX || 0,
      y: touchEvent.locationY || 0,
      pressure: touchEvent.force || 0.5,
      touch_major: touchEvent.majorRadius || 10,
      touch_minor: touchEvent.minorRadius || 8,
      action: this.mapTouchAction(touchEvent.type),
    };

    this.sensorData.touch_events.push(formattedEvent);
    
    // Keep only last 100 events to manage memory
    if (this.sensorData.touch_events.length > 100) {
      this.sensorData.touch_events = this.sensorData.touch_events.slice(-100);
    }
  }

  mapTouchAction(touchType) {
    switch (touchType) {
      case 'press':
      case 'pressIn':
        return 'down';
      case 'pressOut':
      case 'longPress':
        return 'up';
      default:
        return 'move';
    }
  }

  addKeystrokeEvent(keystroke) {
    const formattedEvent = {
      timestamp: Date.now() / 1000,
      key_code: keystroke.keyCode || 0,
      action: keystroke.action || 'down',
      pressure: keystroke.pressure || 0.5,
    };

    this.sensorData.keystroke_events.push(formattedEvent);
    
    // Keep only last 50 keystrokes
    if (this.sensorData.keystroke_events.length > 50) {
      this.sensorData.keystroke_events = this.sensorData.keystroke_events.slice(-50);
    }
  }

  addAppUsageEvent(appName, action) {
    const usageEvent = {
      app_name: appName,
      action: action,
      timestamp: Date.now() / 1000,
    };

    this.sensorData.app_usage.push(usageEvent);
    
    // Keep only last 20 app usage events
    if (this.sensorData.app_usage.length > 20) {
      this.sensorData.app_usage = this.sensorData.app_usage.slice(-20);
    }
  }

  getSensorData() {
    return { ...this.sensorData };
  }

  clearSensorData() {
    this.sensorData = {
      touch_events: [],
      keystroke_events: [],
      motion_data: {},
      audio_data: null,
      image_data: null,
      app_usage: [],
    };
  }

  // Mock methods for audio and camera - would need proper implementation
  async captureAudio(duration = 2000) {
    // This would integrate with react-native-audio-record
    return new Promise(resolve => {
      setTimeout(() => {
        this.sensorData.audio_data = 'base64_mock_audio_data';
        resolve('base64_mock_audio_data');
      }, duration);
    });
  }

  async captureImage() {
    // This would integrate with react-native-camera
    return new Promise(resolve => {
      setTimeout(() => {
        this.sensorData.image_data = 'base64_mock_image_data';
        resolve('base64_mock_image_data');
      }, 1000);
    });
  }
}

export default new SensorManager();