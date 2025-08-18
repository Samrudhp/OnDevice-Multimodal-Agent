import { Platform } from 'react-native';

// Get device-specific utilities that would normally come from expo-device
export const getDeviceId = async (): Promise<string> => {
  try {
    // In a real implementation, use expo-device or react-native-device-info
    return `${Platform.OS}-${Platform.Version}-${Date.now()}`;
  } catch (error) {
    console.error('Error getting device ID:', error);
    return 'unknown-device';
  }
};

export const getDeviceInfo = async () => {
  try {
    return {
      deviceId: await getDeviceId(),
      brand: Platform.OS === 'ios' ? 'Apple' : 'Android',
      model: 'Unknown Model',
      systemVersion: Platform.Version.toString(),
      platform: Platform.OS,
      isEmulator: __DEV__, // Simple heuristic
    };
  } catch (error) {
    console.error('Error getting device info:', error);
    return {
      deviceId: 'unknown',
      brand: 'Unknown',
      model: 'Unknown',
      systemVersion: 'Unknown',
      platform: Platform.OS,
      isEmulator: false,
    };
  }
};

export const formatTimestamp = (timestamp: number): string => {
  return new Date(timestamp * 1000).toISOString();
};

export const generateSessionId = (): string => {
  return `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

export const validateSensorData = (data: any): boolean => {
  if (!data || typeof data !== 'object') return false;
  
  // Basic validation for required sensor data structure
  const hasValidStructure = 
    Array.isArray(data.touch_events) &&
    Array.isArray(data.keystroke_events) &&
    Array.isArray(data.app_usage);
    
  return hasValidStructure;
};

export const sanitizeBase64 = (base64String?: string): string | undefined => {
  if (!base64String) return undefined;
  
  // Remove data URL prefix if present
  const cleanBase64 = base64String.replace(/^data:[^;]+;base64,/, '');
  
  // Basic validation that it's valid base64
  try {
    atob(cleanBase64);
    return cleanBase64;
  } catch (error) {
    console.warn('Invalid base64 string provided');
    return undefined;
  }
};
