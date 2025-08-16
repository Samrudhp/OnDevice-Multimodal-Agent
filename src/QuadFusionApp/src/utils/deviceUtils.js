import DeviceInfo from 'react-native-device-info';
import { Platform } from 'react-native';

export const getDeviceId = async () => {
  try {
    return await DeviceInfo.getUniqueId();
  } catch (error) {
    console.error('Error getting device ID:', error);
    return 'unknown-device';
  }
};

export const getDeviceInfo = async () => {
  try {
    return {
      deviceId: await DeviceInfo.getUniqueId(),
      brand: DeviceInfo.getBrand(),
      model: DeviceInfo.getModel(),
      systemVersion: DeviceInfo.getSystemVersion(),
      platform: Platform.OS,
    };
  } catch (error) {
    console.error('Error getting device info:', error);
    return null;
  }
};

export const formatTimestamp = (timestamp) => {
  return new Date(timestamp * 1000).toISOString();
};