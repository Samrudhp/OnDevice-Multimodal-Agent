// Central API configuration helper
// Provides a single place to get or override the API base URL at runtime.

import { API_BASE_URL as DEFAULT_API_BASE_URL } from '../lib/constants';
import { Platform } from 'react-native';

// Manual defaults: Android emulator (AVD) uses 10.0.2.2 to reach host machine.
// iOS simulator and web use localhost. This provides a working default without env vars.
export const MANUAL_DEFAULT_ANDROID = 'http://10.0.2.2:8000';
// Use the developer machine IP on the local network for physical devices
export const MANUAL_DEFAULT_OTHER = 'http://localhost:8000';

// Use platform-specific manual defaults during development; otherwise use the configured default
const platformDefault = Platform.OS === 'android' ? MANUAL_DEFAULT_ANDROID : MANUAL_DEFAULT_OTHER;

// Start with DEFAULT_API_BASE_URL for production; in development prefer the platform default
let apiBaseUrl: string = __DEV__ ? platformDefault : DEFAULT_API_BASE_URL;

export function setDeviceIp(ipWithPort: string) {
  // Accept forms like '192.168.1.42' or '192.168.1.42:8000'
  if (!ipWithPort) return;
  const hasPort = ipWithPort.includes(':');
  API_CONFIG.DEVICE = hasPort ? `http://${ipWithPort}` : `http://${ipWithPort}:8000`;
}

export function setApiBaseUrl(url: string) {
  runtimeOverride = url;
}

export function clearRuntimeOverride() {
  runtimeOverride = undefined;
}

// Get the appropriate API URL based on environment
export const getApiUrl = (): string => {
  // 1) explicit runtime override (set via setApiBaseUrl or global/env)
  // @ts-ignore
  if (typeof globalThis !== 'undefined' && (globalThis as any).API_BASE_URL) {
    // @ts-ignore
    return (globalThis as any).API_BASE_URL as string;
  }

  if (runtimeOverride) return runtimeOverride;

  // 2) environment variables commonly injected by bundlers
  try {
    // @ts-ignore
    if (typeof process !== 'undefined' && process.env) {
      // common names
      // @ts-ignore
      const envUrl = process.env.API_BASE_URL || process.env.REACT_APP_API_BASE_URL || process.env.EXPO_API_BASE_URL;
      if (envUrl) return envUrl as string;
    }
  } catch (_e) {
    // ignore
  }

  // 3) Dev-specific behavior
  if (typeof __DEV__ !== 'undefined' && __DEV__) {
    // For Android devices, prefer the DEVICE URL (should be set to your PC IP)
    if (Platform.OS === 'android') return API_CONFIG.DEVICE;
    // For web / iOS simulator, use localhost
    return API_CONFIG.LOCAL;
  }

  // Production
  return API_CONFIG.PRODUCTION;
};

// Export the current API URL (evaluated at import time)
export const API_BASE_URL = getApiUrl();

// Keep default export for compatibility with existing imports
export default getApiUrl;
