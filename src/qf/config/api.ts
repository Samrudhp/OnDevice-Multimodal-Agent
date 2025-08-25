// Central API configuration helper
// Provides a single place to get or override the API base URL at runtime.

import { API_BASE_URL as DEFAULT_API_BASE_URL } from '../lib/constants';
import { Platform } from 'react-native';

// Manual defaults: Android emulator (AVD) uses 10.0.2.2 to reach host machine.
// iOS simulator and web use localhost. This provides a working default without env vars.
export const MANUAL_DEFAULT_ANDROID = 'http://127.0.0.1:8000';
// Use the developer machine IP on the local network for physical devices
export const MANUAL_DEFAULT_OTHER = 'http://localhost:8000';

// Use platform-specific manual defaults during development; otherwise use the configured default
const platformDefault = Platform.OS === 'android' ? MANUAL_DEFAULT_ANDROID : MANUAL_DEFAULT_OTHER;

// Start with DEFAULT_API_BASE_URL for production; in development prefer the platform default
let apiBaseUrl: string = __DEV__ ? platformDefault : DEFAULT_API_BASE_URL;

export function getApiBaseUrl(): string {
  return apiBaseUrl;
}

export function setApiBaseUrl(url: string) {
  apiBaseUrl = url;
}

// Default export is the getter for easy imports
export default getApiBaseUrl;
