// Central API configuration helper
// Provides a single place to get or override the API base URL at runtime.

import { API_BASE_URL as DEFAULT_API_BASE_URL } from '../lib/constants';
import { Platform } from 'react-native';

// Manual defaults: Android emulator (AVD) uses 10.0.2.2 to reach host machine.
// iOS simulator and web use localhost. This provides a working default without env vars.
export const MANUAL_DEFAULT_ANDROID = 'http://10.0.2.2:8000';
// Use the developer machine IP on the local network for physical devices
export const MANUAL_DEFAULT_OTHER = 'http://172.16.144.246:8000';

const platformDefault = Platform.OS === 'android' ? MANUAL_DEFAULT_ANDROID : MANUAL_DEFAULT_OTHER;

// Start with manual platform default; fall back to constants if needed
let apiBaseUrl: string = platformDefault || DEFAULT_API_BASE_URL;

export function getApiBaseUrl(): string {
  return apiBaseUrl;
}

export function setApiBaseUrl(url: string) {
  apiBaseUrl = url;
}

// Default export is the getter for easy imports
export default getApiBaseUrl;
