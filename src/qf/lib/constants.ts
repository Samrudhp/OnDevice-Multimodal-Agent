// API Configuration
export const API_BASE_URL = __DEV__ ? 'http://localhost:8000' : 'https://api.quadfusion.com';
export const WS_BASE_URL = __DEV__ ? 'ws://localhost:8000' : 'wss://api.quadfusion.com';

// Risk Assessment Levels
export const RISK_LEVELS = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
} as const;

// QuadFusion Agent Types
export const AGENT_TYPES = {
  TOUCH: 'TouchPatternAgent',
  TYPING: 'TypingBehaviorAgent', 
  VOICE: 'VoiceCommandAgent',
  VISUAL: 'VisualAgent',
  MOVEMENT: 'MovementAgent',
  APP_USAGE: 'AppUsageAgent',
} as const;

// Required Permissions
export const PERMISSIONS = {
  CAMERA: 'camera',
  MICROPHONE: 'microphone',
  LOCATION: 'location',
  MOTION_SENSORS: 'motion-sensors',
} as const;

// Sensor Configuration
export const SENSOR_CONFIG = {
  MOTION_UPDATE_INTERVAL: 100, // milliseconds
  TOUCH_BUFFER_SIZE: 100,
  KEYSTROKE_BUFFER_SIZE: 50,
  APP_USAGE_BUFFER_SIZE: 20,
} as const;

// UI Constants
export const COLORS = {
  PRIMARY: '#2563EB',
  SUCCESS: '#10B981',
  WARNING: '#F59E0B',
  ERROR: '#EF4444',
  BACKGROUND: '#F9FAFB',
  WHITE: '#FFFFFF',
  GRAY_100: '#F3F4F6',
  GRAY_300: '#D1D5DB',
  GRAY_500: '#6B7280',
  GRAY_700: '#374151',
  GRAY_900: '#1F2937',
} as const;

// Animation Durations
export const ANIMATION = {
  FAST: 150,
  NORMAL: 300,
  SLOW: 500,
} as const;

// API Timeouts
export const TIMEOUTS = {
  REQUEST: 30000, // 30 seconds
  UPLOAD: 60000,  // 60 seconds
  WEBSOCKET: 5000, // 5 seconds
} as const;

export type RiskLevel = typeof RISK_LEVELS[keyof typeof RISK_LEVELS];
export type AgentType = typeof AGENT_TYPES[keyof typeof AGENT_TYPES];
export type Permission = typeof PERMISSIONS[keyof typeof PERMISSIONS];
