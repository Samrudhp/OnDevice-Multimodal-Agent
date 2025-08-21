// API Configuration
export const API_BASE_URL = __DEV__ ? 'http://localhost:8000' : 'https://api.quadfusion.com';
export const WS_BASE_URL = __DEV__ ? 'ws://localhost:8000' : 'wss://api.quadfusion.com';

// API Connection Status
export const API_CONNECTION_STATUS = {
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
  CONNECTING: 'connecting',
  ERROR: 'error'
} as const;

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

// UI Constants - Dark Cyber Theme
export const COLORS = {
  PRIMARY: '#00FFFF', // Cyan
  SECONDARY: '#7B42F6', // Purple
  ACCENT: '#00FF9F', // Neon Green
  SUCCESS: '#00FF9F', // Neon Green
  WARNING: '#FFB800', // Amber
  ERROR: '#FF0055', // Neon Red
  BACKGROUND: '#0A0E17', // Dark Blue-Black
  SURFACE: '#141A26', // Slightly lighter dark blue
  CARD: '#1C2333', // Card background
  WHITE: '#FFFFFF',
  BLACK: '#000000',
  GRAY_100: '#E2E8F0',
  GRAY_300: '#94A3B8',
  GRAY_500: '#64748B',
  GRAY_700: '#334155',
  GRAY_900: '#0F172A',
  GLOW: 'rgba(0, 255, 255, 0.5)', // Cyan glow effect
  GRID: 'rgba(0, 255, 255, 0.1)', // Grid lines
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
