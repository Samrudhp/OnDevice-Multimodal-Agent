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

// UI Constants - Modern Cyber Theme
export const COLORS = {
  PRIMARY: '#0CFFE1', // Bright Teal
  SECONDARY: '#8A2BE2', // Electric Purple
  ACCENT: '#36F3FF', // Electric Blue
  SUCCESS: '#00FF9F', // Neon Green
  WARNING: '#FFD700', // Cyber Gold
  ERROR: '#FF2A6D', // Neon Pink
  BACKGROUND: '#080C14', // Deep Space Black
  SURFACE: '#111827', // Dark Slate
  CARD: '#1A202E', // Midnight Blue
  CARD_ALT: '#232A3B', // Alternate Card
  WHITE: '#FFFFFF',
  BLACK: '#000000',
  GRAY_100: '#E2E8F0',
  GRAY_300: '#94A3B8',
  GRAY_500: '#64748B',
  GRAY_700: '#334155',
  GRAY_900: '#0F172A',
  GLOW: 'rgba(12, 255, 225, 0.6)', // Teal glow effect
  GLOW_SECONDARY: 'rgba(138, 43, 226, 0.5)', // Purple glow
  GRID: 'rgba(12, 255, 225, 0.08)', // Grid lines
} as const;

// Animation Durations
export const ANIMATION = {
  FAST: 150,
  NORMAL: 300,
  SLOW: 500,
} as const;

// API Timeouts
export const TIMEOUTS = {
  REQUEST: 60000, // 60 seconds
  UPLOAD: 120000,  // 2 minutes
  WEBSOCKET: 10000, // 10 seconds
} as const;

export type RiskLevel = typeof RISK_LEVELS[keyof typeof RISK_LEVELS];
export type AgentType = typeof AGENT_TYPES[keyof typeof AGENT_TYPES];
export type Permission = typeof PERMISSIONS[keyof typeof PERMISSIONS];
