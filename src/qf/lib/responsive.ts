import { Dimensions, Platform, PixelRatio, ScaledSize } from 'react-native';

// Get device dimensions
const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// Base dimensions (design is based on)
const baseWidth = 375; // iPhone X width
const baseHeight = 812; // iPhone X height

// Scaling factors
const widthScale = SCREEN_WIDTH / baseWidth;
const heightScale = SCREEN_HEIGHT / baseHeight;

// Use the minimum scale for consistent sizing across dimensions
const minScale = Math.min(widthScale, heightScale);

// Device size categories
export enum DeviceSize {
  SMALL = 'small', // e.g. iPhone SE
  MEDIUM = 'medium', // e.g. iPhone X, 11, 12
  LARGE = 'large', // e.g. iPhone Plus, Pro Max
  XLARGE = 'xlarge', // e.g. Tablets
}

// Determine device size category
export const getDeviceSize = (): DeviceSize => {
  if (SCREEN_WIDTH < 350) return DeviceSize.SMALL;
  if (SCREEN_WIDTH < 400) return DeviceSize.MEDIUM;
  if (SCREEN_WIDTH < 600) return DeviceSize.LARGE;
  return DeviceSize.XLARGE;
};

// Current device size
export const deviceSize = getDeviceSize();

// Responsive scaling functions
export const scale = (size: number): number => {
  return PixelRatio.roundToNearestPixel(size * minScale);
};

export const moderateScale = (size: number, factor: number = 0.5): number => {
  return PixelRatio.roundToNearestPixel(size + (scale(size) - size) * factor);
};

// Horizontal scaling (width-based)
export const horizontalScale = (size: number): number => {
  return PixelRatio.roundToNearestPixel(size * widthScale);
};

// Vertical scaling (height-based)
export const verticalScale = (size: number): number => {
  return PixelRatio.roundToNearestPixel(size * heightScale);
};

// Font scaling with size limits to prevent too small/large text
export const fontScale = (size: number): number => {
  const scaledSize = size * minScale;
  const minimumSize = size * 0.8; // Text won't go smaller than 80% of design size
  const maximumSize = size * 1.3; // Text won't go larger than 130% of design size
  
  return PixelRatio.roundToNearestPixel(
    Math.max(minimumSize, Math.min(scaledSize, maximumSize))
  );
};

// Responsive spacing based on device size
export const getResponsiveSpacing = (base: number): number => {
  switch (deviceSize) {
    case DeviceSize.SMALL:
      return scale(base * 0.8);
    case DeviceSize.MEDIUM:
      return scale(base);
    case DeviceSize.LARGE:
      return scale(base * 1.1);
    case DeviceSize.XLARGE:
      return scale(base * 1.3);
    default:
      return scale(base);
  }
};

// Responsive border radius based on device size
export const getResponsiveBorderRadius = (base: number): number => {
  switch (deviceSize) {
    case DeviceSize.SMALL:
      return scale(base * 0.9);
    case DeviceSize.MEDIUM:
      return scale(base);
    case DeviceSize.LARGE:
      return scale(base * 1.1);
    case DeviceSize.XLARGE:
      return scale(base * 1.2);
    default:
      return scale(base);
  }
};

// Responsive shadow size based on device size
export const getResponsiveShadow = (base: number): number => {
  switch (deviceSize) {
    case DeviceSize.SMALL:
      return scale(base * 0.8);
    case DeviceSize.MEDIUM:
      return scale(base);
    case DeviceSize.LARGE:
      return scale(base * 1.2);
    case DeviceSize.XLARGE:
      return scale(base * 1.5);
    default:
      return scale(base);
  }
};

// Listen for dimension changes (e.g., rotation)
export const useDimensionsListener = (callback: (dimensions: ScaledSize) => void) => {
  const subscription = Dimensions.addEventListener('change', ({ window }) => {
    callback(window);
  });
  
  // Return cleanup function to remove event listener
  return () => subscription.remove();
};

// Export screen dimensions for convenience
export const screenDimensions = {
  width: SCREEN_WIDTH,
  height: SCREEN_HEIGHT,
  isSmallDevice: deviceSize === DeviceSize.SMALL,
  isMediumDevice: deviceSize === DeviceSize.MEDIUM,
  isLargeDevice: deviceSize === DeviceSize.LARGE,
  isXLargeDevice: deviceSize === DeviceSize.XLARGE,
  isTablet: SCREEN_WIDTH >= 600,
  isLandscape: SCREEN_WIDTH > SCREEN_HEIGHT,
};