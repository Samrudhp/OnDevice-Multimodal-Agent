import { COLORS } from './constants';
import { StyleSheet } from 'react-native';
import { fontScale, getResponsiveBorderRadius, getResponsiveSpacing, getResponsiveShadow } from './responsive';

// Typography styles
export const TYPOGRAPHY = {
  FONT_FAMILY: {
    PRIMARY: 'System',
    MONO: 'Courier',
  },
  FONT_SIZE: {
    XS: fontScale(10),
    SM: fontScale(12),
    BASE: fontScale(14),
    MD: fontScale(16),
    LG: fontScale(18),
    XL: fontScale(20),
    XXL: fontScale(24),
    XXXL: fontScale(30),
    DISPLAY: fontScale(36),
  },
  FONT_WEIGHT: {
    NORMAL: '400',
    MEDIUM: '500',
    SEMIBOLD: '600',
    BOLD: '700',
  },
  LINE_HEIGHT: {
    TIGHT: 1.2,
    NORMAL: 1.5,
    LOOSE: 1.8,
  },
};

// Spacing system
export const SPACING = {
  XS: getResponsiveSpacing(4),
  SM: getResponsiveSpacing(8),
  MD: getResponsiveSpacing(12),
  BASE: getResponsiveSpacing(16),
  LG: getResponsiveSpacing(20),
  XL: getResponsiveSpacing(24),
  XXL: getResponsiveSpacing(32),
  XXXL: getResponsiveSpacing(48),
};

// Border radius
export const BORDER_RADIUS = {
  XS: getResponsiveBorderRadius(4),
  SM: getResponsiveBorderRadius(8),
  MD: getResponsiveBorderRadius(12),
  LG: getResponsiveBorderRadius(16),
  XL: getResponsiveBorderRadius(24),
  FULL: 9999,
};

// Shadow intensity types
type ShadowIntensity = 'low' | 'medium' | 'high';

// Shadows
export const createShadow = (color: string = COLORS.GLOW, intensity: ShadowIntensity = 'medium') => {
  const shadowIntensity: Record<ShadowIntensity, { opacity: number, offset: { width: number, height: number }, radius: number }> = {
    low: {
      opacity: 0.2,
      offset: { width: 0, height: 2 },
      radius: getResponsiveShadow(4),
    },
    medium: {
      opacity: 0.4,
      offset: { width: 0, height: 4 },
      radius: getResponsiveShadow(8),
    },
    high: {
      opacity: 0.6,
      offset: { width: 0, height: 0 },
      radius: getResponsiveShadow(16),
    },
  };

  const { opacity, offset, radius } = shadowIntensity[intensity] || shadowIntensity.medium;

  return {
    shadowColor: color,
    shadowOffset: offset,
    shadowOpacity: opacity,
    shadowRadius: radius,
    elevation: radius / 2,
  };
};

// Card variant types
type CardVariant = 'primary' | 'secondary' | 'minimal';

// Card styles
export const createCardStyle = (variant: CardVariant = 'primary') => {
  const baseStyle = {
    borderRadius: BORDER_RADIUS.LG,
    padding: SPACING.LG,
    marginBottom: SPACING.LG,
    borderWidth: 1,
  };

  const variants: Record<CardVariant, any> = {
    primary: {
      ...baseStyle,
      backgroundColor: COLORS.CARD,
      borderColor: COLORS.GLOW,
      ...createShadow(COLORS.GLOW, 'medium'),
    },
    secondary: {
      ...baseStyle,
      backgroundColor: COLORS.CARD_ALT,
      borderColor: COLORS.GLOW_SECONDARY,
      ...createShadow(COLORS.GLOW_SECONDARY, 'medium'),
    },
    minimal: {
      ...baseStyle,
      backgroundColor: COLORS.CARD,
      borderColor: COLORS.GRAY_700,
      ...createShadow(COLORS.GLOW, 'low'),
    },
  };

  return variants[variant] || variants.primary;
};

// Button variant types
type ButtonVariant = 'primary' | 'secondary' | 'success' | 'danger' | 'outline' | 'ghost';

// Button styles
export const createButtonStyle = (variant: ButtonVariant = 'primary') => {
  const baseStyle = {
    borderRadius: BORDER_RADIUS.MD,
    paddingVertical: SPACING.MD,
    paddingHorizontal: SPACING.LG,
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
  };

  const variants: Record<ButtonVariant, any> = {
    primary: {
      ...baseStyle,
      backgroundColor: COLORS.PRIMARY,
      ...createShadow(COLORS.GLOW, 'high'),
    },
    secondary: {
      ...baseStyle,
      backgroundColor: COLORS.SECONDARY,
      ...createShadow(COLORS.GLOW_SECONDARY, 'high'),
    },
    success: {
      ...baseStyle,
      backgroundColor: COLORS.SUCCESS,
      ...createShadow(COLORS.SUCCESS, 'medium'),
    },
    danger: {
      ...baseStyle,
      backgroundColor: COLORS.ERROR,
      ...createShadow(COLORS.ERROR, 'medium'),
    },
    outline: {
      ...baseStyle,
      backgroundColor: 'transparent',
      borderWidth: 2,
      borderColor: COLORS.PRIMARY,
      ...createShadow(COLORS.GLOW, 'medium'),
    },
    ghost: {
      ...baseStyle,
      backgroundColor: 'transparent',
    },
  };

  return variants[variant] || variants.primary;
};

// Text variant types
type TextVariant = 'display' | 'title' | 'subtitle' | 'body' | 'caption' | 'button' | 'mono';

// Text styles
export const createTextStyle = (variant: TextVariant = 'body') => {
  const baseStyle = {
    color: COLORS.WHITE,
  };

  const variants: Record<TextVariant, any> = {
    display: {
      ...baseStyle,
      fontSize: TYPOGRAPHY.FONT_SIZE.DISPLAY,
      fontWeight: TYPOGRAPHY.FONT_WEIGHT.BOLD,
      textShadowColor: COLORS.GLOW,
      textShadowOffset: { width: 0, height: 0 },
      textShadowRadius: 8,
    },
    title: {
      ...baseStyle,
      fontSize: TYPOGRAPHY.FONT_SIZE.XXL,
      fontWeight: TYPOGRAPHY.FONT_WEIGHT.BOLD,
      textShadowColor: COLORS.GLOW,
      textShadowOffset: { width: 0, height: 0 },
      textShadowRadius: 4,
    },
    subtitle: {
      ...baseStyle,
      fontSize: TYPOGRAPHY.FONT_SIZE.LG,
      fontWeight: TYPOGRAPHY.FONT_WEIGHT.MEDIUM,
      color: COLORS.GRAY_300,
    },
    body: {
      ...baseStyle,
      fontSize: TYPOGRAPHY.FONT_SIZE.MD,
      fontWeight: TYPOGRAPHY.FONT_WEIGHT.NORMAL,
    },
    caption: {
      ...baseStyle,
      fontSize: TYPOGRAPHY.FONT_SIZE.SM,
      fontWeight: TYPOGRAPHY.FONT_WEIGHT.NORMAL,
      color: COLORS.GRAY_300,
    },
    button: {
      ...baseStyle,
      fontSize: TYPOGRAPHY.FONT_SIZE.MD,
      fontWeight: TYPOGRAPHY.FONT_WEIGHT.SEMIBOLD,
      textShadowColor: 'rgba(0, 0, 0, 0.5)',
      textShadowOffset: { width: 0, height: 1 },
      textShadowRadius: 2,
    },
    mono: {
      ...baseStyle,
      fontFamily: TYPOGRAPHY.FONT_FAMILY.MONO,
      fontSize: TYPOGRAPHY.FONT_SIZE.SM,
      color: COLORS.ACCENT,
    },
  };

  return variants[variant] || variants.body;
};

// Grid background pattern
export const gridBackgroundStyle = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    opacity: 0.4,
    zIndex: -1,
  },
  grid: {
    width: '100%',
    height: '100%',
  },
});

// Common styles
export const commonStyles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BACKGROUND,
  },
  scrollContent: {
    padding: SPACING.LG,
  },
  flexRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  flexBetween: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  flexCenter: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  divider: {
    height: 1,
    backgroundColor: COLORS.GRAY_700,
    marginVertical: SPACING.MD,
  },
});