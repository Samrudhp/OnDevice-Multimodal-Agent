import React, { useEffect } from 'react';
import { Animated, StyleSheet, ViewStyle } from 'react-native';
import { COLORS, ANIMATION } from '../lib/constants';
import { createCardStyle } from '../lib/theme';
import { useFadeInAnimation, useGlowAnimation } from '../lib/animations';

type AnimatedCardProps = {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'minimal';
  glowIntensity?: 'none' | 'low' | 'medium' | 'high';
  animateOnMount?: boolean;
  style?: ViewStyle;
  delay?: number;
};

export const AnimatedCard: React.FC<AnimatedCardProps> = ({
  children,
  variant = 'primary',
  glowIntensity = 'low',
  animateOnMount = true,
  style,
  delay = 0,
}) => {
  // Animation hooks
  const { fadeAnim, startFadeInAnimation } = useFadeInAnimation(ANIMATION.NORMAL);
  const { glowAnim, startGlowAnimation } = useGlowAnimation(0.4, 0.8, ANIMATION.SLOW * 6);

  // Get base card style
  const cardStyle = createCardStyle(variant);

  useEffect(() => {
    if (animateOnMount) {
      // Add delay if specified
      const timer = setTimeout(() => {
        startFadeInAnimation();
        if (glowIntensity !== 'none') {
          startGlowAnimation();
        }
      }, delay);

      return () => clearTimeout(timer);
    }
  }, [animateOnMount, delay]);

  // Determine shadow opacity based on glow intensity
  const shadowOpacity = {
    none: 0,
    low: 0.3,
    medium: 0.5,
    high: 0.8,
  }[glowIntensity];

  // Get glow color based on variant
  const getGlowColor = () => {
    switch (variant) {
      case 'primary':
        return COLORS.GLOW;
      case 'secondary':
        return COLORS.GLOW_SECONDARY;
      case 'minimal':
        return COLORS.GRAY_700;
      default:
        return COLORS.GLOW;
    }
  };

  return (
    <Animated.View
      style={[
        cardStyle,
        styles.container,
        {
          opacity: fadeAnim,
          shadowColor: getGlowColor(),
          shadowOpacity: glowIntensity !== 'none' ? glowAnim : 0,
        },
        style,
      ]}
    >
      {children}
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    shadowOffset: { width: 0, height: 0 },
    shadowRadius: 10,
    elevation: 6,
  },
});