import React, { useEffect } from 'react';
import { Animated, Text, StyleSheet, TextStyle, TextProps } from 'react-native';
import { COLORS } from '../lib/constants';
import { createTextStyle } from '../lib/theme';
import { useFadeInAnimation, useGlowAnimation } from '../lib/animations';
import { ANIMATION } from '../lib/constants';

type AnimatedTextProps = TextProps & {
  children: React.ReactNode;
  variant?: 'title' | 'subtitle' | 'body' | 'caption' | 'button' | 'mono' | 'display';
  color?: string;
  glow?: boolean;
  animateOnMount?: boolean;
  style?: TextStyle;
  delay?: number;
};

export const AnimatedText: React.FC<AnimatedTextProps> = ({
  children,
  variant = 'body',
  color,
  glow = false,
  animateOnMount = true,
  style,
  delay = 0,
  ...props
}) => {
  // Animation hooks
  const { fadeAnim, startFadeInAnimation } = useFadeInAnimation(ANIMATION.NORMAL);
  const { glowAnim, startGlowAnimation } = useGlowAnimation(0.4, 1, ANIMATION.SLOW * 4);

  // Get base text style
  const textStyle = createTextStyle(variant);

  useEffect(() => {
    if (animateOnMount) {
      // Add delay if specified
      const timer = setTimeout(() => {
        startFadeInAnimation();
        if (glow) {
          startGlowAnimation();
        }
      }, delay);

      return () => clearTimeout(timer);
    }
  }, [animateOnMount, delay, glow]);

  // Get glow color based on text color or variant
  const getGlowColor = () => {
    if (color) return color;
    
    switch (variant) {
      case 'title':
        return COLORS.PRIMARY;
      case 'subtitle':
        return COLORS.SECONDARY;
      case 'mono':
        return COLORS.ACCENT;
      case 'display':
        return COLORS.GLOW;
      default:
        return COLORS.PRIMARY;
    }
  };

  return (
    <Animated.Text
      style={[
        textStyle,
        color ? { color } : {},
        glow ? {
          textShadowColor: getGlowColor(),
          textShadowOffset: { width: 0, height: 0 },
          textShadowRadius: 4 * Number(glowAnim),
        } : {},
        { opacity: fadeAnim },
        style,
      ]}
      {...props}
    >
      {children}
    </Animated.Text>
  );
};