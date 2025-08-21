import React from 'react';
import { View, StyleSheet, ViewStyle, StyleProp } from 'react-native';
import { COLORS } from '../lib/constants';
import { createCardStyle } from '../lib/theme';

interface CardProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'minimal';
  style?: StyleProp<ViewStyle>;
  glowIntensity?: 'low' | 'medium' | 'high';
}

export const Card: React.FC<CardProps> = ({
  children,
  variant = 'primary',
  style,
  glowIntensity = 'medium',
}) => {
  const cardStyle = createCardStyle(variant);
  
  // Adjust glow intensity
  const glowIntensityValues = {
    low: 0.3,
    medium: 0.6,
    high: 0.9,
  };
  
  const glowColor = variant === 'secondary' ? COLORS.GLOW_SECONDARY : COLORS.GLOW;
  
  return (
    <View 
      style={[
        cardStyle,
        style,
        {
          shadowOpacity: glowIntensityValues[glowIntensity],
          shadowColor: glowColor,
        }
      ]}
    >
      {children}
    </View>
  );
};

export default Card;