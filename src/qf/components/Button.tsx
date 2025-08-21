import React from 'react';
import { TouchableOpacity, Text, StyleSheet, ViewStyle, TextStyle, StyleProp, ActivityIndicator } from 'react-native';
import { COLORS } from '../lib/constants';
import { createButtonStyle, createTextStyle } from '../lib/theme';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'success' | 'danger' | 'outline' | 'ghost';
  size?: 'small' | 'medium' | 'large';
  icon?: React.ReactNode;
  disabled?: boolean;
  loading?: boolean;
  style?: StyleProp<ViewStyle>;
  textStyle?: StyleProp<TextStyle>;
}

export const Button: React.FC<ButtonProps> = ({
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
  icon,
  disabled = false,
  loading = false,
  style,
  textStyle,
}) => {
  const buttonStyle = createButtonStyle(variant);
  
  const sizeStyles = {
    small: {
      paddingVertical: 8,
      paddingHorizontal: 12,
      fontSize: 12,
    },
    medium: {
      paddingVertical: 12,
      paddingHorizontal: 16,
      fontSize: 14,
    },
    large: {
      paddingVertical: 16,
      paddingHorizontal: 24,
      fontSize: 16,
    },
  };
  
  const textStyleBase = createTextStyle('button');
  
  return (
    <TouchableOpacity
      style={[
        buttonStyle,
        {
          paddingVertical: sizeStyles[size].paddingVertical,
          paddingHorizontal: sizeStyles[size].paddingHorizontal,
          opacity: disabled ? 0.6 : 1,
        },
        style,
      ]}
      onPress={onPress}
      disabled={disabled || loading}
      activeOpacity={0.8}
    >
      {loading ? (
        <ActivityIndicator color={COLORS.WHITE} size="small" />
      ) : (
        <>
          {icon && <>{icon}</>}
          <Text 
            style={[
              textStyleBase,
              { fontSize: sizeStyles[size].fontSize, marginLeft: icon ? 8 : 0 },
              textStyle,
            ]}
          >
            {title}
          </Text>
        </>
      )}
    </TouchableOpacity>
  );
};

export default Button;