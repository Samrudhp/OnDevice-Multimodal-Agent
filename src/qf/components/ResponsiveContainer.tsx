import React from 'react';
import { View, StyleSheet, ViewStyle, StyleProp } from 'react-native';
import { screenDimensions, DeviceSize, deviceSize } from '../lib/responsive';
import { SPACING } from '../lib/theme';

type ResponsiveContainerProps = {
  children: React.ReactNode;
  style?: StyleProp<ViewStyle>;
  padded?: boolean;
  centered?: boolean;
  fullWidth?: boolean;
  fullHeight?: boolean;
  maxWidth?: boolean;
};

export const ResponsiveContainer: React.FC<ResponsiveContainerProps> = ({
  children,
  style,
  padded = false,
  centered = false,
  fullWidth = false,
  fullHeight = false,
  maxWidth = true,
}) => {
  // Determine padding based on device size and padded prop
  const getPadding = () => {
    if (!padded) return 0;
    
    switch (deviceSize) {
      case DeviceSize.SMALL:
        return SPACING.MD;
      case DeviceSize.MEDIUM:
        return SPACING.LG;
      case DeviceSize.LARGE:
        return SPACING.XL;
      case DeviceSize.XLARGE:
        return SPACING.XXL;
      default:
        return SPACING.LG;
    }
  };

  // Determine max width based on device size
  const getMaxWidth = () => {
    if (!maxWidth) return '100%';
    
    switch (deviceSize) {
      case DeviceSize.SMALL:
      case DeviceSize.MEDIUM:
        return '100%';
      case DeviceSize.LARGE:
        return 600;
      case DeviceSize.XLARGE:
        return 800;
      default:
        return '100%';
    }
  };

  return (
    <View
      style={[
        styles.container,
        padded && { padding: getPadding() },
        centered && styles.centered,
        fullWidth && styles.fullWidth,
        fullHeight && styles.fullHeight,
        maxWidth && { maxWidth: getMaxWidth() },
        style,
      ]}
    >
      {children}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    alignSelf: 'center',
  },
  centered: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  fullWidth: {
    width: '100%',
  },
  fullHeight: {
    height: '100%',
  },
});