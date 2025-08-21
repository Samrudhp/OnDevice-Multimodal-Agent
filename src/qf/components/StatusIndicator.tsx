import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import * as Icons from 'lucide-react-native';
import { COLORS } from '../lib/constants';
import { BORDER_RADIUS, SPACING, createShadow } from '../lib/theme';

const CheckCircle = Icons.CheckCircle ?? (() => null);
const AlertTriangle = Icons.AlertTriangle ?? (() => null);
const XCircle = Icons.XCircle ?? (() => null);
const Clock = Icons.Clock ?? (() => null);
const Loader = Icons.Loader ?? (() => null);
const Info = Icons.Info ?? (() => null);

export type StatusType = 'success' | 'error' | 'warning' | 'loading' | 'idle' | 'info';

interface StatusIndicatorProps {
  status: StatusType;
  message: string;
  submessage?: string;
  size?: 'small' | 'medium' | 'large';
}

export default function StatusIndicator({ 
  status, 
  message, 
  submessage, 
  size = 'medium' 
}: StatusIndicatorProps) {
  const getStatusConfig = () => {
    switch (status) {
      case 'success':
        return {
          icon: CheckCircle,
          color: COLORS.SUCCESS,
          backgroundColor: 'rgba(0, 255, 159, 0.15)',
          borderColor: COLORS.SUCCESS,
          glowColor: 'rgba(0, 255, 159, 0.6)',
        };
      case 'error':
        return {
          icon: XCircle,
          color: COLORS.ERROR,
          backgroundColor: 'rgba(255, 42, 109, 0.15)',
          borderColor: COLORS.ERROR,
          glowColor: 'rgba(255, 42, 109, 0.6)',
        };
      case 'warning':
        return {
          icon: AlertTriangle,
          color: COLORS.WARNING,
          backgroundColor: 'rgba(255, 215, 0, 0.15)',
          borderColor: COLORS.WARNING,
          glowColor: 'rgba(255, 215, 0, 0.6)',
        };
      case 'info':
        return {
          icon: Info,
          color: COLORS.ACCENT,
          backgroundColor: 'rgba(54, 243, 255, 0.15)',
          borderColor: COLORS.ACCENT,
          glowColor: 'rgba(54, 243, 255, 0.6)',
        };
      case 'loading':
        return {
          icon: Loader,
          color: COLORS.PRIMARY,
          backgroundColor: 'rgba(12, 255, 225, 0.15)',
          borderColor: COLORS.PRIMARY,
          glowColor: 'rgba(12, 255, 225, 0.6)',
        };
      case 'idle':
      default:
        return {
          icon: Clock,
          color: COLORS.GRAY_500,
          backgroundColor: 'rgba(100, 116, 139, 0.15)',
          borderColor: COLORS.GRAY_500,
          glowColor: 'rgba(100, 116, 139, 0.3)',
        };
    }
  };

  const getSizeConfig = () => {
    switch (size) {
      case 'small':
        return {
          containerPadding: 12,
          iconSize: 16,
          messageSize: 14,
          submessageSize: 12,
        };
      case 'large':
        return {
          containerPadding: 24,
          iconSize: 32,
          messageSize: 18,
          submessageSize: 16,
        };
      case 'medium':
      default:
        return {
          containerPadding: 16,
          iconSize: 24,
          messageSize: 16,
          submessageSize: 14,
        };
    }
  };

  const statusConfig = getStatusConfig();
  const sizeConfig = getSizeConfig();
  const IconComponent = statusConfig.icon;

  const containerStyle = [
    styles.container,
    {
      backgroundColor: statusConfig.backgroundColor,
      borderColor: statusConfig.borderColor,
      padding: sizeConfig.containerPadding,
    },
  ];

  const messageStyle = [
    styles.message,
    {
      color: statusConfig.color,
      fontSize: sizeConfig.messageSize,
    },
  ];

  const submessageStyle = [
    styles.submessage,
    {
      fontSize: sizeConfig.submessageSize,
    },
  ];

  return (
    <View style={containerStyle}>
      <View style={styles.content}>
        <IconComponent 
          size={sizeConfig.iconSize} 
          color={statusConfig.color} 
        />
        <View style={styles.textContainer}>
          <Text style={messageStyle}>{message}</Text>
          {submessage && (
            <Text style={submessageStyle}>{submessage}</Text>
          )}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderWidth: 1,
    borderRadius: BORDER_RADIUS.MD,
    marginVertical: SPACING.SM,
    ...createShadow(COLORS.GLOW, 'medium'),
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  textContainer: {
    flex: 1,
    marginLeft: SPACING.MD,
  },
  message: {
    fontWeight: '600',
    lineHeight: 20,
    color: COLORS.WHITE,
    textShadowColor: 'rgba(0, 0, 0, 0.5)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 2,
  },
  submessage: {
    marginTop: 2,
    lineHeight: 18,
    color: COLORS.GRAY_300,
  },
});
