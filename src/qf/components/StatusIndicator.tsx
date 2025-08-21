import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import * as Icons from 'lucide-react-native';
import { COLORS } from '../lib/constants';

const CheckCircle = Icons.CheckCircle ?? (() => null);
const AlertCircle = Icons.AlertCircle ?? (() => null);
const Clock = Icons.Clock ?? (() => null);
const Loader = Icons.Loader ?? (() => null);

export type StatusType = 'success' | 'error' | 'warning' | 'loading' | 'idle';

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
          backgroundColor: 'rgba(0, 255, 159, 0.1)',
          borderColor: 'rgba(0, 255, 159, 0.3)',
          glowColor: 'rgba(0, 255, 159, 0.5)',
        };
      case 'error':
        return {
          icon: AlertCircle,
          color: COLORS.ERROR,
          backgroundColor: 'rgba(255, 0, 85, 0.1)',
          borderColor: 'rgba(255, 0, 85, 0.3)',
          glowColor: 'rgba(255, 0, 85, 0.5)',
        };
      case 'warning':
        return {
          icon: AlertCircle,
          color: COLORS.WARNING,
          backgroundColor: 'rgba(255, 184, 0, 0.1)',
          borderColor: 'rgba(255, 184, 0, 0.3)',
          glowColor: 'rgba(255, 184, 0, 0.5)',
        };
      case 'loading':
        return {
          icon: Loader,
          color: COLORS.PRIMARY,
          backgroundColor: 'rgba(0, 255, 255, 0.1)',
          borderColor: 'rgba(0, 255, 255, 0.3)',
          glowColor: 'rgba(0, 255, 255, 0.5)',
        };
      case 'idle':
      default:
        return {
          icon: Clock,
          color: COLORS.GRAY_500,
          backgroundColor: 'rgba(100, 116, 139, 0.1)',
          borderColor: 'rgba(100, 116, 139, 0.3)',
          glowColor: 'rgba(100, 116, 139, 0.2)',
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
    borderRadius: 12,
    marginVertical: 4,
    shadowColor: COLORS.PRIMARY,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 5,
  },
  content: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  textContainer: {
    flex: 1,
    marginLeft: 12,
  },
  message: {
    fontWeight: '600',
    lineHeight: 20,
    color: COLORS.WHITE,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
  },
  submessage: {
    marginTop: 4,
    lineHeight: 18,
    opacity: 0.8,
    color: COLORS.GRAY_300,
  },
});
