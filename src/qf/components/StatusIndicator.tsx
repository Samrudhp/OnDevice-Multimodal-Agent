import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import * as Icons from 'lucide-react-native';

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
          color: '#10B981',
          backgroundColor: '#ECFDF5',
          borderColor: '#A7F3D0',
        };
      case 'error':
        return {
          icon: AlertCircle,
          color: '#EF4444',
          backgroundColor: '#FEF2F2',
          borderColor: '#FECACA',
        };
      case 'warning':
        return {
          icon: AlertCircle,
          color: '#F59E0B',
          backgroundColor: '#FFFBEB',
          borderColor: '#FDE68A',
        };
      case 'loading':
        return {
          icon: Loader,
          color: '#2563EB',
          backgroundColor: '#EFF6FF',
          borderColor: '#BFDBFE',
        };
      case 'idle':
      default:
        return {
          icon: Clock,
          color: '#6B7280',
          backgroundColor: '#F9FAFB',
          borderColor: '#E5E7EB',
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
    borderRadius: 8,
    borderWidth: 1,
    marginVertical: 4,
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
  },
  submessage: {
    color: '#6B7280',
    marginTop: 4,
    lineHeight: 18,
  },
});
