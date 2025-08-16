import React from 'react';
import { View, Text } from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';

const StatusIndicator = ({ status, label, size = 'md' }) => {
  const getStatusConfig = (status) => {
    switch (status) {
      case 'success':
      case 'enrolled':
      case 'low':
        return {
          color: 'text-green-600',
          bgColor: 'bg-green-100',
          icon: 'check-circle',
          iconColor: '#10b981'
        };
      case 'warning':
      case 'medium':
        return {
          color: 'text-yellow-600',
          bgColor: 'bg-yellow-100',
          icon: 'warning',
          iconColor: '#f59e0b'
        };
      case 'error':
      case 'high':
        return {
          color: 'text-red-600',
          bgColor: 'bg-red-100',
          icon: 'error',
          iconColor: '#ef4444'
        };
      case 'loading':
      case 'pending':
        return {
          color: 'text-blue-600',
          bgColor: 'bg-blue-100',
          icon: 'hourglass-empty',
          iconColor: '#3b82f6'
        };
      default:
        return {
          color: 'text-gray-600',
          bgColor: 'bg-gray-100',
          icon: 'help',
          iconColor: '#6b7280'
        };
    }
  };

  const config = getStatusConfig(status);
  const iconSize = size === 'sm' ? 16 : size === 'lg' ? 32 : 24;

  return (
    <View className={`flex-row items-center p-3 rounded-lg ${config.bgColor}`}>
      <Icon name={config.icon} size={iconSize} color={config.iconColor} />
      <Text className={`ml-2 font-medium ${config.color}`}>
        {label}
      </Text>
    </View>
  );
};

export default StatusIndicator;