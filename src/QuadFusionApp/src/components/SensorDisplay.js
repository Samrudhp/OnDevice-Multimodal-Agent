import React from 'react';
import { View, Text, ScrollView } from 'react-native';

const SensorDisplay = ({ sensorData }) => {
  if (!sensorData) {
    return (
      <View className="p-4 bg-white rounded-lg shadow-sm">
        <Text className="text-gray-500 text-center">No sensor data available</Text>
      </View>
    );
  }

  const formatMotionData = (data) => {
    if (!data) return 'N/A';
    return `X: ${data[0]?.toFixed(2)}, Y: ${data[1]?.toFixed(2)}, Z: ${data[2]?.toFixed(2)}`;
  };

  return (
    <ScrollView className="bg-white rounded-lg shadow-sm">
      <View className="p-4">
        <Text className="text-lg font-bold text-gray-800 mb-4">Live Sensor Data</Text>
        
        {/* Touch Events */}
        <View className="mb-4">
          <Text className="text-md font-semibold text-gray-700 mb-2">Touch Events</Text>
          <Text className="text-sm text-gray-600">
            Count: {sensorData.touch_events?.length || 0}
          </Text>
          {sensorData.touch_events?.slice(-3).map((event, index) => (
            <View key={index} className="bg-gray-50 p-2 mt-1 rounded">
              <Text className="text-xs text-gray-600">
                Action: {event.action}, X: {event.x?.toFixed(1)}, Y: {event.y?.toFixed(1)}
              </Text>
            </View>
          ))}
        </View>

        {/* Motion Data */}
        <View className="mb-4">
          <Text className="text-md font-semibold text-gray-700 mb-2">Motion Sensors</Text>
          
          <View className="bg-gray-50 p-2 rounded mb-2">
            <Text className="text-sm font-medium text-gray-700">Accelerometer:</Text>
            <Text className="text-xs text-gray-600">
              {formatMotionData(sensorData.motion_data?.accelerometer)}
            </Text>
          </View>

          <View className="bg-gray-50 p-2 rounded mb-2">
            <Text className="text-sm font-medium text-gray-700">Gyroscope:</Text>
            <Text className="text-xs text-gray-600">
              {formatMotionData(sensorData.motion_data?.gyroscope)}
            </Text>
          </View>

          <View className="bg-gray-50 p-2 rounded">
            <Text className="text-sm font-medium text-gray-700">Magnetometer:</Text>
            <Text className="text-xs text-gray-600">
              {formatMotionData(sensorData.motion_data?.magnetometer)}
            </Text>
          </View>
        </View>

        {/* Keystroke Events */}
        <View className="mb-4">
          <Text className="text-md font-semibold text-gray-700 mb-2">Keystroke Events</Text>
          <Text className="text-sm text-gray-600">
            Count: {sensorData.keystroke_events?.length || 0}
          </Text>
        </View>

        {/* App Usage */}
        <View>
          <Text className="text-md font-semibold text-gray-700 mb-2">App Usage</Text>
          <Text className="text-sm text-gray-600">
            Events: {sensorData.app_usage?.length || 0}
          </Text>
        </View>
      </View>
    </ScrollView>
  );
};

export default SensorDisplay;