import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TouchableOpacity, TextInput, Alert } from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import SensorDisplay from '../components/SensorDisplay';
import ProcessingResult from '../components/ProcessingResult';
import StatusIndicator from '../components/StatusIndicator';
import { useSensorData } from '../hooks/useSensorData';
import QuadFusionAPI from '../services/QuadFusionAPI';
import SensorManager from '../services/SensorManager';

const MonitoringScreen = () => {
  const [userId, setUserId] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [processingResult, setProcessingResult] = useState(null);
  const [lastProcessingTime, setLastProcessingTime] = useState(null);
  
  const {
    sensorData,
    isCollecting,
    error,
    startCollection,
    stopCollection,
    getCurrentSensorData,
    clearData
  } = useSensorData();

  // Generate session ID when monitoring starts
  useEffect(() => {
    if (isMonitoring && !sessionId) {
      setSessionId(`session_${Date.now()}`);
    }
  }, [isMonitoring, sessionId]);

  // Auto-process sensor data every 5 seconds when monitoring
  useEffect(() => {
    let interval;
    
    if (isMonitoring && userId && sessionId) {
      interval = setInterval(async () => {
        try {
          const currentData = getCurrentSensorData();
          if (currentData && (
            currentData.touch_events?.length > 0 ||
            currentData.motion_data?.accelerometer ||
            currentData.keystroke_events?.length > 0
          )) {
            const result = await QuadFusionAPI.processRealtimeSensorData(sessionId, currentData);
            setProcessingResult(result);
            setLastProcessingTime(new Date());
          }
        } catch (error) {
          console.error('Processing error:', error);
        }
      }, 5000);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isMonitoring, userId, sessionId, getCurrentSensorData]);

  const startMonitoring = async () => {
    if (!userId.trim()) {
      Alert.alert('Error', 'Please enter a User ID to monitor');
      return;
    }

    try {
      setIsMonitoring(true);
      await startCollection();
      SensorManager.addAppUsageEvent('QuadFusionApp', 'monitoring_started');
    } catch (error) {
      Alert.alert('Error', `Failed to start monitoring: ${error.message}`);
      setIsMonitoring(false);
    }
  };

  const stopMonitoring = () => {
    setIsMonitoring(false);
    stopCollection();
    SensorManager.addAppUsageEvent('QuadFusionApp', 'monitoring_stopped');
  };

  const handleTouchEvent = (event) => {
    if (isMonitoring) {
      SensorManager.addTouchEvent(event.nativeEvent);
    }
  };

  const processManually = async () => {
    if (!userId || !sessionId) return;
    
    try {
      const currentData = getCurrentSensorData();
      const result = await QuadFusionAPI.processRealtimeSensorData(sessionId, currentData);
      setProcessingResult(result);
      setLastProcessingTime(new Date());
    } catch (error) {
      Alert.alert('Processing Error', error.message);
    }
  };

  const resetSession = () => {
    stopMonitoring();
    clearData();
    setProcessingResult(null);
    setSessionId('');
    setLastProcessingTime(null);
  };

  return (
    <ScrollView className="flex-1 bg-gray-50">
      <View className="p-4">
        {/* Control Panel */}
        <View className="bg-white rounded-lg shadow-sm p-4 mb-4">
          <Text className="text-lg font-bold text-gray-800 mb-4">Monitoring Controls</Text>
          
          <View className="mb-3">
            <Text className="text-sm font-medium text-gray-700 mb-2">User ID</Text>
            <TextInput
              value={userId}
              onChangeText={setUserId}
              placeholder="Enter user ID to monitor"
              className="border border-gray-300 rounded-lg px-3 py-2 text-gray-700"
              editable={!isMonitoring}
            />
          </View>

          {sessionId && (
            <View className="mb-3">
              <Text className="text-sm font-medium text-gray-700">Session ID</Text>
              <Text className="text-xs text-gray-500 bg-gray-50 p-2 rounded">{sessionId}</Text>
            </View>
          )}

          <View className="flex-row space-x-3">
            {!isMonitoring ? (
              <TouchableOpacity
                onPress={startMonitoring}
                className="flex-1 bg-green-500 p-3 rounded-lg flex-row items-center justify-center"
              >
                <Icon name="play-arrow" size={20} color="white" />
                <Text className="text-white font-medium ml-2">Start Monitoring</Text>
              </TouchableOpacity>
            ) : (
              <>
                <TouchableOpacity
                  onPress={stopMonitoring}
                  className="flex-1 bg-red-500 p-3 rounded-lg flex-row items-center justify-center"
                >
                  <Icon name="stop" size={20} color="white" />
                  <Text className="text-white font-medium ml-2">Stop</Text>
                </TouchableOpacity>
                
                <TouchableOpacity
                  onPress={processManually}
                  className="flex-1 bg-blue-500 p-3 rounded-lg flex-row items-center justify-center"
                >
                  <Icon name="analytics" size={20} color="white" />
                  <Text className="text-white font-medium ml-2">Process</Text>
                </TouchableOpacity>
              </>
            )}
          </View>

          {isMonitoring && (
            <TouchableOpacity
              onPress={resetSession}
              className="mt-3 bg-gray-500 p-2 rounded-lg flex-row items-center justify-center"
            >
              <Icon name="refresh" size={16} color="white" />
              <Text className="text-white font-medium ml-2">Reset Session</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Status */}
        <View className="mb-4">
          <StatusIndicator
            status={isMonitoring ? 'success' : 'warning'}
            label={isMonitoring ? 'Monitoring Active' : 'Monitoring Inactive'}
          />
          {error && (
            <View className="mt-2">
              <StatusIndicator status="error" label={`Error: ${error}`} />
            </View>
          )}
        </View>

        {/* Interactive Area for Touch Events */}
        {isMonitoring && (
          <TouchableOpacity
            activeOpacity={0.8}
            onPressIn={handleTouchEvent}
            onPressOut={handleTouchEvent}
            className="bg-blue-100 border-2 border-dashed border-blue-300 rounded-lg p-8 mb-4 items-center justify-center"
          >
            <Icon name="touch-app" size={48} color="#3b82f6" />
            <Text className="text-blue-600 font-medium mt-2">Touch & Interact Here</Text>
            <Text className="text-blue-500 text-sm text-center mt-1">
              Tap, swipe, and interact to generate touch pattern data
            </Text>
          </TouchableOpacity>
        )}

        {/* Sensor Data Display */}
        {sensorData && (
          <View className="mb-4">
            <SensorDisplay sensorData={sensorData} />
          </View>
        )}

        {/* Processing Results */}
        {processingResult && (
          <View className="mb-4">
            <ProcessingResult result={processingResult} />
            {lastProcessingTime && (
              <Text className="text-center text-xs text-gray-500 mt-2">
                Last processed: {lastProcessingTime.toLocaleTimeString()}
              </Text>
            )}
          </View>
        )}
      </View>
    </ScrollView>
  );
};

export default MonitoringScreen;