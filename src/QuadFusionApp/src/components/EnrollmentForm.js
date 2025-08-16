import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, Alert } from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import StatusIndicator from './StatusIndicator';
import SensorManager from '../services/SensorManager';
import QuadFusionAPI from '../services/QuadFusionAPI';

const EnrollmentForm = ({ onEnrollmentComplete }) => {
  const [userId, setUserId] = useState('');
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [enrollmentStatus, setEnrollmentStatus] = useState(null);
  const [currentStep, setCurrentStep] = useState('input'); // input, collecting, processing, complete

  const startEnrollment = async () => {
    if (!userId.trim()) {
      Alert.alert('Error', 'Please enter a valid User ID');
      return;
    }

    setIsEnrolling(true);
    setCurrentStep('collecting');
    setEnrollmentStatus('Collecting biometric samples...');

    try {
      // Start collecting sensor data
      await SensorManager.requestPermissions();
      SensorManager.startMotionSensors();

      // Collect samples for enrollment
      setEnrollmentStatus('Recording voice sample...');
      const audioData = await SensorManager.captureAudio(3000);

      setEnrollmentStatus('Capturing face image...');
      const imageData = await SensorManager.captureImage();

      setEnrollmentStatus('Analyzing touch patterns...');
      // Simulate touch data collection
      await new Promise(resolve => setTimeout(resolve, 2000));

      setCurrentStep('processing');
      setEnrollmentStatus('Processing biometric data...');

      // Prepare enrollment data
      const biometricData = {
        voice_samples: [audioData],
        face_images: [imageData],
        typing_samples: [{ keystrokes: [], timings: [] }],
        touch_samples: [{ gestures: [], pressure_data: [] }]
      };

      // Send to API
      const result = await QuadFusionAPI.enrollUser(userId, biometricData);
      
      SensorManager.stopMotionSensors();
      
      if (result.status === 'enrolled') {
        setCurrentStep('complete');
        setEnrollmentStatus('Enrollment completed successfully!');
        setTimeout(() => {
          onEnrollmentComplete?.(userId, result);
        }, 2000);
      } else {
        throw new Error(`Enrollment failed: ${result.status}`);
      }

    } catch (error) {
      console.error('Enrollment error:', error);
      setEnrollmentStatus(`Enrollment failed: ${error.message}`);
      SensorManager.stopMotionSensors();
    } finally {
      setIsEnrolling(false);
    }
  };

  const resetEnrollment = () => {
    setCurrentStep('input');
    setEnrollmentStatus(null);
    setUserId('');
  };

  return (
    <View className="bg-white rounded-lg shadow-sm p-4">
      <Text className="text-lg font-bold text-gray-800 mb-4">User Enrollment</Text>
      
      {currentStep === 'input' && (
        <>
          <View className="mb-4">
            <Text className="text-sm font-medium text-gray-700 mb-2">User ID</Text>
            <TextInput
              value={userId}
              onChangeText={setUserId}
              placeholder="Enter unique user identifier"
              className="border border-gray-300 rounded-lg px-3 py-2 text-gray-700"
              autoCapitalize="none"
            />
          </View>
          
          <TouchableOpacity
            onPress={startEnrollment}
            disabled={isEnrolling || !userId.trim()}
            className={`p-3 rounded-lg flex-row items-center justify-center ${
              isEnrolling || !userId.trim() 
                ? 'bg-gray-300' 
                : 'bg-blue-500'
            }`}
          >
            <Icon name="person-add" size={20} color="white" />
            <Text className="text-white font-medium ml-2">Start Enrollment</Text>
          </TouchableOpacity>
        </>
      )}

      {(currentStep === 'collecting' || currentStep === 'processing') && (
        <View className="items-center">
          <StatusIndicator 
            status="loading" 
            label={enrollmentStatus || 'Processing...'} 
          />
        </View>
      )}

      {currentStep === 'complete' && (
        <View className="items-center">
          <StatusIndicator 
            status="success" 
            label={enrollmentStatus || 'Enrollment Complete!'} 
          />
          <TouchableOpacity
            onPress={resetEnrollment}
            className="mt-4 p-2 bg-blue-500 rounded-lg"
          >
            <Text className="text-white font-medium">Enroll Another User</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
};