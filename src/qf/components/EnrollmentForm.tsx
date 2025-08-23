import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, Alert, StyleSheet, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as Icons from 'lucide-react-native';
import { api, type BiometricEnrollment, type EnrollmentResult } from '../lib/api';
import { sensorManager } from '../lib/sensor-manager';
import { generateSessionId } from '../lib/utils';

const Shield = Icons.Shield ?? (() => null);
const User = Icons.User ?? (() => null);
const CheckCircle = Icons.CheckCircle ?? (() => null);
const AlertCircle = Icons.AlertCircle ?? (() => null);

interface EnrollmentFormProps {
  onEnrollmentComplete?: (result: EnrollmentResult) => void;
  onClose?: () => void;
}

type EnrollmentStep = 'input' | 'collecting' | 'processing' | 'complete' | 'error';

export default function EnrollmentForm({ onEnrollmentComplete, onClose }: EnrollmentFormProps) {
  const [userId, setUserId] = useState('');
  const [currentStep, setCurrentStep] = useState<EnrollmentStep>('input');
  const [statusMessage, setStatusMessage] = useState('');
  const [enrollmentResult, setEnrollmentResult] = useState<EnrollmentResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const startEnrollment = async () => {
    if (!userId.trim()) {
      Alert.alert('Error', 'Please enter a valid User ID');
      return;
    }

    setCurrentStep('collecting');
    setStatusMessage('Requesting permissions...');
    setError(null);

    try {
      // Request permissions
      const permissionsGranted = await sensorManager.requestPermissions();
      if (!permissionsGranted) {
        throw new Error('Required permissions not granted');
      }

      // Start sensor collection
      setStatusMessage('Starting sensor collection...');
      await sensorManager.startMotionSensors();

      // Simulate biometric data collection process
      setStatusMessage('Collecting motion data...');
      await new Promise(resolve => setTimeout(resolve, 3000));

      setStatusMessage('Analyzing touch patterns...');
      await new Promise(resolve => setTimeout(resolve, 2000));

      setStatusMessage('Recording behavioral patterns...');
      await new Promise(resolve => setTimeout(resolve, 2000));

      setCurrentStep('processing');
      setStatusMessage('Processing enrollment data...');

      // Get current sensor data
      const sensorData = sensorManager.getCurrentSensorData();

      // Prepare biometric enrollment data
      const biometricData: BiometricEnrollment = {
        voice_samples: [], // Would be populated with actual audio data
        face_images: [], // Would be populated with actual image data
        typing_samples: [
          {
            keystrokes: sensorData.keystroke_events,
            timings: sensorData.keystroke_events.map(e => e.timestamp),
          }
        ],
        touch_samples: [
          {
            gestures: sensorData.touch_events,
            pressure_data: sensorData.touch_events.map(e => e.pressure),
            motion_data: sensorData.motion_data,
          }
        ],
      };

      // Submit enrollment to API
      const result = await api.enrollUser(userId, biometricData);
      
      setEnrollmentResult(result);
      setCurrentStep('complete');
      setStatusMessage(`Enrollment ${result.status}`);

      // Notify parent component
      onEnrollmentComplete?.(result);

    } catch (error) {
      console.error('Enrollment failed:', error);
      setError(error instanceof Error ? error.message : 'Unknown error occurred');
      setCurrentStep('error');
      setStatusMessage('Enrollment failed');
    } finally {
      // Clean up
      sensorManager.stopMotionSensors();
      sensorManager.resetSensorData();
    }
  };

  const resetForm = () => {
    setCurrentStep('input');
    setUserId('');
    setStatusMessage('');
    setError(null);
    setEnrollmentResult(null);
    sensorManager.stopMotionSensors();
    sensorManager.resetSensorData();
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 'input':
        return (
          <View style={styles.inputContainer}>
            <Text style={styles.inputLabel}>User ID</Text>
                  <TextInput
                    style={styles.textInput}
                    value={userId}
                    onChangeText={setUserId}
                    placeholder="Enter your user ID"
                    autoCapitalize="none"
                    autoCorrect={false}
                    onKeyPress={(e) => {
                      try {
                        sensorManager.addKeystrokeEvent({ keyCode: e.nativeEvent.key, type: 'keydown' });
                      } catch (ex) { /* ignore */ }
                    }}
                  />
            <TouchableOpacity
              style={styles.primaryButton}
              onPress={startEnrollment}
              disabled={!userId.trim()}
            >
              <User size={20} color="#FFFFFF" />
              <Text style={styles.primaryButtonText}>Start Enrollment</Text>
            </TouchableOpacity>
          </View>
        );

      case 'collecting':
      case 'processing':
        return (
          <View style={styles.statusContainer}>
            <ActivityIndicator size="large" color="#2563EB" />
            <Text style={styles.statusText}>{statusMessage}</Text>
            <Text style={styles.statusSubtext}>
              {currentStep === 'collecting' 
                ? 'Please interact normally with your device'
                : 'This may take a few moments...'}
            </Text>
          </View>
        );

      case 'complete':
        return (
          <View style={styles.resultContainer}>
            <CheckCircle size={64} color="#10B981" />
            <Text style={styles.successTitle}>Enrollment Complete!</Text>
            <Text style={styles.successMessage}>{statusMessage}</Text>
            {enrollmentResult && (
              <View style={styles.resultDetails}>
                <Text style={styles.resultLabel}>Enrollment ID:</Text>
                <Text style={styles.resultValue}>{enrollmentResult.enrollment_id}</Text>
                <Text style={styles.resultLabel}>Models Trained:</Text>
                <Text style={styles.resultValue}>
                  {enrollmentResult.models_trained.join(', ')}
                </Text>
              </View>
            )}
            <TouchableOpacity style={styles.primaryButton} onPress={onClose || resetForm}>
              <Text style={styles.primaryButtonText}>Continue</Text>
            </TouchableOpacity>
          </View>
        );

      case 'error':
        return (
          <View style={styles.errorContainer}>
            <AlertCircle size={64} color="#EF4444" />
            <Text style={styles.errorTitle}>Enrollment Failed</Text>
            <Text style={styles.errorMessage}>{error}</Text>
            <View style={styles.buttonContainer}>
              <TouchableOpacity style={styles.secondaryButton} onPress={resetForm}>
                <Text style={styles.secondaryButtonText}>Try Again</Text>
              </TouchableOpacity>
              {onClose && (
                <TouchableOpacity style={styles.primaryButton} onPress={onClose}>
                  <Text style={styles.primaryButtonText}>Close</Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
        );
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Shield size={48} color="#2563EB" />
        <Text style={styles.title}>Biometric Enrollment</Text>
        <Text style={styles.subtitle}>Secure your account with behavioral biometrics</Text>
      </View>

      <View style={styles.content}>
        {renderStepContent()}
      </View>

      {onClose && currentStep === 'input' && (
        <TouchableOpacity style={styles.closeButton} onPress={onClose}>
          <Text style={styles.closeButtonText}>Cancel</Text>
        </TouchableOpacity>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F8FAFC',
  },
  header: {
    alignItems: 'center',
    paddingVertical: 32,
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#1F2937',
    marginTop: 16,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#6B7280',
    marginTop: 8,
    textAlign: 'center',
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
  },
  inputContainer: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  inputLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#D1D5DB',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    marginBottom: 24,
    backgroundColor: '#FFFFFF',
  },
  statusContainer: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 32,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  statusText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
    marginTop: 16,
    textAlign: 'center',
  },
  statusSubtext: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 8,
    textAlign: 'center',
  },
  resultContainer: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 32,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  successTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#10B981',
    marginTop: 16,
    textAlign: 'center',
  },
  successMessage: {
    fontSize: 16,
    color: '#6B7280',
    marginTop: 8,
    textAlign: 'center',
  },
  resultDetails: {
    marginTop: 24,
    width: '100%',
  },
  resultLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginTop: 12,
  },
  resultValue: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 4,
  },
  errorContainer: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 32,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  errorTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#EF4444',
    marginTop: 16,
    textAlign: 'center',
  },
  errorMessage: {
    fontSize: 16,
    color: '#6B7280',
    marginTop: 8,
    textAlign: 'center',
  },
  primaryButton: {
    backgroundColor: '#2563EB',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 24,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  secondaryButton: {
    backgroundColor: '#F3F4F6',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 16,
  },
  secondaryButtonText: {
    color: '#374151',
    fontSize: 16,
    fontWeight: '600',
  },
  buttonContainer: {
    width: '100%',
  },
  closeButton: {
    padding: 16,
    alignItems: 'center',
  },
  closeButtonText: {
    color: '#6B7280',
    fontSize: 16,
    fontWeight: '600',
  },
});
