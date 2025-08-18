import React, { useState, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { Camera, RotateCcw, CircleCheck as CheckCircle, CircleAlert as AlertCircle, Eye } from 'lucide-react-native';
import { StatusBar } from 'expo-status-bar';

export default function CameraTab() {
  const [facing, setFacing] = useState<CameraType>('front');
  const [permission, requestPermission] = useCameraPermissions();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<{
    faceDetected: boolean;
    confidence: number;
    biometricMatch: boolean;
  } | null>(null);
  const cameraRef = useRef<CameraView>(null);

  if (!permission) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <Camera size={48} color="#6B7280" />
          <Text style={styles.loadingText}>Loading camera permissions...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionContainer}>
          <AlertCircle size={64} color="#EF4444" />
          <Text style={styles.permissionTitle}>Camera Access Required</Text>
          <Text style={styles.permissionMessage}>
            QuadFusion needs camera access for biometric authentication and facial recognition.
          </Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Text style={styles.permissionButtonText}>Grant Camera Permission</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  const toggleCameraFacing = () => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  };

  const analyzeFrame = async () => {
    if (isAnalyzing) return;

    setIsAnalyzing(true);

    // Simulate biometric analysis
    setTimeout(() => {
      const faceDetected = Math.random() > 0.2;
      const confidence = faceDetected ? 70 + Math.random() * 30 : Math.random() * 40;
      const biometricMatch = faceDetected && confidence > 85;

      setAnalysisResult({
        faceDetected,
        confidence,
        biometricMatch,
      });
      setIsAnalyzing(false);

      if (biometricMatch) {
        Alert.alert(
          'Authentication Successful',
          `Biometric match confirmed with ${confidence.toFixed(1)}% confidence.`,
          [{ text: 'OK' }]
        );
      } else if (faceDetected) {
        Alert.alert(
          'Authentication Failed',
          `Face detected but biometric match failed (${confidence.toFixed(1)}% confidence).`,
          [{ text: 'Try Again' }]
        );
      } else {
        Alert.alert(
          'No Face Detected',
          'Please position your face clearly within the camera frame.',
          [{ text: 'OK' }]
        );
      }
    }, 2000);
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />
      <View style={styles.header}>
        <Eye size={32} color="#FFFFFF" />
        <Text style={styles.title}>Facial Recognition</Text>
        <Text style={styles.subtitle}>Position your face in the camera frame</Text>
      </View>

      <View style={styles.cameraContainer}>
        <CameraView style={styles.camera} facing={facing} ref={cameraRef}>
          <View style={styles.overlay}>
            <View style={styles.faceFrame} />
            
            {analysisResult && (
              <View style={[
                styles.resultBadge,
                { backgroundColor: analysisResult.biometricMatch ? '#10B981' : '#EF4444' }
              ]}>
                {analysisResult.biometricMatch ? (
                  <CheckCircle size={16} color="#FFFFFF" />
                ) : (
                  <AlertCircle size={16} color="#FFFFFF" />
                )}
                <Text style={styles.resultText}>
                  {analysisResult.biometricMatch ? 'Match' : 'No Match'} - {analysisResult.confidence.toFixed(1)}%
                </Text>
              </View>
            )}
          </View>
        </CameraView>
      </View>

      <View style={styles.controls}>
        <TouchableOpacity style={styles.controlButton} onPress={toggleCameraFacing}>
          <RotateCcw size={24} color="#2563EB" />
          <Text style={styles.controlButtonText}>Flip Camera</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.analyzeButton, isAnalyzing && styles.analyzeButtonActive]}
          onPress={analyzeFrame}
          disabled={isAnalyzing}
        >
          <Eye size={24} color="#FFFFFF" />
          <Text style={styles.analyzeButtonText}>
            {isAnalyzing ? 'Analyzing...' : 'Analyze Face'}
          </Text>
        </TouchableOpacity>
      </View>

      {analysisResult && (
        <View style={styles.resultsCard}>
          <Text style={styles.resultsTitle}>Analysis Results</Text>
          
          <View style={styles.resultItem}>
            <Text style={styles.resultLabel}>Face Detection:</Text>
            <Text style={[
              styles.resultValue,
              { color: analysisResult.faceDetected ? '#10B981' : '#EF4444' }
            ]}>
              {analysisResult.faceDetected ? 'Detected' : 'Not Detected'}
            </Text>
          </View>

          <View style={styles.resultItem}>
            <Text style={styles.resultLabel}>Confidence:</Text>
            <Text style={styles.resultValue}>{analysisResult.confidence.toFixed(1)}%</Text>
          </View>

          <View style={styles.resultItem}>
            <Text style={styles.resultLabel}>Biometric Match:</Text>
            <Text style={[
              styles.resultValue,
              { color: analysisResult.biometricMatch ? '#10B981' : '#EF4444' }
            ]}>
              {analysisResult.biometricMatch ? 'Confirmed' : 'Failed'}
            </Text>
          </View>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F8FAFC',
  },
  loadingText: {
    fontSize: 16,
    color: '#6B7280',
    marginTop: 16,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F8FAFC',
    padding: 40,
  },
  permissionTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#1F2937',
    marginTop: 24,
    marginBottom: 16,
    textAlign: 'center',
  },
  permissionMessage: {
    fontSize: 16,
    color: '#6B7280',
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 32,
  },
  permissionButton: {
    backgroundColor: '#2563EB',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 12,
  },
  permissionButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  header: {
    alignItems: 'center',
    paddingTop: 20,
    paddingBottom: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: '#FFFFFF',
    marginTop: 12,
  },
  subtitle: {
    fontSize: 14,
    color: '#D1D5DB',
    marginTop: 4,
  },
  cameraContainer: {
    flex: 1,
    marginHorizontal: 20,
    marginBottom: 20,
    borderRadius: 16,
    overflow: 'hidden',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
  },
  faceFrame: {
    width: 200,
    height: 200,
    borderWidth: 3,
    borderColor: '#FFFFFF',
    borderRadius: 100,
    backgroundColor: 'transparent',
  },
  resultBadge: {
    position: 'absolute',
    top: 60,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
  },
  resultText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
    marginLeft: 6,
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 20,
    paddingBottom: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
  },
  controlButton: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
    flex: 0.4,
  },
  controlButtonText: {
    color: '#2563EB',
    fontSize: 14,
    fontWeight: '600',
    marginTop: 4,
  },
  analyzeButton: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2563EB',
    borderRadius: 12,
    padding: 16,
    flex: 0.5,
    flexDirection: 'row',
  },
  analyzeButtonActive: {
    backgroundColor: '#F59E0B',
  },
  analyzeButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
    marginLeft: 8,
  },
  resultsCard: {
    backgroundColor: '#FFFFFF',
    margin: 20,
    marginTop: 0,
    borderRadius: 16,
    padding: 20,
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 16,
  },
  resultItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
  },
  resultLabel: {
    fontSize: 16,
    color: '#374151',
    fontWeight: '500',
  },
  resultValue: {
    fontSize: 16,
    fontWeight: '600',
  },
});