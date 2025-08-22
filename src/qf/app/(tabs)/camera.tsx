import React, { useState, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, Animated } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { Camera, RotateCcw, CircleCheck as CheckCircle, CircleAlert as AlertCircle, Eye } from 'lucide-react-native';
import { StatusBar } from 'expo-status-bar';
import { COLORS, ANIMATION } from '../../lib/constants';
import { SPACING, BORDER_RADIUS } from '../../lib/theme';
import GridBackground from '../../components/GridBackground';
import { useGlowAnimation, usePulseAnimation } from '../../lib/animations';

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
  
  const { glowAnim, startGlowAnimation } = useGlowAnimation(0.4, 0.8);
  const { pulseAnim, startPulseAnimation } = usePulseAnimation(0.05);

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
      <GridBackground spacing={30} opacity={0.15} />
      <Animated.View style={[styles.header, { shadowOpacity: glowAnim }]}>
        <Eye size={32} color={COLORS.ACCENT} />
        <Text style={styles.title}>Facial Recognition</Text>
        <Text style={styles.subtitle}>Position your face in the camera frame</Text>
      </Animated.View>

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
    backgroundColor: COLORS.BACKGROUND,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: COLORS.BACKGROUND,
  },
  loadingText: {
    fontSize: 16,
    color: COLORS.GRAY_300,
    marginTop: SPACING.MD,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 2,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: COLORS.BACKGROUND,
    padding: SPACING.XL,
  },
  permissionTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: COLORS.WHITE,
    marginTop: SPACING.LG,
    marginBottom: SPACING.MD,
    textAlign: 'center',
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
  },
  permissionMessage: {
    fontSize: 16,
    color: COLORS.GRAY_300,
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: SPACING.XL,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 2,
  },
  permissionButton: {
    backgroundColor: COLORS.PRIMARY,
    paddingHorizontal: SPACING.LG,
    paddingVertical: SPACING.SM,
    borderRadius: BORDER_RADIUS.MD,
    borderWidth: 1,
    borderColor: COLORS.ACCENT,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
  },
  permissionButtonText: {
    color: COLORS.WHITE,
    fontSize: 16,
    fontWeight: '600',
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
  },
  header: {
    alignItems: 'center',
    paddingTop: SPACING.MD,
    paddingBottom: SPACING.MD,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderBottomWidth: 1,
    borderBottomColor: COLORS.GLOW,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: COLORS.WHITE,
    marginTop: SPACING.SM,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
  },
  subtitle: {
    fontSize: 14,
    color: COLORS.GRAY_300,
    marginTop: SPACING.XS,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 2,
  },
  cameraContainer: {
    flex: 1,
    marginHorizontal: SPACING.MD,
    marginBottom: SPACING.MD,
    borderRadius: BORDER_RADIUS.XL,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: COLORS.GLOW,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
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
    borderWidth: 2,
    borderColor: COLORS.ACCENT,
    borderRadius: 100,
    backgroundColor: 'transparent',
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
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
    paddingHorizontal: SPACING.MD,
    paddingBottom: SPACING.MD,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderTopWidth: 1,
    borderTopColor: COLORS.GLOW,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
  },
  controlButton: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.CARD,
    borderRadius: BORDER_RADIUS.MD,
    padding: SPACING.MD,
    flex: 0.4,
    borderWidth: 1,
    borderColor: COLORS.ACCENT,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
  },
  controlButtonText: {
    color: COLORS.PRIMARY,
    fontSize: 14,
    fontWeight: '600',
    marginTop: SPACING.XS,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
  },
  analyzeButton: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.PRIMARY,
    borderRadius: BORDER_RADIUS.MD,
    padding: SPACING.MD,
    flex: 0.5,
    flexDirection: 'row',
    borderWidth: 1,
    borderColor: COLORS.ACCENT,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
  },
  analyzeButtonActive: {
    backgroundColor: COLORS.WARNING,
  },
  analyzeButtonText: {
    color: COLORS.WHITE,
    fontSize: 14,
    fontWeight: '600',
    marginLeft: SPACING.XS,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
  },
  resultsCard: {
    backgroundColor: COLORS.CARD,
    margin: SPACING.MD,
    marginTop: 0,
    borderRadius: BORDER_RADIUS.LG,
    padding: SPACING.MD,
    borderWidth: 1,
    borderColor: COLORS.GLOW,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: COLORS.WHITE,
    marginBottom: SPACING.MD,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
  },
  resultItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: SPACING.XS,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.GRAY_700,
  },
  resultLabel: {
    fontSize: 16,
    color: COLORS.GRAY_300,
    fontWeight: '500',
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 2,
  },
  resultValue: {
    fontSize: 16,
    fontWeight: '600',
    color: COLORS.WHITE,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 2,
  },
});