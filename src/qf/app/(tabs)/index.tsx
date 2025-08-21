import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert, ScrollView, Animated } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as LocalAuthentication from 'expo-local-authentication';
import * as Icons from 'lucide-react-native';
import { StatusBar } from 'expo-status-bar';
import { COLORS, ANIMATION } from '@/lib/constants';
import EnrollmentForm from '../../components/EnrollmentForm';
import GridBackground from '../../components/GridBackground';
import { SPACING, BORDER_RADIUS } from '../../lib/theme';
import { useGlowAnimation, usePulseAnimation } from '../../lib/animations';

const Shield = Icons.Shield ?? (() => null);
const CheckCircle = Icons.CheckCircle ?? Icons.CircleCheck ?? (() => null);
const XCircle = Icons.XCircle ?? Icons.Circle ?? (() => null);
const Fingerprint = Icons.Fingerprint ?? (() => null);
const Scan = Icons.Scan ?? (() => null);
const AlertTriangle = Icons.AlertTriangle ?? Icons.TriangleAlert ?? (() => null);
const Activity = Icons.Activity ?? (() => null);
const UserPlus = Icons.UserPlus ?? (() => null);

interface AuthResult {
  success: boolean;
  timestamp: string;
  method: string;
  riskScore: number;
}

export default function AuthenticationTab() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [authHistory, setAuthHistory] = useState<AuthResult[]>([]);
  const [biometricType, setBiometricType] = useState<string>('');
  const [showEnrollment, setShowEnrollment] = useState(false);
  
  const { glowAnim, startGlowAnimation } = useGlowAnimation(0.4, 0.8);
  const { pulseAnim, startPulseAnimation } = usePulseAnimation(0.05);

  useEffect(() => {
    checkBiometricSupport();
  }, []);
  
  // Initialize animations on component mount
  useEffect(() => {
    // No continuous animations as per user preference
    // Animations will only be triggered by user interactions
  }, []);

  const checkBiometricSupport = async () => {
    const compatible = await LocalAuthentication.hasHardwareAsync();
    if (compatible) {
      const types = await LocalAuthentication.supportedAuthenticationTypesAsync();
      if (types.includes(LocalAuthentication.AuthenticationType.FINGERPRINT)) {
        setBiometricType('Fingerprint');
      } else if (types.includes(LocalAuthentication.AuthenticationType.FACIAL_RECOGNITION)) {
        setBiometricType('Face ID');
      } else {
        setBiometricType('Biometric');
      }
    }
  };

  const authenticate = async () => {
    setIsAuthenticating(true);
    
    try {
      // Simulate risk assessment
      const riskScore = Math.random() * 100;
      
      const result = await LocalAuthentication.authenticateAsync({
        promptMessage: 'QuadFusion Authentication',
        fallbackLabel: 'Use Passcode',
        disableDeviceFallback: false,
      });

      const authResult: AuthResult = {
        success: result.success,
        timestamp: new Date().toISOString(),
        method: biometricType,
        riskScore: result.success ? riskScore : 95 + Math.random() * 5,
      };

      setAuthHistory(prev => [authResult, ...prev.slice(0, 4)]);
      setIsAuthenticated(result.success);

      if (!result.success) {
        Alert.alert(
          'Authentication Failed',
          'Biometric authentication was not successful. Please try again.',
          [{ text: 'OK' }]
        );
      }
    } catch (error) {
      Alert.alert('Error', 'An error occurred during authentication');
    } finally {
      setIsAuthenticating(false);
    }
  };

  const logout = () => {
    setIsAuthenticated(false);
  };

  const getRiskColor = (score: number) => {
    if (score < 30) return '#10B981'; // Green
    if (score < 70) return '#F59E0B'; // Yellow
    return '#EF4444'; // Red
  };

  const getRiskLabel = (score: number) => {
    if (score < 30) return 'Low Risk';
    if (score < 70) return 'Medium Risk';
    return 'High Risk';
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />
      <GridBackground spacing={30} opacity={0.15} />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <View style={styles.headerTop}>
            <View style={styles.cornerAccent} />
            <View style={[styles.cornerAccent, styles.cornerAccentTopRight]} />
            <View style={[styles.cornerAccent, styles.cornerAccentBottomLeft]} />
            <View style={[styles.cornerAccent, styles.cornerAccentBottomRight]} />
            <Shield size={48} color={COLORS.ACCENT} style={styles.headerIcon} />
            <Animated.Text style={[styles.title, { textShadowRadius: pulseAnim.interpolate({
              inputRange: [1, 1.05],
              outputRange: [4, 8]
            }) }]}>QuadFusion Security</Animated.Text>
            <Text style={styles.subtitle}>Advanced Biometric Authentication</Text>
            <View style={styles.headerDivider} />
          </View>
        </View>

        <Animated.View style={[styles.statusCard, {
          shadowOpacity: glowAnim,
          transform: [{ scale: pulseAnim }]
        }]}>
          <View style={styles.cornerAccent} />
          <View style={[styles.cornerAccent, styles.cornerAccentTopRight]} />
          <View style={[styles.cornerAccent, styles.cornerAccentBottomLeft]} />
          <View style={[styles.cornerAccent, styles.cornerAccentBottomRight]} />
          <View style={styles.statusHeader}>
            {isAuthenticated ? (
              <CheckCircle size={32} color="#10B981" />
            ) : (
              <XCircle size={32} color="#EF4444" />
            )}
            <Text style={[styles.statusText, { color: isAuthenticated ? '#10B981' : '#EF4444' }]}>
              {isAuthenticated ? 'Authenticated' : 'Not Authenticated'}
            </Text>
          </View>

          {!isAuthenticated ? (
            <>
              <TouchableOpacity
                style={[styles.enrollButton]}
                onPress={() => setShowEnrollment(true)}
              >
                <UserPlus size={20} color="#2563EB" />
                <Text style={styles.enrollButtonText}>Enroll New User</Text>
              </TouchableOpacity>
              
              <TouchableOpacity
                style={[styles.authButton, isAuthenticating && styles.authButtonDisabled]}
                onPress={authenticate}
                disabled={isAuthenticating}
              >
                {isAuthenticating ? (
                  <Scan size={24} color="#FFFFFF" />
                ) : (
                  <Fingerprint size={24} color="#FFFFFF" />
                )}
                <Text style={styles.authButtonText}>
                  {isAuthenticating ? 'Authenticating...' : `Authenticate with ${biometricType}`}
                </Text>
              </TouchableOpacity>
            </>
          ) : (
            <TouchableOpacity style={styles.logoutButton} onPress={logout}>
              <Text style={styles.logoutButtonText}>Logout</Text>
            </TouchableOpacity>
          )}
        </View>

        {authHistory.length > 0 && (
          <Animated.View style={[styles.historyCard, {
            shadowOpacity: glowAnim,
            transform: [{ scale: pulseAnim }]
          }]}>
            <View style={styles.cornerAccent} />
            <View style={[styles.cornerAccent, styles.cornerAccentTopRight]} />
            <View style={[styles.cornerAccent, styles.cornerAccentBottomLeft]} />
            <View style={[styles.cornerAccent, styles.cornerAccentBottomRight]} />
            <Text style={styles.historyTitle}>Recent Authentication Attempts</Text>
            {authHistory.map((attempt, index) => (
              <View key={index} style={styles.historyItem}>
                <View style={styles.historyHeader}>
                  {attempt.success ? (
                    <CheckCircle size={20} color="#10B981" />
                  ) : (
                    <XCircle size={20} color="#EF4444" />
                  )}
                  <Text style={styles.historyMethod}>{attempt.method}</Text>
                  <View style={[styles.riskBadge, { backgroundColor: getRiskColor(attempt.riskScore) }]}>
                    <Text style={styles.riskBadgeText}>{getRiskLabel(attempt.riskScore)}</Text>
                  </View>
                </View>
                <Text style={styles.historyTime}>
                  {new Date(attempt.timestamp).toLocaleString()}
                </Text>
                <Text style={styles.riskScore}>
                  Risk Score: {attempt.riskScore.toFixed(1)}%
                </Text>
              </View>
            ))}
          </Animated.View>
        )}

        <Animated.View style={[styles.featuresCard, {
          shadowOpacity: glowAnim,
          transform: [{ scale: pulseAnim }]
        }]}>
          <View style={styles.cornerAccent} />
          <View style={[styles.cornerAccent, styles.cornerAccentTopRight]} />
          <View style={[styles.cornerAccent, styles.cornerAccentBottomLeft]} />
          <View style={[styles.cornerAccent, styles.cornerAccentBottomRight]} />
          <Text style={styles.featuresTitle}>Security Features</Text>
          <View style={styles.featuresList}>
            <View style={styles.featureItem}>
              <Shield size={20} color="#2563EB" />
              <Text style={styles.featureText}>Multi-factor biometric verification</Text>
            </View>
            <View style={styles.featureItem}>
              <Activity size={20} color="#2563EB" />
              <Text style={styles.featureText}>Real-time fraud detection</Text>
            </View>
            <View style={styles.featureItem}>
              <AlertTriangle size={20} color="#2563EB" />
              <Text style={styles.featureText}>Advanced risk assessment</Text>
            </View>
          </View>
        </Animated.View>
      </ScrollView>
      
      {/* Enrollment Modal */}
      {showEnrollment && (
        <EnrollmentForm
          onClose={() => setShowEnrollment(false)}
          onEnrollmentComplete={(result) => {
            console.log('User enrolled:', result.enrollment_id);
            setShowEnrollment(false);
            Alert.alert('Success', 'User enrolled successfully!');
          }}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BACKGROUND,
  },
  scrollContent: {
    padding: SPACING.MD,
  },
  header: {
    alignItems: 'center',
    marginBottom: SPACING.XL,
  },
  headerTop: {
    alignItems: 'center',
    width: '100%',
    position: 'relative',
    paddingVertical: SPACING.LG,
    borderWidth: 1,
    borderColor: COLORS.GLOW,
    borderRadius: BORDER_RADIUS.LG,
    paddingHorizontal: SPACING.MD,
  },
  headerIcon: {
    shadowColor: COLORS.ACCENT,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: COLORS.WHITE,
    marginTop: SPACING.MD,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
  },
  subtitle: {
    fontSize: 16,
    color: COLORS.GRAY_300,
    marginTop: SPACING.XS,
  },
  headerDivider: {
    height: 2,
    backgroundColor: COLORS.GLOW,
    marginTop: SPACING.MD,
    width: '80%',
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 4,
  },
  cornerAccent: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderColor: COLORS.ACCENT,
    top: 0,
    left: 0,
    borderTopWidth: 2,
    borderLeftWidth: 2,
  },
  cornerAccentTopRight: {
    left: undefined,
    right: 0,
    borderTopWidth: 2,
    borderRightWidth: 2,
    borderLeftWidth: 0,
  },
  cornerAccentBottomLeft: {
    top: undefined,
    bottom: 0,
    borderBottomWidth: 2,
    borderLeftWidth: 2,
    borderTopWidth: 0,
  },
  cornerAccentBottomRight: {
    top: undefined,
    left: undefined,
    bottom: 0,
    right: 0,
    borderBottomWidth: 2,
    borderRightWidth: 2,
    borderTopWidth: 0,
    borderLeftWidth: 0,
  },
  statusCard: {
    backgroundColor: COLORS.CARD,
    borderRadius: BORDER_RADIUS.LG,
    padding: SPACING.LG,
    marginBottom: SPACING.LG,
    borderWidth: 1,
    borderColor: COLORS.GLOW,
    shadowColor: COLORS.PRIMARY,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 4,
    position: 'relative',
    overflow: 'hidden',
  },
  statusHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 24,
  },
  statusText: {
    fontSize: 20,
    fontWeight: '600',
    marginLeft: 12,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
  },
  authButton: {
    backgroundColor: COLORS.PRIMARY,
    borderRadius: BORDER_RADIUS.MD,
    padding: SPACING.MD,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.ACCENT,
  },
  authButtonDisabled: {
    opacity: 0.7,
  },
  authButtonText: {
    color: COLORS.WHITE,
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
    textShadowColor: 'rgba(0, 0, 0, 0.5)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 2,
  },
  logoutButton: {
    backgroundColor: COLORS.ERROR,
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 8,
  },
  logoutButtonText: {
    color: COLORS.WHITE,
    fontSize: 16,
    fontWeight: '600',
  },
  historyCard: {
    backgroundColor: COLORS.CARD,
    borderRadius: BORDER_RADIUS.LG,
    padding: SPACING.MD,
    marginBottom: SPACING.LG,
    borderWidth: 1,
    borderColor: COLORS.GLOW,
    shadowColor: COLORS.PRIMARY,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 4,
    position: 'relative',
    overflow: 'hidden',
  },
  historyTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: COLORS.WHITE,
    marginBottom: 16,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
  },
  historyItem: {
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.GRAY_700,
  },
  historyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  historyMethod: {
    fontSize: 16,
    fontWeight: '500',
    color: COLORS.WHITE,
    marginLeft: 8,
    flex: 1,
  },
  riskBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  riskBadgeText: {
    color: COLORS.WHITE,
    fontSize: 12,
    fontWeight: '600',
  },
  historyTime: {
    fontSize: 14,
    color: COLORS.GRAY_300,
    marginTop: 4,
  },
  riskScore: {
    fontSize: 14,
    color: COLORS.WHITE,
    marginTop: 2,
  },
  featuresCard: {
    backgroundColor: COLORS.CARD,
    borderRadius: BORDER_RADIUS.LG,
    padding: SPACING.MD,
    borderWidth: 1,
    borderColor: COLORS.GLOW,
    shadowColor: COLORS.PRIMARY,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 4,
    position: 'relative',
    overflow: 'hidden',
  },
  featuresTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: COLORS.WHITE,
    marginBottom: 16,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
  },
  featuresList: {
    gap: 12,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  featureText: {
    fontSize: 16,
    color: COLORS.WHITE,
    marginLeft: 12,
  },
  enrollButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'transparent',
    borderWidth: 2,
    borderColor: COLORS.PRIMARY,
    paddingVertical: SPACING.MD,
    paddingHorizontal: SPACING.XL,
    borderRadius: BORDER_RADIUS.MD,
    marginBottom: SPACING.MD,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
    elevation: 3,
  },
  enrollButtonText: {
    fontSize: 18,
    fontWeight: '600',
    color: COLORS.PRIMARY,
    marginLeft: 8,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
  },
});