import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as LocalAuthentication from 'expo-local-authentication';
import * as Icons from 'lucide-react-native';
import { StatusBar } from 'expo-status-bar';
import EnrollmentForm from '../../components/EnrollmentForm';

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

  useEffect(() => {
    checkBiometricSupport();
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
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Shield size={48} color="#2563EB" />
          <Text style={styles.title}>QuadFusion Security</Text>
          <Text style={styles.subtitle}>Advanced Biometric Authentication</Text>
        </View>

        <View style={styles.statusCard}>
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
          <View style={styles.historyCard}>
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
          </View>
        )}

        <View style={styles.featuresCard}>
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
        </View>
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
    backgroundColor: '#F8FAFC',
  },
  scrollContent: {
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 32,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#1F2937',
    marginTop: 16,
  },
  subtitle: {
    fontSize: 16,
    color: '#6B7280',
    marginTop: 4,
  },
  statusCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 24,
    marginBottom: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
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
  },
  authButton: {
    backgroundColor: '#2563EB',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  authButtonDisabled: {
    opacity: 0.7,
  },
  authButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  logoutButton: {
    backgroundColor: '#EF4444',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  logoutButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  historyCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  historyTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 16,
  },
  historyItem: {
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  historyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  historyMethod: {
    fontSize: 16,
    fontWeight: '500',
    color: '#374151',
    marginLeft: 8,
    flex: 1,
  },
  riskBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  riskBadgeText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
  },
  historyTime: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 4,
  },
  riskScore: {
    fontSize: 14,
    color: '#374151',
    marginTop: 2,
  },
  featuresCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  featuresTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 16,
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
    color: '#374151',
    marginLeft: 12,
  },
  enrollButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#FFFFFF',
    borderWidth: 2,
    borderColor: '#2563EB',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 12,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  enrollButtonText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#2563EB',
    marginLeft: 8,
  },
});