import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Alert } from 'react-native';
import * as Icons from 'lucide-react-native';
import { SensorManager } from '../lib/sensor-manager';
import { QuadFusionAPI } from '../lib/api';
import ProcessingResultDisplay from '../components/ProcessingResultDisplay';
import StatusIndicator from '../components/StatusIndicator';
import type { SensorData, ProcessingResult } from '../lib/api';

const Play = Icons.Play ?? (() => null);
const Square = Icons.Square ?? (() => null);
const RotateCcw = Icons.RotateCcw ?? (() => null);
const Activity = Icons.Activity ?? (() => null);

interface LiveMonitoringProps {
  onDataCollected?: (data: SensorData) => void;
  onProcessingResult?: (result: ProcessingResult) => void;
}

export default function LiveMonitoring({ 
  onDataCollected, 
  onProcessingResult 
}: LiveMonitoringProps) {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [sensorData, setSensorData] = useState<SensorData | null>(null);
  const [processingResult, setProcessingResult] = useState<ProcessingResult | null>(null);
  const [monitoringDuration, setMonitoringDuration] = useState(0);
  const [sensorManager] = useState(() => new SensorManager());
  const [api] = useState(() => new QuadFusionAPI());

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    
    if (isMonitoring) {
      interval = setInterval(() => {
        setMonitoringDuration((prev) => prev + 1);
      }, 1000);
    }
    
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isMonitoring]);

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const startMonitoring = async () => {
    try {
      // Request permissions
      const hasPermissions = await sensorManager.requestPermissions();
      if (!hasPermissions) {
        Alert.alert(
          'Permissions Required',
          'Please grant all permissions to enable complete monitoring.',
          [
            {
              text: 'Continue Anyway',
              onPress: () => proceedWithMonitoring(),
            },
            {
              text: 'Cancel',
              style: 'cancel',
            },
          ]
        );
        return;
      }

      await proceedWithMonitoring();
    } catch (error) {
      console.error('Error starting monitoring:', error);
      Alert.alert('Error', 'Failed to start monitoring. Please try again.');
    }
  };

  const proceedWithMonitoring = async () => {
    try {
      // Start motion sensors
      await sensorManager.startMotionSensors();
      
      setIsMonitoring(true);
      setMonitoringDuration(0);
      setSensorData(null);
      setProcessingResult(null);
    } catch (error) {
      console.error('Error starting sensors:', error);
      Alert.alert('Error', 'Failed to start sensors. Please try again.');
    }
  };

  const stopMonitoring = async () => {
    try {
      setIsMonitoring(false);
      
      // Stop sensors and get collected data
      sensorManager.stopMotionSensors();
      const collectedData = sensorManager.getSensorData();
      
      if (collectedData) {
        setSensorData(collectedData);
        onDataCollected?.(collectedData);
        
        // Process the data
        await processCollectedData(collectedData);
      }
    } catch (error) {
      console.error('Error stopping monitoring:', error);
      Alert.alert('Error', 'Failed to stop monitoring properly.');
    }
  };

  const processCollectedData = async (data: SensorData) => {
    try {
      setIsProcessing(true);
      
      // Send data for real-time processing (need session ID for real API)
      const sessionId = 'live-monitoring-' + Date.now();
      const result = await api.processRealtimeSensorData(sessionId, data);
      
      setProcessingResult(result);
      onProcessingResult?.(result);
    } catch (error) {
      console.error('Error processing data:', error);
      Alert.alert('Error', 'Failed to process collected data.');
    } finally {
      setIsProcessing(false);
    }
  };

  const resetSession = () => {
    setSensorData(null);
    setProcessingResult(null);
    setMonitoringDuration(0);
  };

  const getMonitoringStatus = () => {
    if (isMonitoring) {
      return {
        status: 'success' as const,
        message: 'Live Monitoring Active',
        submessage: `Duration: ${formatDuration(monitoringDuration)}`,
      };
    }
    
    if (isProcessing) {
      return {
        status: 'loading' as const,
        message: 'Processing Data',
        submessage: 'Analyzing behavioral patterns...',
      };
    }
    
    if (processingResult) {
      return {
        status: 'success' as const,
        message: 'Analysis Complete',
        submessage: `Risk Level: ${processingResult.risk_level}`,
      };
    }
    
    return {
      status: 'idle' as const,
      message: 'Ready to Monitor',
      submessage: 'Start monitoring to collect biometric data',
    };
  };

  const status = getMonitoringStatus();

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Status Display */}
      <View style={styles.statusCard}>
        <StatusIndicator
          status={status.status}
          message={status.message}
          submessage={status.submessage}
          size="large"
        />
      </View>

      {/* Control Panel */}
      <View style={styles.controlCard}>
        <Text style={styles.cardTitle}>Monitoring Controls</Text>
        
        <View style={styles.controlsContainer}>
          {!isMonitoring ? (
            <TouchableOpacity
              style={[styles.controlButton, styles.startButton]}
              onPress={startMonitoring}
              disabled={isProcessing}
            >
              <Play size={20} color="#FFFFFF" />
              <Text style={styles.startButtonText}>Start Monitoring</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              style={[styles.controlButton, styles.stopButton]}
              onPress={stopMonitoring}
            >
              <Square size={20} color="#FFFFFF" />
              <Text style={styles.stopButtonText}>Stop Monitoring</Text>
            </TouchableOpacity>
          )}
          
          <TouchableOpacity
            style={[styles.controlButton, styles.resetButton]}
            onPress={resetSession}
            disabled={isMonitoring || isProcessing}
          >
            <RotateCcw size={20} color="#6B7280" />
            <Text style={styles.resetButtonText}>Reset</Text>
          </TouchableOpacity>
        </View>

        {/* Monitoring Stats */}
        {(isMonitoring || sensorData) && (
          <View style={styles.statsContainer}>
            <View style={styles.stat}>
              <Activity size={16} color="#2563EB" />
              <Text style={styles.statLabel}>Duration</Text>
              <Text style={styles.statValue}>{formatDuration(monitoringDuration)}</Text>
            </View>
            
            {sensorData && (
              <>
                <View style={styles.stat}>
                  <Text style={styles.statLabel}>Touch Events</Text>
                  <Text style={styles.statValue}>{sensorData.touch_events?.length || 0}</Text>
                </View>
                
                <View style={styles.stat}>
                  <Text style={styles.statLabel}>Keystrokes</Text>
                  <Text style={styles.statValue}>{sensorData.keystroke_events?.length || 0}</Text>
                </View>
              </>
            )}
          </View>
        )}
      </View>

      {/* Processing Results */}
      {(processingResult || isProcessing || sensorData) && (
        <ProcessingResultDisplay
          result={processingResult ?? undefined}
          sensorData={sensorData ?? undefined}
          isProcessing={isProcessing}
        />
      )}

      {/* Instructions */}
      {!isMonitoring && !processingResult && (
        <View style={styles.instructionsCard}>
          <Text style={styles.cardTitle}>How It Works</Text>
          <View style={styles.instruction}>
            <Text style={styles.instructionNumber}>1</Text>
            <Text style={styles.instructionText}>
              Tap "Start Monitoring" to begin collecting biometric data
            </Text>
          </View>
          <View style={styles.instruction}>
            <Text style={styles.instructionNumber}>2</Text>
            <Text style={styles.instructionText}>
              Use your device naturally - touch, type, move around
            </Text>
          </View>
          <View style={styles.instruction}>
            <Text style={styles.instructionNumber}>3</Text>
            <Text style={styles.instructionText}>
              Stop monitoring when you have enough data (recommended: 30s+)
            </Text>
          </View>
          <View style={styles.instruction}>
            <Text style={styles.instructionNumber}>4</Text>
            <Text style={styles.instructionText}>
              View the analysis results and risk assessment
            </Text>
          </View>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  statusCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 20,
    margin: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  controlCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
    marginHorizontal: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: 16,
  },
  controlsContainer: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  controlButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    gap: 8,
  },
  startButton: {
    backgroundColor: '#2563EB',
  },
  startButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  stopButton: {
    backgroundColor: '#EF4444',
  },
  stopButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  resetButton: {
    backgroundColor: '#F3F4F6',
    borderWidth: 1,
    borderColor: '#D1D5DB',
  },
  resetButtonText: {
    color: '#6B7280',
    fontSize: 16,
    fontWeight: '600',
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#F3F4F6',
  },
  stat: {
    alignItems: 'center',
    gap: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#6B7280',
  },
  statValue: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1F2937',
  },
  instructionsCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
    marginHorizontal: 16,
    marginBottom: 32,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  instruction: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  instructionNumber: {
    width: 24,
    height: 24,
    backgroundColor: '#2563EB',
    color: '#FFFFFF',
    borderRadius: 12,
    fontSize: 12,
    fontWeight: '700',
    textAlign: 'center',
    lineHeight: 24,
    marginRight: 12,
  },
  instructionText: {
    flex: 1,
    fontSize: 14,
    color: '#374151',
    lineHeight: 20,
  },
});
