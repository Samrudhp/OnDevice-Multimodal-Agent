import React, { useState, useEffect, useRef } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Alert, Animated } from 'react-native';
import * as Icons from 'lucide-react-native';
import { SensorManager } from '../lib/sensor-manager';
import { QuadFusionAPI } from '../lib/api';
import ProcessingResultDisplay from '../components/ProcessingResultDisplay';
import StatusIndicator from '../components/StatusIndicator';
import type { SensorData, ProcessingResult } from '../lib/api';
import { AGENT_TYPES, API_CONNECTION_STATUS, COLORS } from '../lib/constants';
import { SPACING, BORDER_RADIUS } from '../lib/theme';
import { useGlowAnimation, usePulseAnimation } from '../lib/animations';

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
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [apiConnectionStatus, setApiConnectionStatus] = useState<typeof API_CONNECTION_STATUS[keyof typeof API_CONNECTION_STATUS]>(API_CONNECTION_STATUS.DISCONNECTED);
  const [isUsingFallback, setIsUsingFallback] = useState(false);
  const durationTimerRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  
  const { glowAnim, startGlowAnimation } = useGlowAnimation(0.4, 0.8);
  const { pulseAnim, startPulseAnimation } = usePulseAnimation(0.05);

  useEffect(() => {
    // Initialize sensor manager and load models on component mount
    const initialize = async () => {
      await loadModels();
    };
    
    initialize();
    
    // Set up API connection status listener
    const removeListener = api.addConnectionListener((status) => {
      setApiConnectionStatus(status);
      
      // Update fallback state based on connection status
      const usingFallback = status !== API_CONNECTION_STATUS.CONNECTED;
      setIsUsingFallback(usingFallback);
      
      // Log connection status changes
      if (status === API_CONNECTION_STATUS.DISCONNECTED) {
        console.log('API disconnected, using fallback data');
      } else if (status === API_CONNECTION_STATUS.CONNECTED) {
        console.log('Connected to QuadFusion API server');
      } else if (status === API_CONNECTION_STATUS.ERROR) {
        console.log('API connection error, using fallback data');
      } else if (status === API_CONNECTION_STATUS.CONNECTING) {
        console.log('Connecting to QuadFusion API server...');
      }
    });
    
    return () => {
      // Clean up on unmount
      if (durationTimerRef.current) {
        clearInterval(durationTimerRef.current);
      }
      removeListener();
    };
  }, []);
  
  useEffect(() => {
    if (isMonitoring) {
      durationTimerRef.current = setInterval(() => {
        setMonitoringDuration((prev) => prev + 1);
      }, 1000);
    }
    
    return () => {
      if (durationTimerRef.current) {
        clearInterval(durationTimerRef.current);
      }
    };
  }, [isMonitoring]);
  
  // Load models for all agent types
  const loadModels = async () => {
    try {
      setIsLoadingModels(true);
      
      // Get current model status
      const modelStatus = await api.getModelStatus();
      console.log('Model status:', modelStatus);
      
      // Check if any models need to be loaded
      const inactiveAgents = Object.entries(modelStatus.agents_status || {})
        .filter(([_, status]) => !status.is_active)
        .map(([agentType, _]) => agentType);
      
      if (inactiveAgents.length > 0) {
        console.log(`Loading inactive models: ${inactiveAgents.join(', ')}`);
        // Load only inactive models
        const result = await api.loadModels(inactiveAgents);
        console.log('Model loading result:', result);
        
        // Show success message to user
        Alert.alert(
          'Models Loaded', 
          `Successfully loaded ${inactiveAgents.length} behavioral analysis models.`,
          [{ text: 'OK' }]
        );
      } else {
        console.log('All models are already active');
      }
      
      setModelsLoaded(true);
    } catch (error) {
      console.error('Error loading models:', error);
      Alert.alert('Warning', 'Some models could not be loaded. Functionality may be limited.');
    } finally {
      setIsLoadingModels(false);
    }
  };

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
      
      // Generate a unique session ID for this processing request
      const sessionId = 'live-monitoring-' + Date.now();
      console.log(`Processing data with session ID: ${sessionId}`);
      console.log(`Data points: ${data.touch_events.length} touch events, ${data.keystroke_events.length} keystroke events`);
      
      // Send data for real-time processing
      const result = await api.processRealtimeSensorData(sessionId, data);
      
      console.log('Processing result:', result);
      console.log(`Anomaly score: ${result.anomaly_score}, Risk level: ${result.risk_level}`);
      
      // Display detailed agent results
      const agentResults = Object.entries(result.agent_results || {});
      if (agentResults.length > 0) {
        console.log('Individual agent results:');
        agentResults.forEach(([agentType, result]) => {
          console.log(`- ${agentType}: score=${result.anomaly_score}, risk=${result.risk_level}, confidence=${result.confidence}`);
        });
      }
      
      setProcessingResult(result);
      onProcessingResult?.(result);
      
      // Show a toast or alert for high-risk results
      if (result.risk_level === 'high') {
        Alert.alert(
          'High Risk Detected', 
          `Anomalous behavior detected with ${Math.round(result.confidence * 100)}% confidence.`,
          [{ text: 'Review', onPress: () => console.log('Review pressed') }, { text: 'Dismiss' }]
        );
      }
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
    if (isLoadingModels) {
      return {
        status: 'loading' as const,
        message: 'Loading Models',
        submessage: 'Initializing behavioral analysis models...',
      };
    }
    
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
    
    if (!modelsLoaded) {
      return {
        status: 'warning' as const,
        message: 'Models Not Loaded',
        submessage: 'Tap to load behavioral analysis models',
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
      {/* Status Display */}
      <TouchableOpacity 
        style={styles.statusCard} 
        onPress={() => !modelsLoaded && loadModels()}
        disabled={isLoadingModels || isMonitoring || isProcessing}
      >
        <Animated.View style={{
          shadowOpacity: glowAnim,
          transform: [{ scale: pulseAnim }]
        }}>
          <StatusIndicator
            status={status.status}
            message={status.message}
            submessage={status.submessage}
            size="large"
          />
        </Animated.View>
      </TouchableOpacity>
      
      {/* API Connection Status */}
      <View style={[styles.connectionStatus, 
        apiConnectionStatus === API_CONNECTION_STATUS.CONNECTED ? styles.connectedStatus :
        apiConnectionStatus === API_CONNECTION_STATUS.DISCONNECTED ? styles.disconnectedStatus :
        apiConnectionStatus === API_CONNECTION_STATUS.ERROR ? styles.errorStatus :
        styles.connectingStatus
      ]}>
        <Text style={styles.connectionStatusText}>
          {apiConnectionStatus === API_CONNECTION_STATUS.CONNECTED ? 'Connected to API' :
           apiConnectionStatus === API_CONNECTION_STATUS.DISCONNECTED ? 'API Disconnected - Using Fallback' :
           apiConnectionStatus === API_CONNECTION_STATUS.ERROR ? 'API Error - Using Fallback' :
           'Connecting to API...'}
        </Text>
        {isUsingFallback && (
          <Text style={styles.fallbackNotice}>Using local fallback data</Text>
        )}
      </View>

      {/* Control Panel */}
      <View style={styles.controlCard}>
        <Text style={styles.cardTitle}>Monitoring Controls</Text>
        
        <View style={styles.controlsContainer}>
          {!isMonitoring ? (
            <TouchableOpacity
              style={[styles.controlButton, styles.startButton]}
              onPress={startMonitoring}
              disabled={isProcessing || isLoadingModels || !modelsLoaded}
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
    backgroundColor: COLORS.BACKGROUND,
  },
  statusCard: {
    backgroundColor: COLORS.CARD,
    borderRadius: BORDER_RADIUS.LG,
    padding: 20,
    margin: 16,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 12,
    elevation: 8,
    borderWidth: 1,
    borderColor: COLORS.ACCENT,
    position: 'relative',
    overflow: 'hidden',
  },
  connectionStatus: {
    padding: 8,
    borderRadius: 4,
    marginHorizontal: 16,
    marginBottom: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  connectionStatusText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
    textShadowColor: 'rgba(0, 0, 0, 0.5)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 2,
  },
  connectedStatus: {
    backgroundColor: 'rgba(0, 255, 159, 0.2)',
    borderColor: COLORS.SUCCESS,
    borderWidth: 1,
  },
  disconnectedStatus: {
    backgroundColor: 'rgba(255, 184, 0, 0.2)',
    borderColor: COLORS.WARNING,
    borderWidth: 1,
  },
  errorStatus: {
    backgroundColor: 'rgba(255, 0, 85, 0.2)',
    borderColor: COLORS.ERROR,
    borderWidth: 1,
  },
  connectingStatus: {
    backgroundColor: 'rgba(123, 66, 246, 0.2)',
    borderColor: COLORS.SECONDARY,
    borderWidth: 1,
  },
  controlCard: {
    backgroundColor: COLORS.CARD,
    borderRadius: BORDER_RADIUS.LG,
    padding: SPACING.MD,
    marginHorizontal: 16,
    marginBottom: 16,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.6,
    shadowRadius: 12,
    elevation: 8,
    borderWidth: 1,
    borderColor: COLORS.ACCENT,
    position: 'relative',
    overflow: 'hidden',
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: COLORS.WHITE,
    marginBottom: 16,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 6,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.ACCENT,
    paddingBottom: 12,
    letterSpacing: 0.5,
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
    borderRadius: BORDER_RADIUS.MD,
    gap: 8,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 12,
    elevation: 8,
    borderWidth: 1,
    borderColor: COLORS.ACCENT,
    position: 'relative',
    overflow: 'hidden',
  },
  startButton: {
    backgroundColor: COLORS.SUCCESS,
  },
  startButtonText: {
    color: COLORS.WHITE,
    fontSize: 16,
    fontWeight: '600',
    textShadowColor: 'rgba(0, 0, 0, 0.5)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 2,
  },
  stopButton: {
    backgroundColor: COLORS.ERROR,
  },
  stopButtonText: {
    color: COLORS.WHITE,
    fontSize: 16,
    fontWeight: '600',
  },
  fallbackNotice: {
    color: '#FFFFFF',
    fontSize: 10,
    fontStyle: 'italic',
    opacity: 0.8,
    marginTop: 2,
  },
  resetButton: {
    backgroundColor: COLORS.GRAY_700,
    borderWidth: 1,
    borderColor: COLORS.GRAY_700,
  },
  resetButtonText: {
    color: COLORS.GRAY_300,
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
    backgroundColor: COLORS.CARD,
    borderRadius: BORDER_RADIUS.LG,
    padding: SPACING.MD,
    marginHorizontal: 16,
    marginBottom: 32,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 8,
    borderWidth: 1,
    borderColor: COLORS.ACCENT,
    position: 'relative',
    overflow: 'hidden',
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
