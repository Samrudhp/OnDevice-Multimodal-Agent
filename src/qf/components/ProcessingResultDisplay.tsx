import React from 'react';
import { View, Text, StyleSheet, ScrollView, Dimensions } from 'react-native';
import * as Icons from 'lucide-react-native';
import type { ProcessingResult, SensorData } from '../lib/api';
import StatusIndicator from './StatusIndicator';
import { COLORS } from '../lib/constants';

const { width } = Dimensions.get('window');

const Activity = Icons.Activity ?? (() => null);
const Zap = Icons.Zap ?? (() => null);
const Mic = Icons.Mic ?? (() => null);
const Camera = Icons.Camera ?? (() => null);
const Smartphone = Icons.Smartphone ?? (() => null);
const Eye = Icons.Eye ?? (() => null);

interface ProcessingResultDisplayProps {
  result?: ProcessingResult;
  sensorData?: SensorData;
  isProcessing?: boolean;
}

export default function ProcessingResultDisplay({ 
  result, 
  sensorData, 
  isProcessing = false 
}: ProcessingResultDisplayProps) {
  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low':
        return COLORS.SUCCESS;
      case 'medium':
        return COLORS.WARNING;
      case 'high':
        return COLORS.ERROR;
      default:
        return COLORS.GRAY_500;
    }
  };

  const getAgentIcon = (agentName: string) => {
    switch (agentName) {
      case 'TouchPatternAgent':
        return Activity;
      case 'TypingBehaviorAgent':
        return Zap;
      case 'VoiceCommandAgent':
        return Mic;
      case 'VisualAgent':
        return Eye;
      case 'MovementAgent':
        return Smartphone;
      case 'AppUsageAgent':
        return Camera;
      default:
        return Activity;
    }
  };

  if (isProcessing) {
    return (
      <View style={styles.container}>
        <StatusIndicator
          status="loading"
          message="Processing biometric data..."
          submessage="Analyzing behavioral patterns"
          size="large"
        />
      </View>
    );
  }

  if (!result) {
    return (
      <View style={styles.container}>
        <StatusIndicator
          status="idle"
          message="No processing results available"
          submessage="Start data collection to see analysis"
        />
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Overall Risk Assessment */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Text style={styles.cardTitle}>Risk Assessment</Text>
          <View style={[styles.riskBadge, { backgroundColor: getRiskColor(result.risk_level) }]}>
            <Text style={styles.riskBadgeText}>
              {result.risk_level.toUpperCase()}
            </Text>
          </View>
        </View>
        
        <View style={styles.metricsContainer}>
          <View style={styles.metric}>
            <Text style={styles.metricLabel}>Anomaly Score</Text>
            <View style={styles.scoreContainer}>
              <View 
                style={[styles.scoreBar, { width: `${result.anomaly_score * 100}%`, backgroundColor: getRiskColor(result.risk_level) }]} 
              />
              <Text style={styles.metricValue}>
                {(result.anomaly_score * 100).toFixed(1)}%
              </Text>
            </View>
          </View>
          <View style={styles.metric}>
            <Text style={styles.metricLabel}>Confidence</Text>
            <Text style={styles.metricValue}>
              {(result.confidence * 100).toFixed(1)}%
            </Text>
          </View>
          <View style={styles.metric}>
            <Text style={styles.metricLabel}>Processing Time</Text>
            <Text style={styles.metricValue}>
              {result.processing_time_ms.toFixed(0)}ms
            </Text>
          </View>
        </View>
      </View>

      {/* Agent Results */}
      {result.agent_results && Object.keys(result.agent_results).length > 0 && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Agent Analysis</Text>
          {Object.entries(result.agent_results).map(([agentName, agentResult]) => {
            const IconComponent = getAgentIcon(agentName);
            const displayName = agentName.replace('Agent', '').replace(/([A-Z])/g, ' $1').trim();
            
            return (
              <View key={agentName} style={styles.agentResult}>
                <View style={styles.agentHeader}>
                  <IconComponent size={20} color="#2563EB" />
                  <Text style={styles.agentName}>{displayName}</Text>
                  <Text style={[styles.agentScore, { color: getRiskColor(agentResult.risk_level) }]}>
                    {(agentResult.anomaly_score * 100).toFixed(0)}%
                  </Text>
                </View>
                
                {agentResult.features_analyzed && agentResult.features_analyzed.length > 0 && (
                  <View style={styles.featuresContainer}>
                    <Text style={styles.featuresLabel}>Features Analyzed:</Text>
                    <Text style={styles.featuresText}>
                      {agentResult.features_analyzed.join(', ')}
                    </Text>
                  </View>
                )}
              </View>
            );
          })}
        </View>
      )}

      {/* Sensor Data Summary */}
      {sensorData && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Sensor Data Summary</Text>
          
          <View style={styles.sensorSummary}>
            <View style={styles.sensorStat}>
              <Activity size={16} color="#6B7280" />
              <Text style={styles.sensorStatText}>
                {sensorData.touch_events?.length || 0} touch events
              </Text>
            </View>
            
            <View style={styles.sensorStat}>
              <Zap size={16} color="#6B7280" />
              <Text style={styles.sensorStatText}>
                {sensorData.keystroke_events?.length || 0} keystrokes
              </Text>
            </View>
            
            {sensorData.motion_data && (
              <View style={styles.sensorStat}>
                <Smartphone size={16} color="#6B7280" />
                <Text style={styles.sensorStatText}>Motion data collected</Text>
              </View>
            )}
            
            {sensorData.audio_data && (
              <View style={styles.sensorStat}>
                <Mic size={16} color="#6B7280" />
                <Text style={styles.sensorStatText}>
                  Audio ({sensorData.audio_duration?.toFixed(1)}s)
                </Text>
              </View>
            )}
            
            {sensorData.image_data && (
              <View style={styles.sensorStat}>
                <Camera size={16} color="#6B7280" />
                <Text style={styles.sensorStatText}>Image captured</Text>
              </View>
            )}
          </View>
        </View>
      )}

      {/* Metadata */}
      {result.metadata && Object.keys(result.metadata).length > 0 && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Additional Information</Text>
          {Object.entries(result.metadata).map(([key, value]) => (
            <View key={key} style={styles.metadataItem}>
              <Text style={styles.metadataKey}>{key}:</Text>
              <Text style={styles.metadataValue}>
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </Text>
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: COLORS.BACKGROUND,
  },
  card: {
    backgroundColor: COLORS.CARD,
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: COLORS.GLOW,
    shadowColor: COLORS.PRIMARY,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.GRAY_700,
    paddingBottom: 12,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: COLORS.WHITE,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
  },
  riskBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    shadowColor: COLORS.PRIMARY,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 4,
  },
  riskBadgeText: {
    color: COLORS.WHITE,
    fontSize: 12,
    fontWeight: '700',
  },
  metricsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
  },
  metric: {
    flex: 1,
    minWidth: width / 2 - 32,
    marginBottom: 16,
    alignItems: 'center',
  },
  metricLabel: {
    fontSize: 12,
    color: COLORS.GRAY_300,
    marginBottom: 4,
  },
  metricValue: {
    fontSize: 18,
    fontWeight: '700',
    color: COLORS.WHITE,
  },
  scoreContainer: {
    position: 'relative',
    height: 32,
    backgroundColor: COLORS.GRAY_700,
    borderRadius: 16,
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center',
  },
  scoreBar: {
    position: 'absolute',
    left: 0,
    top: 0,
    height: '100%',
    borderRadius: 16,
    opacity: 0.8,
  },
  agentResult: {
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  agentHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  agentName: {
    flex: 1,
    fontSize: 16,
    fontWeight: '600',
    color: '#374151',
    marginLeft: 8,
  },
  agentScore: {
    fontSize: 16,
    fontWeight: '700',
  },
  featuresContainer: {
    marginTop: 4,
    paddingLeft: 28,
  },
  featuresLabel: {
    fontSize: 12,
    color: '#6B7280',
    marginBottom: 2,
  },
  featuresText: {
    fontSize: 12,
    color: '#9CA3AF',
  },
  sensorSummary: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  sensorStat: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F9FAFB',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  sensorStatText: {
    fontSize: 12,
    color: '#6B7280',
    marginLeft: 4,
  },
  metadataItem: {
    flexDirection: 'row',
    paddingVertical: 4,
  },
  metadataKey: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    minWidth: 100,
  },
  metadataValue: {
    flex: 1,
    fontSize: 14,
    color: '#6B7280',
  },
});
