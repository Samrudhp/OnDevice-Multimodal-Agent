import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import * as Icons from 'lucide-react-native';
import type { ProcessingResult, SensorData } from '../lib/api';
import StatusIndicator from './StatusIndicator';

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
        return '#10B981';
      case 'medium':
        return '#F59E0B';
      case 'high':
        return '#EF4444';
      default:
        return '#6B7280';
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
            <Text style={styles.metricValue}>
              {(result.anomaly_score * 100).toFixed(1)}%
            </Text>
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
  },
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1F2937',
  },
  riskBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  riskBadgeText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '700',
  },
  metricsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  metric: {
    flex: 1,
    alignItems: 'center',
  },
  metricLabel: {
    fontSize: 12,
    color: '#6B7280',
    marginBottom: 4,
  },
  metricValue: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1F2937',
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
