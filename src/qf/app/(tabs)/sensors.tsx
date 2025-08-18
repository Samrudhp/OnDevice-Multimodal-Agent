import React, { useState } from 'react';
import { View, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import LiveMonitoring from '../../components/LiveMonitoring';
import type { SensorData, ProcessingResult } from '../../lib/api';

export default function SensorsTab() {
  const [lastResult, setLastResult] = useState<ProcessingResult | null>(null);
  const [lastSensorData, setLastSensorData] = useState<SensorData | null>(null);

  const handleDataCollected = (data: SensorData) => {
    setLastSensorData(data);
    console.log('Sensor data collected:', {
      touchEvents: data.touch_events?.length || 0,
      keystrokes: data.keystroke_events?.length || 0,
      hasMotion: !!data.motion_data,
      hasAudio: !!data.audio_data,
      hasImage: !!data.image_data,
    });
  };

  const handleProcessingResult = (result: ProcessingResult) => {
    setLastResult(result);
    console.log('Processing result:', {
      riskLevel: result.risk_level,
      anomalyScore: result.anomaly_score,
      confidence: result.confidence,
      processingTime: result.processing_time_ms,
    });
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <StatusBar style="dark" />
      <View style={styles.content}>
        <LiveMonitoring
          onDataCollected={handleDataCollected}
          onProcessingResult={handleProcessingResult}
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  content: {
    flex: 1,
    paddingBottom: 20,
  },
});
