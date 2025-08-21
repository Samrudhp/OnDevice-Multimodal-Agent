import React, { useState, useEffect } from 'react';
import { View, StyleSheet, Text, Animated } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import LiveMonitoring from '../../components/LiveMonitoring';
import type { SensorData, ProcessingResult } from '../../lib/api';
import { COLORS, ANIMATION } from '../../lib/constants';
import { SPACING, BORDER_RADIUS, createCardStyle, createTextStyle } from '../../lib/theme';
import GridBackground from '../../components/GridBackground';
import { useGlowAnimation, usePulseAnimation } from '../../lib/animations';
import * as Icons from 'lucide-react-native';

export default function SensorsTab() {
  const [lastResult, setLastResult] = useState<ProcessingResult | null>(null);
  const [lastSensorData, setLastSensorData] = useState<SensorData | null>(null);
  
  const { glowAnim, startGlowAnimation } = useGlowAnimation(0.4, 0.8);
  const { pulseAnim, startPulseAnimation } = usePulseAnimation(0.05);
  
  useEffect(() => {
    // No continuous animations
  }, []);

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
      <StatusBar style="light" />
      <GridBackground spacing={30} opacity={0.15} />
      <View style={styles.header}>
        <View style={styles.titleContainer}>
          <Icons.Activity size={24} color={COLORS.ACCENT} style={styles.titleIcon} />
          <Animated.Text style={[styles.title, { textShadowRadius: pulseAnim.interpolate({
            inputRange: [1, 1.05],
            outputRange: [4, 8]
          }) }]}>
            Sensor Monitoring
          </Animated.Text>
        </View>
        <Text style={styles.subtitle}>Real-time data collection and analysis</Text>
        <View style={styles.headerDivider} />
      </View>
      <Animated.View style={[styles.content, {
        shadowOpacity: glowAnim,
        transform: [{ scale: pulseAnim }]
      }]}>
        <View style={styles.cornerAccent} />
        <View style={[styles.cornerAccent, styles.cornerAccentTopRight]} />
        <View style={[styles.cornerAccent, styles.cornerAccentBottomLeft]} />
        <View style={[styles.cornerAccent, styles.cornerAccentBottomRight]} />
        <LiveMonitoring
          onDataCollected={handleDataCollected}
          onProcessingResult={handleProcessingResult}
        />
      </Animated.View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BACKGROUND,
  },
  header: {
    paddingTop: SPACING.XL,
    paddingHorizontal: SPACING.LG,
    marginBottom: SPACING.MD,
    paddingBottom: SPACING.MD,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.5,
    shadowRadius: 8,
  },
  titleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: SPACING.XS,
  },
  titleIcon: {
    marginRight: SPACING.SM,
    shadowColor: COLORS.ACCENT,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 5,
  },
  title: {
    ...createTextStyle('title'),
    textAlign: 'center',
  },
  subtitle: {
    ...createTextStyle('subtitle'),
    textAlign: 'center',
    color: COLORS.ACCENT,
    opacity: 0.8,
  },
  headerDivider: {
    height: 2,
    backgroundColor: COLORS.GLOW,
    marginTop: SPACING.MD,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 4,
  },
  content: {
    flex: 1,
    paddingHorizontal: SPACING.MD,
    paddingBottom: SPACING.LG,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 15,
    borderWidth: 1,
    borderColor: COLORS.GLOW,
    borderRadius: BORDER_RADIUS.LG,
    margin: SPACING.MD,
    position: 'relative',
    overflow: 'hidden',
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
});
