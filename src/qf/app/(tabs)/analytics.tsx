import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, Dimensions, Animated } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ChartBar as BarChart3, TrendingUp, Shield, TriangleAlert as AlertTriangle, Users, Clock } from 'lucide-react-native';
import { StatusBar } from 'expo-status-bar';
import { VictoryChart, VictoryLine, VictoryArea, VictoryAxis, VictoryTheme } from 'victory-native';
import { COLORS, ANIMATION } from '../../lib/constants';
import { createCardStyle, createTextStyle, SPACING, BORDER_RADIUS } from '../../lib/theme';
import { useGlowAnimation, usePulseAnimation } from '../../lib/animations';

const screenWidth = Dimensions.get('window').width;

interface AnalyticsData {
  time: string;
  authentications: number;
  fraudAttempts: number;
  riskScore: number;
}

export default function AnalyticsTab() {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData[]>([]);
  
  // Animations
  const { glowAnim, startGlowAnimation } = useGlowAnimation(0.3, 0.7, ANIMATION.SLOW * 2);
  const { pulseAnim, startPulseAnimation } = usePulseAnimation(0.1, ANIMATION.NORMAL * 2);
  
  useEffect(() => {
    // No continuous animations
  }, []);

  useEffect(() => {
    // Generate sample analytics data
    const generateData = () => {
      const data: AnalyticsData[] = [];
      const now = new Date();
      
      for (let i = 23; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 60 * 60 * 1000);
        data.push({
          time: time.toISOString(),
          authentications: Math.floor(Math.random() * 50) + 10,
          fraudAttempts: Math.floor(Math.random() * 8),
          riskScore: Math.random() * 100,
        });
      }
      
      setAnalyticsData(data);
    };

    generateData();
    const interval = setInterval(generateData, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const totalAuthentications = analyticsData.reduce((sum, item) => sum + item.authentications, 0);
  const totalFraudAttempts = analyticsData.reduce((sum, item) => sum + item.fraudAttempts, 0);
  const averageRiskScore = analyticsData.reduce((sum, item) => sum + item.riskScore, 0) / analyticsData.length;
  const successRate = ((totalAuthentications - totalFraudAttempts) / totalAuthentications) * 100;

  const chartData = analyticsData.map((item, index) => ({
    x: index,
    y: item.authentications,
    y0: 0,
  }));

  const riskChartData = analyticsData.map((item, index) => ({
    x: index,
    y: item.riskScore,
  }));

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Animated.View style={[styles.header, {
          shadowColor: COLORS.GLOW,
          shadowOpacity: glowAnim,
          shadowRadius: 20,
          shadowOffset: { width: 0, height: 0 }
        }]}>
          <Animated.View style={{
            transform: [{ scale: pulseAnim }],
            shadowColor: COLORS.PRIMARY,
            shadowOpacity: 0.8,
            shadowRadius: 10,
            shadowOffset: { width: 0, height: 0 }
          }}>
            <BarChart3 size={48} color={COLORS.PRIMARY} />
          </Animated.View>
          <Text style={styles.title}>Security Analytics</Text>
          <Text style={styles.subtitle}>24-hour fraud detection overview</Text>
        </Animated.View>

        <View style={styles.statsGrid}>
          <Animated.View style={[styles.statCard, createCardStyle('primary'), {
            shadowOpacity: glowAnim
          }]}>
            <Users size={32} color={COLORS.PRIMARY} />
            <Text style={styles.statValue}>{totalAuthentications.toLocaleString()}</Text>
            <Text style={styles.statLabel}>Total Authentications</Text>
          </Animated.View>

          <Animated.View style={[styles.statCard, createCardStyle('secondary'), {
            shadowColor: COLORS.ERROR,
            shadowOpacity: glowAnim
          }]}>
            <AlertTriangle size={32} color={COLORS.ERROR} />
            <Text style={styles.statValue}>{totalFraudAttempts}</Text>
            <Text style={styles.statLabel}>Fraud Attempts</Text>
          </Animated.View>

          <Animated.View style={[styles.statCard, createCardStyle('primary'), {
            shadowColor: COLORS.SUCCESS,
            shadowOpacity: glowAnim
          }]}>
            <Shield size={32} color={COLORS.SUCCESS} />
            <Text style={styles.statValue}>{successRate.toFixed(1)}%</Text>
            <Text style={styles.statLabel}>Success Rate</Text>
          </Animated.View>

          <Animated.View style={[styles.statCard, createCardStyle('secondary'), {
            shadowColor: COLORS.WARNING,
            shadowOpacity: glowAnim
          }]}>
            <TrendingUp size={32} color={COLORS.WARNING} />
            <Text style={styles.statValue}>{averageRiskScore.toFixed(0)}</Text>
            <Text style={styles.statLabel}>Avg Risk Score</Text>
          </Animated.View>
        </View>

        <View style={[styles.chartCard, createCardStyle('primary')]}>
          <View style={styles.chartHeader}>
            <Text style={styles.chartTitle}>Authentication Volume</Text>
            <View style={styles.chartLegend}>
              <View style={styles.legendItem}>
                <View style={[styles.legendColor, { backgroundColor: COLORS.PRIMARY }]} />
                <Text style={styles.legendText}>Authentications</Text>
              </View>
            </View>
          </View>
          
          <View style={styles.chartContainer}>
            <VictoryChart
              theme={VictoryTheme.material}
              width={screenWidth - 60}
              height={200}
              padding={{ left: 50, top: 20, right: 20, bottom: 40 }}
            >
              <VictoryAxis dependentAxis tickFormat={(t) => `${Math.round(t)}`} />
              <VictoryAxis fixLabelOverlap={true} />
              <VictoryArea
                data={chartData}
                style={{
                  data: { fill: COLORS.PRIMARY, fillOpacity: 0.3, stroke: COLORS.PRIMARY, strokeWidth: 2 },
                }}
                animate={{
                  duration: 1000,
                  onLoad: { duration: 500 },
                }}
              />
            </VictoryChart>
          </View>
        </View>

        <View style={[styles.chartCard, createCardStyle('secondary')]}>
          <View style={styles.chartHeader}>
            <Text style={styles.chartTitle}>Risk Score Trend</Text>
            <View style={styles.chartLegend}>
              <View style={styles.legendItem}>
                <View style={[styles.legendColor, { backgroundColor: COLORS.WARNING }]} />
                <Text style={styles.legendText}>Risk Level</Text>
              </View>
            </View>
          </View>
          
          <View style={styles.chartContainer}>
            <VictoryChart
              theme={VictoryTheme.material}
              width={screenWidth - 60}
              height={200}
              padding={{ left: 50, top: 20, right: 20, bottom: 40 }}
            >
              <VictoryAxis dependentAxis tickFormat={(t) => `${Math.round(t)}%`} />
              <VictoryAxis fixLabelOverlap={true} />
              <VictoryLine
                data={riskChartData}
                style={{
                  data: { stroke: COLORS.WARNING, strokeWidth: 3 },
                }}
                animate={{
                  duration: 1000,
                  onLoad: { duration: 500 },
                }}
              />
            </VictoryChart>
          </View>
        </View>

        <View style={[styles.alertsCard, createCardStyle('primary')]}>
          <View style={styles.alertsHeader}>
            <Animated.View style={{
              transform: [{ scale: pulseAnim }],
              shadowColor: COLORS.ERROR,
              shadowOpacity: 0.8,
              shadowRadius: 10,
              shadowOffset: { width: 0, height: 0 }
            }}>
              <AlertTriangle size={24} color={COLORS.ERROR} />
            </Animated.View>
            <Text style={styles.alertsTitle}>Security Alerts</Text>
          </View>

          <View style={styles.alertItem}>
            <Animated.View style={[styles.alertIndicator, {
              backgroundColor: COLORS.ERROR,
              shadowColor: COLORS.ERROR,
              shadowOpacity: glowAnim,
              shadowRadius: 5,
              shadowOffset: { width: 0, height: 0 }
            }]} />
            <View style={styles.alertContent}>
              <Text style={styles.alertText}>High risk authentication attempt detected</Text>
              <Text style={styles.alertTime}>2 minutes ago</Text>
            </View>
          </View>

          <View style={styles.alertItem}>
            <Animated.View style={[styles.alertIndicator, {
              backgroundColor: COLORS.WARNING,
              shadowColor: COLORS.WARNING,
              shadowOpacity: glowAnim,
              shadowRadius: 5,
              shadowOffset: { width: 0, height: 0 }
            }]} />
            <View style={styles.alertContent}>
              <Text style={styles.alertText}>Multiple failed biometric scans from same device</Text>
              <Text style={styles.alertTime}>15 minutes ago</Text>
            </View>
          </View>

          <View style={styles.alertItem}>
            <Animated.View style={[styles.alertIndicator, {
              backgroundColor: COLORS.SUCCESS,
              shadowColor: COLORS.SUCCESS,
              shadowOpacity: glowAnim,
              shadowRadius: 5,
              shadowOffset: { width: 0, height: 0 }
            }]} />
            <View style={styles.alertContent}>
              <Text style={styles.alertText}>System security scan completed successfully</Text>
              <Text style={styles.alertTime}>1 hour ago</Text>
            </View>
          </View>
        </View>

        <View style={[styles.insightsCard, createCardStyle('secondary')]}>
          <View style={styles.insightsHeader}>
            <Clock size={24} color={COLORS.SECONDARY} />
            <Text style={styles.insightsTitle}>Key Insights</Text>
          </View>

          <View style={styles.insightsList}>
            <View style={styles.insightItem}>
              <Text style={styles.insightText}>
                • Authentication volume increased by 23% compared to yesterday
              </Text>
            </View>
            <View style={styles.insightItem}>
              <Text style={styles.insightText}>
                • Fraud detection accuracy improved to 94.2%
              </Text>
            </View>
            <View style={styles.insightItem}>
              <Text style={styles.insightText}>
                • Peak authentication hours: 9-11 AM and 2-4 PM
              </Text>
            </View>
            <View style={styles.insightItem}>
              <Text style={styles.insightText}>
                • Biometric sensor confidence scores averaging 87%
              </Text>
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BACKGROUND,
  },
  scrollContent: {
    padding: SPACING.LG,
  },
  header: {
    alignItems: 'center',
    marginBottom: SPACING.XXL,
  },
  title: {
    ...createTextStyle('title'),
    marginTop: SPACING.MD,
  },
  subtitle: {
    ...createTextStyle('subtitle'),
    marginTop: SPACING.XS,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: SPACING.XL,
  },
  statCard: {
    width: '48%',
    alignItems: 'center',
    marginBottom: SPACING.MD,
    padding: SPACING.MD,
  },
  statValue: {
    ...createTextStyle('title'),
    fontSize: 24,
    marginTop: SPACING.SM,
    marginBottom: SPACING.XS,
  },
  statLabel: {
    ...createTextStyle('caption'),
    textAlign: 'center',
  },
  chartCard: {
    marginBottom: SPACING.XL,
    padding: SPACING.LG,
  },
  chartHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SPACING.MD,
  },
  chartTitle: {
    ...createTextStyle('subtitle'),
    textTransform: 'uppercase',
    letterSpacing: 1.2,
  },
  chartLegend: {
    flexDirection: 'row',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  legendColor: {
    width: 12,
    height: 12,
    borderRadius: 2,
    marginRight: SPACING.XS,
  },
  legendText: {
    ...createTextStyle('caption'),
  },
  chartContainer: {
    alignItems: 'center',
  },
  alertsCard: {
    marginBottom: SPACING.XL,
    padding: SPACING.LG,
  },
  alertsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: SPACING.MD,
  },
  alertsTitle: {
    ...createTextStyle('subtitle'),
    marginLeft: SPACING.SM,
    textTransform: 'uppercase',
    letterSpacing: 1.2,
  },
  alertItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: SPACING.MD,
  },
  alertIndicator: {
    width: 10,
    height: 10,
    borderRadius: BORDER_RADIUS.FULL,
    marginTop: 6,
    marginRight: SPACING.MD,
  },
  alertContent: {
    flex: 1,
  },
  alertText: {
    ...createTextStyle('body'),
    marginBottom: 2,
  },
  alertTime: {
    ...createTextStyle('caption'),
  },
  insightsCard: {
    padding: SPACING.LG,
  },
  insightsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: SPACING.MD,
  },
  insightsTitle: {
    ...createTextStyle('subtitle'),
    marginLeft: SPACING.SM,
    textTransform: 'uppercase',
    letterSpacing: 1.2,
  },
  insightsList: {
    gap: SPACING.SM,
  },
  insightItem: {
    paddingVertical: SPACING.XS,
  },
  insightText: {
    ...createTextStyle('body'),
    lineHeight: 20,
  },
});