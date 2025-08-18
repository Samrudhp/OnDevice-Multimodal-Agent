import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, Dimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ChartBar as BarChart3, TrendingUp, Shield, TriangleAlert as AlertTriangle, Users, Clock } from 'lucide-react-native';
import { StatusBar } from 'expo-status-bar';
import { VictoryChart, VictoryLine, VictoryArea, VictoryAxis, VictoryTheme } from 'victory-native';

const screenWidth = Dimensions.get('window').width;

interface AnalyticsData {
  time: string;
  authentications: number;
  fraudAttempts: number;
  riskScore: number;
}

export default function AnalyticsTab() {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData[]>([]);

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
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <BarChart3 size={48} color="#2563EB" />
          <Text style={styles.title}>Security Analytics</Text>
          <Text style={styles.subtitle}>24-hour fraud detection overview</Text>
        </View>

        <View style={styles.statsGrid}>
          <View style={[styles.statCard, { backgroundColor: '#EFF6FF' }]}>
            <Users size={32} color="#2563EB" />
            <Text style={styles.statValue}>{totalAuthentications.toLocaleString()}</Text>
            <Text style={styles.statLabel}>Total Authentications</Text>
          </View>

          <View style={[styles.statCard, { backgroundColor: '#FEF2F2' }]}>
            <AlertTriangle size={32} color="#EF4444" />
            <Text style={styles.statValue}>{totalFraudAttempts}</Text>
            <Text style={styles.statLabel}>Fraud Attempts</Text>
          </View>

          <View style={[styles.statCard, { backgroundColor: '#ECFDF5' }]}>
            <Shield size={32} color="#10B981" />
            <Text style={styles.statValue}>{successRate.toFixed(1)}%</Text>
            <Text style={styles.statLabel}>Success Rate</Text>
          </View>

          <View style={[styles.statCard, { backgroundColor: '#FEF3C7' }]}>
            <TrendingUp size={32} color="#F59E0B" />
            <Text style={styles.statValue}>{averageRiskScore.toFixed(0)}</Text>
            <Text style={styles.statLabel}>Avg Risk Score</Text>
          </View>
        </View>

        <View style={styles.chartCard}>
          <View style={styles.chartHeader}>
            <Text style={styles.chartTitle}>Authentication Volume</Text>
            <View style={styles.chartLegend}>
              <View style={styles.legendItem}>
                <View style={[styles.legendColor, { backgroundColor: '#2563EB' }]} />
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
                  data: { fill: '#2563EB', fillOpacity: 0.3, stroke: '#2563EB', strokeWidth: 2 },
                }}
                animate={{
                  duration: 1000,
                  onLoad: { duration: 500 },
                }}
              />
            </VictoryChart>
          </View>
        </View>

        <View style={styles.chartCard}>
          <View style={styles.chartHeader}>
            <Text style={styles.chartTitle}>Risk Score Trend</Text>
            <View style={styles.chartLegend}>
              <View style={styles.legendItem}>
                <View style={[styles.legendColor, { backgroundColor: '#F59E0B' }]} />
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
                  data: { stroke: '#F59E0B', strokeWidth: 3 },
                }}
                animate={{
                  duration: 1000,
                  onLoad: { duration: 500 },
                }}
              />
            </VictoryChart>
          </View>
        </View>

        <View style={styles.alertsCard}>
          <View style={styles.alertsHeader}>
            <AlertTriangle size={24} color="#EF4444" />
            <Text style={styles.alertsTitle}>Security Alerts</Text>
          </View>

          <View style={styles.alertItem}>
            <View style={styles.alertIndicator} />
            <View style={styles.alertContent}>
              <Text style={styles.alertText}>High risk authentication attempt detected</Text>
              <Text style={styles.alertTime}>2 minutes ago</Text>
            </View>
          </View>

          <View style={styles.alertItem}>
            <View style={[styles.alertIndicator, { backgroundColor: '#F59E0B' }]} />
            <View style={styles.alertContent}>
              <Text style={styles.alertText}>Multiple failed biometric scans from same device</Text>
              <Text style={styles.alertTime}>15 minutes ago</Text>
            </View>
          </View>

          <View style={styles.alertItem}>
            <View style={[styles.alertIndicator, { backgroundColor: '#10B981' }]} />
            <View style={styles.alertContent}>
              <Text style={styles.alertText}>System security scan completed successfully</Text>
              <Text style={styles.alertTime}>1 hour ago</Text>
            </View>
          </View>
        </View>

        <View style={styles.insightsCard}>
          <View style={styles.insightsHeader}>
            <Clock size={24} color="#2563EB" />
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
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  statCard: {
    width: '48%',
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  statValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#1F2937',
    marginTop: 8,
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#6B7280',
    textAlign: 'center',
    fontWeight: '500',
  },
  chartCard: {
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
  chartHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  chartTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
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
    marginRight: 6,
  },
  legendText: {
    fontSize: 12,
    color: '#6B7280',
  },
  chartContainer: {
    alignItems: 'center',
  },
  alertsCard: {
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
  alertsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  alertsTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
    marginLeft: 8,
  },
  alertItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: 12,
  },
  alertIndicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#EF4444',
    marginTop: 6,
    marginRight: 12,
  },
  alertContent: {
    flex: 1,
  },
  alertText: {
    fontSize: 14,
    color: '#374151',
    marginBottom: 2,
  },
  alertTime: {
    fontSize: 12,
    color: '#6B7280',
  },
  insightsCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  insightsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  insightsTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
    marginLeft: 8,
  },
  insightsList: {
    gap: 8,
  },
  insightItem: {
    paddingVertical: 4,
  },
  insightText: {
    fontSize: 14,
    color: '#374151',
    lineHeight: 20,
  },
});