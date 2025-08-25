import React, { useState } from 'react';
import { View, Text, TouchableOpacity, ScrollView, StyleSheet, Alert } from 'react-native';
import * as Icons from 'lucide-react-native';
import { 
  runComprehensiveValidation, 
  generateValidationReport,
  type ComprehensiveTestResult,
  type ValidationResult 
} from '../lib/validation';
import { COLORS } from '../lib/constants';
import { SPACING, BORDER_RADIUS } from '../lib/theme';

const CheckCircle = Icons.CheckCircle ?? (() => null);
const XCircle = Icons.XCircle ?? (() => null);
const Play = Icons.Play ?? (() => null);
const FileText = Icons.FileText ?? (() => null);
const Loader = Icons.Loader ?? (() => null);

interface ValidationPanelProps {
  onValidationComplete?: (results: ComprehensiveTestResult) => void;
}

export default function ValidationPanel({ onValidationComplete }: ValidationPanelProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<ComprehensiveTestResult | null>(null);
  const [showDetails, setShowDetails] = useState<Record<string, boolean>>({});

  const runValidation = async () => {
    try {
      setIsRunning(true);
      setResults(null);
      
      const validationResults = await runComprehensiveValidation();
      setResults(validationResults);
      onValidationComplete?.(validationResults);
      
      // Show overall result
      Alert.alert(
        validationResults.overall.success ? 'Validation Passed' : 'Validation Issues Found',
        validationResults.overall.message,
        [{ text: 'OK' }]
      );
    } catch (error) {
      console.error('Validation failed:', error);
      Alert.alert('Validation Error', 'Failed to run validation tests');
    } finally {
      setIsRunning(false);
    }
  };

  const generateReport = () => {
    if (!results) return;
    
    const report = generateValidationReport(results);
    console.log('Validation Report:\n', report);
    
    Alert.alert(
      'Report Generated',
      'Validation report has been logged to console. Check developer tools for details.',
      [{ text: 'OK' }]
    );
  };

  const toggleDetails = (testName: string) => {
    setShowDetails(prev => ({
      ...prev,
      [testName]: !prev[testName]
    }));
  };

  const renderTestResult = (name: string, result: ValidationResult) => {
    const isExpanded = showDetails[name];
    
    return (
      <View key={name} style={styles.testResult}>
        <TouchableOpacity 
          style={styles.testHeader}
          onPress={() => toggleDetails(name)}
        >
          <View style={styles.testInfo}>
            {result.success ? (
              <CheckCircle size={20} color={COLORS.SUCCESS} />
            ) : (
              <XCircle size={20} color={COLORS.ERROR} />
            )}
            <Text style={styles.testName}>{name}</Text>
          </View>
          <Text style={[
            styles.testStatus,
            { color: result.success ? COLORS.SUCCESS : COLORS.ERROR }
          ]}>
            {result.success ? 'PASS' : 'FAIL'}
          </Text>
        </TouchableOpacity>
        
        <Text style={styles.testMessage}>{result.message}</Text>
        
        {isExpanded && (
          <View style={styles.testDetails}>
            {result.details && (
              <View style={styles.detailsSection}>
                <Text style={styles.detailsTitle}>Details:</Text>
                <Text style={styles.detailsText}>
                  {JSON.stringify(result.details, null, 2)}
                </Text>
              </View>
            )}
            
            {result.errors && result.errors.length > 0 && (
              <View style={styles.errorsSection}>
                <Text style={styles.errorsTitle}>Errors:</Text>
                {result.errors.map((error, index) => (
                  <Text key={index} style={styles.errorText}>• {error}</Text>
                ))}
              </View>
            )}
          </View>
        )}
      </View>
    );
  };

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      <View style={styles.header}>
        <Text style={styles.title}>QuadFusion Validation</Text>
        <Text style={styles.subtitle}>
          Test all components and ensure perfect operation
        </Text>
      </View>

      <View style={styles.controls}>
        <TouchableOpacity
          style={[styles.button, styles.runButton]}
          onPress={runValidation}
          disabled={isRunning}
        >
          {isRunning ? (
            <Loader size={20} color={COLORS.WHITE} />
          ) : (
            <Play size={20} color={COLORS.WHITE} />
          )}
          <Text style={styles.runButtonText}>
            {isRunning ? 'Running Tests...' : 'Run Validation'}
          </Text>
        </TouchableOpacity>

        {results && (
          <TouchableOpacity
            style={[styles.button, styles.reportButton]}
            onPress={generateReport}
          >
            <FileText size={20} color={COLORS.PRIMARY} />
            <Text style={styles.reportButtonText}>Generate Report</Text>
          </TouchableOpacity>
        )}
      </View>

      {results && (
        <View style={styles.results}>
          <View style={[
            styles.overallResult,
            { backgroundColor: results.overall.success ? COLORS.SUCCESS + '20' : COLORS.ERROR + '20' }
          ]}>
            <Text style={[
              styles.overallStatus,
              { color: results.overall.success ? COLORS.SUCCESS : COLORS.ERROR }
            ]}>
              {results.overall.success ? '✅ ALL TESTS PASSED' : '❌ ISSUES FOUND'}
            </Text>
            <Text style={styles.overallMessage}>{results.overall.message}</Text>
          </View>

          <View style={styles.testResults}>
            {renderTestResult('API Connection', results.apiConnection)}
            {renderTestResult('Sensor Collection', results.sensorCollection)}
            {renderTestResult('Data Transmission', results.dataTransmission)}
            {renderTestResult('Agent Processing', results.agentProcessing)}
            {renderTestResult('Model Status', results.modelStatus)}
          </View>
        </View>
      )}

      {!results && !isRunning && (
        <View style={styles.instructions}>
          <Text style={styles.instructionsTitle}>Validation Tests</Text>
          <View style={styles.instruction}>
            <Text style={styles.instructionNumber}>1</Text>
            <Text style={styles.instructionText}>
              API Connection - Tests backend connectivity and health
            </Text>
          </View>
          <View style={styles.instruction}>
            <Text style={styles.instructionNumber}>2</Text>
            <Text style={styles.instructionText}>
              Sensor Collection - Validates sensor data gathering
            </Text>
          </View>
          <View style={styles.instruction}>
            <Text style={styles.instructionNumber}>3</Text>
            <Text style={styles.instructionText}>
              Data Transmission - Tests API communication
            </Text>
          </View>
          <View style={styles.instruction}>
            <Text style={styles.instructionNumber}>4</Text>
            <Text style={styles.instructionText}>
              Agent Processing - Validates all 6 agents work correctly
            </Text>
          </View>
          <View style={styles.instruction}>
            <Text style={styles.instructionNumber}>5</Text>
            <Text style={styles.instructionText}>
              Model Status - Checks if all models are trained and ready
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
  header: {
    padding: SPACING.LG,
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: COLORS.WHITE,
    marginBottom: 8,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 6,
  },
  subtitle: {
    fontSize: 16,
    color: COLORS.GRAY_300,
    textAlign: 'center',
  },
  controls: {
    flexDirection: 'row',
    paddingHorizontal: SPACING.LG,
    gap: 12,
    marginBottom: SPACING.LG,
  },
  button: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: BORDER_RADIUS.MD,
    gap: 8,
  },
  runButton: {
    backgroundColor: COLORS.PRIMARY,
  },
  runButtonText: {
    color: COLORS.WHITE,
    fontSize: 16,
    fontWeight: '600',
  },
  reportButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: COLORS.PRIMARY,
  },
  reportButtonText: {
    color: COLORS.PRIMARY,
    fontSize: 16,
    fontWeight: '600',
  },
  results: {
    paddingHorizontal: SPACING.LG,
  },
  overallResult: {
    padding: SPACING.MD,
    borderRadius: BORDER_RADIUS.LG,
    marginBottom: SPACING.LG,
    borderWidth: 1,
    borderColor: COLORS.ACCENT,
  },
  overallStatus: {
    fontSize: 18,
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 8,
  },
  overallMessage: {
    fontSize: 14,
    color: COLORS.GRAY_300,
    textAlign: 'center',
  },
  testResults: {
    gap: 12,
  },
  testResult: {
    backgroundColor: COLORS.CARD,
    borderRadius: BORDER_RADIUS.MD,
    padding: SPACING.MD,
    borderWidth: 1,
    borderColor: COLORS.ACCENT,
  },
  testHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  testInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  testName: {
    fontSize: 16,
    fontWeight: '600',
    color: COLORS.WHITE,
  },
  testStatus: {
    fontSize: 14,
    fontWeight: '700',
  },
  testMessage: {
    fontSize: 14,
    color: COLORS.GRAY_300,
    marginBottom: 8,
  },
  testDetails: {
    borderTopWidth: 1,
    borderTopColor: COLORS.ACCENT,
    paddingTop: 12,
  },
  detailsSection: {
    marginBottom: 12,
  },
  detailsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: COLORS.PRIMARY,
    marginBottom: 4,
  },
  detailsText: {
    fontSize: 12,
    color: COLORS.GRAY_300,
    fontFamily: 'monospace',
    backgroundColor: COLORS.GRAY_900,
    padding: 8,
    borderRadius: 4,
  },
  errorsSection: {
    marginBottom: 12,
  },
  errorsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: COLORS.ERROR,
    marginBottom: 4,
  },
  errorText: {
    fontSize: 12,
    color: COLORS.ERROR,
    marginBottom: 2,
  },
  instructions: {
    padding: SPACING.LG,
  },
  instructionsTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: COLORS.WHITE,
    marginBottom: 16,
    textAlign: 'center',
  },
  instruction: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  instructionNumber: {
    width: 24,
    height: 24,
    backgroundColor: COLORS.PRIMARY,
    color: COLORS.WHITE,
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
    color: COLORS.GRAY_300,
    lineHeight: 20,
  },
});