/**
 * QuadFusion App Validation and Testing Utilities
 * 
 * This module provides comprehensive validation and testing functions
 * to ensure the QuadFusion app runs perfectly with all agents working correctly.
 */

import { api, type SensorData, type ProcessingResult, AGENT_TYPES } from './api';
import { sensorManager } from './sensor-manager';
import { generateSessionId } from './utils';

export interface ValidationResult {
  success: boolean;
  message: string;
  details?: any;
  errors?: string[];
}

export interface ComprehensiveTestResult {
  overall: ValidationResult;
  apiConnection: ValidationResult;
  sensorCollection: ValidationResult;
  dataTransmission: ValidationResult;
  agentProcessing: ValidationResult;
  modelStatus: ValidationResult;
}

/**
 * Test API connection and basic functionality
 */
export async function validateAPIConnection(): Promise<ValidationResult> {
  try {
    const healthCheck = await api.healthCheck();
    
    if (healthCheck.status === 'healthy') {
      return {
        success: true,
        message: 'API connection successful',
        details: {
          quadfusion_available: healthCheck.quadfusion_available,
          agents_initialized: healthCheck.agents_initialized,
          active_sessions: healthCheck.active_sessions
        }
      };
    } else {
      return {
        success: false,
        message: 'API health check failed',
        details: healthCheck
      };
    }
  } catch (error) {
    return {
      success: false,
      message: 'Failed to connect to API',
      errors: [error instanceof Error ? error.message : String(error)]
    };
  }
}

/**
 * Test sensor data collection
 */
export async function validateSensorCollection(): Promise<ValidationResult> {
  try {
    // Request permissions
    const hasPermissions = await sensorManager.requestPermissions();
    if (!hasPermissions) {
      return {
        success: false,
        message: 'Sensor permissions not granted',
        errors: ['Required permissions for sensors not available']
      };
    }

    // Start motion sensors
    await sensorManager.startMotionSensors();
    
    // Wait a bit for data collection
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Add some test touch events
    sensorManager.addTouchEvent({
      locationX: 100,
      locationY: 200,
      force: 0.8,
      majorRadius: 15,
      minorRadius: 12,
      type: 'press'
    });
    
    // Add test keystroke events
    sensorManager.addKeystrokeEvent({
      keyCode: 65,
      type: 'keydown',
      pressure: 0.6
    });
    
    // Add app usage event
    sensorManager.addAppUsageEvent('test_app', 'open');
    
    // Get collected data
    const sensorData = sensorManager.getCurrentSensorData();
    
    // Stop sensors
    sensorManager.stopMotionSensors();
    
    const validation = {
      touch_events: sensorData.touch_events.length > 0,
      keystroke_events: sensorData.keystroke_events.length > 0,
      motion_data: !!sensorData.motion_data,
      app_usage: sensorData.app_usage.length > 0
    };
    
    const allValid = Object.values(validation).every(v => v);
    
    return {
      success: allValid,
      message: allValid ? 'Sensor collection working correctly' : 'Some sensor data missing',
      details: {
        validation,
        data_summary: {
          touch_events: sensorData.touch_events.length,
          keystroke_events: sensorData.keystroke_events.length,
          has_motion_data: !!sensorData.motion_data,
          app_usage_events: sensorData.app_usage.length
        }
      }
    };
  } catch (error) {
    return {
      success: false,
      message: 'Sensor collection failed',
      errors: [error instanceof Error ? error.message : String(error)]
    };
  }
}

/**
 * Test data transmission to API
 */
export async function validateDataTransmission(): Promise<ValidationResult> {
  try {
    // Generate test sensor data
    const testSensorData: SensorData = {
      touch_events: [
        {
          timestamp: Date.now() / 1000,
          x: 540,
          y: 960,
          pressure: 0.8,
          touch_major: 15,
          touch_minor: 12,
          action: 'down'
        }
      ],
      keystroke_events: [
        {
          timestamp: Date.now() / 1000,
          key_code: 65,
          action: 'down',
          pressure: 0.6
        }
      ],
      motion_data: {
        accelerometer: [0.1, -0.05, 9.81],
        gyroscope: [0.01, 0.02, -0.01],
        magnetometer: [23.4, -12.1, 45.6],
        timestamp: Date.now() / 1000
      },
      app_usage: [
        {
          app_name: 'test_app',
          action: 'open',
          timestamp: Date.now() / 1000
        }
      ]
    };

    const sessionId = generateSessionId();
    const result = await api.processRealtimeSensorData(sessionId, testSensorData);
    
    // Validate response structure
    const hasRequiredFields = !!(
      typeof result.anomaly_score === 'number' &&
      result.risk_level &&
      typeof result.confidence === 'number' &&
      result.agent_results &&
      typeof result.processing_time_ms === 'number'
    );
    
    return {
      success: hasRequiredFields,
      message: hasRequiredFields ? 'Data transmission successful' : 'Invalid response structure',
      details: {
        response_structure: {
          has_anomaly_score: typeof result.anomaly_score === 'number',
          has_risk_level: !!result.risk_level,
          has_confidence: typeof result.confidence === 'number',
          has_agent_results: !!result.agent_results,
          has_processing_time: typeof result.processing_time_ms === 'number'
        },
        sample_result: {
          anomaly_score: result.anomaly_score,
          risk_level: result.risk_level,
          confidence: result.confidence,
          agents_count: Object.keys(result.agent_results).length
        }
      }
    };
  } catch (error) {
    return {
      success: false,
      message: 'Data transmission failed',
      errors: [error instanceof Error ? error.message : String(error)]
    };
  }
}

/**
 * Test agent processing and outputs
 */
export async function validateAgentProcessing(): Promise<ValidationResult> {
  try {
    // Generate comprehensive test data
    const testSensorData: SensorData = {
      touch_events: Array.from({ length: 10 }, (_, i) => ({
        timestamp: Date.now() / 1000 + i * 0.1,
        x: 100 + i * 50,
        y: 200 + i * 30,
        pressure: 0.5 + Math.random() * 0.5,
        touch_major: 10 + Math.random() * 10,
        touch_minor: 8 + Math.random() * 8,
        action: i % 3 === 0 ? 'down' : i % 3 === 1 ? 'move' : 'up'
      })),
      keystroke_events: Array.from({ length: 5 }, (_, i) => ({
        timestamp: Date.now() / 1000 + i * 0.2,
        key_code: 65 + i,
        action: i % 2 === 0 ? 'down' : 'up',
        pressure: 0.4 + Math.random() * 0.4
      })),
      motion_data: {
        accelerometer: [Math.random() * 0.2 - 0.1, Math.random() * 0.2 - 0.1, 9.8 + Math.random() * 0.2],
        gyroscope: [Math.random() * 0.1 - 0.05, Math.random() * 0.1 - 0.05, Math.random() * 0.1 - 0.05],
        magnetometer: [20 + Math.random() * 10, -15 + Math.random() * 10, 40 + Math.random() * 10],
        timestamp: Date.now() / 1000
      },
      audio_data: 'dGVzdF9hdWRpb19kYXRh', // base64 encoded test data
      sample_rate: 16000,
      audio_duration: 2.0,
      image_data: 'dGVzdF9pbWFnZV9kYXRh', // base64 encoded test data
      camera_type: 'front',
      app_usage: [
        { app_name: 'banking_app', action: 'open', timestamp: Date.now() / 1000 },
        { app_name: 'social_app', action: 'switch_to', timestamp: Date.now() / 1000 + 1 }
      ]
    };

    const sessionId = generateSessionId();
    const result = await api.processRealtimeSensorData(sessionId, testSensorData);
    
    // Check if all expected agents are present
    const expectedAgents = Object.values(AGENT_TYPES);
    const presentAgents = Object.keys(result.agent_results);
    const missingAgents = expectedAgents.filter(agent => !presentAgents.includes(agent));
    
    // Validate agent results structure
    const agentValidation: Record<string, boolean> = {};
    for (const [agentName, agentResult] of Object.entries(result.agent_results)) {
      agentValidation[agentName] = !!(
        typeof agentResult.anomaly_score === 'number' &&
        agentResult.risk_level &&
        typeof agentResult.confidence === 'number' &&
        Array.isArray(agentResult.features_analyzed) &&
        typeof agentResult.processing_time_ms === 'number'
      );
    }
    
    const allAgentsValid = Object.values(agentValidation).every(v => v);
    
    return {
      success: allAgentsValid && missingAgents.length === 0,
      message: allAgentsValid && missingAgents.length === 0 
        ? 'All agents processing correctly' 
        : 'Some agents missing or invalid',
      details: {
        expected_agents: expectedAgents.length,
        present_agents: presentAgents.length,
        missing_agents: missingAgents,
        agent_validation: agentValidation,
        sample_scores: Object.fromEntries(
          Object.entries(result.agent_results).map(([name, result]) => [
            name, 
            { 
              anomaly_score: result.anomaly_score, 
              risk_level: result.risk_level,
              features_count: result.features_analyzed.length
            }
          ])
        )
      }
    };
  } catch (error) {
    return {
      success: false,
      message: 'Agent processing validation failed',
      errors: [error instanceof Error ? error.message : String(error)]
    };
  }
}

/**
 * Test model status and initialization
 */
export async function validateModelStatus(): Promise<ValidationResult> {
  try {
    const modelStatus = await api.getModelStatus();
    
    if (!modelStatus.models) {
      return {
        success: false,
        message: 'Invalid model status response',
        errors: ['No models field in response']
      };
    }
    
    const models = modelStatus.models;
    const modelNames = Object.keys(models);
    const trainedModels = Object.entries(models)
      .filter(([_, status]) => status.is_trained)
      .map(([name, _]) => name);
    
    const allTrained = modelNames.length > 0 && trainedModels.length === modelNames.length;
    
    return {
      success: allTrained,
      message: allTrained 
        ? 'All models are trained and ready' 
        : `${trainedModels.length}/${modelNames.length} models trained`,
      details: {
        total_models: modelNames.length,
        trained_models: trainedModels.length,
        model_status: models,
        trained_model_names: trainedModels
      }
    };
  } catch (error) {
    return {
      success: false,
      message: 'Model status validation failed',
      errors: [error instanceof Error ? error.message : String(error)]
    };
  }
}

/**
 * Run comprehensive validation of the entire system
 */
export async function runComprehensiveValidation(): Promise<ComprehensiveTestResult> {
  console.log('Starting comprehensive QuadFusion validation...');
  
  const results: ComprehensiveTestResult = {
    overall: { success: false, message: 'Validation in progress' },
    apiConnection: await validateAPIConnection(),
    sensorCollection: await validateSensorCollection(),
    dataTransmission: await validateDataTransmission(),
    agentProcessing: await validateAgentProcessing(),
    modelStatus: await validateModelStatus()
  };
  
  // Determine overall success
  const allTests = [
    results.apiConnection,
    results.sensorCollection,
    results.dataTransmission,
    results.agentProcessing,
    results.modelStatus
  ];
  
  const successCount = allTests.filter(test => test.success).length;
  const totalTests = allTests.length;
  
  results.overall = {
    success: successCount === totalTests,
    message: `Validation complete: ${successCount}/${totalTests} tests passed`,
    details: {
      passed_tests: successCount,
      total_tests: totalTests,
      success_rate: (successCount / totalTests) * 100
    }
  };
  
  console.log('Comprehensive validation complete:', results.overall);
  
  return results;
}

/**
 * Generate a detailed validation report
 */
export function generateValidationReport(results: ComprehensiveTestResult): string {
  const sections = [
    '# QuadFusion App Validation Report',
    `Generated: ${new Date().toISOString()}`,
    '',
    `## Overall Status: ${results.overall.success ? '✅ PASS' : '❌ FAIL'}`,
    `${results.overall.message}`,
    ''
  ];
  
  const testSections = [
    { name: 'API Connection', result: results.apiConnection },
    { name: 'Sensor Collection', result: results.sensorCollection },
    { name: 'Data Transmission', result: results.dataTransmission },
    { name: 'Agent Processing', result: results.agentProcessing },
    { name: 'Model Status', result: results.modelStatus }
  ];
  
  testSections.forEach(({ name, result }) => {
    sections.push(`## ${name}: ${result.success ? '✅ PASS' : '❌ FAIL'}`);
    sections.push(`**Message:** ${result.message}`);
    
    if (result.details) {
      sections.push('**Details:**');
      sections.push('```json');
      sections.push(JSON.stringify(result.details, null, 2));
      sections.push('```');
    }
    
    if (result.errors && result.errors.length > 0) {
      sections.push('**Errors:**');
      result.errors.forEach(error => sections.push(`- ${error}`));
    }
    
    sections.push('');
  });
  
  return sections.join('\n');
}