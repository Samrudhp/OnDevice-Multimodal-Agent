import React from 'react';
import { View, Text, ScrollView } from 'react-native';
import StatusIndicator from './StatusIndicator';

const ProcessingResult = ({ result }) => {
  if (!result) {
    return (
      <View className="p-4 bg-white rounded-lg shadow-sm">
        <Text className="text-gray-500 text-center">No processing results available</Text>
      </View>
    );
  }

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'low': return 'text-green-600';
      case 'medium': return 'text-yellow-600';
      case 'high': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <ScrollView className="bg-white rounded-lg shadow-sm">
      <View className="p-4">
        <Text className="text-lg font-bold text-gray-800 mb-4">Processing Results</Text>
        
        {/* Overall Status */}
        <View className="mb-4">
          <StatusIndicator 
            status={result.processing_result?.risk_level || 'unknown'} 
            label={`Risk Level: ${result.processing_result?.risk_level || 'Unknown'}`}
          />
        </View>

        {/* Scores */}
        <View className="mb-4">
          <Text className="text-md font-semibold text-gray-700 mb-2">Overall Scores</Text>
          <View className="bg-gray-50 p-3 rounded">
            <Text className="text-sm text-gray-600 mb-1">
              Anomaly Score: <Text className="font-medium">
                {result.processing_result?.anomaly_score?.toFixed(3) || 'N/A'}
              </Text>
            </Text>
            <Text className="text-sm text-gray-600 mb-1">
              Confidence: <Text className="font-medium">
                {result.processing_result?.confidence?.toFixed(3) || 'N/A'}
              </Text>
            </Text>
            <Text className="text-sm text-gray-600">
              Processing Time: <Text className="font-medium">
                {result.processing_result?.processing_time_ms?.toFixed(1) || 'N/A'}ms
              </Text>
            </Text>
          </View>
        </View>

        {/* Agent Results */}
        {result.processing_result?.agent_results && (
          <View className="mb-4">
            <Text className="text-md font-semibold text-gray-700 mb-2">Agent Analysis</Text>
            {Object.entries(result.processing_result.agent_results).map(([agentName, agentResult]) => (
              <View key={agentName} className="bg-gray-50 p-3 rounded mb-2">
                <Text className="text-sm font-semibold text-gray-700 mb-1">
                  {agentName.replace(/([A-Z])/g, ' $1').trim()}
                </Text>
                <Text className="text-xs text-gray-600">
                  Anomaly Score: {agentResult.anomaly_score?.toFixed(3) || 'N/A'}
                </Text>
                {agentResult.features_analyzed && (
                  <Text className="text-xs text-gray-600 mt-1">
                    Features: {agentResult.features_analyzed.join(', ')}
                  </Text>
                )}
              </View>
            ))}
          </View>
        )}
      </View>
    </ScrollView>
  );
};

export default ProcessingResult;