import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TouchableOpacity, RefreshControl } from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import StatusIndicator from '../components/StatusIndicator';
import QuadFusionAPI from '../services/QuadFusionAPI';

const HomeScreen = () => {
  const [modelStatus, setModelStatus] = useState(null);
  const [config, setConfig] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchData = async () => {
    setIsLoading(true);
    try {
      const [modelResponse, configResponse] = await Promise.all([
        QuadFusionAPI.getModelStatus(),
        QuadFusionAPI.getConfig()
      ]);
      
      setModelStatus(modelResponse);
      setConfig(configResponse);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Failed to fetch data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const getModelStatusIndicator = (model) => {
    if (!model) return 'unknown';
    return model.is_trained ? 'success' : 'warning';
  };

  return (
    <ScrollView 
      className="flex-1 bg-gray-50"
      refreshControl={
        <RefreshControl refreshing={isLoading} onRefresh={fetchData} />
      }
    >
      <View className="p-4">
        {/* Header */}
        <View className="bg-blue-500 rounded-lg p-4 mb-4">
          <Text className="text-white text-2xl font-bold">QuadFusion</Text>
          <Text className="text-blue-100 text-sm mt-1">
            Multi-Modal Behavioral Fraud Detection
          </Text>
        </View>

        {/* System Status */}
        <View className="bg-white rounded-lg shadow-sm p-4 mb-4">
          <Text className="text-lg font-bold text-gray-800 mb-3">System Status</Text>
          
          {modelStatus ? (
            <>
              {Object.entries(modelStatus.models).map(([modelName, model]) => (
                <View key={modelName} className="mb-2">
                  <StatusIndicator
                    status={getModelStatusIndicator(model)}
                    label={`${modelName.charAt(0).toUpperCase() + modelName.slice(1)} Model`}
                  />
                  <Text className="text-xs text-gray-500 ml-8 mt-1">
                    Training samples: {model.training_samples || 0}
                  </Text>
                </View>
              ))}
            </>
          ) : (
            <StatusIndicator status="loading" label="Loading system status..." />
          )}
        </View>

        {/* Agent Configuration */}
        {config && (
          <View className="bg-white rounded-lg shadow-sm p-4 mb-4">
            <Text className="text-lg font-bold text-gray-800 mb-3">Agent Weights</Text>
            {Object.entries(config.agent_weights || {}).map(([agent, weight]) => (
              <View key={agent} className="flex-row justify-between items-center mb-2 bg-gray-50 p-2 rounded">
                <Text className="text-sm font-medium text-gray-700">
                  {agent.replace('Agent', '').replace(/([A-Z])/g, ' $1').trim()}
                </Text>
                <Text className="text-sm text-blue-600 font-medium">
                  {weight?.toFixed(2) || '0.00'}
                </Text>
              </View>
            ))}
          </View>
        )}

        {/* Risk Thresholds */}
        {config && (
          <View className="bg-white rounded-lg shadow-sm p-4 mb-4">
            <Text className="text-lg font-bold text-gray-800 mb-3">Risk Thresholds</Text>
            {Object.entries(config.risk_thresholds || {}).map(([level, threshold]) => (
              <View key={level} className="flex-row justify-between items-center mb-2 bg-gray-50 p-2 rounded">
                <Text className="text-sm font-medium text-gray-700 capitalize">
                  {level} Risk
                </Text>
                <Text className="text-sm text-red-600 font-medium">
                  {threshold?.toFixed(2) || '0.00'}
                </Text>
              </View>
            ))}
          </View>
        )}

        {/* Quick Actions */}
        <View className="bg-white rounded-lg shadow-sm p-4">
          <Text className="text-lg font-bold text-gray-800 mb-3">Quick Actions</Text>
          
          <TouchableOpacity className="flex-row items-center p-3 bg-blue-50 rounded-lg mb-2">
            <Icon name="refresh" size={24} color="#3b82f6" />
            <Text className="ml-3 text-blue-700 font-medium">Refresh Models</Text>
          </TouchableOpacity>
          
          <TouchableOpacity className="flex-row items-center p-3 bg-green-50 rounded-lg mb-2">
            <Icon name="settings" size={24} color="#10b981" />
            <Text className="ml-3 text-green-700 font-medium">Configure Settings</Text>
          </TouchableOpacity>
          
          <TouchableOpacity className="flex-row items-center p-3 bg-purple-50 rounded-lg">
            <Icon name="analytics" size={24} color="#8b5cf6" />
            <Text className="ml-3 text-purple-700 font-medium">View Analytics</Text>
          </TouchableOpacity>
        </View>

        {/* Last Updated */}
        {lastUpdated && (
          <Text className="text-center text-xs text-gray-500 mt-4">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </Text>
        )}
      </View>
    </ScrollView>
  );
};

export default HomeScreen;