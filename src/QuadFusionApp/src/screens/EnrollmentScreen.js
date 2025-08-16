import React, { useState } from 'react';
import { View, ScrollView, Alert } from 'react-native';
import EnrollmentForm from '../components/EnrollmentForm';
import StatusIndicator from '../components/StatusIndicator';

const EnrollmentScreen = () => {
  const [enrolledUsers, setEnrolledUsers] = useState([]);

  const handleEnrollmentComplete = (userId, result) => {
    setEnrolledUsers(prev => [...prev, { userId, result, timestamp: new Date() }]);
    Alert.alert(
      'Enrollment Complete',
      `User ${userId} has been successfully enrolled with ${result.models_trained?.length || 0} models trained.`,
      [{ text: 'OK' }]
    );
  };

  return (
    <ScrollView className="flex-1 bg-gray-50">
      <View className="p-4">
        <EnrollmentForm onEnrollmentComplete={handleEnrollmentComplete} />
        
        {enrolledUsers.length > 0 && (
          <View className="mt-6 bg-white rounded-lg shadow-sm p-4">
            <Text className="text-lg font-bold text-gray-800 mb-3">Recently Enrolled Users</Text>
            {enrolledUsers.slice().reverse().map((enrollment, index) => (
              <View key={index} className="mb-3 p-3 bg-gray-50 rounded-lg">
                <Text className="font-medium text-gray-800 mb-1">
                  User: {enrollment.userId}
                </Text>
                <StatusIndicator 
                  status={enrollment.result.status} 
                  label={`Status: ${enrollment.result.status}`}
                  size="sm"
                />
                <Text className="text-xs text-gray-500 mt-2">
                  Enrolled: {enrollment.timestamp.toLocaleString()}
                </Text>
                <Text className="text-xs text-gray-500">
                  Models: {enrollment.result.models_trained?.join(', ') || 'None'}
                </Text>
              </View>
            ))}
          </View>
        )}
      </View>
    </ScrollView>
  );
};

export default EnrollmentScreen;