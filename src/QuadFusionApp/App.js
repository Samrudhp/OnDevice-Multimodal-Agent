import React, { useState } from 'react';
import {
  SafeAreaView,
  StatusBar,
  View,
  Text,
  TouchableOpacity,
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Icon from 'react-native-vector-icons/MaterialIcons';

import HomeScreen from './src/screens/HomeScreen';
import EnrollmentScreen from './src/screens/EnrollmentScreen';
import MonitoringScreen from './src/screens/MonitoringScreen';

const Tab = createBottomTabNavigator();

function App() {
  return (
    <SafeAreaView className="flex-1 bg-gray-50">
      <StatusBar barStyle="dark-content" backgroundColor="#f9fafb" />
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={({route}) => ({
            tabBarIcon: ({focused, color, size}) => {
              let iconName;
              
              if (route.name === 'Home') {
                iconName = 'home';
              } else if (route.name === 'Enrollment') {
                iconName = 'person-add';
              } else if (route.name === 'Monitoring') {
                iconName = 'security';
              }
              
              return <Icon name={iconName} size={size} color={color} />;
            },
            tabBarActiveTintColor: '#3b82f6',
            tabBarInactiveTintColor: 'gray',
            headerStyle: {
              backgroundColor: '#3b82f6',
            },
            headerTintColor: '#fff',
            headerTitleStyle: {
              fontWeight: 'bold',
            },
          })}
        >
          <Tab.Screen 
            name="Home" 
            component={HomeScreen}
            options={{title: 'QuadFusion'}}
          />
          <Tab.Screen 
            name="Enrollment" 
            component={EnrollmentScreen}
            options={{title: 'User Enrollment'}}
          />
          <Tab.Screen 
            name="Monitoring" 
            component={MonitoringScreen}
            options={{title: 'Live Monitoring'}}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </SafeAreaView>
  );
}

export default App;