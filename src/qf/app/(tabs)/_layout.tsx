import { Tabs } from 'expo-router';
import * as Icons from 'lucide-react-native';
import { COLORS } from '../../lib/constants';
import { StyleSheet } from 'react-native';

const Shield = Icons.Shield ?? (() => null);
const Camera = Icons.Camera ?? (() => null);
const Settings = Icons.Settings ?? (() => null);
const BarChart3 = Icons['ChartBar'] ?? (() => null);
const Activity = Icons.Activity ?? (() => null);

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor: COLORS.SURFACE,
    borderTopWidth: 1,
    borderTopColor: COLORS.GLOW,
    paddingBottom: 8,
    paddingTop: 8,
    height: 80,
    elevation: 0,
    shadowOpacity: 0,
    borderTopRightRadius: 16,
    borderTopLeftRadius: 16,
  },
  tabBarLabel: {
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 4,
  },
  tabBarIcon: {
    marginTop: 4,
  },
});

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: COLORS.PRIMARY,
        tabBarInactiveTintColor: COLORS.GRAY_500,
        tabBarStyle: styles.tabBar,
        tabBarLabelStyle: styles.tabBarLabel,
        tabBarIconStyle: styles.tabBarIcon,
        tabBarBackground: () => (
          <></>  // We'll use the style for background
        ),
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: 'Authentication',
          tabBarIcon: ({ size, color }) => (
            <Shield size={size} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="sensors"
        options={{
          title: 'Sensors',
          tabBarIcon: ({ size, color }) => (
            <Activity size={size} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="camera"
        options={{
          title: 'Camera',
          tabBarIcon: ({ size, color }) => (
            <Camera size={size} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="analytics"
        options={{
          title: 'Analytics',
          tabBarIcon: ({ size, color }) => (
            <BarChart3 size={size} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: 'Settings',
          tabBarIcon: ({ size, color }) => (
            <Settings size={size} color={color} />
          ),
        }}
      />
    </Tabs>
  );
}