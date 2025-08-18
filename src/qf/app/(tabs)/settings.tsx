import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Switch, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Settings, Shield, Bell, Database, Smartphone, Lock, User, CircleHelp as HelpCircle, ChevronRight, Info } from 'lucide-react-native';
import { StatusBar } from 'expo-status-bar';

export default function SettingsTab() {
  const [biometricEnabled, setBiometricEnabled] = useState(true);
  const [pushNotifications, setPushNotifications] = useState(true);
  const [dataCollection, setDataCollection] = useState(true);
  const [highSecurityMode, setHighSecurityMode] = useState(false);

  const handleSecurityModeToggle = (value: boolean) => {
    if (value) {
      Alert.alert(
        'Enable High Security Mode',
        'This will require additional verification steps and may affect app performance. Continue?',
        [
          { text: 'Cancel', style: 'cancel' },
          { 
            text: 'Enable', 
            onPress: () => setHighSecurityMode(true),
            style: 'destructive' 
          }
        ]
      );
    } else {
      setHighSecurityMode(false);
    }
  };

  const clearData = () => {
    Alert.alert(
      'Clear All Data',
      'This will permanently delete all biometric data and authentication history. This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Clear Data', 
          onPress: () => Alert.alert('Data Cleared', 'All data has been cleared successfully.'),
          style: 'destructive' 
        }
      ]
    );
  };

  const exportData = () => {
    Alert.alert(
      'Export Data',
      'Your biometric patterns and authentication history will be exported to a secure file.',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Export', 
          onPress: () => Alert.alert('Data Exported', 'Your data has been exported successfully.')
        }
      ]
    );
  };

  const SettingItem = ({ 
    icon, 
    title, 
    subtitle, 
    onPress, 
    hasSwitch = false, 
    switchValue, 
    onSwitchChange 
  }: {
    icon: any;
    title: string;
    subtitle?: string;
    onPress?: () => void;
    hasSwitch?: boolean;
    switchValue?: boolean;
    onSwitchChange?: (value: boolean) => void;
  }) => (
    <TouchableOpacity 
      style={styles.settingItem} 
      onPress={onPress}
      disabled={hasSwitch}
    >
      <View style={styles.settingContent}>
        <View style={styles.settingIcon}>
          {icon}
        </View>
        <View style={styles.settingText}>
          <Text style={styles.settingTitle}>{title}</Text>
          {subtitle && <Text style={styles.settingSubtitle}>{subtitle}</Text>}
        </View>
      </View>
      {hasSwitch ? (
        <Switch
          value={switchValue}
          onValueChange={onSwitchChange}
          trackColor={{ false: '#D1D5DB', true: '#2563EB' }}
          thumbColor="#FFFFFF"
        />
      ) : (
        <ChevronRight size={20} color="#9CA3AF" />
      )}
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Settings size={48} color="#2563EB" />
          <Text style={styles.title}>Settings</Text>
          <Text style={styles.subtitle}>Configure your security preferences</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Security Settings</Text>
          <View style={styles.settingsCard}>
            <SettingItem
              icon={<Shield size={24} color="#2563EB" />}
              title="Biometric Authentication"
              subtitle="Use fingerprint or face recognition"
              hasSwitch
              switchValue={biometricEnabled}
              onSwitchChange={setBiometricEnabled}
            />
            <SettingItem
              icon={<Lock size={24} color="#EF4444" />}
              title="High Security Mode"
              subtitle="Enhanced verification and monitoring"
              hasSwitch
              switchValue={highSecurityMode}
              onSwitchChange={handleSecurityModeToggle}
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Privacy & Data</Text>
          <View style={styles.settingsCard}>
            <SettingItem
              icon={<Database size={24} color="#10B981" />}
              title="Data Collection"
              subtitle="Allow sensor data collection for fraud detection"
              hasSwitch
              switchValue={dataCollection}
              onSwitchChange={setDataCollection}
            />
            <SettingItem
              icon={<Bell size={24} color="#F59E0B" />}
              title="Push Notifications"
              subtitle="Receive security alerts and updates"
              hasSwitch
              switchValue={pushNotifications}
              onSwitchChange={setPushNotifications}
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account</Text>
          <View style={styles.settingsCard}>
            <SettingItem
              icon={<User size={24} color="#6B7280" />}
              title="Profile Information"
              subtitle="Manage your account details"
              onPress={() => Alert.alert('Profile', 'Profile management coming soon')}
            />
            <SettingItem
              icon={<Smartphone size={24} color="#6B7280" />}
              title="Device Management"
              subtitle="Registered devices and sessions"
              onPress={() => Alert.alert('Devices', 'Device management coming soon')}
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Data Management</Text>
          <View style={styles.settingsCard}>
            <SettingItem
              icon={<Database size={24} color="#2563EB" />}
              title="Export Data"
              subtitle="Download your biometric patterns"
              onPress={exportData}
            />
            <SettingItem
              icon={<Database size={24} color="#EF4444" />}
              title="Clear All Data"
              subtitle="Permanently delete all stored data"
              onPress={clearData}
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Support</Text>
          <View style={styles.settingsCard}>
            <SettingItem
              icon={<HelpCircle size={24} color="#6B7280" />}
              title="Help Center"
              subtitle="Get help with QuadFusion"
              onPress={() => Alert.alert('Help', 'Help center coming soon')}
            />
            <SettingItem
              icon={<Info size={24} color="#6B7280" />}
              title="About"
              subtitle="Version 1.0.0 - QuadFusion Security"
              onPress={() => Alert.alert('About', 'QuadFusion Biometric Authentication System\nVersion 1.0.0\n\nAdvanced multi-sensor fraud detection platform.')}
            />
          </View>
        </View>

        <View style={styles.systemInfo}>
          <Text style={styles.systemInfoTitle}>System Information</Text>
          <View style={styles.systemInfoGrid}>
            <View style={styles.systemInfoItem}>
              <Text style={styles.systemInfoLabel}>Sensors Active</Text>
              <Text style={styles.systemInfoValue}>5/5</Text>
            </View>
            <View style={styles.systemInfoItem}>
              <Text style={styles.systemInfoLabel}>Security Level</Text>
              <Text style={styles.systemInfoValue}>{highSecurityMode ? 'High' : 'Standard'}</Text>
            </View>
            <View style={styles.systemInfoItem}>
              <Text style={styles.systemInfoLabel}>Data Encrypted</Text>
              <Text style={styles.systemInfoValue}>Yes</Text>
            </View>
            <View style={styles.systemInfoItem}>
              <Text style={styles.systemInfoLabel}>Last Sync</Text>
              <Text style={styles.systemInfoValue}>Just now</Text>
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
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 12,
    paddingHorizontal: 4,
  },
  settingsCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  settingContent: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  settingIcon: {
    marginRight: 16,
  },
  settingText: {
    flex: 1,
  },
  settingTitle: {
    fontSize: 16,
    fontWeight: '500',
    color: '#1F2937',
    marginBottom: 2,
  },
  settingSubtitle: {
    fontSize: 14,
    color: '#6B7280',
  },
  systemInfo: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  systemInfoTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 16,
  },
  systemInfoGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  systemInfoItem: {
    width: '48%',
    marginBottom: 12,
  },
  systemInfoLabel: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 4,
  },
  systemInfoValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
  },
});