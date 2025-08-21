import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Switch, Alert, Animated } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Settings, Shield, Bell, Database, Smartphone, Lock, User, CircleHelp as HelpCircle, ChevronRight, Info } from 'lucide-react-native';
import { StatusBar } from 'expo-status-bar';
import { COLORS, ANIMATION } from '../../lib/constants';
import { createCardStyle, createTextStyle, SPACING, BORDER_RADIUS } from '../../lib/theme';
import { useGlowAnimation } from '../../lib/animations';

export default function SettingsTab() {
  const [biometricEnabled, setBiometricEnabled] = useState(true);
  const [pushNotifications, setPushNotifications] = useState(true);
  const [dataCollection, setDataCollection] = useState(true);
  const [highSecurityMode, setHighSecurityMode] = useState(false);
  
  // Animation for glowing effect
  const { glowAnim, startGlowAnimation } = useGlowAnimation(0.3, 0.7, ANIMATION.SLOW * 2);
  
  useEffect(() => {
    // No continuous animations
  }, []);

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
    onSwitchChange,
    variant = 'primary'
  }: {
    icon: any;
    title: string;
    subtitle?: string;
    onPress?: () => void;
    hasSwitch?: boolean;
    switchValue?: boolean;
    onSwitchChange?: (value: boolean) => void;
    variant?: 'primary' | 'secondary' | 'danger';
  }) => {
    // Determine colors based on variant
    const getIconColor = () => {
      switch(variant) {
        case 'primary': return COLORS.PRIMARY;
        case 'secondary': return COLORS.SECONDARY;
        case 'danger': return COLORS.ERROR;
        default: return COLORS.PRIMARY;
      }
    };
    
    const getGlowColor = () => {
      switch(variant) {
        case 'primary': return COLORS.GLOW;
        case 'secondary': return COLORS.GLOW_SECONDARY;
        case 'danger': return COLORS.ERROR;
        default: return COLORS.GLOW;
      }
    };
    
    return (
      <TouchableOpacity 
        style={[styles.settingItem, { borderColor: getGlowColor() }]} 
        onPress={onPress}
        disabled={hasSwitch}
      >
        <View style={styles.settingContent}>
          <Animated.View 
            style={[styles.settingIcon, { 
              shadowColor: getGlowColor(),
              shadowOpacity: glowAnim,
              shadowRadius: 10,
              shadowOffset: { width: 0, height: 0 }
            }]}
          >
            {React.cloneElement(icon, { color: getIconColor() })}
          </Animated.View>
          <View style={styles.settingText}>
            <Text style={styles.settingTitle}>{title}</Text>
            {subtitle && <Text style={styles.settingSubtitle}>{subtitle}</Text>}
          </View>
        </View>
        {hasSwitch ? (
          <Switch
            value={switchValue}
            onValueChange={onSwitchChange}
            trackColor={{ false: COLORS.GRAY_700, true: getIconColor() }}
            thumbColor={COLORS.WHITE}
          />
        ) : (
          <ChevronRight size={20} color={COLORS.GRAY_300} />
        )}
      </TouchableOpacity>
    );
  }

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
          <Settings size={48} color={COLORS.PRIMARY} />
          <Text style={styles.title}>Settings</Text>
          <Text style={styles.subtitle}>Configure your security preferences</Text>
        </Animated.View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Security Settings</Text>
          <View style={[styles.settingsCard, createCardStyle('primary')]}>
            <SettingItem
              icon={<Shield size={24} />}
              title="Biometric Authentication"
              subtitle="Use fingerprint or face recognition"
              hasSwitch
              switchValue={biometricEnabled}
              onSwitchChange={setBiometricEnabled}
              variant="primary"
            />
            <SettingItem
              icon={<Lock size={24} />}
              title="High Security Mode"
              subtitle="Enhanced verification and monitoring"
              hasSwitch
              switchValue={highSecurityMode}
              onSwitchChange={handleSecurityModeToggle}
              variant="danger"
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Privacy & Data</Text>
          <View style={[styles.settingsCard, createCardStyle('secondary')]}>
            <SettingItem
              icon={<Database size={24} />}
              title="Data Collection"
              subtitle="Allow sensor data collection for fraud detection"
              hasSwitch
              switchValue={dataCollection}
              onSwitchChange={setDataCollection}
              variant="secondary"
            />
            <SettingItem
              icon={<Bell size={24} />}
              title="Push Notifications"
              subtitle="Receive security alerts and updates"
              hasSwitch
              switchValue={pushNotifications}
              onSwitchChange={setPushNotifications}
              variant="secondary"
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account</Text>
          <View style={[styles.settingsCard, createCardStyle('minimal')]}>
            <SettingItem
              icon={<User size={24} />}
              title="Profile Information"
              subtitle="Manage your account details"
              onPress={() => Alert.alert('Profile', 'Profile management coming soon')}
              variant="primary"
            />
            <SettingItem
              icon={<Smartphone size={24} />}
              title="Device Management"
              subtitle="Registered devices and sessions"
              onPress={() => Alert.alert('Devices', 'Device management coming soon')}
              variant="primary"
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Data Management</Text>
          <View style={[styles.settingsCard, createCardStyle('secondary')]}>
            <SettingItem
              icon={<Database size={24} />}
              title="Export Data"
              subtitle="Download your biometric patterns"
              onPress={exportData}
              variant="secondary"
            />
            <SettingItem
              icon={<Database size={24} />}
              title="Clear All Data"
              subtitle="Permanently delete all stored data"
              onPress={clearData}
              variant="danger"
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Support</Text>
          <View style={[styles.settingsCard, createCardStyle('minimal')]}>
            <SettingItem
              icon={<HelpCircle size={24} />}
              title="Help Center"
              subtitle="Get help with QuadFusion"
              onPress={() => Alert.alert('Help', 'Help center coming soon')}
              variant="primary"
            />
            <SettingItem
              icon={<Info size={24} />}
              title="About"
              subtitle="Version 1.0.0 - QuadFusion Security"
              onPress={() => Alert.alert('About', 'QuadFusion Biometric Authentication System\nVersion 1.0.0\n\nAdvanced multi-sensor fraud detection platform.')}
              variant="primary"
            />
          </View>
        </View>

        <View style={[styles.systemInfo, createCardStyle('primary')]}>
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
  section: {
    marginBottom: SPACING.XL,
  },
  sectionTitle: {
    ...createTextStyle('subtitle'),
    marginBottom: SPACING.MD,
    paddingHorizontal: SPACING.XS,
    textTransform: 'uppercase',
    letterSpacing: 1.2,
  },
  settingsCard: {
    // Card styles are applied from createCardStyle
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: SPACING.MD,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.GRAY_700,
  },
  settingContent: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  settingIcon: {
    marginRight: SPACING.MD,
    padding: SPACING.XS,
    borderRadius: BORDER_RADIUS.FULL,
  },
  settingText: {
    flex: 1,
  },
  settingTitle: {
    ...createTextStyle('body'),
    marginBottom: 2,
  },
  settingSubtitle: {
    ...createTextStyle('caption'),
  },
  systemInfo: {
    padding: SPACING.LG,
  },
  systemInfoTitle: {
    ...createTextStyle('subtitle'),
    marginBottom: SPACING.MD,
    textTransform: 'uppercase',
    letterSpacing: 1.2,
  },
  systemInfoGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  systemInfoItem: {
    width: '48%',
    marginBottom: SPACING.MD,
  },
  systemInfoLabel: {
    ...createTextStyle('caption'),
    marginBottom: SPACING.XS,
  },
  systemInfoValue: {
    ...createTextStyle('body'),
    color: COLORS.PRIMARY,
  },
});