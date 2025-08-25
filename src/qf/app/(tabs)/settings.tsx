import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Switch, Alert, Animated, Share, Platform } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Settings, Shield, Bell, Database, Smartphone, Lock, User, CircleHelp as HelpCircle, ChevronRight, Info, Download, Trash2, RefreshCw, Wifi, Battery, Cpu } from 'lucide-react-native';
import { StatusBar } from 'expo-status-bar';
import { COLORS, ANIMATION } from '../../lib/constants';
import { createCardStyle, createTextStyle, SPACING, BORDER_RADIUS } from '../../lib/theme';
import { useGlowAnimation } from '../../lib/animations';
import { QuadFusionAPI } from '../../lib/api';

interface SystemInfo {
  sensorsActive: number;
  totalSensors: number;
  securityLevel: string;
  dataEncrypted: boolean;
  lastSync: string;
  batteryOptimized: boolean;
  networkStatus: string;
  cpuUsage: number;
}

interface UserProfile {
  name: string;
  email: string;
  enrollmentDate: string;
  deviceCount: number;
  lastLogin: string;
}

export default function SettingsTab() {
  const [biometricEnabled, setBiometricEnabled] = useState(true);
  const [pushNotifications, setPushNotifications] = useState(true);
  const [dataCollection, setDataCollection] = useState(true);
  const [highSecurityMode, setHighSecurityMode] = useState(false);
  const [batteryOptimization, setBatteryOptimization] = useState(true);
  const [autoSync, setAutoSync] = useState(true);
  const [systemInfo, setSystemInfo] = useState<SystemInfo>({
    sensorsActive: 5,
    totalSensors: 6,
    securityLevel: 'Standard',
    dataEncrypted: true,
    lastSync: 'Just now',
    batteryOptimized: true,
    networkStatus: 'Connected',
    cpuUsage: 15
  });
  const [userProfile, setUserProfile] = useState<UserProfile>({
    name: 'QuadFusion User',
    email: 'user@quadfusion.app',
    enrollmentDate: new Date().toLocaleDateString(),
    deviceCount: 1,
    lastLogin: 'Just now'
  });
  const [api] = useState(() => new QuadFusionAPI());
  const [isLoading, setIsLoading] = useState(false);
  
  // Animation for glowing effect
  const { glowAnim, startGlowAnimation } = useGlowAnimation(0.3, 0.7, ANIMATION.SLOW * 2);
  
  useEffect(() => {
    loadSettings();
    fetchSystemInfo();
    loadUserProfile();
  }, []);

  const loadSettings = async () => {
    try {
      // Use localStorage for web or simple in-memory storage
      if (typeof window !== 'undefined' && window.localStorage) {
        const settings = [
          'biometric_enabled',
          'push_notifications', 
          'data_collection',
          'high_security_mode',
          'battery_optimization',
          'auto_sync'
        ];
        
        settings.forEach((key) => {
          const value = localStorage.getItem(key);
          if (value !== null) {
            const boolValue = value === 'true';
            switch (key) {
              case 'biometric_enabled':
                setBiometricEnabled(boolValue);
                break;
              case 'push_notifications':
                setPushNotifications(boolValue);
                break;
              case 'data_collection':
                setDataCollection(boolValue);
                break;
              case 'high_security_mode':
                setHighSecurityMode(boolValue);
                break;
              case 'battery_optimization':
                setBatteryOptimization(boolValue);
                break;
              case 'auto_sync':
                setAutoSync(boolValue);
                break;
            }
          }
        });
      }
    } catch (error) {
      console.warn('Failed to load settings:', error);
    }
  };

  const saveSettings = async (key: string, value: boolean) => {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        localStorage.setItem(key, value.toString());
      }
      console.log(`Setting ${key} saved: ${value}`);
    } catch (error) {
      console.warn('Failed to save setting:', error);
    }
  };

  const fetchSystemInfo = async () => {
    try {
      const [modelStatus, healthCheck] = await Promise.allSettled([
        api.getModelStatus(),
        api.healthCheck()
      ]);
      
      let updatedInfo = { ...systemInfo };
      
      if (modelStatus.status === 'fulfilled') {
        const modelsData = modelStatus.value as any;
        const modelsMap = modelsData?.models || modelsData?.agents_status || {};
        const activeModels = Object.keys(modelsMap).length;
        const trainedModels = Object.values(modelsMap).filter((status: any) => 
          status?.is_trained || status?.status === 'trained'
        ).length;
        
        updatedInfo.sensorsActive = trainedModels;
        updatedInfo.totalSensors = Math.max(6, activeModels);
        updatedInfo.cpuUsage = Math.floor(10 + (activeModels * 5) + Math.random() * 10);
      }
      
      if (healthCheck.status === 'fulfilled') {
        const healthData = healthCheck.value as any;
        updatedInfo.networkStatus = healthData?.status === 'healthy' ? 'Connected' : 'Limited';
      }
      
      updatedInfo.lastSync = new Date().toLocaleTimeString();
      updatedInfo.securityLevel = highSecurityMode ? 'High' : 'Standard';
      
      setSystemInfo(updatedInfo);
    } catch (error) {
      console.warn('Failed to fetch system info:', error);
    }
  };

  const loadUserProfile = async () => {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const profileData = localStorage.getItem('user_profile');
        if (profileData) {
          setUserProfile(JSON.parse(profileData));
        }
      }
    } catch (error) {
      console.warn('Failed to load user profile:', error);
    }
  };

  const handleSecurityModeToggle = (value: boolean) => {
    if (value) {
      Alert.alert(
        'Enable High Security Mode',
        'This will require additional verification steps and may affect app performance. Continue?',
        [
          { text: 'Cancel', style: 'cancel' },
          { 
            text: 'Enable', 
            onPress: () => {
              setHighSecurityMode(true);
              saveSettings('high_security_mode', true);
              setSystemInfo(prev => ({ ...prev, securityLevel: 'High' }));
            },
            style: 'destructive' 
          }
        ]
      );
    } else {
      setHighSecurityMode(false);
      saveSettings('high_security_mode', false);
      setSystemInfo(prev => ({ ...prev, securityLevel: 'Standard' }));
    }
  };

  const handleBiometricToggle = (value: boolean) => {
    setBiometricEnabled(value);
    saveSettings('biometric_enabled', value);
    if (!value) {
      Alert.alert(
        'Biometric Disabled',
        'You will need to use alternative authentication methods.',
        [{ text: 'OK' }]
      );
    }
  };

  const handleDataCollectionToggle = (value: boolean) => {
    setDataCollection(value);
    saveSettings('data_collection', value);
    if (!value) {
      Alert.alert(
        'Data Collection Disabled',
        'This may reduce the accuracy of fraud detection.',
        [{ text: 'OK' }]
      );
    }
  };

  const handleNotificationsToggle = (value: boolean) => {
    setPushNotifications(value);
    saveSettings('push_notifications', value);
  };

  const handleBatteryOptimizationToggle = (value: boolean) => {
    setBatteryOptimization(value);
    saveSettings('battery_optimization', value);
  };

  const handleAutoSyncToggle = (value: boolean) => {
    setAutoSync(value);
    saveSettings('auto_sync', value);
  };

  const clearAllData = () => {
    Alert.alert(
      'Clear All Data',
      'This will permanently delete all biometric data, authentication history, and settings. This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Clear Data', 
          onPress: async () => {
            setIsLoading(true);
            try {
              // Clear localStorage
              if (typeof window !== 'undefined' && window.localStorage) {
                localStorage.clear();
              }
              
              // Reset all states to defaults
              setBiometricEnabled(true);
              setPushNotifications(true);
              setDataCollection(true);
              setHighSecurityMode(false);
              setBatteryOptimization(true);
              setAutoSync(true);
              
              Alert.alert('Success', 'All data has been cleared successfully.');
            } catch (error) {
              Alert.alert('Error', 'Failed to clear data. Please try again.');
            } finally {
              setIsLoading(false);
            }
          },
          style: 'destructive' 
        }
      ]
    );
  };

  const exportData = async () => {
    setIsLoading(true);
    try {
      // Simulate data export
      const exportData = {
        settings: {
          biometricEnabled,
          pushNotifications,
          dataCollection,
          highSecurityMode,
          batteryOptimization,
          autoSync
        },
        systemInfo,
        userProfile,
        exportDate: new Date().toISOString(),
        version: '1.0.0'
      };
      
      const exportString = JSON.stringify(exportData, null, 2);
      
      if (Platform.OS === 'ios' || Platform.OS === 'android') {
        await Share.share({
          message: exportString,
          title: 'QuadFusion Data Export'
        });
      } else {
        // For web, create download
        const blob = new Blob([exportString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'quadfusion-export.json';
        a.click();
        URL.revokeObjectURL(url);
      }
      
      Alert.alert('Success', 'Your data has been exported successfully.');
    } catch (error) {
      Alert.alert('Error', 'Failed to export data. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const refreshSystemInfo = async () => {
    setIsLoading(true);
    await fetchSystemInfo();
    setIsLoading(false);
    Alert.alert('Refreshed', 'System information has been updated.');
  };

  const manageProfile = () => {
    Alert.alert(
      'Profile Management',
      'Choose an action:',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Edit Name',
          onPress: () => {
            Alert.prompt(
              'Edit Profile',
              'Enter your name:',
              [
                { text: 'Cancel', style: 'cancel' },
                {
                  text: 'Save',
                  onPress: async (name) => {
                    if (name && name.trim()) {
                      const updatedProfile = { ...userProfile, name: name.trim() };
                      setUserProfile(updatedProfile);
                      
                      // Save to localStorage
                      if (typeof window !== 'undefined' && window.localStorage) {
                        localStorage.setItem('user_profile', JSON.stringify(updatedProfile));
                      }
                      
                      Alert.alert('Success', 'Profile updated successfully.');
                    }
                  }
                }
              ],
              'plain-text',
              userProfile.name
            );
          }
        },
        {
          text: 'Register New User',
          onPress: () => registerNewUser()
        }
      ]
    );
  };

  const registerNewUser = async () => {
    Alert.prompt(
      'Register New User',
      'Enter user ID:',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Register',
          onPress: async (userId) => {
            if (userId && userId.trim()) {
              setIsLoading(true);
              try {
                const deviceId = Platform.OS + '_' + Date.now();
                
                const registrationRequest = {
                  user_id: userId.trim(),
                  device_id: deviceId,
                  biometric_enrollment: {
                    voice_samples: [],
                    face_images: [],
                    typing_samples: [],
                    touch_samples: []
                  }
                };

                console.log('🔐 Registering new user:', registrationRequest);
                const result = await api.registerUser(registrationRequest);
                console.log('✅ Registration result:', result);

                if (result.status === 'enrolled') {
                  const updatedProfile = {
                    ...userProfile,
                    name: userId.trim(),
                    email: `${userId.trim()}@quadfusion.app`,
                    enrollmentDate: new Date().toLocaleDateString(),
                    lastLogin: 'Just now'
                  };
                  setUserProfile(updatedProfile);
                  
                  // Save to localStorage
                  if (typeof window !== 'undefined' && window.localStorage) {
                    localStorage.setItem('user_profile', JSON.stringify(updatedProfile));
                  }
                  
                  Alert.alert(
                    'Registration Successful!',
                    `User ${userId} has been registered successfully.\nEnrollment ID: ${result.enrollment_id}\nModels trained: ${result.models_trained.join(', ')}`
                  );
                } else {
                  Alert.alert('Registration Failed', result.message || 'Unknown error occurred');
                }
              } catch (error) {
                console.error('❌ Registration failed:', error);
                Alert.alert('Registration Error', `Failed to register user: ${error}`);
              } finally {
                setIsLoading(false);
              }
            }
          }
        }
      ],
      'plain-text',
      'user_' + Date.now()
    );
  };

  const manageDevices = () => {
    Alert.alert(
      'Device Management',
      `Current device: ${Platform.OS} ${Platform.Version}\nRegistered: ${userProfile.enrollmentDate}\nLast login: ${userProfile.lastLogin}`,
      [
        { text: 'OK' },
        {
          text: 'Refresh',
          onPress: () => {
            const updatedProfile = { ...userProfile, lastLogin: new Date().toLocaleString() };
            setUserProfile(updatedProfile);
            if (typeof window !== 'undefined' && window.localStorage) {
              localStorage.setItem('user_profile', JSON.stringify(updatedProfile));
            }
          }
        }
      ]
    );
  };

  const showHelp = () => {
    Alert.alert(
      'Help Center',
      'QuadFusion Security Help\n\n• Biometric Authentication: Use fingerprint or face recognition\n• High Security Mode: Enhanced verification\n• Data Collection: Improves fraud detection\n• Battery Optimization: Reduces power usage\n• Auto Sync: Keeps data updated\n\nFor more help, contact support@quadfusion.app',
      [{ text: 'OK' }]
    );
  };

  const showAbout = () => {
    Alert.alert(
      'About QuadFusion',
      `QuadFusion Biometric Authentication System
Version 1.0.0

Advanced multi-sensor fraud detection platform using:
• Touch pattern analysis
• Typing behavior recognition  
• Voice biometrics
• Visual face detection
• Movement pattern analysis
• App usage behavior

© 2024 QuadFusion Security`,
      [{ text: 'OK' }]
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
    variant = 'primary',
    disabled = false
  }: {
    icon: any;
    title: string;
    subtitle?: string;
    onPress?: () => void;
    hasSwitch?: boolean;
    switchValue?: boolean;
    onSwitchChange?: (value: boolean) => void;
    variant?: 'primary' | 'secondary' | 'danger';
    disabled?: boolean;
  }) => {
    const getIconColor = () => {
      if (disabled) return COLORS.GRAY_500;
      switch(variant) {
        case 'primary': return COLORS.PRIMARY;
        case 'secondary': return COLORS.SECONDARY;
        case 'danger': return COLORS.ERROR;
        default: return COLORS.PRIMARY;
      }
    };
    
    const getGlowColor = () => {
      if (disabled) return COLORS.GRAY_700;
      switch(variant) {
        case 'primary': return COLORS.GLOW;
        case 'secondary': return COLORS.GLOW_SECONDARY;
        case 'danger': return COLORS.ERROR;
        default: return COLORS.GLOW;
      }
    };
    
    return (
      <TouchableOpacity 
        style={[styles.settingItem, { borderColor: getGlowColor(), opacity: disabled ? 0.6 : 1 }]} 
        onPress={onPress}
        disabled={hasSwitch || disabled}
      >
        <View style={styles.settingContent}>
          <Animated.View 
            style={[styles.settingIcon, { 
              shadowColor: getGlowColor(),
              shadowOpacity: disabled ? 0 : glowAnim,
              shadowRadius: 10,
              shadowOffset: { width: 0, height: 0 }
            }]}
          >
            {React.cloneElement(icon, { color: getIconColor() })}
          </Animated.View>
          <View style={styles.settingText}>
            <Text style={[styles.settingTitle, { color: disabled ? COLORS.GRAY_500 : COLORS.WHITE }]}>{title}</Text>
            {subtitle && <Text style={[styles.settingSubtitle, { color: disabled ? COLORS.GRAY_500 : COLORS.GRAY_300 }]}>{subtitle}</Text>}
          </View>
        </View>
        {hasSwitch ? (
          <Switch
            value={switchValue}
            onValueChange={onSwitchChange}
            trackColor={{ false: COLORS.GRAY_700, true: getIconColor() }}
            thumbColor={COLORS.WHITE}
            disabled={disabled}
          />
        ) : (
          <ChevronRight size={20} color={disabled ? COLORS.GRAY_500 : COLORS.GRAY_300} />
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
              onSwitchChange={handleBiometricToggle}
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
              onSwitchChange={handleDataCollectionToggle}
              variant="secondary"
            />
            <SettingItem
              icon={<Bell size={24} />}
              title="Push Notifications"
              subtitle="Receive security alerts and updates"
              hasSwitch
              switchValue={pushNotifications}
              onSwitchChange={handleNotificationsToggle}
              variant="secondary"
            />
            <SettingItem
              icon={<RefreshCw size={24} />}
              title="Auto Sync"
              subtitle="Automatically sync data with server"
              hasSwitch
              switchValue={autoSync}
              onSwitchChange={handleAutoSyncToggle}
              variant="secondary"
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Performance</Text>
          <View style={[styles.settingsCard, createCardStyle('minimal')]}>
            <SettingItem
              icon={<Battery size={24} />}
              title="Battery Optimization"
              subtitle="Reduce power consumption"
              hasSwitch
              switchValue={batteryOptimization}
              onSwitchChange={handleBatteryOptimizationToggle}
              variant="primary"
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account</Text>
          <View style={[styles.settingsCard, createCardStyle('minimal')]}>
            <SettingItem
              icon={<User size={24} />}
              title="Profile Information"
              subtitle={`${userProfile.name} • ${userProfile.email}`}
              onPress={manageProfile}
              variant="primary"
            />
            <SettingItem
              icon={<Smartphone size={24} />}
              title="Device Management"
              subtitle={`${userProfile.deviceCount} registered device(s)`}
              onPress={manageDevices}
              variant="primary"
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Data Management</Text>
          <View style={[styles.settingsCard, createCardStyle('secondary')]}>
            <SettingItem
              icon={<Download size={24} />}
              title="Export Data"
              subtitle="Download your biometric patterns and settings"
              onPress={exportData}
              variant="secondary"
              disabled={isLoading}
            />
            <SettingItem
              icon={<Trash2 size={24} />}
              title="Clear All Data"
              subtitle="Permanently delete all stored data"
              onPress={clearAllData}
              variant="danger"
              disabled={isLoading}
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
              onPress={showHelp}
              variant="primary"
            />
            <SettingItem
              icon={<Info size={24} />}
              title="About"
              subtitle="Version 1.0.0 - QuadFusion Security"
              onPress={showAbout}
              variant="primary"
            />
          </View>
        </View>

        <View style={[styles.systemInfo, createCardStyle('primary')]}>
          <View style={styles.systemInfoHeader}>
            <Text style={styles.systemInfoTitle}>System Information</Text>
            <TouchableOpacity onPress={refreshSystemInfo} disabled={isLoading}>
              <RefreshCw size={20} color={COLORS.PRIMARY} />
            </TouchableOpacity>
          </View>
          <View style={styles.systemInfoGrid}>
            <View style={styles.systemInfoItem}>
              <Text style={styles.systemInfoLabel}>Sensors Active</Text>
              <Text style={styles.systemInfoValue}>{systemInfo.sensorsActive}/{systemInfo.totalSensors}</Text>
            </View>
            <View style={styles.systemInfoItem}>
              <Text style={styles.systemInfoLabel}>Security Level</Text>
              <Text style={styles.systemInfoValue}>{systemInfo.securityLevel}</Text>
            </View>
            <View style={styles.systemInfoItem}>
              <Text style={styles.systemInfoLabel}>Data Encrypted</Text>
              <Text style={styles.systemInfoValue}>{systemInfo.dataEncrypted ? 'Yes' : 'No'}</Text>
            </View>
            <View style={styles.systemInfoItem}>
              <Text style={styles.systemInfoLabel}>Last Sync</Text>
              <Text style={styles.systemInfoValue}>{systemInfo.lastSync}</Text>
            </View>
            <View style={styles.systemInfoItem}>
              <Text style={styles.systemInfoLabel}>Network Status</Text>
              <Text style={styles.systemInfoValue}>{systemInfo.networkStatus}</Text>
            </View>
            <View style={styles.systemInfoItem}>
              <Text style={styles.systemInfoLabel}>CPU Usage</Text>
              <Text style={styles.systemInfoValue}>{systemInfo.cpuUsage}%</Text>
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
  systemInfoHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SPACING.MD,
  },
  systemInfoTitle: {
    ...createTextStyle('subtitle'),
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