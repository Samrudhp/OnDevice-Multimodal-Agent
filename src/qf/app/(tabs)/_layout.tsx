import { Tabs } from 'expo-router';
import * as Icons from 'lucide-react-native';
import { COLORS, ANIMATION } from '../../lib/constants';
import { StyleSheet, View, Animated } from 'react-native';
import { BORDER_RADIUS, SPACING } from '../../lib/theme';
import GridBackground from '../../components/GridBackground';
import { screenDimensions, deviceSize, DeviceSize } from '../../lib/responsive';
import { useGlowAnimation, usePulseAnimation } from '../../lib/animations';
import { useEffect, useRef } from 'react';

const Shield = Icons.ShieldCheck ?? (() => null);
const Camera = Icons.Camera ?? (() => null);
const Settings = Icons.Settings ?? (() => null);
const BarChart3 = Icons.BarChart3 ?? (() => null);
const Activity = Icons.Activity ?? (() => null);
const Home = Icons.Home ?? (() => null);

// Get responsive tab bar height based on device size
const getTabBarHeight = () => {
  switch (deviceSize) {
    case DeviceSize.SMALL:
      return 75;
    case DeviceSize.MEDIUM:
      return 85;
    case DeviceSize.LARGE:
      return 90;
    case DeviceSize.XLARGE:
      return 100;
    default:
      return 85;
  }
};

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor: COLORS.SURFACE,
    borderTopWidth: 1,
    borderTopColor: COLORS.GLOW,
    paddingBottom: SPACING.SM,
    paddingTop: SPACING.SM,
    height: getTabBarHeight(),
    elevation: 10,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: -3 },
    shadowOpacity: 0.6,
    shadowRadius: 10,
    borderTopRightRadius: BORDER_RADIUS.XL,
    borderTopLeftRadius: BORDER_RADIUS.XL,
    position: 'relative',
    overflow: 'hidden',
    borderLeftWidth: 1,
    borderRightWidth: 1,
    borderLeftColor: COLORS.GLOW,
    borderRightColor: COLORS.GLOW,
  },
  tabBarBackground: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    opacity: 0.15,
  },
  tabBarDivider: {
    position: 'absolute',
    top: 10,
    left: '20%',
    right: '20%',
    height: 1,
    backgroundColor: COLORS.ACCENT,
    opacity: 0.4,
  },
  tabBarLabel: {
    fontSize: screenDimensions.isSmallDevice ? 10 : 12,
    fontWeight: '600',
    marginBottom: screenDimensions.isSmallDevice ? 0 : 2,
    color: COLORS.WHITE,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  tabBarLabelActive: {
    color: COLORS.PRIMARY,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 8,
    letterSpacing: 1.5,
  },
  tabBarIcon: {
    marginTop: screenDimensions.isSmallDevice ? 0 : SPACING.XS,
  },
  tabBarIconContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: SPACING.XS,
  },
  tabBarIconActive: {
    backgroundColor: 'rgba(0, 255, 255, 0.1)',
    borderRadius: BORDER_RADIUS.LG,
    padding: SPACING.XS,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 5,
  },
  activeIndicator: {
    position: 'absolute',
    top: -3,
    width: screenDimensions.isSmallDevice ? 20 : 30,
    height: 3,
    backgroundColor: COLORS.PRIMARY,
    borderRadius: BORDER_RADIUS.FULL,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 5,
    zIndex: 10,
  },
  cornerAccent: {
    position: 'absolute',
    width: 15,
    height: 15,
    borderColor: COLORS.ACCENT,
    zIndex: 10,
    opacity: 0.7,
  },
  topLeft: {
    top: 5,
    left: 5,
    borderTopWidth: 2,
    borderLeftWidth: 2,
  },
  topRight: {
    top: 5,
    right: 5,
    borderTopWidth: 2,
    borderRightWidth: 2,
  },
});

export default function TabLayout() {
  // Animations for the tab bar
  const { glowAnim, startGlowAnimation } = useGlowAnimation(0.5, 1, ANIMATION.SLOW * 2);
  const { pulseAnim, startPulseAnimation } = usePulseAnimation(0.05, ANIMATION.NORMAL * 2);
  
  // Animation for the active tab indicator
  const indicatorAnim = useRef(new Animated.Value(0)).current;
  // Use scaleX on a fixed-width indicator instead of animating width directly
  const indicatorScale = indicatorAnim.interpolate({
    inputRange: [0, 15],
    outputRange: [1, 1.5],
    extrapolate: 'clamp'
  });
  
  useEffect(() => {
    // No continuous animations
  }, []);
  
  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: COLORS.PRIMARY,
        tabBarInactiveTintColor: COLORS.GRAY_500,
        tabBarStyle: [styles.tabBar, {
          shadowOpacity: glowAnim,
        }],
        tabBarLabelStyle: styles.tabBarLabel,
        tabBarIconStyle: styles.tabBarIcon,
        headerShown: false,
        tabBarBackground: () => (
          <>
            <GridBackground spacing={15} opacity={0.2} />
            <Animated.View style={styles.tabBarDivider} />
            <Animated.View 
              style={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                height: 1,
                backgroundColor: COLORS.GLOW,
                opacity: glowAnim,
              }}
            />
            <Animated.View style={[styles.cornerAccent, styles.topLeft, { opacity: glowAnim }]} />
            <Animated.View style={[styles.cornerAccent, styles.topRight, { opacity: glowAnim }]} />
          </>
        ),
        tabBarItemStyle: {
          paddingVertical: SPACING.XS,
        },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Home',
          tabBarIcon: ({ color, focused }) => (
            <View style={styles.tabBarIconContainer}>
              {focused && (
                <Animated.View 
                  style={[styles.activeIndicator, {
                    width: screenDimensions.isSmallDevice ? 20 : 30,
                    transform: [{ scaleX: indicatorScale }],
                    opacity: glowAnim
                  }]} 
                />
              )}
              <Animated.View style={[focused ? styles.tabBarIconActive : null, {
                transform: [{ scale: focused ? pulseAnim : 1 }],
              }]}>
                <Home size={screenDimensions.isSmallDevice ? 20 : 24} color={focused ? COLORS.PRIMARY : COLORS.GRAY_500} />
              </Animated.View>
            </View>
          ),
        }}
      />
      <Tabs.Screen
        name="analytics"
        options={{
          title: 'Analytics',
          tabBarIcon: ({ color, focused }) => (
            <View style={styles.tabBarIconContainer}>
              {focused && (
                <Animated.View 
                  style={[styles.activeIndicator, {
                    width: screenDimensions.isSmallDevice ? 20 : 30,
                    transform: [{ scaleX: indicatorScale }],
                    opacity: glowAnim
                  }]} 
                />
              )}
              <Animated.View style={[focused ? styles.tabBarIconActive : null, {
                transform: [{ scale: focused ? pulseAnim : 1 }],
              }]}>
                <BarChart3 size={screenDimensions.isSmallDevice ? 20 : 24} color={focused ? COLORS.PRIMARY : COLORS.GRAY_500} />
              </Animated.View>
            </View>
          ),
        }}
      />
      <Tabs.Screen
        name="scan"
        options={{
          title: 'Scan',
          tabBarIcon: ({ color, focused }) => (
            <View style={styles.tabBarIconContainer}>
              {focused && (
                <Animated.View 
                  style={[styles.activeIndicator, {
                    width: screenDimensions.isSmallDevice ? 20 : 30,
                    transform: [{ scaleX: indicatorScale }],
                    opacity: glowAnim
                  }]} 
                />
              )}
              <Animated.View style={[focused ? styles.tabBarIconActive : null, {
                transform: [{ scale: focused ? pulseAnim : 1 }],
              }]}>
                <Camera size={screenDimensions.isSmallDevice ? 20 : 24} color={focused ? COLORS.PRIMARY : COLORS.GRAY_500} />
              </Animated.View>
            </View>
          ),
        }}
      />
      <Tabs.Screen
        name="monitor"
        options={{
          title: 'Monitor',
          tabBarIcon: ({ color, focused }) => (
            <View style={styles.tabBarIconContainer}>
              {focused && (
                <Animated.View 
                  style={[styles.activeIndicator, {
                    width: screenDimensions.isSmallDevice ? 20 : 30,
                    transform: [{ scaleX: indicatorScale }],
                    opacity: glowAnim
                  }]} 
                />
              )}
              <Animated.View style={[focused ? styles.tabBarIconActive : null, {
                transform: [{ scale: focused ? pulseAnim : 1 }],
              }]}>
                <Activity size={screenDimensions.isSmallDevice ? 20 : 24} color={focused ? COLORS.PRIMARY : COLORS.GRAY_500} />
              </Animated.View>
            </View>
          ),
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: 'Settings',
          tabBarIcon: ({ color, focused }) => (
            <View style={styles.tabBarIconContainer}>
              {focused && (
                <Animated.View 
                  style={[styles.activeIndicator, {
                    width: screenDimensions.isSmallDevice ? 20 : 30,
                    transform: [{ scaleX: indicatorScale }],
                    opacity: glowAnim
                  }]} 
                />
              )}
              <Animated.View style={[focused ? styles.tabBarIconActive : null, {
                transform: [{ scale: focused ? pulseAnim : 1 }],
              }]}>
                <Settings size={screenDimensions.isSmallDevice ? 20 : 24} color={focused ? COLORS.PRIMARY : COLORS.GRAY_500} />
              </Animated.View>
            </View>
          ),
        }}
      />
    </Tabs>
  );
}