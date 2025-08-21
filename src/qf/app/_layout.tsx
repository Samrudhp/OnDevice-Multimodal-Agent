import { useEffect } from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useColorScheme, View, Animated, StyleSheet, Text } from 'react-native';
import { useFrameworkReady } from '@/hooks/useFrameworkReady';
import { COLORS, ANIMATION } from '@/lib/constants';
import GridBackground from '@/components/GridBackground';
import { commonStyles, SPACING, createTextStyle } from '@/lib/theme';
import { ResponsiveContainer } from '@/components/ResponsiveContainer';
import { useDimensionsListener } from '@/lib/responsive';
import { useGlowAnimation, usePulseAnimation } from '@/lib/animations';

export default function RootLayout() {
  useFrameworkReady();
  
  // Force dark mode for cyber theme
  const colorScheme = 'dark';
  
  const { glowAnim, startGlowAnimation } = useGlowAnimation(0.4, 0.8);
  const { pulseAnim, startPulseAnimation } = usePulseAnimation(0.03);
  
  // Listen for dimension changes (e.g., rotation)
  useEffect(() => {
    const unsubscribe = useDimensionsListener((dimensions) => {
      // This will be called when the screen dimensions change
      // We can use this to adjust layouts if needed
    });
    
    // Clean up subscription when component unmounts
    return () => unsubscribe();
  }, []);

  return (
    <View style={commonStyles.container}>
      <GridBackground spacing={30} opacity={0.2} />
      
      {/* Cyber border effect */}
      <Animated.View style={[styles.topBorder, { opacity: glowAnim }]} />
      <Animated.View style={[styles.bottomBorder, { opacity: glowAnim }]} />
      <Animated.View style={[styles.leftBorder, { opacity: glowAnim }]} />
      <Animated.View style={[styles.rightBorder, { opacity: glowAnim }]} />
      
      {/* Corner accents */}
      <Animated.View style={[styles.cornerAccent, styles.topLeft, { opacity: glowAnim }]} />
      <Animated.View style={[styles.cornerAccent, styles.topRight, { opacity: glowAnim }]} />
      <Animated.View style={[styles.cornerAccent, styles.bottomLeft, { opacity: glowAnim }]} />
      <Animated.View style={[styles.cornerAccent, styles.bottomRight, { opacity: glowAnim }]} />
      
      <ResponsiveContainer fullHeight>
        <Animated.View style={[styles.appContainer, {
          transform: [{ scale: pulseAnim }],
          shadowOpacity: glowAnim
        }]}>
          <Stack 
            screenOptions={{
              headerShown: false,
              contentStyle: { backgroundColor: 'transparent' },
              animation: 'fade',
              animationDuration: 200,
            }}
          >
            <Stack.Screen name="+not-found" />
          </Stack>
        </Animated.View>
      </ResponsiveContainer>
      
      {/* App branding */}
      <Animated.View style={[styles.brandingContainer, {
        opacity: glowAnim,
        transform: [{ scale: pulseAnim }]
      }]}>
        <Text style={styles.brandingText}>QUADFUSION</Text>
        <Text style={styles.versionText}>v2.0.4</Text>
      </Animated.View>
      
      <StatusBar style="light" />
    </View>
  );
}

const styles = StyleSheet.create({
  appContainer: {
    flex: 1,
    shadowColor: COLORS.GLOW,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.6,
    shadowRadius: 20,
  },
  topBorder: {
    position: 'absolute',
    top: 0,
    left: 20,
    right: 20,
    height: 2,
    backgroundColor: COLORS.GLOW,
    zIndex: 10,
  },
  bottomBorder: {
    position: 'absolute',
    bottom: 0,
    left: 20,
    right: 20,
    height: 2,
    backgroundColor: COLORS.GLOW,
    zIndex: 10,
  },
  leftBorder: {
    position: 'absolute',
    top: 20,
    bottom: 20,
    left: 0,
    width: 2,
    backgroundColor: COLORS.GLOW,
    zIndex: 10,
  },
  rightBorder: {
    position: 'absolute',
    top: 20,
    bottom: 20,
    right: 0,
    width: 2,
    backgroundColor: COLORS.GLOW,
    zIndex: 10,
  },
  cornerAccent: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderColor: COLORS.ACCENT,
    zIndex: 10,
  },
  topLeft: {
    top: 0,
    left: 0,
    borderTopWidth: 2,
    borderLeftWidth: 2,
  },
  topRight: {
    top: 0,
    right: 0,
    borderTopWidth: 2,
    borderRightWidth: 2,
  },
  bottomLeft: {
    bottom: 0,
    left: 0,
    borderBottomWidth: 2,
    borderLeftWidth: 2,
  },
  bottomRight: {
    bottom: 0,
    right: 0,
    borderBottomWidth: 2,
    borderRightWidth: 2,
  },
  brandingContainer: {
    position: 'absolute',
    bottom: SPACING.MD,
    right: SPACING.MD,
    alignItems: 'flex-end',
  },
  brandingText: {
    ...createTextStyle('mono'),
    fontSize: 12,
    letterSpacing: 2,
    color: COLORS.ACCENT,
    textShadowColor: COLORS.GLOW,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 4,
  },
  versionText: {
    ...createTextStyle('mono'),
    fontSize: 10,
    color: COLORS.GRAY_500,
    marginTop: 2,
  },
});
