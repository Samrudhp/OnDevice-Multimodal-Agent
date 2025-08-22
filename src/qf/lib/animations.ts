import { Animated, Easing } from 'react-native';
import { ANIMATION } from './constants';

// Pulse animation - creates a subtle pulsing effect
export const usePulseAnimation = (intensity = 0.05, duration = ANIMATION.NORMAL * 2) => {
  const pulseAnim = new Animated.Value(1);

  const startPulseAnimation = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1 + intensity,
          duration: duration,
          easing: Easing.inOut(Easing.sin),
          useNativeDriver: false,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: duration,
          easing: Easing.inOut(Easing.sin),
          useNativeDriver: false,
        }),
      ])
    ).start();
  };

  return { pulseAnim, startPulseAnimation };
};

// Glow animation - creates a glowing effect
export const useGlowAnimation = (minOpacity = 0.5, maxOpacity = 1, duration = ANIMATION.SLOW * 2) => {
  const glowAnim = new Animated.Value(minOpacity);

  const startGlowAnimation = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(glowAnim, {
            toValue: maxOpacity,
            duration: duration,
            easing: Easing.inOut(Easing.sin),
            useNativeDriver: false,
          }),
        Animated.timing(glowAnim, {
            toValue: minOpacity,
            duration: duration,
            easing: Easing.inOut(Easing.sin),
            useNativeDriver: false,
          }),
      ])
    ).start();
  };

  return { glowAnim, startGlowAnimation };
};

// Fade in animation
export const useFadeInAnimation = (duration = ANIMATION.NORMAL) => {
  const fadeAnim = new Animated.Value(0);

  const startFadeInAnimation = () => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: duration,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: false,
    }).start();
  };

  return { fadeAnim, startFadeInAnimation };
};

// Slide up animation
export const useSlideUpAnimation = (distance = 50, duration = ANIMATION.NORMAL) => {
  const slideAnim = new Animated.Value(distance);

  const startSlideUpAnimation = () => {
    Animated.timing(slideAnim, {
      toValue: 0,
      duration: duration,
      easing: Easing.out(Easing.back(1.5)),
      useNativeDriver: false,
    }).start();
  };

  return { slideAnim, startSlideUpAnimation };
};

// Scale animation
export const useScaleAnimation = (startScale = 0.95, endScale = 1, duration = ANIMATION.FAST) => {
  const scaleAnim = new Animated.Value(startScale);

  const startScaleAnimation = () => {
    Animated.timing(scaleAnim, {
      toValue: endScale,
      duration: duration,
      easing: Easing.out(Easing.back(1.5)),
      useNativeDriver: false,
    }).start();
  };

  return { scaleAnim, startScaleAnimation };
};

// Button press animation
export const useButtonPressAnimation = () => {
  const pressAnim = new Animated.Value(1);

  const onPressIn = () => {
    Animated.timing(pressAnim, {
      toValue: 0.95,
      duration: ANIMATION.FAST,
      easing: Easing.inOut(Easing.quad),
      useNativeDriver: false,
    }).start();
  };

  const onPressOut = () => {
    Animated.timing(pressAnim, {
      toValue: 1,
      duration: ANIMATION.FAST,
      easing: Easing.inOut(Easing.quad),
      useNativeDriver: false,
    }).start();
  };

  return { pressAnim, onPressIn, onPressOut };
};

// Rotating animation (for loading indicators)
export const useRotateAnimation = (duration = 1500) => {
  const rotateAnim = new Animated.Value(0);

  const startRotateAnimation = () => {
    Animated.loop(
      Animated.timing(rotateAnim, {
        toValue: 1,
        duration: duration,
        easing: Easing.linear,
        useNativeDriver: false,
      })
    ).start();
  };

  const interpolatedRotate = rotateAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  return { rotateAnim, interpolatedRotate, startRotateAnimation };
};