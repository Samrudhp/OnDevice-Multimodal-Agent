import React from 'react';
import { View, StyleSheet } from 'react-native';
import Svg, { Line, Rect } from 'react-native-svg';
import { COLORS } from '../lib/constants';
import { gridBackgroundStyle } from '../lib/theme';

interface GridBackgroundProps {
  spacing?: number;
  opacity?: number;
  color?: string;
}

export const GridBackground: React.FC<GridBackgroundProps> = ({
  spacing = 20,
  opacity = 0.4,
  color = COLORS.GRID,
}) => {
  return (
    <View style={[gridBackgroundStyle.container, { opacity }]}>
      <Svg style={gridBackgroundStyle.grid}>
        <Rect
          x="0"
          y="0"
          width="100%"
          height="100%"
          fill={COLORS.BACKGROUND}
        />
        
        {/* Horizontal lines */}
        {Array.from({ length: 100 }).map((_, i) => (
          <Line
            key={`h-${i}`}
            x1="0"
            y1={i * spacing}
            x2="100%"
            y2={i * spacing}
            stroke={color}
            strokeWidth="0.5"
          />
        ))}
        
        {/* Vertical lines */}
        {Array.from({ length: 100 }).map((_, i) => (
          <Line
            key={`v-${i}`}
            x1={i * spacing}
            y1="0"
            x2={i * spacing}
            y2="100%"
            stroke={color}
            strokeWidth="0.5"
          />
        ))}
      </Svg>
    </View>
  );
};

export default GridBackground;