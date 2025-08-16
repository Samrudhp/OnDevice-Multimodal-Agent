import { useState, useEffect, useCallback } from 'react';
import SensorManager from '../services/SensorManager';

export const useSensorData = () => {
  const [sensorData, setSensorData] = useState(null);
  const [isCollecting, setIsCollecting] = useState(false);
  const [error, setError] = useState(null);

  const startCollection = useCallback(async () => {
    try {
      const hasPermissions = await SensorManager.requestPermissions();
      if (!hasPermissions) {
        throw new Error('Required permissions not granted');
      }

      SensorManager.startMotionSensors();
      setIsCollecting(true);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Failed to start sensor collection:', err);
    }
  }, []);

  const stopCollection = useCallback(() => {
    SensorManager.stopMotionSensors();
    setIsCollecting(false);
  }, []);

  const getCurrentSensorData = useCallback(() => {
    return SensorManager.getSensorData();
  }, []);

  const clearData = useCallback(() => {
    SensorManager.clearSensorData();
    setSensorData(null);
  }, []);

  useEffect(() => {
    let interval;
    
    if (isCollecting) {
      interval = setInterval(() => {
        const data = SensorManager.getSensorData();
        setSensorData(data);
      }, 1000); // Update every second
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isCollecting]);

  return {
    sensorData,
    isCollecting,
    error,
    startCollection,
    stopCollection,
    getCurrentSensorData,
    clearData,
  };
};