# QuadFusion Real Data Verification Guide

This guide shows you exactly what happens when you start monitoring and analyze camera data, with detailed logging to verify real data collection and transmission.

## 🔍 What to Expect When You Start Monitoring

### 1. **Starting Monitoring Process**
When you tap "Start Monitoring", you'll see these console logs:

```
🚀 Starting monitoring - initializing sensors...
✅ Motion sensors started
🎤 Capturing audio sample...
✅ Audio sample captured: {duration: 1.5, sampleRate: 16000, dataSize: 12345}
📷 Camera capture would happen here (requires camera integration)
✅ Monitoring started successfully
```

**Alert Message**: "Monitoring Started - All sensors are now active and collecting data. Interact with your device normally."

### 2. **Real-Time Data Collection**
As you interact with the device, you'll see:

```
👆 Touch event captured: down at (540.5, 960.0) pressure=0.80
👆 Touch event captured: move at (542.1, 958.3) pressure=0.75
👆 Touch event captured: up at (545.0, 955.0) pressure=0.60
⌨️ Keystroke captured: key=65 action=down pressure=0.50
⌨️ Keystroke captured: key=65 action=up pressure=0.45
```

### 3. **Motion Sensor Data**
Motion sensors automatically collect data every 100ms:
- **Accelerometer**: Detects device movement and orientation
- **Gyroscope**: Measures rotation and angular velocity  
- **Magnetometer**: Detects magnetic field changes

### 4. **When You Stop Monitoring**
You'll see detailed data transmission logs:

```
🔄 Processing collected sensor data...
📊 Session ID: live-monitoring-1234567890
📤 Data being sent to API:
  • Touch events: 15
  • Keystroke events: 8
  • Motion data: Available
  • Audio data: Available (12345 chars)
  • Image data: Not available
  • App usage events: 2
  • Motion details: accel=[0.120, -0.050, 9.810]
🚀 Sending data to backend API...
✅ Received processing result from API
📥 API Response Summary:
  • Anomaly score: 0.25
  • Risk level: low
  • Confidence: 0.87
  • Processing time: 145ms
  • Agents processed: 6
🤖 Individual Agent Results:
  • TouchPatternAgent:
    - Anomaly score: 0.20
    - Risk level: low
    - Confidence: 0.90
    - Features: [pressure_variance, swipe_speed, tremor_score]
    - Processing time: 40ms
  • TypingBehaviorAgent:
    - Anomaly score: 0.15
    - Risk level: low
    - Confidence: 0.85
    - Features: [dwell_times, flight_times, rhythm]
    - Processing time: 30ms
  [... and so on for all 6 agents]
```

**Alert Message**: "Analysis Complete - Risk Level: LOW, Confidence: 87%, Agents: 6"

## 📷 What to Expect When You Analyze Camera

### 1. **Camera Analysis Process**
When you tap "Analyze Face" in the camera tab:

```
📷 Starting camera analysis...
📸 Capturing image from camera...
✅ Image captured successfully:
  • Image size: 45678 characters
  • Camera facing: front
  • Image dimensions: 1080x1920
✅ Image data stored in sensor manager
🚀 Sending image data to API for analysis (Session: camera-capture-1234567890)
📤 Sensor data being sent:
  • Image data: Available
  • Camera type: front
  • Touch events: 3
  • Motion data: Available
```

### 2. **API Processing Response**
```
✅ Received analysis result from API:
  • Overall risk level: low
  • Confidence: 0.82
  • Agents processed: 6
👁️ Visual Agent Results:
  • Anomaly score: 0.30
  • Features analyzed: [face_match, scene_consistency]
  • Metadata: {face_detected: true, image_data_available: true}
📷 Camera analysis completed
```

**Alert Message**: "Analysis Complete - Face detected with 82.0% confidence"

### 3. **If Camera Capture Fails**
```
📷 Camera capture not available, using simulated analysis
🎭 Simulated results: {faceDetected: true, confidence: 75.2, biometricMatch: false}
```

**Alert Message**: "Simulation Mode - Camera capture not available. Using simulated facial analysis."

## 🔧 How to Verify Real Data is Being Sent

### 1. **Check Console Logs**
Open your development console (Metro bundler or browser dev tools) and look for:
- ✅ **Green checkmarks**: Successful operations
- 📤 **Upload arrows**: Data being sent to API
- 📥 **Download arrows**: Data received from API
- 🤖 **Robot icons**: Agent processing results

### 2. **Verify Network Requests**
In browser dev tools Network tab, look for:
- `POST /api/v1/process/realtime` - Real-time processing requests
- `POST /api/v1/admin/train_all` - Agent initialization
- `GET /api/v1/models/status` - Model status checks

### 3. **Check Backend Logs**
In your backend server console, you should see:
```
INFO - Processing sensor data for session: live-monitoring-1234567890
INFO - TouchPatternAgent processed 15 touch events
INFO - VisualAgent processed image data (45678 chars)
INFO - Returning processing result with 6 agent results
```

## 🚨 Troubleshooting Real Data Issues

### Problem: No Touch Events Captured
**Symptoms**: Console shows `Touch events: 0`
**Solution**: 
- Ensure you're interacting with the screen while monitoring
- Check that touch event handlers are properly attached
- Try scrolling or tapping buttons during monitoring

### Problem: No Motion Data
**Symptoms**: Console shows `Motion data: Not available`
**Solution**:
- Test on a physical device (simulators have limited sensor support)
- Grant motion sensor permissions
- Move the device during monitoring

### Problem: No Audio Data
**Symptoms**: Console shows `Audio data: Not available`
**Solution**:
- Grant microphone permissions
- Ensure device has working microphone
- Check audio recording implementation

### Problem: No Camera Data
**Symptoms**: Console shows `Image data: Not available`
**Solution**:
- Grant camera permissions
- Ensure camera is working
- Try both front and rear cameras
- Test on physical device

### Problem: API Not Receiving Data
**Symptoms**: All data shows as "Available" but API returns empty results
**Solution**:
- Check network connectivity
- Verify backend server is running on correct port
- Check API base URL configuration
- Look for CORS or network errors

## 📊 Expected Data Volumes

### Typical Monitoring Session (30 seconds):
- **Touch Events**: 10-50 events (depending on interaction)
- **Keystroke Events**: 5-20 events (if typing)
- **Motion Data**: 300+ samples (10Hz sampling rate)
- **Audio Data**: ~24,000 characters (1.5 seconds of audio)
- **Image Data**: 40,000-80,000 characters (if camera used)

### API Response Times:
- **Real-time Processing**: 50-200ms
- **Agent Processing**: 25-70ms per agent
- **Total Response**: Usually under 300ms

## ✅ Verification Checklist

Before considering the system "working correctly":

- [ ] Console shows sensor initialization messages
- [ ] Touch events are logged when interacting with screen
- [ ] Motion data shows real accelerometer values (not all zeros)
- [ ] Audio data is captured and shows realistic size
- [ ] Camera analysis captures real image data
- [ ] API requests are sent with actual sensor data
- [ ] Backend responds with processing results
- [ ] All 6 agents return individual results
- [ ] UI displays analysis results correctly
- [ ] Network tab shows successful API calls
- [ ] Backend logs show data processing

## 🎯 Success Indicators

**You know the system is working correctly when:**

1. **Console logs show real data**: Actual coordinates, sensor values, and data sizes
2. **API calls succeed**: Network requests return 200 status codes
3. **Agent results vary**: Different anomaly scores and confidence levels
4. **Processing times are realistic**: 50-300ms response times
5. **Data sizes make sense**: Audio ~24k chars, images ~50k chars
6. **Alerts show real values**: Actual confidence percentages and risk levels

The enhanced logging system now provides complete visibility into data collection, transmission, and processing, so you can verify that real sensor data is being captured and sent to the API for analysis.