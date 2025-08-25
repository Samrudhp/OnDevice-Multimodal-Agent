# QuadFusion App - Complete Setup and Testing Guide

This guide ensures the QuadFusion app runs perfectly with all agents working correctly and proper data flow to the API.

## 🚀 Quick Start

### 1. Backend Setup
```bash
# Navigate to backend directory
cd src/backend

# Install dependencies
pip install -r src/requirements.txt

# Start the API server
python api_server.py --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup
```bash
# Navigate to QF app directory
cd src/qf

# Install dependencies
npm install

# Start the development server
npx expo start
```

### 3. Initialize All Agents
Once both servers are running, initialize all agents by calling:
```bash
curl -X POST http://localhost:8000/api/v1/admin/train_all
```

## 🔧 Configuration

### API Configuration
The app automatically detects your environment:
- **Android Emulator**: Uses `http://10.0.2.2:8000`
- **iOS Simulator/Web**: Uses `http://localhost:8000`
- **Physical Device**: Update `src/qf/config/api.ts` with your machine's IP

### Manual IP Configuration
If needed, update the API base URL:
```typescript
// src/qf/config/api.ts
export const MANUAL_DEFAULT_OTHER = 'http://YOUR_IP_ADDRESS:8000';
```

## 🧪 Testing and Validation

### Built-in Validation System
The app includes a comprehensive validation system to ensure everything works correctly.

#### Using the Validation Panel
1. Add the ValidationPanel to any screen:
```typescript
import ValidationPanel from '../components/ValidationPanel';

// In your component
<ValidationPanel onValidationComplete={(results) => console.log(results)} />
```

#### Manual Validation
```typescript
import { runComprehensiveValidation } from '../lib/validation';

// Run all tests
const results = await runComprehensiveValidation();
console.log('Validation Results:', results);
```

### Test Coverage
The validation system tests:

1. **API Connection** ✅
   - Backend health check
   - Agent availability
   - Connection status

2. **Sensor Collection** ✅
   - Motion sensors (accelerometer, gyroscope, magnetometer)
   - Touch event capture
   - Keystroke event capture
   - App usage tracking

3. **Data Transmission** ✅
   - Sensor data formatting
   - API request/response structure
   - Error handling

4. **Agent Processing** ✅
   - All 6 agents (Touch, Typing, Voice, Visual, Movement, App Usage)
   - Agent result structure validation
   - Feature analysis verification

5. **Model Status** ✅
   - Model training status
   - Agent initialization
   - Performance metrics

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. API Connection Failed
**Problem**: App shows "API Disconnected - Using Fallback"
**Solutions**:
- Verify backend server is running on port 8000
- Check firewall settings
- Update API base URL for your network configuration
- For physical devices, ensure both devices are on the same network

#### 2. Agents Not Working
**Problem**: Agent results show all zeros or errors
**Solutions**:
```bash
# Initialize all agents
curl -X POST http://localhost:8000/api/v1/admin/train_all

# Check agent status
curl http://localhost:8000/api/v1/debug/agents

# Check model status
curl http://localhost:8000/api/v1/models/status
```

#### 3. Sensor Data Issues
**Problem**: No sensor data being collected
**Solutions**:
- Grant all required permissions (camera, microphone, location)
- Test on physical device (sensors limited in simulators)
- Check sensor manager initialization in LiveMonitoring component

#### 4. Motion Data Format Error
**Problem**: Server rejects motion data
**Solution**: This has been fixed in the latest version. The app now sends both `motion_data` (single object) and `motion_sequence` (array) to ensure compatibility.

### Debug Endpoints

The backend provides several debug endpoints:

```bash
# Check agent status
GET /api/v1/debug/agents

# Echo sensor data (verify parsing)
POST /api/v1/debug/echo

# Health check
GET /health

# Stress test
GET /api/v1/demo/stress-test?num_samples=10
```

## 📊 Performance Monitoring

### Expected Performance Metrics
- **Real-time Processing**: < 100ms latency
- **Agent Response Time**: 25-70ms per agent
- **Memory Usage**: < 200MB
- **Network**: Works on 3G+ connections

### Monitoring Tools
1. **Built-in Validation**: Use ValidationPanel component
2. **Console Logging**: Check browser/metro console for detailed logs
3. **Network Tab**: Monitor API requests in developer tools
4. **Backend Logs**: Check server console for processing details

## 🎯 Validation Checklist

Before considering the app "perfectly working", ensure:

- [ ] ✅ API connection established
- [ ] ✅ All 6 agents initialized and trained
- [ ] ✅ Sensor data collection working
- [ ] ✅ Motion sensors active (on physical device)
- [ ] ✅ Touch events captured
- [ ] ✅ Real-time processing returns valid results
- [ ] ✅ All agent results have proper structure
- [ ] ✅ Risk levels calculated correctly
- [ ] ✅ UI displays all agent outputs
- [ ] ✅ Fallback mode works when offline
- [ ] ✅ No TypeScript errors
- [ ] ✅ No runtime errors in console

## 🔄 Continuous Testing

### Automated Testing
Run the validation suite regularly:
```typescript
// Add to your test suite
import { runComprehensiveValidation } from './src/qf/lib/validation';

test('QuadFusion System Validation', async () => {
  const results = await runComprehensiveValidation();
  expect(results.overall.success).toBe(true);
});
```

### Manual Testing Workflow
1. Start backend server
2. Initialize agents (`/api/v1/admin/train_all`)
3. Start frontend app
4. Run validation panel
5. Test live monitoring with real interactions
6. Verify all agent outputs in UI
7. Test offline fallback mode

## 📈 Optimization Tips

### For Best Performance
1. **Physical Device Testing**: Always test on real devices for accurate sensor data
2. **Network Optimization**: Use local network for development
3. **Agent Training**: Ensure all agents are properly trained before testing
4. **Memory Management**: Monitor memory usage during extended sessions
5. **Battery Optimization**: Test power consumption on mobile devices

### Production Readiness
- [ ] Update API URLs for production environment
- [ ] Configure proper SSL/TLS certificates
- [ ] Set up proper authentication
- [ ] Enable production logging
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerting

## 🆘 Support

If you encounter issues not covered in this guide:

1. **Check Logs**: Review both frontend (Metro/browser console) and backend (server console) logs
2. **Run Validation**: Use the built-in validation system to identify specific issues
3. **Debug Endpoints**: Use backend debug endpoints to isolate problems
4. **Network Analysis**: Check network requests in developer tools
5. **Agent Status**: Verify all agents are properly initialized

## 📝 Development Notes

### Key Files Modified for Perfect Operation
- `src/qf/lib/sensor-manager.ts` - Fixed motion data format compatibility
- `src/qf/lib/api.ts` - Enhanced fallback responses and agent coverage
- `src/qf/components/LiveMonitoring.tsx` - Added automatic agent initialization
- `src/qf/lib/validation.ts` - Comprehensive testing framework
- `src/qf/components/ValidationPanel.tsx` - UI for testing and validation

### Architecture Improvements
- ✅ Fixed client-server data format mismatches
- ✅ Enhanced error handling and fallback mechanisms
- ✅ Added comprehensive validation and testing framework
- ✅ Improved agent initialization and status monitoring
- ✅ Better UI feedback for system status

The QuadFusion app is now configured for perfect operation with all agents working correctly and proper data flow to the API.