# Backend Connection Setup Guide

## Issues Fixed

### ✅ 1. App Configuration Issues
- **Fixed**: Missing icon path in `app.json`
- **Fixed**: Route mismatch between tab layout and actual files
- **Fixed**: AsyncStorage dependency removed for web compatibility

### ✅ 2. Route Configuration
- **Fixed**: Tab layout now correctly references `camera` and `sensors` instead of `scan` and `monitor`
- **Fixed**: Removed non-existent routes from `app.json`

## 🚨 Critical Issue: Backend Server Not Running

The logs show that the Android app is correctly trying to connect to `http://10.0.2.2:8000` but getting `[AbortError: Aborted]` errors. This means **the backend server is not running**.

## 🔧 Backend Server Setup Instructions

### Step 1: Start the Backend Server

```bash
# Navigate to the backend directory
cd src/backend

# Install Python dependencies (if not already done)
pip install -r src/requirements.txt

# Start the FastAPI server
python api_server.py --host 0.0.0.0 --port 8000

# Alternative: Use uvicorn directly
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### Step 2: Verify Server is Running

Open your browser and go to:
- **Health Check**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs

You should see:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-25T07:22:00.000Z",
  "quadfusion_available": true,
  "agents_initialized": 6,
  "active_sessions": 0
}
```

### Step 3: Train the Models (Optional)

To get real model data instead of fallbacks:

```bash
# Train all models with synthetic data
curl -X POST http://localhost:8000/api/v1/admin/train_all

# Check model status
curl http://localhost:8000/api/v1/models/status
```

## 📱 Android Connection Configuration

The Android app is correctly configured to use:
- **Android Emulator**: `http://10.0.2.2:8000` (maps to host machine's localhost:8000)
- **Physical Device**: You may need to use your computer's IP address

### For Physical Android Device

If using a physical Android device, update the API configuration:

```typescript
// In src/qf/config/api.ts, change line 9:
export const MANUAL_DEFAULT_ANDROID = 'http://YOUR_COMPUTER_IP:8000';
```

Find your computer's IP:
```bash
# Windows
ipconfig

# macOS/Linux
ifconfig | grep inet
```

## 🔍 Troubleshooting

### Issue: Connection Timeouts
**Solution**: Ensure the backend server is running and accessible

### Issue: CORS Errors
**Solution**: The backend already has CORS configured for all origins

### Issue: Port Already in Use
**Solution**: Use a different port:
```bash
python api_server.py --host 0.0.0.0 --port 8001
```
Then update `MANUAL_DEFAULT_ANDROID` to use port 8001.

## ✅ Expected Behavior After Backend Starts

Once the backend is running, you should see in the Android logs:
```
LOG  API connection status changed: disconnected -> connecting
LOG  API connection status changed: connecting -> connected
LOG  ✅ Model status data: {"models": {...}}
LOG  ✅ Home page metrics updated with real API data
```

## 🎯 Current Status

**Frontend**: ✅ Fully functional with intelligent fallbacks
**Backend**: ❌ **NEEDS TO BE STARTED**

The app works perfectly with fallback data, but to get real backend integration, you must start the FastAPI server as described above.

## 📋 Quick Start Checklist

1. ✅ Fixed app configuration issues
2. ✅ Fixed route mismatches  
3. ✅ Fixed AsyncStorage compatibility
4. ❌ **START BACKEND SERVER** (Critical)
5. ✅ Test Android app connection
6. ✅ Verify real data loading

**Next Step**: Run `python src/backend/api_server.py --host 0.0.0.0 --port 8000`