# QuadFusion Backend Server Startup Guide

## 🚨 CRITICAL: Backend Server Must Be Started

The Android app logs show connection attempts to `http://10.0.2.2:8000` but getting `[AbortError: Aborted]` - this means the backend server is not running.

## 📋 Step-by-Step Server Startup

### Step 1: Open Terminal in Backend Directory
```bash
cd src/backend
```

### Step 2: Install Dependencies (if not done)
```bash
pip install fastapi uvicorn numpy pydantic
```

### Step 3: Start the Server
```bash
python api_server.py --host 0.0.0.0 --port 8000
```

**Alternative method:**
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4: Verify Server is Running
You should see output like:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 5: Test Server Health
Open browser and go to: http://localhost:8000/health

You should see:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-25T07:28:00.000Z",
  "quadfusion_available": true,
  "agents_initialized": 6,
  "active_sessions": 0
}
```

## 🔧 Troubleshooting

### Issue: "Address already in use"
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python api_server.py --host 0.0.0.0 --port 8001
```

### Issue: "Module not found"
```bash
# Install missing dependencies
pip install -r src/requirements.txt
```

### Issue: Python not found
```bash
# Use python3 instead
python3 api_server.py --host 0.0.0.0 --port 8000
```

## ✅ Expected Android App Behavior After Server Starts

Once the server is running, your Android logs should change from:
```
ERROR  API Request failed for /api/v1/models/status: [AbortError: Aborted]
```

To:
```
LOG  API connection status changed: connecting -> connected
LOG  ✅ Real API data received from backend
LOG  ✅ Model status loaded successfully
```

## 🚀 Quick Test Commands

After server starts, test these endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Model status
curl http://localhost:8000/api/v1/models/status

# Train models (optional)
curl -X POST http://localhost:8000/api/v1/admin/train_all
```

## 📱 Android Connection Details

- **Android Emulator**: Uses `10.0.2.2:8000` to reach host machine's `localhost:8000`
- **Physical Device**: May need your computer's IP address instead

The app is correctly configured - it just needs the server to be running!