# OnDevide MultiModal multi-agent System

<div align="center">

**Multi-Modal Behavioral Fraud Detection System**

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React Native](https://img.shields.io/badge/React%20Native-0.79-61DAFB.svg)](https://reactnative.dev/)
[![Expo](https://img.shields.io/badge/Expo-53.0-000020.svg)](https://expo.dev/)

**Real-time fraud detection using behavioral biometrics and AI-powered multi-agent analysis**

[Features](#-features) • [Architecture](#-architecture) • [Getting Started](#-getting-started) • [Demo](#-demo) • [Documentation](#-documentation)

</div>

---

## 🎯 Overview

QuadFusion is an advanced **multi-modal behavioral fraud detection system** that leverages AI and machine learning to identify fraudulent activities through behavioral biometrics. Unlike traditional authentication methods, QuadFusion continuously monitors user behavior patterns across multiple dimensions:

- **Touch Patterns** - Swipe dynamics, tap pressure, gesture recognition
- **Typing Behavior** - Keystroke dynamics, rhythm analysis, timing patterns
- **Voice Authentication** - Speaker identification, voice pattern analysis
- **Visual Biometrics** - Face recognition, scene analysis
- **Motion Analysis** - Accelerometer, gyroscope, magnetometer data
- **App Usage Patterns** - Usage frequency, navigation patterns, temporal analysis

The system uses a **multi-agent architecture** where specialized AI agents analyze different behavioral aspects and a coordinator agent fuses their decisions for robust fraud detection.

---

## ✨ Features

### 🔒 **Multi-Modal Authentication**
- Continuous behavioral biometric monitoring
- Real-time anomaly detection
- Risk scoring with confidence levels
- Session-based fraud analysis

### 🤖 **AI-Powered Multi-Agent System**
- **6 Specialized Agents:**
  - Touch Pattern Agent
  - Typing Behavior Agent
  - Voice Command Agent
  - Visual Agent
  - Movement Agent
  - App Usage Agent
- **Coordinator Agent** for intelligent decision fusion
- Lightweight models optimized for mobile deployment

### 📱 **Mobile-First Design**
- React Native + Expo for cross-platform support
- Real-time sensor data collection
- Live monitoring dashboard
- Beautiful, responsive UI with animations
- Offline-capable with local processing

### 🛡️ **Privacy & Security**
- End-to-end encryption for biometric data
- On-device processing where possible
- Secure data storage and transmission
- GDPR-compliant data handling

### 📊 **Developer Experience**
- RESTful API with comprehensive documentation
- Easy integration with existing apps
- Detailed logging and monitoring
- Performance metrics and analytics

---

## 🏗️ Architecture

### System Components

```
QuadFusion/
├── Backend (Python)          # AI/ML Processing Server
│   ├── API Server           # FastAPI REST endpoints
│   ├── Multi-Agent System   # 6 specialized + 1 coordinator
│   ├── Models               # ML models (LSTM, CNN, etc.)
│   ├── Data Pipeline        # Collection, preprocessing, encryption
│   └── Mobile Deployment    # ONNX/TFLite conversion
│
└── Frontend (React Native)   # Mobile Application
    ├── Sensor Managers      # Data collection
    ├── Live Monitoring      # Real-time dashboard
    ├── UI Components        # Responsive, animated UI
    └── API Client           # Backend communication
```

### Multi-Agent Architecture

```
User Interaction Data
        ↓
┌───────────────────────────────────┐
│   Specialized Agent Layer         │
├───────────────────────────────────┤
│ • TouchPatternAgent    (20%)      │
│ • TypingBehaviorAgent  (15%)      │
│ • VoiceCommandAgent    (20%)      │
│ • VisualAgent          (25%)      │
│ • MovementAgent        (10%)      │
│ • AppUsageAgent        (10%)      │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│   Coordinator Agent               │
│   • Weighted fusion               │
│   • Confidence aggregation        │
│   • Risk level determination      │
└───────────────────────────────────┘
        ↓
   Fraud Decision
```

### Technology Stack

**Backend:**
- Python 3.10+
- FastAPI (REST API)
- TensorFlow & PyTorch (Deep Learning)
- Scikit-learn (ML algorithms)
- ONNX/TFLite (Mobile optimization)
- Librosa (Audio processing)
- OpenCV & MediaPipe (Computer Vision)

**Frontend:**
- React Native 0.79
- Expo 53.0
- TypeScript
- Expo Sensors, Camera, Audio
- Victory Native (Charts)
- React Navigation

---

## 🚀 Getting Started

### Prerequisites

- **Backend:** Python 3.10+, pip
- **Frontend:** Node.js 18+, npm/yarn
- **Mobile:** Expo Go app (for testing) or Expo CLI

### Quick Start

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Samrudhp/OnDevice-Multimodal-Agent.git
cd QuadFusion
```

#### 2️⃣ Backend Setup

```bash
cd src/backend/src

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start API server
cd ..
python api_server.py
```

The backend server will start at \`http://127.0.0.1:8000\`

**API Documentation:** Visit \`http://127.0.0.1:8000/docs\` for interactive API docs

#### 3️⃣ Frontend Setup

```bash
cd src/qf

# Install dependencies
npm install

# Start development server
npm run dev
```

Scan the QR code with Expo Go app to run on your device.

### Configuration

#### Backend Configuration
Edit \`src/backend/src/config.yaml\`:

```yaml
agents:
  coordinator:
    agent_weights:
      TouchPatternAgent: 0.2
      TypingBehaviorAgent: 0.15
      VoiceCommandAgent: 0.2
      VisualAgent: 0.25
      AppUsageAgent: 0.1
      MovementAgent: 0.1
    risk_thresholds:
      low: 0.3
      medium: 0.6
      high: 0.8
```
---

## 📖 Usage

### Running the Demo

```bash
# Terminal 1: Start backend
cd src/backend
python api_server.py

# Terminal 2: Start frontend
cd src/qf
npm run dev
```

### API Examples

#### Real-time Fraud Detection

```bash
curl -X POST http://127.0.0.1:8000/api/v1/process/realtime \\
  -H "Content-Type: application/json" \\
  -d '{
    "session_id": "session-123",
    "sensor_data": {
      "touch_events": [...],
      "keystroke_events": [...],
      "motion_data": {...},
      "audio_data": "base64...",
      "image_data": "base64..."
    }
  }'
```

**Response:**
```json
{
  "anomaly_score": 0.23,
  "risk_level": "low",
  "confidence": 0.87,
  "agent_results": {
    "MovementAgent": {
      "anomaly_score": 0.15,
      "risk_level": "low",
      "confidence": 0.9
    },
    "TouchPatternAgent": {...},
    ...
  }
}
```

---

## 🎬 Demo

### Live Monitoring Dashboard

The mobile app provides real-time visualization of:
- **Sensor data collection** (touch, motion, audio, camera)
- **Agent analysis results** with individual scores
- **Risk assessment** with confidence levels
- **Processing metrics** and performance stats

### Screenshots

*(Add screenshots of your mobile app here)*

---

## 📚 Documentation

- **[API Specification](src/backend/API_SPECIFICATION.md)** - Complete API reference
- **[Backend Setup](src/backend/START_SERVER.md)** - Detailed backend setup guide
- **[Mobile Setup](src/qf/SETUP_AND_TESTING.md)** - Frontend setup and testing
- **[Model Documentation](src/backend/src/models.md)** - ML model details
- **[Architecture Docs](docs/)** - System architecture and design

---

## 🔧 Development

### Project Structure

```
QuadFusion/
├── src/
│   ├── backend/
│   │   ├── api_server.py              # Main API server
│   │   ├── API_SPECIFICATION.md       # API docs
│   │   └── src/
│   │       ├── agents/                # Multi-agent system
│   │       │   ├── coordinator_agent.py
│   │       │   ├── touch_pattern_agent.py
│   │       │   ├── typing_behavior_agent.py
│   │       │   ├── voice_command_agent.py
│   │       │   ├── visual_agent.py
│   │       │   ├── movement_agent.py
│   │       │   └── app_usage_agent.py
│   │       ├── models/                # ML models
│   │       ├── data/                  # Data pipeline
│   │       ├── mobile_deployment/     # Model conversion
│   │       ├── training/              # Model training
│   │       └── utils/                 # Utilities
│   │
│   └── qf/                            # React Native app
│       ├── app/                       # Expo Router pages
│       ├── components/                # UI components
│       ├── lib/                       # Utilities
│       │   ├── sensor-manager.ts      # Sensor data collection
│       │   ├── api.ts                 # API client
│       │   └── audio-recorder.ts      # Audio recording
│       └── config/                    # Configuration
│
├── docs/                              # Documentation
└── README.md                          # This file
```



## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (\`git checkout -b feature/AmazingFeature\`)
3. Commit your changes (\`git commit -m 'Add some AmazingFeature'\`)
4. Push to the branch (\`git push origin feature/AmazingFeature\`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built for Samsung EnnovateX 2025 AI Challenge
- TensorFlow and PyTorch communities
- Expo and React Native teams
- Open-source ML model contributors

---

## 📧 Contact

**Project Repository:** [https://github.com/Samrudhp/OnDevice-Multimodal-Agent](https://github.com/Samrudhp/OnDevice-Multimodal-Agent)


---

<div align="center">

**Built with ❤️ using AI and Multi-Agent Systems**

*Protecting users through behavioral intelligence*
