# QuadFusion Setup Guide 🚀

**Complete setup instructions for the QuadFusion fraud detection system from scratch**

---

## 📋 **Table of Contents**

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [System Configuration](#system-configuration)
7. [Running the System](#running-the-system)
8. [Testing & Validation](#testing--validation)
9. [Mobile Deployment](#mobile-deployment)
10. [Troubleshooting](#troubleshooting)

---

## 🔧 **Prerequisites**

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux Ubuntu 18.04+
- **Python**: 3.8 or higher (3.9 recommended)
- **RAM**: Minimum 8GB (16GB recommended for training)
- **Storage**: At least 5GB free space
- **GPU**: Optional but recommended for faster training (CUDA compatible)

### Software Dependencies
- Git (for version control)
- Python package manager (pip)
- Virtual environment tool (venv or conda)

---

## 🌟 **Environment Setup**

### Step 1: Create Python Virtual Environment

**Option A: Using venv (Recommended)**
```bash
# Navigate to project directory
cd e:\Sagar\Hackathons\EnnovateX\QuadFusion

# Create virtual environment
python -m venv quadfusion_env

# Activate virtual environment
# Windows:
quadfusion_env\Scripts\activate
# macOS/Linux:
source quadfusion_env/bin/activate
```

**Option B: Using conda**
```bash
# Create conda environment
conda create -n quadfusion python=3.9
conda activate quadfusion
```

### Step 2: Verify Python Installation
```bash
python --version
# Should show Python 3.8+ 
```

---

## 📦 **Installation**

### Step 1: Install Core Dependencies
```bash
# Navigate to src directory
cd src

# Install all required packages
pip install -r requirements.txt
```

### Step 2: Install Additional Mobile Dependencies (Optional)
```bash
# For TensorFlow Lite conversion
pip install tensorflow-lite-converter

# For ONNX conversion
pip install onnxruntime onnxruntime-tools

# For PyTorch Mobile
pip install torch torchvision torchaudio
```

### Step 3: Verify Installation
```bash
# Test imports
python -c "import numpy, sklearn, torch, tensorflow; print('✅ All core libraries installed successfully')"
```

---

## 📊 **Data Preparation**

### Step 1: Create Data Directories
```bash
# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/synthetic
mkdir -p pretrained_models
mkdir -p mobile_models
```

### Step 2: Generate Synthetic Training Data
```bash
# Run data generation script
python demo/simulate_sensor_data.py
```

**This will create:**
- `data/synthetic/touch_patterns.csv` - Touch interaction data
- `data/synthetic/typing_patterns.csv` - Keystroke timing data  
- `data/synthetic/voice_features.csv` - Audio feature data
- `data/synthetic/visual_embeddings.npy` - Face embedding data
- `data/synthetic/movement_data.csv` - Accelerometer/gyroscope data
- `data/synthetic/app_usage.json` - App usage patterns

### Step 3: Prepare Real Data (Optional)
If you have real data, place it in the `data/raw/` directory:

**Touch Data Format:**
```csv
timestamp,x,y,pressure,user_id,is_fraud
1629123456,100.5,200.3,0.8,user1,0
```

**Typing Data Format:**
```csv
user_id,keystroke_sequence,dwell_times,flight_times,is_fraud
user1,"hello",[100,150,120,80,90],[50,30,40,35],0
```

**Voice Data Format:**
- Audio files (.wav) in `data/raw/voice/`
- Metadata CSV with user_id, file_path, is_fraud

---

## 🏋️ **Model Training**

### Option A: Use Pretrained Models (Recommended - Faster Setup) 🚀

**QuadFusion supports leveraging existing pretrained models for faster deployment:**

#### Step 1: Download Pretrained Base Models

**Face Recognition (Visual Agent):**
```bash
# Download pretrained face recognition model
python -c "
import face_recognition
import pickle
import os
os.makedirs('pretrained_models', exist_ok=True)
print('✅ Face recognition models downloaded automatically')
"
```

**Voice/Speaker Recognition:**
```bash
# Download pretrained speaker verification model
python -c "
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir='pretrained_models/speaker_verification'
)
print('✅ Speaker verification model downloaded')
"
```

**Movement Pattern Analysis:**
```bash
# Download pretrained activity recognition model
python -c "
import timm
model = timm.create_model('mobilenetv3_small_100', pretrained=True)
import torch
torch.save(model.state_dict(), 'pretrained_models/movement_base.pth')
print('✅ Movement analysis base model downloaded')
"
```

**Typing Pattern Analysis:**
```bash
# Download pretrained language model for keystroke analysis
python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('microsoft/DialoGPT-small')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model.save_pretrained('pretrained_models/typing_base')
tokenizer.save_pretrained('pretrained_models/typing_base')
print('✅ Typing analysis base model downloaded')
"
```

#### Step 2: Fine-tune with Fraud Detection Data
```bash
# Fine-tune pretrained models for fraud detection
python training/finetune_pretrained_models.py
```

**Advantages of Pretrained Models:**
- ✅ **Faster Setup**: No training from scratch required
- ✅ **Better Performance**: Pretrained on large datasets
- ✅ **Lower Resource Requirements**: Less computational power needed
- ✅ **Proven Accuracy**: State-of-the-art base models
- ✅ **Transfer Learning**: Adapts to fraud detection quickly

#### Step 3: Validate Pretrained Models
```bash
python -c "
from demo.demo_pipeline import DemoPipeline
pipeline = DemoPipeline(use_pretrained=True)
pipeline.validate_pretrained_models()
print('✅ All pretrained models validated')
"
```

---

### Option B: Train From Scratch (Advanced Users) 🔬

**For custom requirements or research purposes:**
Edit `config.yaml` to adjust training parameters:
```yaml
# Training Configuration
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  validation_split: 0.2
  mobile_optimization: true
```

### Step 2: Train Individual Models

**Train Touch Pattern Model:**
```bash
python training/train_touch_model.py
```
**Output:** `pretrained_models/touch_model.pkl` (≈8MB)

**Train Typing Behavior Model:**
```bash
python training/train_typing_model.py
```
**Output:** `pretrained_models/typing_model.pth` (≈5MB)

**Train Voice Authentication Model:**
```bash
python training/train_voice_model.py
```
**Output:** `pretrained_models/voice_model.pkl` (≈12MB)

**Train Visual Recognition Model:**
```bash
python training/train_visual_model.py
```
**Output:** `pretrained_models/visual_model.pth` (≈15MB)

**Train Movement Pattern Model:**
```bash
python training/train_movement_model.py
```
**Output:** `pretrained_models/movement_model.pth` (≈8MB)

### Step 3: Model Quantization for Mobile
```bash
python training/model_quantization.py
```
**Output:** Optimized models in `mobile_models/` (each <10MB)

---

## ⚙️ **System Configuration**

### Step 1: Configure Agent Settings
Edit `config.yaml` to enable/disable agents and set thresholds:

```yaml
agents:
  touch_pattern:
    enabled: true
    contamination: 0.1        # Anomaly threshold
    n_estimators: 50          # Forest size
  
  typing_behavior:
    enabled: true
    sequence_length: 20       # Keystroke sequence length
    lstm_units: 32           # Model complexity
    threshold: 0.1           # Anomaly threshold
  
  voice_command:
    enabled: true
    sample_rate: 16000       # Audio sampling rate
    n_mfcc: 13              # Feature count
    speaker_threshold: 0.7   # Authentication threshold
  
  visual:
    enabled: true
    image_size: 224          # Input image resolution
    similarity_threshold: 0.7 # Face matching threshold
  
  movement:
    enabled: true
    sequence_length: 100     # Sensor data window
    sampling_rate: 50        # Hz
```

### Step 2: Configure Security Settings
```yaml
security:
  encryption_key_path: "keys/master.key"
  enable_encryption: true
  differential_privacy: true
  privacy_budget: 1.0
```

### Step 3: Configure Performance Settings
```yaml
performance:
  max_inference_time_ms: 100
  max_memory_usage_mb: 500
  battery_optimization: true
  cpu_threads: 2
```

---

## 🚀 **Running the System**

### Step 1: Initialize System
```bash
# Initialize database and encryption
python -c "
from data.database_manager import DatabaseManager
from data.encryption import EncryptionManager
db = DatabaseManager()
db.initialize_database()
enc = EncryptionManager()
enc.generate_master_key()
print('✅ System initialized successfully')
"
```

### Step 2: Run Complete Demo Pipeline
```bash
python demo/demo_pipeline.py
```

**This will:**
- Load all trained models
- Initialize all agents
- Process demo data
- Show fraud detection results
- Display performance metrics

### Step 3: Run Individual Agent Tests
```bash
# Test touch pattern detection
python -c "
from agents.touch_pattern_agent import TouchPatternAgent
agent = TouchPatternAgent()
agent.load_model('pretrained_models/touch_model.pkl')
print('✅ Touch agent ready')
"

# Test typing behavior detection  
python -c "
from agents.typing_behavior_agent import TypingBehaviorAgent
agent = TypingBehaviorAgent()
agent.load_model('pretrained_models/typing_model.pth')
print('✅ Typing agent ready')
"
```

### Step 4: Run Coordinator System
```bash
python -c "
from agents.coordinator_agent import CoordinatorAgent
coordinator = CoordinatorAgent()
coordinator.initialize_agents()
print('✅ Multi-agent system ready')
"
```

---

## 🧪 **Testing & Validation**

### Step 1: Run Unit Tests
```bash
# Test all components
python -m pytest tests/ -v

# Test specific modules
python -m pytest tests/test_models.py -v
python -m pytest tests/test_agents.py -v
python -m pytest tests/test_mobile_deployment.py -v
```

### Step 2: Performance Benchmarks
```bash
# Run performance tests
python tests/test_models.py
```

**Expected Results:**
- ✅ Inference time: <100ms per model
- ✅ Memory usage: <500MB total
- ✅ Model size: <50MB total
- ✅ Accuracy: >95% on test data

### Step 3: Mobile Optimization Validation
```bash
# Test mobile model conversion
python mobile_deployment/model_validator.py
```

**This validates:**
- ONNX conversion success
- TensorFlow Lite compatibility
- PyTorch Mobile optimization
- Model size requirements
- Inference speed requirements

---

## 📱 **Mobile Deployment**

### Step 1: Convert Models for Mobile

**Convert to ONNX:**
```bash
python mobile_deployment/onnx_converter.py
```

**Convert to TensorFlow Lite:**
```bash
python mobile_deployment/tflite_converter.py
```

**Convert to PyTorch Mobile:**
```bash
python mobile_deployment/pytorch_mobile_converter.py
```

### Step 2: Mobile Integration Example
```bash
python demo/mobile_integration_example.py
```

**Output Files:**
- `mobile_models/touch_model.onnx`
- `mobile_models/typing_model.tflite`
- `mobile_models/voice_model.ptl`
- `mobile_models/visual_model.onnx`
- `mobile_models/movement_model.tflite`

### Step 3: Deploy to Mobile Platform

**For Android:**
1. Copy `.tflite` and `.onnx` files to `assets/models/`
2. Use TensorFlow Lite Java API or ONNX Runtime Android
3. Implement real-time sensor data collection

**For iOS:**
1. Copy `.onnx` and `.ptl` files to app bundle
2. Use ONNX Runtime iOS or PyTorch Mobile
3. Implement Core Motion and Touch data collection

---

## 🔍 **System Usage Examples**

### Real-time Fraud Detection
```python
from agents.coordinator_agent import CoordinatorAgent

# Initialize system
coordinator = CoordinatorAgent()
coordinator.load_all_models()

# Process user interaction
touch_data = [(100, 200, 0.8), (105, 205, 0.7)]  # x, y, pressure
typing_data = "hello world"
voice_data = np.array([...])  # Audio features

# Get fraud assessment
result = coordinator.assess_user_authenticity({
    'touch': touch_data,
    'typing': typing_data,
    'voice': voice_data
})

print(f"Fraud Probability: {result['fraud_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

### Batch Processing
```python
from demo.demo_pipeline import DemoPipeline

# Process multiple users
pipeline = DemoPipeline()
results = pipeline.process_batch_data('data/test_batch.csv')
pipeline.generate_report(results)
```

---

## 🚨 **Troubleshooting**

### Common Issues & Solutions

**1. Import Errors**
```
Error: ModuleNotFoundError: No module named 'torch'
Solution: pip install torch torchvision torchaudio
```

**2. Model Loading Errors**
```
Error: FileNotFoundError: pretrained_models/touch_model.pkl
Solution: Run training scripts first: python training/train_touch_model.py
```

**3. Memory Issues**
```
Error: CUDA out of memory
Solution: Reduce batch_size in config.yaml or use CPU: device='cpu'
```

**4. Performance Issues**
```
Error: Inference time >100ms
Solution: Run model quantization: python training/model_quantization.py
```

**5. Data Format Errors**
```
Error: Invalid data format
Solution: Check data format requirements in data_preparation section
```

### Debug Mode
```bash
# Enable debug logging
python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['system']['debug'] = True
config['system']['log_level'] = 'DEBUG'
with open('config.yaml', 'w') as f:
    yaml.dump(config, f)
print('✅ Debug mode enabled')
"
```

### Performance Monitoring
```bash
# Monitor system resources
python -c "
from utils.metrics import PerformanceMonitor
monitor = PerformanceMonitor()
monitor.start_monitoring()
# Run your code here
monitor.stop_monitoring()
monitor.print_report()
"
```

---

## 📈 **Expected System Performance**

### Mobile Optimization Targets
- **Inference Time**: <100ms total (all 5 models combined)
- **Model Size**: <50MB total (compressed models)
- **Memory Usage**: <500MB RAM peak
- **Battery Impact**: <2% per hour of continuous use
- **Accuracy**: >95% fraud detection with <5% false positives

### Throughput Benchmarks
- **Touch Events**: 1000+ events/second processing
- **Keystroke Analysis**: Real-time typing pattern analysis
- **Voice Processing**: 16kHz audio stream analysis
- **Face Recognition**: 30 FPS video stream processing
- **Motion Analysis**: 50Hz sensor data processing

---

## 🎯 **Next Steps**

1. **Production Deployment**
   - Implement API endpoints for remote access
   - Add database scaling for high-volume data
   - Implement distributed processing

2. **Enhanced Security**
   - Add hardware-based key storage
   - Implement secure enclaves for model execution
   - Add audit logging and compliance features

3. **Model Improvements**
   - Implement federated learning for privacy-preserving updates
   - Add continuous learning capabilities
   - Implement adversarial attack detection

4. **Platform Integration**
   - Develop native mobile SDKs
   - Create web browser extensions
   - Add IoT device support

---

## 📞 **Support**

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review log files in `logs/` directory
3. Run diagnostic tests: `python tests/test_models.py`
4. Enable debug mode for detailed logging

---

**🎉 Congratulations! Your QuadFusion fraud detection system is now ready for action!**

The system provides real-time, privacy-preserving fraud detection across multiple behavioral modalities with mobile-optimized performance. All models run locally on-device without cloud communication, ensuring maximum privacy and security.
