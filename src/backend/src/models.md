# Model mappings for QuadFusion

This file lists, per agent/modality, the concrete model type the code uses (or intends to use), the framework, and the expected save/load filenames the agent runtime looks for.

Format: Agent (file) — Model type — Framework — Expected saved filename(s) / notes

- Typing (TypingBehaviorAgent - `src/agents/typing_behavior_agent.py`)
  - Model: LSTM autoencoder for keystroke-dynamics anomaly detection
  - Framework: TensorFlow / Keras
  - Files: `{filepath}_autoencoder.h5` and `{filepath}_components.joblib` (joblib contains scaler, thresholds, baseline metadata)

- Touch (TouchPatternAgent - `src/agents/touch_pattern_agent.py`)
  - Model: IsolationForest (unsupervised anomaly detection)
  - Framework: scikit-learn
  - Files: `joblib` dump at the given `filepath` (contains isolation_forest, scaler, baseline_data)

- Movement (MovementAgent - `src/agents/movement_agent.py`)
  - Model: 1D CNN autoencoder + statistical baseline patterns
  - Framework: TensorFlow / Keras
  - Files: `{filepath}_cnn.h5` (Keras model) and `{filepath}_data.json` (baseline patterns, scaler stats, config)

- Voice / Audio (VoiceCommandAgent - `src/agents/voice_command_agent.py`)
  - STT: placeholder for Tiny-Whisper / Tiny Whisper (production integration)
  - Speaker ID: MFCC features + SVM (scikit-learn)
  - Frameworks: `librosa` for features, scikit-learn for classifier; (optional) Whisper for STT
  - Files: `joblib` dump at the given `filepath` (stores `speaker_classifier`, `mfcc_scaler`, enrolled embeddings, speaker_id)

- Visual (VisualAgent - `src/agents/visual_agent.py`)
  - Intended model: CLIP-Tiny (PyTorch) for face/scene embeddings
  - Current implementation: simplified OpenCV + handcrafted embedding generator (no CLIP inference in the shipped code)
  - Frameworks: intended — PyTorch/CLIP; current — OpenCV + numpy (torch is imported but CLIP is not loaded)
  - Files (prod): stored embeddings (joblib/npz) or serialized CLIP model; current code keeps embeddings in memory

- App usage (AppUsageAgent - `src/agents/app_usage_agent.py`)
  - Model: Statistical baselines / z-score analysis (no heavy ML)
  - Framework: numpy / plain Python
  - Files: JSON state file via `save_model(filepath)` (contains `usage_stats`, `hourly_stats`, etc.)

- Coordinator / Fusion (CoordinatorAgent - `src/agents/coordinator_agent.py`)
  - Model: Weighted rule-based fusion; optional quantized tiny LLM for fusion (placeholder)
  - Frameworks: rule-based Python; `src/models/tiny_llm.py` exists for tiny-LLM experiments
  - Files: JSON state file via `save_model(filepath)` (saves `agent_weights`, `decision_history`)

Notes & quick pointers
- Training scripts live in `src/training/` and instantiate the loaders from `src/training/dataset_loaders.py`. Typical training scripts: `train_movement_model.py`, `train_visual_model.py`, `train_typing_model.py`, `train_voice_model.py`, `train_touch_model.py`.
- The repository provides a `SyntheticDataGenerator` in `src/training/dataset_loaders.py` with `generate_*` methods. Use it to produce small datasets for quick local training and to generate example checkpoints.
- Where to place checkpoints: create a `pretrained_models/` (or similar) directory at the repo root and place files with the `filepath` base the agents expect (for example `pretrained_models/typing_autoencoder` would yield `pretrained_models/typing_autoencoder_autoencoder.h5` and `pretrained_models/typing_autoencoder_components.joblib` if you call agents' `load_model('pretrained_models/typing_autoencoder')`).

If you want, I can generate short synthetic-data training runs and write example checkpoints into `pretrained_models/` so the demo uses the real models instead of stubs — tell me to `generate synthetic checkpoints` and I will run the short training jobs.
