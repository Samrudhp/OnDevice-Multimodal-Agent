Repository model artifact status and next steps

Summary
-------
- I searched the repository (excluding the `.venv`) and found no committed trained model checkpoint files (e.g. `*.pth`, `*.pt`, `*.joblib`, `saved_model/`, `*.tflite`) in the project tree. That means the demo fell back to stub implementations because the code didn't find expected model artifacts.

Where the code looks for models
-----------------------------
The demo and agents expect model files at or derived from these names/locations (examples found in the code / docs):

- pretrained_models/typing_model.pth
- pretrained_models/visual_model.pth
- pretrained_models/movement_model.pth
- best_typing_model.pth, best_movement_model.pth, best_visual_model.pth (training scripts)
- typing_model.pth, movement_model.pth, visual_model.pth (training outputs)
- touch_model.pkl (touch agent)
- speaker_id.tflite, motion_cnn.tflite (mobile integration examples)
- models/saved_model/ (SavedModel directory used by mobile converters)
- models/quadfusion_model.tflite (mobile model validator)

Why the demo used stubs
-----------------------
- The demo's startup code checks whether model loader methods succeed. If load fails (files missing or invalid), the pipeline prints messages like "QuadFusion components not available - using stubs" and continues using lightweight stub implementations. Since there are no committed checkpoints in this repo, the loaders returned False and the stubs were used.

How to get real models
----------------------
You have two options:

1) Provide pre-trained artifacts
   - Copy your model files into the repository (or into a path the demo expects), e.g.:
     - mkdir -p pretrained_models
     - cp /path/to/movement_model.pth pretrained_models/movement_model.pth
   - For TFLite/ONNX/mobile flows, copy `.tflite` / `.onnx` files into the assets/models or `models/tflite` folders as needed.

2) Train models locally using the included training scripts
   - The repository contains training scripts in `src/training/` (examples: `train_movement_model.py`, `train_visual_model.py`, `train_typing_model.py`). Running those scripts will produce checkpoint files (the scripts save files such as `best_movement_model.pth` or `movement_model.pth`).
   - Example (assumes your virtualenv is active and dependencies installed):
     - python -m src.training.train_movement_model
     - python -m src.training.train_visual_model
     - python -m src.training.train_typing_model
   - If you want tiny/faster runs for a quick test, pass the script arguments to reduce epochs / dataset size (check the script's argparse help).

Notes and tips
--------------
- Model artifacts are usually git-ignored in this project, so they won't be present in the repo by design. Keep large binaries out of git; share them via an artifacts storage (S3, Google Drive, internal server) and copy them into `pretrained_models/` locally.
- If you'd like, I can run a small end-to-end demo training to produce a tiny model (fast, few epochs) and wire it so the demo uses it instead of stubs — tell me which model you want first (movement / visual / typing / speaker).

What I did in this session
--------------------------
- Searched the repo for model artifacts and loader calls.
- Created this `RUNNING.md` with findings and next steps.

Next steps (pick one)
---------------------
- I can run a quick training job to produce a tiny demo model (you'll need to confirm you want that and which model to train).
- I can add a script to download prebuilt demo checkpoints into `pretrained_models/` if you want to host them somewhere.
- I can update the demo to print exact paths it tried to load at runtime to make debugging easier.
