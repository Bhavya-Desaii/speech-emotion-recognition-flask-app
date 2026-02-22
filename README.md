# Speech Emotion Recognition

A Flask web application that predicts human emotion from speech audio using a pre-trained TensorFlow model. Supports live browser recording and WAV file uploads.

## Features

- Real-time emotion prediction from microphone recordings
- WAV file upload support
- REST API with health check and app info endpoints
- Structured logging to console and file

## Predicted Emotions

`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad`

---

## Project Structure

```
Speech-Emotion-Recognition/
├── app.py                  # Flask app, routes, and request handling
├── inference.py            # Model loading and emotion prediction
├── feature_extractor.py    # Audio feature extraction (ZCR, RMSE, MFCC)
├── audio_processor.py      # Audio loading, validation, and format conversion
├── logger.py               # Centralized logging configuration
├── config.py               # All paths, parameters, and app settings
├── quick_predict.py        # Command-line utility for single-file prediction
├── requirements.txt
├── model_data/             # See Model Setup below
│   ├── model_fp32/         # TensorFlow SavedModel
│   ├── scaler.pkl          # StandardScaler from training
│   └── Enc_labels.sav      # Label encoder
├── audio_samples/          # Sample audio files for testing
├── static/
│   ├── css/                # Frontend styles
│   └── js/                 # Frontend scripts
├── templates/
│   └── index.html          # Frontend UI
└── logs/
    └── app.log             # Auto-created on first run
```

## File Responsibilities

| File | Responsibility |
|---|---|
| `config.py` | Single source of truth for all paths, audio parameters, and Flask settings |
| `logger.py` | Creates named loggers used across all modules; controls log level and output targets |
| `audio_processor.py` | Converts raw audio bytes to numpy arrays; validates format and duration; handles non-WAV conversion via pydub/ffmpeg |
| `feature_extractor.py` | Extracts ZCR, RMSE, and MFCC features from audio; pads/truncates to target length; applies StandardScaler |
| `inference.py` | Loads TensorFlow SavedModel and label encoder at startup; runs prediction and returns emotion with confidence scores |
| `quick_predict.py` | Command-line utility to run the full inference pipeline on a single audio file |
| `app.py` | Defines Flask routes (`/predict`, `/health`, `/info`); orchestrates the audio → features → prediction pipeline |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

### 2. Create and activate a virtual environment

```bash
python -m venv speechenv

# Windows
speechenv\Scripts\activate

# Linux/macOS
source speechenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download model files

Model files are not included in the repository due to size. Download `model_data.zip` from the [latest release](https://github.com/yourusername/Speech-Emotion-Recognition/releases/latest), extract it, and place it in the project root:

```
Speech-Emotion-Recognition/
├── model_data/
│   ├── model_fp32/
│   ├── scaler.pkl
│   └── Enc_labels.sav
```

### 5. ffmpeg Setup

ffmpeg is required for non-WAV audio conversion.

**Windows** — Download from [ffmpeg.org](https://ffmpeg.org/download.html) and update the paths in `audio_processor.py`:

```python
FFMPEG_PATH = r"C:\path\to\ffmpeg.exe"
FFPROBE_PATH = r"C:\path\to\ffprobe.exe"
```

**Linux/macOS** — Install via package manager and ffmpeg will be picked up automatically:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

---

## Configuration

All settings are in `config.py`. Key parameters:

```python
FLASK_DEBUG = True          # Set False in production
FLASK_PORT = 5000
LOG_LEVEL = 'INFO'          # DEBUG for verbose output
LOAD_MODEL_AT_STARTUP = True
```

---

## Running the App

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

### Testing individual modules

```bash
python feature_extractor.py                   # Test feature extraction pipeline
python audio_processor.py                     # Test audio loading and validation
python inference.py                           # Test model loading and prediction
python quick_predict.py                       # Test full pipeline on sample audio
python quick_predict.py path/to/file.wav      # Test on a specific file
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend UI |
| `POST` | `/predict` | Accepts audio file, returns emotion prediction |
| `GET` | `/health` | Returns model status (`healthy` / `unhealthy`) |
| `GET` | `/info` | Returns app config and model info |

### `/predict` Request

```
POST /predict
Content-Type: multipart/form-data

audio: <WAV file>
```

### `/predict` Response

```json
{
  "success": true,
  "emotion": "happy",
  "confidence": 87.43,
  "probabilities": {
    "angry": 2.1,
    "disgust": 1.3,
    "fear": 0.8,
    "happy": 87.43,
    "neutral": 5.2,
    "sad": 3.17
  },
  "predicted_index": 3
}
```

---

## Logging

Logs are written to both the console and `logs/app.log`. The console level depends on `FLASK_DEBUG`:

- **Debug mode on** — `DEBUG` and above shown in terminal
- **Debug mode off** — `WARNING` and above shown in terminal; `INFO` and above always written to file

To get verbose output without enabling Flask debug mode, set `LOG_LEVEL = 'DEBUG'` in `config.py`.