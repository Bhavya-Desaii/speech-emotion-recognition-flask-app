import os
import io

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base directory (where app.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model and preprocessor paths
MODEL_PATH = os.path.join(BASE_DIR, "model_data/model_fp32")
SCALER_PATH = os.path.join(BASE_DIR, "model_data/scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "model_data/Enc_labels.sav")
SAMPLE_AUDIO_PATH = os.path.join(BASE_DIR, "audio_samples/1001_DFA_ANG_XX.wav")

FFMPEG_PATH = r"C:\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"  # ← UPDATE THIS
FFPROBE_PATH = r"C:\ffmpeg-8.0.1-full_build\bin\ffprobe.exe"  # ← UPDATE THIS


# ============================================================================
# AUDIO PARAMETERS (MUST MATCH TRAINING!)
# ============================================================================

# Audio loading parameters
DURATION = 2.5              # seconds - audio duration to load
OFFSET = 0.6                # seconds - offset from start
# Note: Sample rate is NOT specified - librosa uses default 22050 Hz

# Recording parameters for web app
RECORDING_DURATION = 3      # seconds - how long user records


# ============================================================================
# FILE UPLOAD SETTINGS (NEW)
# ============================================================================

# Allowed audio file extensions (for upload feature)
ALLOWED_AUDIO_EXTENSIONS = ['.wav']

# Maximum upload file size (5 MB for demo)
MAX_UPLOAD_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


# ============================================================================
# FEATURE EXTRACTION PARAMETERS (MUST MATCH TRAINING!)
# ============================================================================

# Frame parameters for MFCC, ZCR and RMSE
FRAME_LENGTH = 2048
HOP_LENGTH = 512
TARGET_FEATURES = 2376      # Total features after padding/truncation


# ============================================================================
# EMOTION LABELS
# ============================================================================

# Default emotion labels (fallback if encoder fails)
DEFAULT_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']


# ============================================================================
# FLASK SETTINGS
# ============================================================================

# Server settings
FLASK_HOST = '0.0.0.0'      # Accessible from network (use 'localhost' for local only)
FLASK_PORT = 5000           # Port number
FLASK_DEBUG = True          # Debug mode (set False in production)

# Upload settings
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max file size


# ============================================================================
# LOGGING SETTINGS (Optional)
# ============================================================================

LOG_LEVEL = 'INFO'                                  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = os.path.join(BASE_DIR, "logs/app.log")


# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

LOAD_MODEL_AT_STARTUP = True    # Load model when app starts (recommended)