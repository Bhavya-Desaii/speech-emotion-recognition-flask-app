import config
import numpy as np
import librosa
import pickle
from logger import setup_logger

logger = setup_logger(__name__)


def load_scaler(scaler_path=config.SCALER_PATH):
    """
    Load the saved StandardScaler pickle file.
    This is the BEST option - uses the exact scaler from training!
    """
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded from %s", scaler_path)
        return scaler

    except FileNotFoundError:
        logger.warning("Scaler file not found at %s — predictions will run unscaled", scaler_path)
        return None
    except Exception as e:
        logger.warning("Could not load scaler: %s", e)
        return None


def zcr(data, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH):
    """Zero Crossing Rate - EXACTLY as in training"""
    zcr_feat = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr_feat)


def rmse(data, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH):
    """RMSE - EXACTLY as in training"""
    rmse_feat = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse_feat)


def mfcc(data, sr, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH, flatten=True):
    """
    MFCC - EXACTLY as in training
    CRITICAL: Does NOT pass n_fft or hop_length to librosa.feature.mfcc!
    Uses librosa's default parameters!
    """
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr)  # Uses defaults!
    return np.ravel(mfcc_feature.T) if flatten else np.squeeze(mfcc_feature.T)


def extract_features(data, sr, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH):
    """
    Extract features EXACTLY as in training
    Combines: ZCR + RMSE + MFCC
    """
    try:
        result = np.array([])
        result = np.hstack((result,
                            zcr(data, frame_length, hop_length),
                            rmse(data, frame_length, hop_length),
                            mfcc(data, sr, frame_length, hop_length)
                            ))
        logger.debug("Extracted raw features, shape: %s", result.shape)
        return result
    except Exception as e:
        logger.error("Feature extraction failed: %s", e, exc_info=True)
        return None


# Load once at module level
_SCALER = load_scaler(config.SCALER_PATH)


def prepare_features(features, scaler=None):
    if features is None:
        logger.error("prepare_features received None — feature extraction likely failed upstream")
        return None

    original_len = len(features)
    if len(features) < config.TARGET_FEATURES:
        features = np.pad(features, (0, config.TARGET_FEATURES - len(features)), mode='constant')
        logger.warning("Features padded from %d to %d", original_len, config.TARGET_FEATURES)
    elif len(features) > config.TARGET_FEATURES:
        features = features[:config.TARGET_FEATURES]
        logger.warning("Features truncated from %d to %d", original_len, config.TARGET_FEATURES)

    if scaler is None:
        scaler = _SCALER
    if scaler is not None:
        features_scaled = scaler.transform(features.reshape(1, -1))
        features = features_scaled.flatten()
        logger.debug("Scaler applied successfully")
    else:
        logger.warning("No scaler available — features are unscaled")

    features = features.reshape(config.TARGET_FEATURES, 1)
    logger.debug("prepare_features output shape: %s", features.shape)
    return features


if __name__ == "__main__":
    data, sr = librosa.load(config.SAMPLE_AUDIO_PATH, duration=config.DURATION, offset=config.OFFSET)
    logger.info("Loaded sample audio, sr=%d", sr)

    features = extract_features(data, sr, config.FRAME_LENGTH, config.HOP_LENGTH)
    if features is not None:
        logger.info("Raw features extracted, shape: %s", features.shape)

    features = prepare_features(features)
    if features is not None:
        logger.info("Pipeline complete, final shape: %s", features.shape)
    else:
        logger.error("Pipeline failed — no features produced")