"""
quick_predict.py — Command-line utility for single-file emotion prediction.
Delegates all logic to the proper modules.

Usage:
    python quick_predict.py
    python quick_predict.py path/to/audio.wav
"""

import sys
import os
from logger import setup_logger
import config
import audio_processor
import feature_extractor
import inference

logger = setup_logger(__name__)


def predict_file(audio_path):
    """
    Run the full prediction pipeline on a single audio file.

    Args:
        audio_path: Path to a WAV audio file

    Returns:
        tuple: (emotion, confidence) or (None, None) on failure
    """
    logger.info("Processing: %s", os.path.basename(audio_path))

    # Load audio
    try:
        audio_data, sr = audio_processor.load_audio_from_file(audio_path)
    except Exception as e:
        logger.error("Failed to load audio file: %s", e)
        return None, None

    # Extract features
    features = feature_extractor.extract_features(audio_data, sr)
    if features is None:
        return None, None

    # Prepare features (pad/truncate, scale, reshape)
    prepared = feature_extractor.prepare_features(features)
    if prepared is None:
        return None, None

    # Predict
    result = inference.predict_emotion(prepared)

    if not result['success']:
        logger.error("Prediction failed: %s", result.get('error'))
        return None, None

    logger.info("Emotion: %s | Confidence: %.2f%%", result['emotion'].upper(), result['confidence'])

    # Log full probability breakdown
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for emotion, prob in sorted_probs:
        logger.debug("  %-10s %.2f%%", emotion, prob)

    return result['emotion'], result['confidence']


if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else config.SAMPLE_AUDIO_PATH

    if not os.path.exists(audio_file):
        logger.error("Audio file not found: %s", audio_file)
        sys.exit(1)

    inference.initialize_models()
    predict_file(audio_file)