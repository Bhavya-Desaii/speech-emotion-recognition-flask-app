# inference.py

import tensorflow as tf
import numpy as np
import pickle
import config
import sys
from logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# MODULE-LEVEL VARIABLES (Loaded once at startup)
# ============================================================================

model = None
infer = None
emotion_labels = None
_initialized = False


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_model():
    """
    Load TensorFlow SavedModel.

    Returns:
        tuple: (model, inference_function)

    Raises:
        Exception: If model loading fails
    """
    try:
        logger.debug("Loading model from: %s", config.MODEL_PATH)
        loaded_model = tf.saved_model.load(config.MODEL_PATH)

        # Get the inference signature
        inference_fn = loaded_model.signatures["serving_default"]

        logger.info("Model loaded successfully from %s", config.MODEL_PATH)
        return loaded_model, inference_fn

    except FileNotFoundError:
        logger.error("Model not found at: %s", config.MODEL_PATH)
        raise FileNotFoundError(f"Model not found at: {config.MODEL_PATH}")

    except Exception as e:
        logger.error("Failed to load model: %s", e, exc_info=True)
        raise Exception(f"Failed to load model: {e}")


def load_label_encoder():
    """
    Load label encoder and extract emotion classes.

    Returns:
        list: Emotion label names

    Raises:
        Exception: If encoder loading fails
    """
    try:
        logger.debug("Loading label encoder from: %s", config.LABEL_ENCODER_PATH)

        with open(config.LABEL_ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)

        labels = encoder.classes_.tolist()
        logger.info("Label encoder loaded. Emotions: %s", labels)
        return labels

    except FileNotFoundError:
        logger.warning(
            "Label encoder not found at %s — falling back to default emotions: %s",
            config.LABEL_ENCODER_PATH, config.DEFAULT_EMOTIONS
        )
        return config.DEFAULT_EMOTIONS

    except Exception as e:
        logger.warning(
            "Failed to load label encoder: %s — falling back to default emotions: %s",
            e, config.DEFAULT_EMOTIONS
        )
        return config.DEFAULT_EMOTIONS


def initialize_models():
    """
    Initialize all models at startup.
    This should be called ONCE when Flask app starts.

    Returns:
        bool: True if successful, False otherwise
    """
    global model, infer, emotion_labels, _initialized

    logger.info("Initializing emotion recognition models...")

    try:
        model, infer = load_model()
        emotion_labels = load_label_encoder()
        _initialized = True
        logger.info("All models initialized successfully")
        return True

    except Exception as e:
        logger.critical(
            "Model initialization failed: %s. Check that model exists at '%s' "
            "and label encoder exists at '%s'",
            e, config.MODEL_PATH, config.LABEL_ENCODER_PATH,
            exc_info=True
        )
        _initialized = False
        return False


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_emotion(features):
    """
    Predict emotion from prepared features.

    Args:
        features: numpy array of shape (2376, 1) - prepared features

    Returns:
        dict: {
            'success': bool,
            'emotion': str,
            'confidence': float,
            'probabilities': dict,
            'predicted_index': int
        }
        or dict with 'success': False and 'error' message if failed
    """
    if not _initialized:
        logger.error("predict_emotion called before models were initialized")
        return {
            'success': False,
            'error': 'Models not initialized. Call initialize_models() first.'
        }

    try:
        # Validate input shape
        if features.shape != (config.TARGET_FEATURES, 1):
            logger.error("Invalid feature shape: %s. Expected (%d, 1)", features.shape, config.TARGET_FEATURES)
            return {
                'success': False,
                'error': f'Invalid feature shape: {features.shape}. Expected ({config.TARGET_FEATURES}, 1)'
            }

        # Add batch dimension: (2376, 1) -> (1, 2376, 1)
        input_tensor = tf.convert_to_tensor(features[np.newaxis, :, :], dtype=tf.float32)
        logger.debug("Input tensor shape: %s", input_tensor.shape)

        # Run inference
        prediction = infer(input_tensor)

        # Extract probabilities from output dictionary
        output_key = list(prediction.keys())[0]
        probabilities = prediction[output_key].numpy()[0]

        # Get predicted emotion
        predicted_idx = np.argmax(probabilities)
        predicted_emotion = emotion_labels[predicted_idx]
        confidence = float(probabilities[predicted_idx] * 100)

        # Create probability dictionary
        prob_dict = {
            emotion: float(prob * 100)
            for emotion, prob in zip(emotion_labels, probabilities)
        }

        logger.debug("Prediction: %s (%.2f%% confidence)", predicted_emotion, confidence)

        return {
            'success': True,
            'emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': prob_dict,
            'predicted_index': int(predicted_idx)
        }

    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        return {
            'success': False,
            'error': f'Prediction failed: {e}'
        }


def get_model_info():
    """
    Get information about loaded models.

    Returns:
        dict: Model information
    """
    if not _initialized:
        logger.warning("get_model_info called but models are not initialized")
        return {
            'initialized': False,
            'error': 'Models not initialized'
        }

    return {
        'initialized': True,
        'model_path': config.MODEL_PATH,
        'label_encoder_path': config.LABEL_ENCODER_PATH,
        'emotion_labels': emotion_labels,
        'num_emotions': len(emotion_labels),
        'expected_feature_shape': (config.TARGET_FEATURES, 1)
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logger.info("--- Inference Module Test Start ---")

    # Test 1: Initialize models
    logger.info("TEST 1: Model Initialization")
    success = initialize_models()

    if not success:
        logger.error("Model initialization failed. Cannot proceed with tests.")
        sys.exit(1)

    # Test 2: Get model info
    logger.info("TEST 2: Model Information")
    info = get_model_info()
    logger.info("Initialized: %s | Emotions: %s | Expected shape: %s",
                info['initialized'], info['emotion_labels'], info['expected_feature_shape'])

    # Test 3: Prediction with real audio file
    logger.info("TEST 3: Prediction with Real Audio")

    try:
        import feature_extractor
        import audio_processor
        import os

        test_audio_file = "audio_samples/1001_DFA_ANG_XX.wav"

        if os.path.exists(test_audio_file):
            logger.info("Loading audio: %s", test_audio_file)

            audio_data, sr = audio_processor.load_audio_from_file(test_audio_file)
            logger.info("Audio loaded: %d samples at %d Hz", len(audio_data), sr)

            features = feature_extractor.extract_features(audio_data, sr)
            logger.info("Features extracted: %d features", len(features))

            prepared_features = feature_extractor.prepare_features(features)
            logger.info("Features prepared, shape: %s", prepared_features.shape)

            result = predict_emotion(prepared_features)

            if result['success']:
                logger.info("Predicted emotion: %s (%.2f%% confidence)",
                            result['emotion'].upper(), result['confidence'])

                # Check against filename label (CREMA-D format)
                label_map = {'_ANG_': 'angry', '_SAD_': 'sad', '_HAP_': 'happy',
                             '_DIS_': 'disgust', '_FEA_': 'fear', '_NEU_': 'neutral'}
                expected = next((v for k, v in label_map.items() if k in test_audio_file), None)

                if expected:
                    if result['emotion'] == expected:
                        logger.info("Prediction matches filename label '%s'", expected)
                    else:
                        logger.warning("Prediction '%s' differs from filename label '%s'",
                                       result['emotion'], expected)

                sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
                for emotion, prob in sorted_probs:
                    logger.debug("  %-10s %.2f%%", emotion, prob)
            else:
                logger.error("Prediction failed: %s", result['error'])
        else:
            logger.warning("Test audio file not found at %s — skipping real audio test", test_audio_file)

    except ImportError as e:
        logger.warning("Cannot import required modules: %s — skipping real audio test", e)
    except Exception as e:
        logger.error("Real audio test failed: %s", e, exc_info=True)

    logger.info("--- Inference Module Test Complete ---")