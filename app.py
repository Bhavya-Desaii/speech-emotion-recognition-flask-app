# app.py

from flask import Flask, render_template, request, jsonify
import io
import os
import config
import audio_processor
import feature_extractor
import inference
from logger import setup_logger

logger = setup_logger(__name__)

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """
    Check if file has allowed extension.

    Args:
        filename: Name of the file

    Returns:
        bool: True if allowed, False otherwise
    """
    if '.' not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in config.ALLOWED_AUDIO_EXTENSIONS


# ============================================================================
# STARTUP - LOAD MODELS
# ============================================================================

logger.info("Starting Flask app — Speech Emotion Recognition")

if config.LOAD_MODEL_AT_STARTUP:
    logger.info("Loading models at startup...")
    success = inference.initialize_models()

    if not success:
        # WARNING not CRITICAL here: app still starts and serves routes,
        # but every prediction will fail until models are fixed
        logger.warning(
            "Models failed to load at startup. App will start but predictions will fail. "
            "Check model paths: model='%s', encoder='%s'",
            config.MODEL_PATH, config.LABEL_ENCODER_PATH
        )
else:
    logger.warning("Model loading at startup is disabled in config — models will not be loaded")


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    Receives audio file from frontend, processes it, and returns emotion prediction.

    Expected request:
        - Form data with 'audio' file

    Returns:
        JSON: {
            'success': bool,
            'emotion': str,
            'confidence': float,
            'probabilities': dict,
            'predicted_index': int
        }
        or error response
    """
    try:
        # Check if audio file is in request
        if 'audio' not in request.files:
            logger.warning("Prediction request received with no audio file")
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']

        if audio_file.filename == '':
            logger.warning("Prediction request received with empty filename")
            return jsonify({'success': False, 'error': 'Empty audio file'}), 400

        if not allowed_file(audio_file.filename):
            logger.warning("Rejected file with disallowed extension: %s", audio_file.filename)
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Only {", ".join(config.ALLOWED_AUDIO_EXTENSIONS)} files are allowed.'
            }), 400

        logger.info("Prediction request — file: '%s', content-type: %s",
                    audio_file.filename, audio_file.content_type)

        audio_bytes = audio_file.read()
        logger.debug("Audio bytes read: %d bytes", len(audio_bytes))

        if len(audio_bytes) > config.MAX_UPLOAD_FILE_SIZE:
            logger.warning("Rejected oversized upload: %d bytes (limit: %d bytes)",
                           len(audio_bytes), config.MAX_UPLOAD_FILE_SIZE)
            return jsonify({
                'success': False,
                'error': f'File too large. Maximum size is {config.MAX_UPLOAD_FILE_SIZE // (1024*1024)} MB.'
            }), 400

        # Step 1: Process audio (bytes → numpy array)
        logger.debug("Step 1: Audio processing")
        audio_data, sr, audio_info = audio_processor.process_audio_for_inference(audio_bytes)

        if audio_data is None:
            error_msg = audio_info.get('error', 'Audio processing failed')
            logger.warning("Audio processing failed: %s", error_msg)
            return jsonify({'success': False, 'error': error_msg}), 400

        logger.debug("Step 1 complete — sr: %d Hz, duration: %.2fs, samples: %d",
                     sr, audio_info.get('duration', 0), audio_info.get('num_samples', 0))

        # Step 2: Extract features
        logger.debug("Step 2: Feature extraction")
        features = feature_extractor.extract_features(
            audio_data,
            sr,
            frame_length=config.FRAME_LENGTH,
            hop_length=config.HOP_LENGTH
        )

        if features is None:
            logger.error("Feature extraction returned None for file: %s", audio_file.filename)
            return jsonify({'success': False, 'error': 'Feature extraction failed'}), 500

        logger.debug("Step 2 complete — %d features extracted", len(features))

        # Step 3: Prepare features (pad, scale, reshape)
        logger.debug("Step 3: Feature preparation")
        prepared_features = feature_extractor.prepare_features(features)

        if prepared_features is None:
            logger.error("Feature preparation returned None for file: %s", audio_file.filename)
            return jsonify({'success': False, 'error': 'Feature preparation failed'}), 500

        logger.debug("Step 3 complete — prepared shape: %s", prepared_features.shape)

        # Step 4: Run inference
        logger.debug("Step 4: Running inference")
        result = inference.predict_emotion(prepared_features)

        if not result['success']:
            error_msg = result.get('error', 'Prediction failed')
            logger.error("Inference failed for file '%s': %s", audio_file.filename, error_msg)
            return jsonify({'success': False, 'error': error_msg}), 500

        logger.info("Prediction complete — emotion: %s, confidence: %.2f%%",
                    result['emotion'], result['confidence'])

        # Log full probability breakdown at DEBUG so it's available when needed
        # but doesn't clutter production logs
        for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            logger.debug("  %-10s %.2f%%", emotion, prob)

        return jsonify(result), 200

    except Exception as e:
        logger.error("Unhandled exception in /predict: %s", e, exc_info=True)
        return jsonify({'success': False, 'error': f'Server error: {e}'}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    Returns 200 if models are loaded and ready, 503 otherwise.
    """
    try:
        model_info = inference.get_model_info()

        if model_info.get('initialized', False):
            return jsonify({
                'status': 'healthy',
                'models_loaded': True,
                'model_info': model_info
            }), 200
        else:
            logger.warning("Health check failed — models not initialized")
            return jsonify({
                'status': 'unhealthy',
                'models_loaded': False,
                'error': 'Models not initialized'
            }), 503

    except Exception as e:
        logger.error("Health check raised an exception: %s", e, exc_info=True)
        return jsonify({'status': 'unhealthy', 'models_loaded': False, 'error': str(e)}), 503


@app.route('/info', methods=['GET'])
def info():
    """Get application information and current config."""
    return jsonify({
        'app': 'Speech Emotion Recognition',
        'version': '1.0.0',
        'recording_duration': config.RECORDING_DURATION,
        'emotions': config.DEFAULT_EMOTIONS,
        'allowed_extensions': config.ALLOWED_AUDIO_EXTENSIONS,
        'max_upload_size_mb': config.MAX_UPLOAD_FILE_SIZE // (1024 * 1024),
        'model_info': inference.get_model_info()
    }), 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    logger.warning("Request rejected — payload too large (HTTP 413)")
    return jsonify({'success': False, 'error': 'Audio file too large. Maximum size is 16 MB.'}), 413


@app.errorhandler(404)
def not_found(error):
    logger.debug("404 — endpoint not found: %s", request.path)
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error("Unhandled 500 error: %s", error, exc_info=True)
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    logger.info(
        "Starting Flask development server — host: %s, port: %d, debug: %s, allowed types: %s",
        config.FLASK_HOST, config.FLASK_PORT, config.FLASK_DEBUG,
        ", ".join(config.ALLOWED_AUDIO_EXTENSIONS)
    )

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )