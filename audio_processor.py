# audio_processor.py

import io
import numpy as np
import librosa
import config
import os
from logger import setup_logger

logger = setup_logger(__name__)

def convert_to_wav(audio_bytes):
    """
    Convert any audio format (Ogg, WebM, MP3, etc.) to WAV using pydub.
    """
    try:
        from pydub import AudioSegment

        if os.path.exists(config.FFMPEG_PATH):
            AudioSegment.converter = config.FFMPEG_PATH
            AudioSegment.ffprobe = config.FFPROBE_PATH
            logger.debug("ffmpeg configured from: %s", config.FFMPEG_PATH)
        else:
            # WARNING: app may still work if ffmpeg is on system PATH, but likely won't
            logger.warning("ffmpeg not found at configured path: %s — audio conversion may fail", config.FFMPEG_PATH)

        logger.debug("Converting audio bytes to WAV format")
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

        wav_io = io.BytesIO()
        audio.export(wav_io, format='wav')
        wav_io.seek(0)
        wav_bytes = wav_io.read()

        logger.debug("Conversion successful, WAV size: %d bytes", len(wav_bytes))
        return wav_bytes

    except Exception as e:
        logger.error("Failed to convert audio to WAV: %s", e, exc_info=True)
        raise


def load_audio_from_bytes(audio_bytes):
    """
    Load audio from raw bytes (from web upload).

    Args:
        audio_bytes: Raw audio file bytes (WAV format from browser)

    Returns:
        tuple: (audio_data, sample_rate)
            - audio_data: numpy array of audio samples
            - sample_rate: sample rate (will be 22050 from librosa default)

    Raises:
        Exception: If audio loading fails
    """
    try:
        audio_file = io.BytesIO(audio_bytes)
        data, sr = librosa.load(
            audio_file,
            duration=config.DURATION,
            offset=config.OFFSET
            # No sr parameter - uses librosa default (22050 Hz)
        )
        logger.debug("Audio loaded from bytes: %d samples at %d Hz", len(data), sr)
        return data, sr

    except Exception as e:
        logger.error("Failed to load audio from bytes: %s", e, exc_info=True)
        raise


def load_audio_from_file(file_path):
    """
    Load audio from file path (for testing).

    Args:
        file_path: Path to audio file

    Returns:
        tuple: (audio_data, sample_rate)
    """
    try:
        data, sr = librosa.load(
            file_path,
            duration=config.DURATION,
            offset=config.OFFSET
        )
        logger.debug("Audio loaded from file '%s': %d samples at %d Hz", file_path, len(data), sr)
        return data, sr

    except Exception as e:
        logger.error("Failed to load audio from file '%s': %s", file_path, e, exc_info=True)
        raise


def validate_audio_duration(audio_data, sr, min_duration=1.0):
    """
    Validate that audio has sufficient duration.

    Args:
        audio_data: numpy array of audio samples
        sr: sample rate
        min_duration: minimum required duration in seconds

    Returns:
        tuple: (is_valid, actual_duration)
    """
    actual_duration = len(audio_data) / sr
    is_valid = actual_duration >= min_duration
    return is_valid, actual_duration


def validate_audio_format(audio_data):
    """
    Validate audio data format.

    Args:
        audio_data: numpy array of audio samples

    Returns:
        tuple: (is_valid, message)
    """
    if audio_data is None or len(audio_data) == 0:
        return False, "Audio data is empty"

    if np.isnan(audio_data).any():
        return False, "Audio contains NaN values"

    if np.isinf(audio_data).any():
        return False, "Audio contains Inf values"

    max_amplitude = np.abs(audio_data).max()
    if max_amplitude == 0:
        return False, "Audio is silent (all zeros)"

    return True, f"Valid audio (max amplitude: {max_amplitude:.4f})"


def get_audio_info(audio_data, sr):
    """
    Get detailed information about audio data.

    Args:
        audio_data: numpy array of audio samples
        sr: sample rate

    Returns:
        dict: Audio information
    """
    duration = len(audio_data) / sr
    return {
        'sample_rate': sr,
        'duration': duration,
        'num_samples': len(audio_data),
        'max_amplitude': float(np.abs(audio_data).max()),
        'mean_amplitude': float(np.abs(audio_data).mean()),
        'rms_amplitude': float(np.sqrt(np.mean(audio_data**2)))
    }


def process_audio_for_inference(audio_bytes):
    """
    Complete pipeline: bytes → validated audio data.

    This is the main function to use in the Flask app.

    Args:
        audio_bytes: Raw audio bytes from browser

    Returns:
        tuple: (audio_data, sr, info_dict) or (None, None, error_dict)
    """
    try:
        logger.debug("Starting audio processing pipeline, input size: %d bytes", len(audio_bytes))

        # Check if already WAV format
        if audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE':
            logger.debug("Audio is already WAV format, skipping conversion")
            wav_bytes = audio_bytes
        else:
            logger.debug("Non-WAV audio detected, converting...")
            wav_bytes = convert_to_wav(audio_bytes)

        # Load audio
        audio_data, sr = load_audio_from_bytes(wav_bytes)
        logger.info("Audio loaded: %d samples at %d Hz", len(audio_data), sr)

        # Validate format
        is_valid_format, format_msg = validate_audio_format(audio_data)
        if not is_valid_format:
            logger.warning("Audio failed format validation: %s", format_msg)
            return None, None, {'error': f"Invalid format: {format_msg}"}
        logger.debug("Format validation passed: %s", format_msg)

        # Validate duration
        is_valid_duration, actual_duration = validate_audio_duration(audio_data, sr)
        if not is_valid_duration:
            # WARNING not ERROR: we continue processing, short audio may still yield a result
            logger.warning("Audio duration %.2fs is below recommended minimum", actual_duration)
        else:
            logger.debug("Duration valid: %.2fs", actual_duration)

        info = get_audio_info(audio_data, sr)
        info['valid'] = True

        logger.debug("Audio pipeline complete: duration=%.2fs, max_amplitude=%.4f",
                     info['duration'], info['max_amplitude'])
        return audio_data, sr, info

    except Exception as e:
        logger.error("Audio processing pipeline failed: %s", e, exc_info=True)
        return None, None, {'error': f"Audio processing failed: {e}"}


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logger.info("--- Audio Processor Test Start ---")

    test_file = "audio_samples/1001_DFA_ANG_XX.wav"

    if not os.path.exists(test_file):
        logger.error("Test file not found: %s", test_file)
    else:
        logger.info("Test file: %s", test_file)

        # Test 1: Load from file
        logger.info("TEST 1: Load from file path")
        try:
            audio, sr = load_audio_from_file(test_file)
            logger.info("Loaded audio shape: %s, sample rate: %d", audio.shape, sr)
        except Exception as e:
            logger.error("TEST 1 failed: %s", e)

        # Test 2: Load from bytes
        logger.info("TEST 2: Load from bytes (simulating web upload)")
        try:
            with open(test_file, 'rb') as f:
                audio_bytes = f.read()
            logger.info("Read %d bytes from file", len(audio_bytes))

            audio, sr = load_audio_from_bytes(audio_bytes)
            logger.info("Loaded from bytes — shape: %s, sr: %d", audio.shape, sr)
        except Exception as e:
            logger.error("TEST 2 failed: %s", e)

        # Test 3: Validate audio
        logger.info("TEST 3: Audio validation")
        try:
            is_valid, duration = validate_audio_duration(audio, sr)
            logger.info("Duration validation: %s (%.2fs)", is_valid, duration)

            is_valid, msg = validate_audio_format(audio)
            logger.info("Format validation: %s — %s", is_valid, msg)
        except Exception as e:
            logger.error("TEST 3 failed: %s", e)

        # Test 4: Audio info
        logger.info("TEST 4: Audio information")
        try:
            info = get_audio_info(audio, sr)
            logger.info(
                "sr=%d Hz | duration=%.2fs | samples=%d | max_amp=%.4f | mean_amp=%.4f | rms=%.4f",
                info['sample_rate'], info['duration'], info['num_samples'],
                info['max_amplitude'], info['mean_amplitude'], info['rms_amplitude']
            )
        except Exception as e:
            logger.error("TEST 4 failed: %s", e)

        # Test 5: Full pipeline
        logger.info("TEST 5: Complete processing pipeline")
        try:
            with open(test_file, 'rb') as f:
                audio_bytes = f.read()

            audio_data, sr, info = process_audio_for_inference(audio_bytes)

            if audio_data is not None:
                logger.info("Pipeline successful — shape: %s, sr: %d, info: %s",
                            audio_data.shape, sr, info)
            else:
                logger.error("Pipeline failed: %s", info)
        except Exception as e:
            logger.error("TEST 5 failed: %s", e)

    logger.info("--- Audio Processor Test Complete ---")