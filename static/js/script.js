// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
    RECORDING_DURATION: 3, // seconds (matches backend config)
    API_ENDPOINT: '/predict', // Flask endpoint
    SAMPLE_RATE: 16000, // Browser recording sample rate
    MAX_FILE_SIZE: 5 * 1024 * 1024, // 5 MB
    ALLOWED_EXTENSIONS: ['.wav']
};

// Emotion icons mapping
const EMOTION_ICONS = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢'
};

// ============================================================================
// STATE
// ============================================================================

let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;
let isRecording = false;
let countdownInterval = null;
let selectedFile = null; // NEW

// ============================================================================
// DOM ELEMENTS
// ============================================================================

const elements = {
    recordButton: document.getElementById('recordButton'),
    recordingStatus: document.getElementById('recordingStatus'),
    countdown: document.getElementById('countdown'),
    audioInfo: document.getElementById('audioInfo'),
    playButton: document.getElementById('playButton'),
    loadingSpinner: document.getElementById('loadingSpinner'),
    resultsSection: document.getElementById('resultsSection'),
    errorSection: document.getElementById('errorSection'),
    errorMessage: document.getElementById('errorMessage'),
    tryAgainButton: document.getElementById('tryAgainButton'),
    retryButton: document.getElementById('retryButton'),
    emotionIcon: document.getElementById('emotionIcon'),
    emotionName: document.getElementById('emotionName'),
    confidence: document.getElementById('confidence'),
    probabilityBars: document.getElementById('probabilityBars'),
    audioPlayback: document.getElementById('audioPlayback'),
    // NEW - Upload elements
    fileInput: document.getElementById('fileInput'),
    fileInfo: document.getElementById('fileInfo'),
    fileName: document.getElementById('fileName'),
    uploadButton: document.getElementById('uploadButton'),
    clearFileButton: document.getElementById('clearFileButton')
};

// ============================================================================
// INITIALIZATION
// ============================================================================

async function initialize() {
    // Check browser compatibility
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError('Your browser does not support audio recording. Please use a modern browser.');
        return;
    }

    // Add event listeners - Recording
    elements.recordButton.addEventListener('click', handleRecordButtonClick);
    elements.playButton.addEventListener('click', playRecording);
    elements.tryAgainButton.addEventListener('click', resetUI);
    elements.retryButton.addEventListener('click', resetUI);

    // NEW - Add event listeners - Upload
    elements.fileInput.addEventListener('change', handleFileSelect);
    elements.uploadButton.addEventListener('click', handleUploadButtonClick);
    elements.clearFileButton.addEventListener('click', clearSelectedFile);

    console.log('✓ App initialized successfully');
}

// ============================================================================
// RECORDING FUNCTIONS
// ============================================================================

async function handleRecordButtonClick() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        // Request microphone permission
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                sampleRate: CONFIG.SAMPLE_RATE,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        // Reset state
        audioChunks = [];
        recordedBlob = null;

        // Create MediaRecorder
        mediaRecorder = new MediaRecorder(stream);

        // Handle data available event
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        // Handle recording stop
        mediaRecorder.onstop = async () => {
            // Create blob from chunks
            recordedBlob = new Blob(audioChunks, { type: 'audio/wav' });
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());

            // Show audio info
            elements.recordingStatus.classList.add('hidden');
            elements.audioInfo.classList.remove('hidden');

            // Send to backend for prediction
            await sendAudioForPrediction(recordedBlob);
        };

        // Start recording
        mediaRecorder.start();
        isRecording = true;

        // Update UI
        elements.recordButton.classList.add('recording');
        elements.recordButton.querySelector('.button-text').textContent = 'Recording...';
        elements.recordingStatus.classList.remove('hidden');
        elements.audioInfo.classList.add('hidden');

        // Start countdown
        startCountdown();

        console.log('✓ Recording started');

    } catch (error) {
        console.error('✗ Error starting recording:', error);
        showError('Could not access microphone. Please allow microphone access and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;

        // Update UI
        elements.recordButton.classList.remove('recording');
        elements.recordButton.querySelector('.button-text').textContent = 'Start Recording';

        // Stop countdown
        if (countdownInterval) {
            clearInterval(countdownInterval);
        }

        console.log('✓ Recording stopped');
    }
}

function startCountdown() {
    let timeLeft = CONFIG.RECORDING_DURATION;
    elements.countdown.textContent = timeLeft;

    countdownInterval = setInterval(() => {
        timeLeft--;
        elements.countdown.textContent = timeLeft;

        if (timeLeft <= 0) {
            clearInterval(countdownInterval);
            stopRecording();
        }
    }, 1000);
}

// ============================================================================
// FILE UPLOAD FUNCTIONS (NEW)
// ============================================================================

function handleFileSelect(event) {
    const file = event.target.files[0];
    
    if (!file) {
        return;
    }

    // Validate file extension
    const fileName = file.name.toLowerCase();
    const isValidExtension = CONFIG.ALLOWED_EXTENSIONS.some(ext => fileName.endsWith(ext));
    
    if (!isValidExtension) {
        showError(`Invalid file type. Only ${CONFIG.ALLOWED_EXTENSIONS.join(', ')} files are allowed.`);
        elements.fileInput.value = ''; // Clear input
        return;
    }

    // Validate file size
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        const maxSizeMB = CONFIG.MAX_FILE_SIZE / (1024 * 1024);
        showError(`File too large. Maximum size is ${maxSizeMB} MB.`);
        elements.fileInput.value = ''; // Clear input
        return;
    }

    // Store selected file
    selectedFile = file;

    // Update UI
    elements.fileName.textContent = file.name;
    elements.fileInfo.classList.remove('hidden');

    console.log('✓ File selected:', file.name);
}

function clearSelectedFile() {
    // Clear file input
    elements.fileInput.value = '';
    selectedFile = null;

    // Hide file info
    elements.fileInfo.classList.add('hidden');

    console.log('✓ File cleared');
}

async function handleUploadButtonClick() {
    if (!selectedFile) {
        showError('Please select a file first.');
        return;
    }

    // Send uploaded file for prediction
    await sendAudioForPrediction(selectedFile);
}

// ============================================================================
// AUDIO PLAYBACK
// ============================================================================

function playRecording() {
    if (recordedBlob) {
        const audioURL = URL.createObjectURL(recordedBlob);
        elements.audioPlayback.src = audioURL;
        elements.audioPlayback.play();
        console.log('▶ Playing recording');
    }
}

// ============================================================================
// PREDICTION
// ============================================================================

async function sendAudioForPrediction(audioBlob) {
    try {
        // Show loading spinner
        elements.loadingSpinner.classList.remove('hidden');
        elements.audioInfo.classList.add('hidden');
        elements.fileInfo.classList.add('hidden');

        // Convert blob to WAV format (if needed)
        const wavBlob = await convertToWav(audioBlob);

        // Create FormData
        const formData = new FormData();
        formData.append('audio', wavBlob, 'recording.wav');

        console.log('📤 Sending audio to backend...');

        // Send to Flask backend
        const response = await fetch(CONFIG.API_ENDPOINT, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        console.log('📥 Received response:', result);

        // Hide loading spinner
        elements.loadingSpinner.classList.add('hidden');

        // Check if prediction was successful
        if (result.success) {
            displayResults(result);
        } else {
            showError(result.error || 'Prediction failed');
        }

    } catch (error) {
        console.error('✗ Error sending audio:', error);
        elements.loadingSpinner.classList.add('hidden');
        showError('Failed to analyze audio. Please try again.');
    }
}

async function convertToWav(blob) {
    // For now, return the blob as-is
    // The backend will handle format conversion if needed
    // You can add WAV conversion here if needed using a library
    return blob;
}

// ============================================================================
// DISPLAY RESULTS
// ============================================================================

function displayResults(result) {
    // Update emotion display
    const emotion = result.emotion;
    const confidence = result.confidence;
    const probabilities = result.probabilities;

    // Set emotion icon
    elements.emotionIcon.textContent = EMOTION_ICONS[emotion] || '😊';

    // Set emotion name
    elements.emotionName.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);

    // Set confidence
    elements.confidence.textContent = confidence.toFixed(1);

    // Create probability bars
    createProbabilityBars(probabilities);

    // Show results section
    elements.resultsSection.classList.remove('hidden');

    console.log('✓ Results displayed');
}

function createProbabilityBars(probabilities) {
    // Clear existing bars
    elements.probabilityBars.innerHTML = '';

    // Sort emotions by probability (descending)
    const sortedEmotions = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1]);

    // Create bars
    sortedEmotions.forEach(([emotion, probability]) => {
        const barDiv = document.createElement('div');
        barDiv.className = 'probability-bar';

        barDiv.innerHTML = `
            <div class="bar-label">${emotion}</div>
            <div class="bar-container">
                <div class="bar-fill" style="width: ${probability}%">
                    ${probability.toFixed(1)}%
                </div>
            </div>
        `;

        elements.probabilityBars.appendChild(barDiv);
    });
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorSection.classList.remove('hidden');
    elements.loadingSpinner.classList.add('hidden');
    elements.audioInfo.classList.add('hidden');
    elements.fileInfo.classList.add('hidden');
    elements.resultsSection.classList.add('hidden');
}

// ============================================================================
// UI RESET
// ============================================================================

function resetUI() {
    // Hide all sections
    elements.recordingStatus.classList.add('hidden');
    elements.audioInfo.classList.add('hidden');
    elements.fileInfo.classList.add('hidden');
    elements.loadingSpinner.classList.add('hidden');
    elements.resultsSection.classList.add('hidden');
    elements.errorSection.classList.add('hidden');

    // Reset button
    elements.recordButton.classList.remove('recording');
    elements.recordButton.querySelector('.button-text').textContent = 'Start Recording';

    // Reset recording state
    audioChunks = [];
    recordedBlob = null;
    isRecording = false;

    // Reset upload state (NEW)
    elements.fileInput.value = '';
    selectedFile = null;

    console.log('✓ UI reset');
}

// ============================================================================
// START APP
// ============================================================================

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
} else {
    initialize();
}