// ===================================
// Configuration
// ===================================

const API_BASE_URL = 'http://localhost:5000';
let selectedFile = null;

// ===================================
// DOM Elements
// ===================================

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeFileBtn = document.getElementById('removeFile');
const analyzeButton = document.getElementById('analyzeButton');

const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

const sliceIndex = document.getElementById('sliceIndex');
const inputImages = document.getElementById('inputImages');
const predictionImages = document.getElementById('predictionImages');
const overlayImage = document.getElementById('overlayImage');

const newAnalysisButton = document.getElementById('newAnalysisButton');
const retryButton = document.getElementById('retryButton');

// ===================================
// File Upload Handling
// ===================================

// Click to upload (but not when clicking the label)
uploadArea.addEventListener('click', (e) => {
    // Don't trigger if clicking on the label (it has its own trigger)
    if (e.target.tagName !== 'LABEL' && !e.target.closest('label')) {
        fileInput.click();
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// Remove file
removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFileSelection();
});

// ===================================
// File Selection
// ===================================

function handleFileSelect(file) {
    if (!file) return;

    // Validate file type
    const validExtensions = ['.nii', '.nii.gz'];
    const isValid = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

    if (!isValid) {
        showError('Invalid file type. Please upload a NIfTI file (.nii or .nii.gz)');
        return;
    }

    // Validate file size (500MB max)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File too large. Maximum size is 500MB');
        return;
    }

    selectedFile = file;

    // Update UI
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    uploadArea.style.display = 'none';
    fileInfo.style.display = 'flex';
    analyzeButton.disabled = false;
}

function clearFileSelection() {
    selectedFile = null;
    fileInput.value = '';

    uploadArea.style.display = 'block';
    fileInfo.style.display = 'none';
    analyzeButton.disabled = true;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ===================================
// Analysis
// ===================================

analyzeButton.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Hide all sections
    hideAllSections();

    // Show loading
    loadingSection.style.display = 'block';

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Send request to backend
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || data.error || 'Prediction failed');
        }

        // Display results
        displayResults(data.results);

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred during analysis. Please try again.');
    }
});

// ===================================
// Results Display
// ===================================

function displayResults(results) {
    hideAllSections();

    // Update slice index
    sliceIndex.textContent = results.slice_index;

    // Clear previous results
    inputImages.innerHTML = '';
    predictionImages.innerHTML = '';
    overlayImage.innerHTML = '';

    // Display visualizations
    results.visualizations.forEach(viz => {
        const imageCard = createImageCard(viz.name, viz.image);

        if (viz.type === 'input') {
            inputImages.appendChild(imageCard);
        } else if (viz.type === 'prediction') {
            predictionImages.appendChild(imageCard);
        } else if (viz.type === 'overlay') {
            overlayImage.appendChild(imageCard);
        }
    });

    // Show results section
    resultsSection.style.display = 'block';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function createImageCard(label, base64Image) {
    const card = document.createElement('div');
    card.className = 'image-card';

    const img = document.createElement('img');
    img.src = `data:image/png;base64,${base64Image}`;
    img.alt = label;

    const labelDiv = document.createElement('div');
    labelDiv.className = 'image-label';
    labelDiv.textContent = label;

    card.appendChild(img);
    card.appendChild(labelDiv);

    return card;
}

// ===================================
// Error Handling
// ===================================

function showError(message) {
    hideAllSections();

    errorMessage.textContent = message;
    errorSection.style.display = 'block';

    // Scroll to error
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ===================================
// UI State Management
// ===================================

function hideAllSections() {
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
}

function resetToUpload() {
    hideAllSections();
    clearFileSelection();

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ===================================
// Button Handlers
// ===================================

newAnalysisButton.addEventListener('click', resetToUpload);
retryButton.addEventListener('click', resetToUpload);

// ===================================
// Health Check on Load
// ===================================

async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (response.ok) {
            console.log('✓ API is healthy:', data.message);
        } else {
            console.warn('⚠ API health check failed:', data.message);
        }
    } catch (error) {
        console.error('✗ Cannot connect to API:', error.message);
        console.log('Make sure the Flask backend is running on http://localhost:5000');
    }
}

// Check API health when page loads
window.addEventListener('load', () => {
    checkAPIHealth();
    console.log('Brain Tumor Segmentation App Loaded');
});

// ===================================
// Keyboard Shortcuts
// ===================================

document.addEventListener('keydown', (e) => {
    // ESC to reset
    if (e.key === 'Escape') {
        if (resultsSection.style.display === 'block' || errorSection.style.display === 'block') {
            resetToUpload();
        }
    }

    // Enter to analyze (if file selected)
    if (e.key === 'Enter' && selectedFile && !analyzeButton.disabled) {
        analyzeButton.click();
    }
});

// ===================================
// Prevent default drag behavior on document
// ===================================

document.addEventListener('dragover', (e) => {
    e.preventDefault();
});

document.addEventListener('drop', (e) => {
    e.preventDefault();
});
