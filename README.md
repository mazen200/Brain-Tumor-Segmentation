# Brain Tumor Segmentation

AI-powered brain tumor segmentation using SwinUNETR deep learning architecture. This project provides a complete web-based solution for analyzing MRI scans and identifying tumor regions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)


## 🎯 Features

- **Advanced Deep Learning**: SwinUNETR architecture for accurate tumor segmentation
- **Multi-Modal MRI Support**: Processes 4-channel MRI data (FLAIR, T1, T1ce, T2)
- **3-Class Segmentation**: Identifies Tumor Core (TC), Whole Tumor (WT), and Enhancing Tumor (ET)
- **Modern Web Interface**: Beautiful, responsive UI with drag-and-drop file upload
- **Real-time Visualization**: Interactive display of segmentation results
- **RESTful API**: Flask-based backend for easy integration

## 📁 Project Structure

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Modern web browser (Chrome, Firefox, Edge, Safari)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mazen200/Brain-Tumor-Segmentation.git
   cd Brain-Tumor-Segmentation
   ```

2. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. **Download the pre-trained model**

   Download out trained model (the link will be provided here soon)
   place it in the `Models/` directory as `best_model-13.pth`.


4. **Start the Flask server**
   ```bash
   python app.py
   ```
   
   The API will be available at `http://localhost:5000`

5. **Open the frontend**
   
   Open `frontend/index.html` in your web browser, or serve it using a local server:
   ```bash
   cd frontend
   python -m http.server 8000
   ```
   
   Then navigate to `http://localhost:8000`

## 💻 Usage

### Web Interface

1. **Upload MRI Scan**: Drag and drop or browse to select a NIfTI file (.nii or .nii.gz)
2. **Analyze**: Click the "Analyze MRI Scan" button
3. **View Results**: Explore the segmentation results with interactive visualizations

### API Endpoints

#### Health Check
```bash
GET http://localhost:5000/health
```

#### Predict Segmentation
```bash
POST http://localhost:5000/predict
Content-Type: multipart/form-data

file: <NIfTI file>
```

**Response:**
```json
{
  "success": true,
  "message": "Segmentation completed successfully",
  "results": {
    "slice_index": 64,
    "visualizations": [
      {
        "type": "input",
        "name": "FLAIR",
        "image": "<base64-encoded-image>"
      },
      ...
    ]
  }
}
```

## 🧠 Model Architecture

The project uses **SwinUNETR** (Swin Transformer-based UNETR), a state-of-the-art architecture for medical image segmentation:

- **Input**: 4-channel MRI volumes (FLAIR, T1, T1ce, T2)
- **Output**: 3-channel segmentation masks (TC, WT, ET)
- **Image Size**: 128×128×128 voxels
- **Inference**: Sliding window with 50% overlap
- **Loss Function**: Dice Loss
- **Optimizer**: AdamW with Cosine Annealing

### Tumor Classes

1. **Tumor Core (TC)**: Necrotic and non-enhancing tumor core
2. **Whole Tumor (WT)**: Complete tumor extent including edema
3. **Enhancing Tumor (ET)**: Actively enhancing tumor regions

## 🎨 Frontend Features

- **Dark Theme**: Modern, eye-friendly dark interface
- **Glassmorphism**: Frosted glass effect on cards
- **Gradient Accents**: Vibrant purple-blue gradients
- **Smooth Animations**: Micro-interactions and transitions
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Drag & Drop**: Intuitive file upload experience

## 🔧 Configuration

### Backend Configuration

Edit `backend/app.py` to customize:

```python
# Model path
MODEL_PATH = 'path/to/your/model.pth'

# Upload settings
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
UPLOAD_FOLDER = '/path/to/uploads'

# Server settings
app.run(host='0.0.0.0', port=5000, debug=True)
```

### Frontend Configuration

Edit `frontend/script.js` to customize:

```javascript
// API endpoint
const API_BASE_URL = 'http://localhost:5000';
```

## 📊 Dataset

The model was trained on the **BraTS (Brain Tumor Segmentation) dataset**, which contains:

- Multi-institutional MRI scans
- Expert-annotated tumor segmentations
- Multiple MRI modalities (FLAIR, T1, T1ce, T2)
- Diverse tumor types and sizes

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with 6GB+ VRAM (optional but recommended)
- **Storage**: 2GB for model and dependencies
- **OS**: Windows, Linux, or macOS

## 🐛 Troubleshooting

### Common Issues

**1. Model not loading**
- Ensure `Models/best_model-13.pth` exists
- Check file path in `app.py`

**2. CUDA out of memory**
- Reduce batch size in inference
- Use CPU instead: Set `device='cpu'` in `model_loader.py`

**3. API connection failed**
- Verify Flask server is running on port 5000
- Check CORS settings in `app.py`
- Update `API_BASE_URL` in `script.js`

**4. File upload fails**
- Check file format (.nii or .nii.gz)
- Verify file size < 500MB
- Ensure file contains 4 channels
