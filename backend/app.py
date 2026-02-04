from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
from werkzeug.utils import secure_filename
from model_loader import get_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'nii', 'nii.gz'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Model path - adjust this to your model location
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Models', 'best_model-13.pth')

# Initialize model (lazy loading)
model = None

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           any(filename.lower().endswith(ext) for ext in ['.nii', '.nii.gz'])

def get_model_instance():
    """Get or initialize model instance"""
    global model
    if model is None:
        try:
            model = get_model(MODEL_PATH)
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise
    return model

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if model can be loaded
        get_model_instance()
        return jsonify({
            'status': 'healthy',
            'message': 'API is running and model is loaded'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'message': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Accepts NIfTI file upload and returns segmentation results
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Please upload a NIfTI file'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a file to upload'
            }), 400
        
        # Validate file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Please upload a NIfTI file (.nii or .nii.gz)'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Get model instance
            model_instance = get_model_instance()
            
            # Process file
            print(f"Processing file: {filename}")
            results = model_instance.process_file(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'message': 'Segmentation completed successfully',
                'results': results
            }), 200
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    File upload endpoint (for validation before prediction)
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Only .nii and .nii.gz files are allowed'
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'File is valid',
            'filename': file.filename
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Brain Tumor Segmentation API',
        'version': '1.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/upload': 'POST - Validate file upload',
            '/predict': 'POST - Run segmentation prediction'
        }
    }), 200

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large',
        'message': f'Maximum file size is {MAX_FILE_SIZE / (1024*1024)}MB'
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Brain Tumor Segmentation API")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Max file size: {MAX_FILE_SIZE / (1024*1024)}MB")
    print("=" * 60)
    print("Starting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("=" * 60)
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
