# File: app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def denoise_image(img, method='gaussian', strength=5):
    """
    Apply different denoising methods to an image
    
    Parameters:
    - img: Input image
    - method: Denoising method (gaussian, median, bilateral, nlm, etc.)
    - strength: Strength of denoising effect
    
    Returns:
    - Denoised image
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(img, (strength, strength), 0)
    elif method == 'median':
        return cv2.medianBlur(img, strength)
    elif method == 'bilateral':
        d = strength * 2 + 1  # Diameter of each pixel neighborhood
        sigma_color = strength  # Filter sigma in the color space
        sigma_space = strength  # Filter sigma in the coordinate space
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    elif method == 'nlm':  # Non-local means denoising
        h = strength  # Filter strength
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    else:
        return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/denoise', methods=['POST'])
def api_denoise():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Get parameters
        method = request.form.get('method', 'gaussian')
        strength = int(request.form.get('strength', 5))
        
        # Read and process the image
        img_stream = file.read()
        nparr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Apply denoising
        denoised_img = denoise_image(img, method, strength)
        
        # Convert the image back to base64 for sending it to the client
        _, buffer = cv2.imencode('.png', denoised_img)
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'original_size': img.shape,
            'denoised_size': denoised_img.shape,
            'denoised_image': encoded_img
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

# This is required for Vercel deployment
if __name__ == '__main__':
    app.run(debug=True)

# File: requirements.txt
flask==2.0.1
numpy==1.22.0
opencv-python-headless==4.5.5.64
Werkzeug==2.0.1
gunicorn==20.1.0

# File: templates/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Denoiser</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .upload-section {
            border: 2px dashed #ccc;
            padding: 20px;
            text-align: center;
            cursor: pointer;
        }
        .upload-section:hover {
            border-color: #777;
        }
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
        }
        .control-group {
            flex: 1;
            min-width: 200px;
        }
        .image-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .image-box {
            flex: 1;
            min-width: 300px;
            text-align: center;
        }
        .image-box img {
            max-width: 100%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 15px;
            cursor: pointer;
            font-size: 16px;
            border-radius: 4px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        select, input {
            padding: 8px;
            width: 100%;
            margin-top: 5px;
            box-sizing: border-box;
        }
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 2s linear infinite;
            margin: 20px auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .download-btn {
            background-color: #337ab7;
            margin-top: 10px;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <h1>Image Denoiser</h1>
    <div class="container">
        <div class="upload-section" id="upload-area">
            <p>Drop your image here or click to upload</p>
            <input type="file" id="file-input" accept=".jpg,.jpeg,.png" style="display: none;">
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="method">Denoising Method:</label>
                <select id="method">
                    <option value="gaussian">Gaussian Blur</option>
                    <option value="median">Median Blur</option>
                    <option value="bilateral">Bilateral Filter</option>
                    <option value="nlm">Non-Local Means</option>
                </select>
            </div>
            <div class="control-group">
                <label for="strength">Strength (1-15):</label>
                <input type="range" id="strength" min="1" max="15" value="5">
                <span id="strength-value">5</span>
            </div>
            <div class="control-group">
                <button id="denoise-btn" disabled>Apply Denoising</button>
            </div>
        </div>
        
        <div class="loader" id="loader"></div>
        
        <div class="image-container">
            <div class="image-box">
                <h3>Original Image</h3>
                <img id="original-image" src="" alt="Original image will appear here" class="hidden">
            </div>
            <div class="image-box">
                <h3>Denoised Image</h3>
                <img id="denoised-image" src="" alt="Denoised image will appear here" class="hidden">
                <button id="download-btn" class="download-btn hidden">Download Denoised Image</button>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const uploadArea = document.getElementById('upload-area');
            const fileInput = document.getElementById('file-input');
            const originalImage = document.getElementById('original-image');
            const denoisedImage = document.getElementById('denoised-image');
            const denoiseBtn = document.getElementById('denoise-btn');
            const downloadBtn = document.getElementById('download-btn');
            const methodSelect = document.getElementById('method');
            const strengthSlider = document.getElementById('strength');
            const strengthValue = document.getElementById('strength-value');
            const loader = document.getElementById('loader');
            
            let selectedFile = null;
            
            // Update strength value display
            strengthSlider.addEventListener('input', function() {
                strengthValue.textContent = this.value;
            });
            
            // Handle file selection via click
            uploadArea.addEventListener('click', function() {
                fileInput.click();
            });
            
            // Handle file selection via file input
            fileInput.addEventListener('change', function() {
                handleFileSelect(this.files);
            });
            
            // Handle drag and drop
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                e.stopPropagation();
                this.style.borderColor = '#45a049';
            });
            
            uploadArea.addEventListener('dragleave', function(e) {
                e.preventDefault();
                e.stopPropagation();
                this.style.borderColor = '#ccc';
            });
            
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                e.stopPropagation();
                this.style.borderColor = '#ccc';
                handleFileSelect(e.dataTransfer.files);
            });
            
            // Handle the denoise button click
            denoiseBtn.addEventListener('click', function() {
                if (selectedFile) {
                    denoiseImage();
                }
            });
            
            // Handle the download button click
            downloadBtn.addEventListener('click', function() {
                if (denoisedImage.src) {
                    const link = document.createElement('a');
                    link.href = denoisedImage.src;
                    link.download = 'denoised_image.png';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }
            });
            
            function handleFileSelect(files) {
                if (files.length > 0) {
                    selectedFile = files[0];
                    
                    if (selectedFile.type.match('image.*')) {
                        const reader = new FileReader();
                        
                        reader.onload = function(e) {
                            originalImage.src = e.target.result;
                            originalImage.classList.remove('hidden');
                            denoiseBtn.disabled = false;
                        };
                        
                        reader.readAsDataURL(selectedFile);
                    } else {
                        alert('Please select an image file (JPEG or PNG).');
                    }
                }
            }
            
            function denoiseImage() {
                // Show the loader
                loader.style.display = 'block';
                
                // Hide the denoised image while processing
                denoisedImage.classList.add('hidden');
                downloadBtn.classList.add('hidden');
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                formData.append('method', methodSelect.value);
                formData.append('strength', strengthSlider.value);
                
                fetch('/api/denoise', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.status === 'success') {
                        // Display the denoised image
                        denoisedImage.src = 'data:image/png;base64,' + data.denoised_image;
                        denoisedImage.classList.remove('hidden');
                        downloadBtn.classList.remove('hidden');
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while processing the image. Please try again.');
                })
                .finally(() => {
                    // Hide the loader
                    loader.style.display = 'none';
                });
            }
        });
    </script>
</body>
</html>

# File: vercel.json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python",
      "config": {
        "runtime": "python3.9"
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}