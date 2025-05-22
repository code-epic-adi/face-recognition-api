import os
import base64
import json
from io import BytesIO
from PIL import Image
import numpy as np
import face_recognition
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/api/compare_faces', methods=['POST'])
def compare_faces():
    try:
        # Get request data
        data = request.json
        
        if not data or 'submitted_image' not in data or 'reference_image' not in data:
            return jsonify({
                "is_match": False,
                "similarity_score": 0.0,
                "error": "Both submitted_image and reference_image are required"
            })
        
        # Decode base64 images
        try:
            submitted_image_data = data['submitted_image']
            reference_image_data = data['reference_image']
            
            # Handle data URLs (remove prefix if present)
            if ',' in submitted_image_data:
                submitted_image_data = submitted_image_data.split(',', 1)[1]
            if ',' in reference_image_data:
                reference_image_data = reference_image_data.split(',', 1)[1]
            
            submitted_image_bytes = base64.b64decode(submitted_image_data)
            reference_image_bytes = base64.b64decode(reference_image_data)
        except Exception as e:
            return jsonify({
                "is_match": False,
                "similarity_score": 0.0,
                "error": f"Invalid image format: {str(e)}"
            })
        
        # Load images
        try:
            submitted_pil = Image.open(BytesIO(submitted_image_bytes))
            reference_pil = Image.open(BytesIO(reference_image_bytes))
            
            # Convert to RGB if needed
            if submitted_pil.mode != 'RGB':
                submitted_pil = submitted_pil.convert('RGB')
            if reference_pil.mode != 'RGB':
                reference_pil = reference_pil.convert('RGB')
            
            # Convert to numpy arrays
            submitted_image = np.array(submitted_pil)
            reference_image = np.array(reference_pil)
        except Exception as e:
            return jsonify({
                "is_match": False,
                "similarity_score": 0.0,
                "error": f"Invalid image data: {str(e)}"
            })
        
        # Detect faces
        submitted_face_locations = face_recognition.face_locations(submitted_image)
        reference_face_locations = face_recognition.face_locations(reference_image)
        
        # Check if faces were detected
        if not submitted_face_locations:
            return jsonify({
                "is_match": False,
                "similarity_score": 0.0,
                "error": "No face detected in submitted image"
            })
        
        if not reference_face_locations:
            return jsonify({
                "is_match": False,
                "similarity_score": 0.0,
                "error": "No face detected in reference image"
            })
        
        # Get face encodings
        submitted_face_encoding = face_recognition.face_encodings(submitted_image, submitted_face_locations)[0]
        reference_face_encoding = face_recognition.face_encodings(reference_image, reference_face_locations)[0]
        
        # Compare faces
        # Calculate face distance (lower means more similar)
        face_distance = face_recognition.face_distance([reference_face_encoding], submitted_face_encoding)[0]
        
        # Convert distance to similarity score (0 to 1, higher is more similar)
        similarity_score = 1.0 - min(face_distance, 1.0)
        
        # Determine if it's a match (threshold can be adjusted)
        threshold = 0.6
        is_match = similarity_score >= threshold
        
        return jsonify({
            "is_match": bool(is_match),
            "similarity_score": float(similarity_score),
            "error": None
        })
        
    except Exception as e:
        return jsonify({
            "is_match": False,
            "similarity_score": 0.0,
            "error": str(e)
        })

# Simple test endpoint
@app.route('/test')
def test():
    return jsonify({"status": "API is working!"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
