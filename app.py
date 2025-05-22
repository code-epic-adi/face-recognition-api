import streamlit as st
import base64
import io
import numpy as np
from PIL import Image
import json
from io import BytesIO
from skimage import feature, transform, color

st.set_page_config(page_title="Face Recognition API", layout="wide")
st.title("Face Recognition API")

# Function to detect faces using HOG from scikit-image
def detect_faces_hog(image):
    # Convert to grayscale
    img_gray = np.array(image.convert('L'))
    
    # Resize for faster processing
    img_resized = transform.resize(img_gray, (256, 256), preserve_range=True).astype(np.uint8)
    
    # Extract HOG features
    hog_features = feature.hog(
        img_resized, 
        orientations=8, 
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), 
        visualize=False
    )
    
    # Check if there's enough feature information (very naive approach)
    feature_score = np.sum(hog_features > 0.1)
    has_face = feature_score > 100
    
    return has_face

# Function to compare images using HOG features
def compare_faces_hog(img1, img2):
    # Resize images to same dimensions
    img1 = img1.resize((128, 128))
    img2 = img2.resize((128, 128))
    
    # Convert to grayscale
    img1_gray = np.array(img1.convert('L'))
    img2_gray = np.array(img2.convert('L'))
    
    # Extract HOG features
    hog1 = feature.hog(
        img1_gray, 
        orientations=8, 
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), 
        visualize=False
    )
    
    hog2 = feature.hog(
        img2_gray, 
        orientations=8, 
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), 
        visualize=False
    )
    
    # Calculate cosine similarity
    dot_product = np.dot(hog1, hog2)
    norm1 = np.linalg.norm(hog1)
    norm2 = np.linalg.norm(hog2)
    
    if norm1 == 0 or norm2 == 0:
        similarity = 0
    else:
        similarity = dot_product / (norm1 * norm2)
    
    # Determine if it's a match
    threshold = 0.8
    is_match = similarity >= threshold
    
    return is_match, similarity

# API endpoints
tab1, tab2 = st.tabs(["Web Interface", "API Documentation"])

with tab1:
    # Upload reference image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reference Image")
        ref_image_file = st.file_uploader("Upload reference image", type=["jpg", "jpeg", "png"], key="ref")
        if ref_image_file:
            ref_image = Image.open(ref_image_file)
            st.image(ref_image, caption="Reference Image", use_column_width=True)
    
    with col2:
        st.subheader("Test Image")
        test_image_file = st.file_uploader("Upload test image", type=["jpg", "jpeg", "png"], key="test")
        if test_image_file:
            test_image = Image.open(test_image_file)
            st.image(test_image, caption="Test Image", use_column_width=True)
    
    # Process images
    if ref_image_file and test_image_file:
        if st.button("Compare Faces"):
            with st.spinner("Processing..."):
                # Convert to PIL Images
                ref_image = Image.open(ref_image_file)
                test_image = Image.open(test_image_file)
                
                # Detect faces
                ref_has_face = detect_faces_hog(ref_image)
                test_has_face = detect_faces_hog(test_image)
                
                if not ref_has_face:
                    st.error("No face detected in reference image")
                elif not test_has_face:
                    st.error("No face detected in test image")
                else:
                    # Compare faces
                    is_match, similarity = compare_faces_hog(ref_image, test_image)
                    
                    # Display results
                    result_color = "green" if is_match else "red"
                    st.markdown(f"<h2 style='text-align: center; color: {result_color};'>{'MATCH' if is_match else 'NO MATCH'}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center;'>Similarity: {similarity:.2f}</h3>", unsafe_allow_html=True)
                    
                    st.warning("Note: This is using HOG features for face comparison and may not be as accurate as specialized face recognition systems")

with tab2:
    st.header("API Documentation")
    
    st.subheader("Compare Faces API")
    st.markdown("""
    **Endpoint:** `/api/compare_faces`
    
    **Method:** POST
    
    **Request Format:**
    ```json
    {
        "submitted_image": "base64_encoded_image",
        "reference_image": "base64_encoded_image"
    }
    ```
    
    **Response Format:**
    ```json
    {
        "is_match": true,
        "similarity_score": 0.85,
        "error": null
    }
    ```
    
    **Example Usage:**
    ```kotlin
    // Android Kotlin Example
    val apiUrl = "https://huggingface.co/spaces/yourusername/face-recognition-api/api/compare_faces"
    val jsonBody = JSONObject().apply {
        put("submitted_image", submittedImageBase64)
        put("reference_image", referenceImageBase64)
    }
    
    // Make API call using your preferred HTTP client
    ```
    """)

# Add a footer with disclaimer
st.markdown("---")
st.caption("Disclaimer: This is a simplified face comparison tool using HOG features and not a production-ready face recognition system.")
