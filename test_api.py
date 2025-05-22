import requests
import base64
import json
import os

def test_api():
    """
    Test the face recognition API by sending a request with two images.
    You can use this to test if your API is working correctly.
    """
    # Replace with your API URL
    api_url = "https://your-render-app.onrender.com/api/compare_faces"
    
    # For local testing
    # api_url = "http://localhost:8000/api/compare_faces"
    
    # Test if API is up
    try:
        test_response = requests.get("https://your-render-app.onrender.com/test")
        print(f"API test endpoint response: {test_response.json()}")
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return
    
    # Path to your test images
    # Replace these with paths to your own test images
    submitted_image_path = "submitted_image.jpg"
    reference_image_path = "reference_image.jpg"
    
    # Check if files exist
    if not os.path.exists(submitted_image_path) or not os.path.exists(reference_image_path):
        print(f"Error: Test image files not found. Please provide valid image paths.")
        return
    
    # Read and encode images
    with open(submitted_image_path, "rb") as image_file:
        submitted_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    with open(reference_image_path, "rb") as image_file:
        reference_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Prepare request data
    payload = {
        "submitted_image": submitted_image_base64,
        "reference_image": reference_image_base64
    }
    
    # Send request
    try:
        response = requests.post(api_url, json=payload)
        
        # Print response
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error making API request: {e}")

if __name__ == "__main__":
    test_api()
