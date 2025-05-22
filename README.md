# Face Recognition API

A Flask-based API for face recognition that can be deployed on Render.

## Features

- Compare two face images and determine if they match
- Returns similarity score and match status
- Simple HTTP API that can be integrated with any client application

## API Usage

### Compare Faces

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

## Deployment

This API is deployed on Render using Docker.

## License

MIT
