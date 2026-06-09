# AI Fakeness Detector
AI-powered system that compares an original image with a suspected one to detect and quantify manipulation. Using deep learning-based feature extraction and similarity analysis, it generates a “fakeness score” indicating the level of alteration. The system is designed to handle common challenges like cropping, compression, and minor edits, ensuring reliable detection. This helps users quickly verify authenticity and identify tampered content.

## Features
* Image comparison using AI
* Fakeness score generation
* Detects image tampering
* Handles cropping, resizing, and compression
* Fast similarity-based matching
* Supports image and video frame analysis
  
## Tech Stack
* JavaScript
* FastAPI
* PyTorch
* Hugging Face Transformers
* FAISS
* OpenCV
* Pillow (PIL)
  
## How It Works
* Upload the original image.
* Upload the suspected image.
* Extract features from both images.
* Compare the features using similarity search.
* Generate a Fakeness Score.
* Display the result.

## Flow Diagram

  <img width="1413" height="536" alt="image" src="https://github.com/user-attachments/assets/4244c921-9709-42d8-98b3-e5d4f3b25a42" />

