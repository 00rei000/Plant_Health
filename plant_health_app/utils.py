# Loại bỏ: Blueberry, Orange, Raspberry, Soybean, Squash (chỉ có 1 class không train được)
PLANT_TYPE_MAPPING = {
    "Apple___Apple_scab": "Apple",
    "Apple___Black_rot": "Apple",
    "Apple___Cedar_apple_rust": "Apple",
    "Apple___healthy": "Apple",
    "Cherry_(including_sour)___healthy": "Cherry",
    "Cherry_(including_sour)___Powdery_mildew": "Cherry",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": "Corn",
    "Corn_(maize)___Common_rust_": "Corn",
    "Corn_(maize)___healthy": "Corn",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn",
    "Grape___Black_rot": "Grape",
    "Grape___Esca_(Black_Measles)": "Grape",
    "Grape___healthy": "Grape",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape",
    "Peach___Bacterial_spot": "Peach",
    "Peach___healthy": "Peach",
    "Pepper___bell___Bacterial_spot": "Pepper",
    "Pepper___bell___healthy": "Pepper",
    "Potato___Early_blight": "Potato",
    "Potato___healthy": "Potato",
    "Potato___Late_blight": "Potato",
    "Strawberry___healthy": "Strawberry",
    "Strawberry___Leaf_scorch": "Strawberry",
    "Tomato___Bacterial_spot": "Tomato",
    "Tomato___Early_blight": "Tomato",
    "Tomato___healthy": "Tomato",
    "Tomato___Late_blight": "Tomato",
    "Tomato___Leaf_Mold": "Tomato",
    "Tomato___Septoria_leaf_spot": "Tomato",
    "Tomato___Spider_mites_Two-spotted_spider_mite": "Tomato",
    "Tomato___Target_Spot": "Tomato",
    "Tomato___Tomato_mosaic_virus": "Tomato",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato"
}

# Loại bỏ: Blueberry, Orange, Raspberry, Soybean, Squash
CLASS_NAME_TO_DISEASE = {
    'Apple___Apple_scab': 'Apple scab',
    'Apple___Black_rot': 'Black rot',
    'Grape___Black_rot': 'Black rot',
    'Apple___Cedar_apple_rust': 'Cedar apple rust',
    'Apple___Healthy': 'Healthy',
    'Cherry_(including_sour)___Healthy': 'Healthy',
    'Corn_(maize)___Healthy': 'Healthy',
    'Grape___Healthy': 'Healthy',
    'Peach___Healthy': 'Healthy',
    'Pepper___bell___Healthy': 'Healthy',
    'Potato___Healthy': 'Healthy',
    'Strawberry___Healthy': 'Healthy',
    'Tomato___Healthy': 'Healthy',
    'Cherry_(including_sour)___Powdery_mildew': 'Powdery mildew',
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot': 'Cercospora leaf spot gray leaf spot',
    'Corn_(maize)___Common_rust_': 'Common rust',
    'Corn_(maize)___Northern_Leaf_Blight': 'Northern leaf blight',
    'Grape___Esca_(Black_Measles)': 'Esca (black measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Leaf blight (isariopsis leaf spot)',
    'Peach___Bacterial_spot': 'Bacterial spot',
    'Pepper___bell___Bacterial_spot': 'Bacterial spot',
    'Tomato___Bacterial_spot': 'Bacterial spot',
    'Potato___Early_blight': 'Early blight',
    'Tomato___Early_blight': 'Early blight',
    'Potato___Late_blight': 'Late blight',
    'Tomato___Late_blight': 'Late blight',
    'Strawberry___Leaf_scorch': 'Leaf scorch',
    'Tomato___Leaf_Mold': 'Leaf mold',
    'Tomato___Septoria_leaf_spot': 'Septoria leaf spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite': 'Spider mites two-spotted spider mite',
    'Tomato___Target_Spot': 'Target spot',
    'Tomato___Tomato_mosaic_virus': 'Tomato mosaic virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato yellow leaf curl virus'
}


import cv2
import numpy as np
from PIL import Image
import hashlib
from pathlib import Path


def calculate_blur_score(image_path):
    """
    Calculate blur score using Laplacian variance.
    Lower score = more blurry
    Typical threshold: 100-200
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return laplacian_var
    except Exception as e:
        print(f"Error calculating blur score for {image_path}: {e}")
        return None


def is_image_blurry(image_path, threshold=100.0):
    """
    Check if image is blurry based on threshold.
    """
    score = calculate_blur_score(image_path)
    if score is None:
        return False
    return score < threshold


def calculate_image_hash(image_path):
    """
    Calculate perceptual hash of image for duplicate detection.
    Uses average hash algorithm.
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Resize to 8x8
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Get pixel data
        pixels = list(img.getdata())
        
        # Calculate average
        avg = sum(pixels) / len(pixels)
        
        # Create hash
        hash_bits = ''.join('1' if pixel > avg else '0' for pixel in pixels)
        
        # Convert to hex
        hash_hex = hex(int(hash_bits, 2))[2:].zfill(16)
        
        return hash_hex
    except Exception as e:
        print(f"Error calculating hash for {image_path}: {e}")
        return None


def find_duplicate_images(image_paths):
    """
    Find duplicate images based on perceptual hash.
    Returns a dictionary mapping hash to list of image paths.
    """
    hash_dict = {}
    
    for image_path in image_paths:
        image_hash = calculate_image_hash(image_path)
        if image_hash:
            if image_hash not in hash_dict:
                hash_dict[image_hash] = []
            hash_dict[image_hash].append(image_path)
    
    # Return only hashes with duplicates
    duplicates = {k: v for k, v in hash_dict.items() if len(v) > 1}
    
    return duplicates