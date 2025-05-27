# plant_health_app/utils.py

# Ánh xạ loại cây từ tên bệnh
PLANT_TYPE_MAPPING = {
    "apple scab": "Apple",
    "cedar apple rust": "Apple",
    "black rot": "Grape",
    "bacterial spot": "Pepper",
    "cercospora leaf spot gray leaf spot": "Corn",
    "common rust": "Corn",
    "early blight": "Potato",
    "esca (black measles)": "Grape",
    "haunglongbing (citrus greening)": "Orange",
    "healthy": "Unknown",
    "late blight": "Tomato",
    "leaf blight (isariopsis leaf spot)": "Grape",
    "leaf mold": "Tomato",
    "leaf scorch": "Unknown",
    "northern leaf blight": "Corn",
    "powdery mildew": "Cherry",
    "septoria leaf spot": "Tomato",
    "spider mites two-spotted spider mite": "Tomato",
    "target spot": "Tomato",
    "tomato mosaic virus": "Tomato",
    "tomato yellow leaf curl virus": "Tomato",
}