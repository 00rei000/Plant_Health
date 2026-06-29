# Django imports
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login as auth_login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User, Group
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib import messages
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.http import JsonResponse, HttpResponse
from django.db.models import Count, Q, Avg
from django.db.models.functions import TruncMonth
from django.utils import timezone
from django.conf import settings
from django.core import serializers as django_serializers
from django.core.files import File
from django.urls import reverse
from django import forms
import time

# Python standard library imports
import os
import re
import csv
import json
import random
import difflib
import traceback
import threading
import zipfile
import mimetypes
import glob
import uuid
from datetime import datetime
from pathlib import Path

# Third-party imports
import requests
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import numpy as np
from PIL import Image
from ultralytics import YOLO
import django

# Local imports
from .utils import calculate_blur_score, calculate_image_hash
from .models import (
    Feedback, DiseaseLibrary, UserProfile, PredictionHistory, 
    Notification, DeletedObject, BlogPost, BlogComment,
    TrainingDataset, TrainingDatasetImage, ExportTask,
    PlantTypeModel, DiseaseModel, SegmentationModel
)

# Đọc group_classes.json
with open(os.path.join(settings.BASE_DIR, 'plant_health_app', 'notebook', 'plant_disease', 'group_classes.json'), 'r', encoding='utf-8') as f:
    GROUPS = json.load(f)

# Đọc class_names.json (thứ tự lớp phân loại loại cây - phải khớp với thứ tự khi huấn luyện plant type model)
PLANT_TYPE_CLASSES = None
try:
    with open(os.path.join(settings.BASE_DIR, 'plant_health_app', 'notebook', 'plant_type', 'class_names.json'), 'r', encoding='utf-8') as f:
        PLANT_TYPE_CLASSES = json.load(f)
except Exception as e:
    # fallback: use GROUPS keys order but warn (this may cause label-index mismatch with trained model)
    PLANT_TYPE_CLASSES = list(GROUPS.keys())
    print(f"Warning: could not load class_names.json, falling back to GROUPS keys order. Error: {e}")


def _normalize_name(s: str) -> str:
    if not s:
        return ''
    return re.sub(r'[^a-z0-9]', '', s.lower())


def normalize_plant_type(value: str) -> str:
    if not value:
        return value
    raw_value = value.strip()
    if raw_value.lower() == 'auto':
        return 'auto'
    normalized = _normalize_name(raw_value)
    if normalized == 'potatofield':
        return 'Potato'
    if normalized == 'potato':
        return 'Potato'
    return raw_value


def get_disease_details_dict(disease_name, plant_type):
    """Return a dict for disease details.

    Tries these strategies in order:
      1. exact name + exact plant_type (case-insensitive)
      2. exact name (case-insensitive)
      3. icontains name
      4. normalized exact match (remove non-alphanum)
      5. fuzzy match on existing names
    """
    # 1 & 2 & 3
    details = DiseaseLibrary.objects.filter(name__iexact=disease_name, plant_type__iexact=plant_type).first()
    if not details:
        details = DiseaseLibrary.objects.filter(name__iexact=disease_name).first()
    if not details:
        details = DiseaseLibrary.objects.filter(name__icontains=disease_name).first()

    # 4: normalized exact match
    if not details:
        try:
            norm_target = _normalize_name(disease_name)
            for obj in DiseaseLibrary.objects.all():
                if _normalize_name(obj.name) == norm_target:
                    details = obj
                    break
        except Exception:
            details = None

    # 5: fuzzy matching on names
    if not details:
        try:
            names = list(DiseaseLibrary.objects.values_list('name', flat=True))
            matches = difflib.get_close_matches(disease_name, names, n=1, cutoff=0.75)
            if not matches:
                # also try normalized names
                norm_map = { _normalize_name(n): n for n in names }
                nmatch = difflib.get_close_matches(_normalize_name(disease_name), list(norm_map.keys()), n=1, cutoff=0.85)
                if nmatch:
                    matched_name = norm_map[nmatch[0]]
                else:
                    matched_name = None
            else:
                matched_name = matches[0]

            if matched_name:
                details = DiseaseLibrary.objects.filter(name__iexact=matched_name).first()
                if details:
                    print(f"Info: fuzzy-matched disease '{disease_name}' -> '{matched_name}'")
        except Exception:
            details = None

    if details:
        image_url = getattr(details.image, 'url', None) or '/static/images/placeholder.jpg'
        return {
            'name': details.name,
            'plant_type': details.plant_type or plant_type,
            'description': details.description or 'Không có thông tin mô tả.',
            'symptoms': details.symptoms or 'Không có thông tin triệu chứng.',
            'treatment': details.treatment or 'Không có thông tin điều trị. Vui lòng liên hệ chuyên gia.',
            'image_url': image_url,
            'db_obj': details,
        }

    return {
        'name': disease_name,
        'plant_type': plant_type,
        'description': 'Không có thông tin mô tả.',
        'symptoms': 'Không có thông tin triệu chứng.',
        'treatment': 'Không có thông tin điều trị. Vui lòng liên hệ chuyên gia.',
        'image_url': '/static/images/placeholder.jpg',
        'db_obj': None,
    }

# Định nghĩa biến đổi ảnh
data_transforms = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Biến toàn cục để lưu trữ mô hình
disease_models = {}
plant_type_model = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load YOLO model for Potato leaf detection
YOLO_POTATO_MODEL = None
YOLO_POTATO_MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    'plant_health_app',
    'notebook',
    'preprocessing',
    'models',
    'yolo_potato.pt'
)
try:
    YOLO_POTATO_MODEL = YOLO(YOLO_POTATO_MODEL_PATH)
    print("Đã tải YOLO Potato model")
except Exception as e:
    print(f"Lỗi khi tải YOLO Potato model: {e}")

def load_plant_type_model():
    global plant_type_model
    if plant_type_model is None:
        # Try to load from database (default model)
        default_model = PlantTypeModel.objects.filter(
            is_default=True,
            is_active=True
        ).first()
        
        if default_model:
            # Load from database
            file_path = default_model.file_path
            if os.path.isabs(file_path):
                model_path = file_path
            else:
                model_path = os.path.join(settings.BASE_DIR, file_path)
                if not os.path.exists(model_path):
                    media_path = os.path.join(settings.MEDIA_ROOT, file_path)
                    if os.path.exists(media_path):
                        model_path = media_path
            num_classes = default_model.num_classes
            print(f"Loading Plant Type model from DB: {default_model.name}")
        else:
            # Fallback to old path
            model_path = os.path.join(
                settings.BASE_DIR, 
                'plant_health_app', 
                'notebook', 
                'plant_type', 
                'models', 
                'best', 
                'plant_type_model.pth'
            )
            num_classes = len(PLANT_TYPE_CLASSES)
            print(f"No default model in DB, using fallback path: {model_path}")
        
        # Sử dụng EfficientNet-B4 (khớp với model đã train)
        plant_type_model = efficientnet_b4(weights=None)
        plant_type_model.classifier[1] = nn.Linear(plant_type_model.classifier[1].in_features, num_classes)

        print(f"Đang kiểm tra đường dẫn: {model_path}")
        if not os.path.exists(model_path):
            print(f"File không tồn tại tại: {model_path}")
        try:
            plant_type_model.load_state_dict(torch.load(model_path, map_location=device))
            plant_type_model = plant_type_model.to(device)
            plant_type_model.eval()
            print(f"Đã tải mô hình EfficientNet-B4 phân loại cây từ {model_path}")
        except Exception as e:
            print(f"Lỗi khi tải mô hình phân loại cây: {e}")
            plant_type_model = None
    return plant_type_model

def load_disease_model(plant_type):
    global disease_models
    plant_type = normalize_plant_type(plant_type)
    
    # BƯỚC 1: ĐỌC DỮ LIỆU TỪ FILE JSON ĐỂ LẤY NHÃN ĐỘNG
    json_path = os.path.join(settings.BASE_DIR, 'plant_health_app', 'notebook', 'plant_disease', 'group_classes.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            groups_data = json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file group_classes.json: {e}")
        groups_data = {}

    # BƯỚC 2: TÌM BẢN GHI MÔ HÌNH TRONG DB
    default_model = DiseaseModel.objects.filter(
        plant_type__iexact=plant_type,  # Case-insensitive
        is_default=True,
        is_active=True
    ).first()
    
    if not default_model:
        default_model = DiseaseModel.objects.filter(plant_type__iexact=plant_type, is_active=True).last()

    # BƯỚC 3: XÁC ĐỊNH BỘ NHÃN (Dựa vào label_group hoặc dùng plant_type làm mặc định)
    if default_model and getattr(default_model, 'label_group', None):
        target_group_key = default_model.label_group
    else:
        target_group_key = plant_type
        
    class_names = groups_data.get(target_group_key, [])

    # BƯỚC 4: LOAD TRỌNG SỐ VÀO BIẾN CACHE
    if plant_type not in disease_models:
        if default_model:
            # Load from database - file_path is relative to BASE_DIR, not MEDIA_ROOT
            file_path = default_model.file_path
            if os.path.isabs(file_path):
                model_path = file_path
            else:
                model_path = os.path.join(settings.BASE_DIR, file_path)
                if not os.path.exists(model_path):
                    media_path = os.path.join(settings.MEDIA_ROOT, file_path)
                    if os.path.exists(media_path):
                        model_path = media_path
            
            # Ưu tiên lấy num_classes từ DB, nếu không có thì lấy số lượng class từ JSON
            num_classes = default_model.num_classes or len(class_names)
            print(f"Loading {plant_type} Disease model from DB: {default_model.name}")
            print(f"Model path: {model_path}")
        else:
            # Fallback to old path
            model_filename = f'{plant_type.lower()}_model.pth'
            model_path = os.path.join(
                settings.BASE_DIR,
                'plant_health_app',
                'notebook',
                'plant_disease',
                'models',
                'best',
                model_filename
            )
            if not os.path.exists(model_path):
                alt_filename = f'{plant_type.lower()}.pth'
                alt_path = os.path.join(
                    settings.BASE_DIR,
                    'plant_health_app',
                    'notebook',
                    'plant_disease',
                    'models',
                    'best',
                    alt_filename
                )
                if os.path.exists(alt_path):
                    model_path = alt_path
            
            num_classes = len(class_names)
            print(f"No default {plant_type} model in DB, using fallback: {model_path}")
        
        # Sử dụng EfficientNet-B4 (khớp với model đã train)
        model = efficientnet_b4(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        try:
            # Try to load a state_dict first
            state = torch.load(model_path, map_location=device)
            # If successful and state is a dict, try to load
            if isinstance(state, dict):
                try:
                    model.load_state_dict(state)
                except Exception as e:
                    print(f"Warning: load_state_dict shape mismatch for {plant_type}: {e}")
            model = model.to(device)
            model.eval()
            disease_models[plant_type] = model
            print(f"Đã tải mô hình bệnh cho {plant_type} từ {model_path}")
        except Exception as e:
            # If loading failed, check if the file is a placeholder/text explaining no model
            reason = str(e)
            placeholder_text = ''
            try:
                with open(model_path, 'r', encoding='utf-8', errors='ignore') as fh:
                    placeholder_text = fh.read(1024)
            except Exception:
                placeholder_text = ''

            if 'No model' in placeholder_text or 'No best model' in placeholder_text or 'Single class' in placeholder_text:
                # If plant has only a single disease class, create a dummy model that always predicts class 0
                if num_classes == 1:
                    class DummySingleClassModel(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.fc = torch.nn.Linear(1, 1)

                        def forward(self, x):
                            b = x.shape[0]
                            logits = torch.full((b, 1), -10.0, device=x.device)
                            logits[:, 0] = 10.0
                            return logits

                    dummy = DummySingleClassModel().to(device).eval()
                    disease_models[plant_type] = dummy
                    print(f"Detected placeholder model file for {plant_type}. Using DummySingleClassModel (always predicts the single class).")
                else:
                    print(f"Placeholder model file found for {plant_type} but expected {num_classes} classes. Setting disease model to None.")
                    disease_models[plant_type] = None
            else:
                print(f"Lỗi khi tải mô hình bệnh cho {plant_type}: {reason}")
                disease_models[plant_type] = None
                
    # QUAN TRỌNG: Trả về Tuple (model, class_names) thay vì chỉ mỗi model
    return disease_models.get(plant_type), class_names, target_group_key

def predict_disease(image_path, user_plant_type):
    try:
        def _run_disease_model(plant_key, pil_image):
            # 1. ĐÓN NHẬN CẢ MODEL VÀ NHÃN TỪ HÀM ĐÃ NÂNG CẤP
            disease_model, class_names, used_key = load_disease_model(plant_key)
            
            if disease_model is None:
                raise Exception(f"Không thể tải mô hình bệnh cho {plant_key}.")
            if not class_names:
                raise Exception(f"LỖI ÁNH XẠ: Database yêu cầu bộ nhãn có tên là '{used_key}', nhưng trong file group_classes.json KHÔNG CÓ chữ này!")

            tensor_image = data_transforms(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = disease_model(tensor_image)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                # 2. SỬ DỤNG BỘ NHÃN ĐỘNG (CLASS_NAMES) THAY VÌ GROUPS
                disease = class_names[predicted_idx.item()]
                disease_confidence = confidence.item() * 100

                print(f"Probabilities for {plant_key} diseases:")
                for i, prob in enumerate(probabilities[0]):
                    print(f"Class {class_names[i]}: {prob.item() * 100:.2f}%")

            return {
                'disease': disease,
                'confidence': disease_confidence,
                'plant_type': plant_key,
            }

        def _save_cropped_image(pil_image):
            crop_dir = os.path.join(settings.MEDIA_ROOT, 'potato_crops')
            os.makedirs(crop_dir, exist_ok=True)
            filename = f"potato_crop_{uuid.uuid4().hex}.jpg"
            abs_path = os.path.join(crop_dir, filename)
            pil_image.save(abs_path, format='JPEG')
            rel_path = os.path.join('potato_crops', filename).replace('\\', '/')
            return abs_path, f"{settings.MEDIA_URL}{rel_path}"

        image = Image.open(image_path).convert('RGB')
        user_plant_type = normalize_plant_type(user_plant_type)

        # Nếu user_plant_type là 'auto', dự đoán loại cây trước
        if user_plant_type.lower() == 'auto':
            plant_model = load_plant_type_model()
            if plant_model is None:
                raise Exception("Không thể tải mô hình phân loại cây.")
            tensor_image = data_transforms(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = plant_model(tensor_image)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                try:
                    plant_type = PLANT_TYPE_CLASSES[predicted_idx.item()]
                except Exception:
                    # Giữ nguyên fallback của bạn nếu GROUPS còn tồn tại ở đầu file
                    plant_type = list(GROUPS.keys())[predicted_idx.item()]
                    print("Warning: predicted index mapped using GROUPS keys as fallback; consider regenerating plant_classes.json to match training order.")
                plant_type = normalize_plant_type(plant_type)
                plant_confidence = confidence.item() * 100
                print(f"Dự đoán loại cây: {plant_type} ({plant_confidence:.2f}%)")
        else:
            plant_type = normalize_plant_type(user_plant_type)
            plant_confidence = None

        cropped_image_abs_path = None
        image_for_prediction = image

        if plant_type == 'Potato':
            cropped_image_for_display = image
            image_for_prediction = image

            if YOLO_POTATO_MODEL is not None:
                try:
                    results = YOLO_POTATO_MODEL(image_path, verbose=False)
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        confs = boxes.conf.cpu().numpy()
                        best_idx = confs.argmax()
                        xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy.tolist()
                        
                        # 1. CẮT ẢNH SẠCH (Nguyên bản) để đưa vào EfficientNet chẩn đoán bệnh
                        image_for_prediction = image.crop((x1, y1, x2, y2))
                        
                        # 2. VẼ BOUNDING BOX & MASK ĐỂ HIỂN THỊ LÊN WEB
                        plotted_array = results[0].plot() 
                        plotted_image = Image.fromarray(plotted_array[..., ::-1]) # Chuyển BGR sang RGB
                        cropped_image_for_display = plotted_image.crop((x1, y1, x2, y2))
                    else:
                        print("YOLO không phát hiện được lá khoai tây, sử dụng ảnh gốc.")
                except Exception as e:
                    print(f"YOLO detection error: {e}")

            # Lưu ảnh ĐÃ VẼ MASK vào thư mục potato_crops và Database
            cropped_image_abs_path, _ = _save_cropped_image(cropped_image_for_display)
        disease_result = _run_disease_model(plant_type, image_for_prediction)
        
        # Trả về nguyên bản 5 biến để không làm hỏng các hàm khác đang gọi predict_disease
        return disease_result['disease'], disease_result['confidence'], plant_type, plant_confidence, cropped_image_abs_path
    except Exception as e:
        raise Exception(f"Lỗi khi dự đoán: {str(e)}")

def is_admin(user):
    return user.is_authenticated and user.is_staff

def is_farmer(user):
    return user.is_authenticated and user.groups.filter(name='Farmer').exists()

def is_expert(user):
    return user.is_authenticated and user.groups.filter(name='Expert').exists()

def is_admin_or_expert(user):
    return user.is_authenticated and (user.is_staff or user.groups.filter(name='Expert').exists())

def is_farmer_or_expert(user):
    return user.is_authenticated and (
        user.groups.filter(name='Farmer').exists() or
        user.groups.filter(name='Expert').exists()
    )

def home(request):
    if request.method == 'GET':
        return render(request, 'home.html', {})
    return redirect('home')

def get_weather_icon(weather_main, weather_id, default_icon):
    """
    Return Bootstrap Icons class based on weather condition
    """
    # Rain conditions (500-531)
    if 500 <= weather_id <= 531 or weather_main == 'Rain':
        if weather_id >= 520:  # Heavy rain
            return 'cloud-rain-heavy-fill'
        return 'cloud-rain-fill'
    
    # Drizzle (300-321)
    elif 300 <= weather_id <= 321 or weather_main == 'Drizzle':
        return 'cloud-drizzle-fill'
    
    # Thunderstorm (200-232)
    elif 200 <= weather_id <= 232 or weather_main == 'Thunderstorm':
        return 'cloud-lightning-rain-fill'
    
    # Snow (600-622)
    elif 600 <= weather_id <= 622 or weather_main == 'Snow':
        return 'cloud-snow-fill'
    
    # Clouds (801-804)
    elif weather_main == 'Clouds':
        if weather_id == 801:  # Few clouds
            return 'cloud-sun-fill'
        elif weather_id == 802:  # Scattered clouds
            return 'cloud-sun-fill'
        elif weather_id == 803:  # Broken clouds
            return 'clouds-fill'
        else:  # Overcast
            return 'cloud-fill'
    
    # Clear (800)
    elif weather_main == 'Clear':
        # Check if day or night based on icon
        if 'n' in default_icon:
            return 'moon-stars-fill'
        return 'sun-fill'
    
    # Mist, Fog, Haze (701-781)
    elif weather_id >= 701 and weather_id < 800:
        return 'cloud-haze-fill'
    
    # Default
    return 'cloud-sun-fill'

# Danh sách thủ đô các quốc gia
CAPITAL_CITIES = {
    'hanoi': {'name': 'Hà Nội', 'country': 'Việt Nam', 'lat': 21.0285, 'lon': 105.8542},
    'tokyo': {'name': 'Tokyo', 'country': 'Nhật Bản', 'lat': 35.6762, 'lon': 139.6503},
    'sapporo': {'name': 'Sapporo', 'country': 'Nhật Bản (Hokkaido)', 'lat': 43.0642, 'lon': 141.3469},
    'beijing': {'name': 'Bắc Kinh', 'country': 'Trung Quốc', 'lat': 39.9042, 'lon': 116.4074},
    'seoul': {'name': 'Seoul', 'country': 'Hàn Quốc', 'lat': 37.5665, 'lon': 126.9780},
    'bangkok': {'name': 'Bangkok', 'country': 'Thái Lan', 'lat': 13.7563, 'lon': 100.5018},
    'singapore': {'name': 'Singapore', 'country': 'Singapore', 'lat': 1.3521, 'lon': 103.8198},
    'kualalumpur': {'name': 'Kuala Lumpur', 'country': 'Malaysia', 'lat': 3.1390, 'lon': 101.6869},
    'jakarta': {'name': 'Jakarta', 'country': 'Indonesia', 'lat': -6.2088, 'lon': 106.8456},
    'manila': {'name': 'Manila', 'country': 'Philippines', 'lat': 14.5995, 'lon': 120.9842},
    'newdelhi': {'name': 'New Delhi', 'country': 'Ấn Độ', 'lat': 28.6139, 'lon': 77.2090},
    'london': {'name': 'London', 'country': 'Anh', 'lat': 51.5074, 'lon': -0.1278},
    'paris': {'name': 'Paris', 'country': 'Pháp', 'lat': 48.8566, 'lon': 2.3522},
    'berlin': {'name': 'Berlin', 'country': 'Đức', 'lat': 52.5200, 'lon': 13.4050},
    'moscow': {'name': 'Moscow', 'country': 'Nga', 'lat': 55.7558, 'lon': 37.6173},
    'washington': {'name': 'Washington DC', 'country': 'Mỹ', 'lat': 38.9072, 'lon': -77.0369},
    'ottawa': {'name': 'Ottawa', 'country': 'Canada', 'lat': 45.4215, 'lon': -75.6972},
    'canberra': {'name': 'Canberra', 'country': 'Úc', 'lat': -35.2809, 'lon': 149.1300},
    'sydney': {'name': 'Sydney', 'country': 'Úc', 'lat': -33.8688, 'lon': 151.2093},
}

# Dịch mô tả thời tiết sang tiếng Việt
WEATHER_TRANSLATIONS = {
    'clear sky': 'Trời quang đãng',
    'few clouds': 'Ít mây',
    'scattered clouds': 'Mây rải rác',
    'broken clouds': 'Nhiều mây',
    'overcast clouds': 'U ám',
    'light rain': 'Mưa nhẹ',
    'moderate rain': 'Mưa vừa',
    'heavy rain': 'Mưa to',
    'light snow': 'Tuyết nhẹ',
    'snow': 'Có tuyết',
    'mist': 'Sương mù',
    'fog': 'Sương mù dày',
    'thunderstorm': 'Giông bão',
}

def get_weather(request):
    """
    API endpoint to get current weather and 8-day forecast
    Uses OpenWeatherMap API (free tier)
    """
    # Get city from request, default to Hanoi
    city_key = request.GET.get('city', 'hanoi').lower()
    city_data = CAPITAL_CITIES.get(city_key, CAPITAL_CITIES['hanoi'])
    
    lat = city_data['lat']
    lon = city_data['lon']
    
    # OpenWeatherMap API key
    api_key = '5855609682a3bec2ac0b6bfe3d7ceea8'
    
    # Try to get weather data
    try:
        # Current weather endpoint
        current_url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=vi'
        current_response = requests.get(current_url, timeout=5)
        
        # 8-day forecast endpoint (One Call API)
        forecast_url = f'http://api.openweathermap.org/data/2.5/forecast/daily?lat={lat}&lon={lon}&cnt=8&appid={api_key}&units=metric&lang=vi'
        forecast_response = requests.get(forecast_url, timeout=5)
        
        if current_response.status_code == 200:
            data = current_response.json()
            
            # Get weather condition
            weather_main = data['weather'][0]['main']
            weather_id = data['weather'][0]['id']
            weather_desc = data['weather'][0]['description']
            
            # Translate to Vietnamese
            weather_desc_vn = WEATHER_TRANSLATIONS.get(weather_desc.lower(), weather_desc.capitalize())
            
            # Get custom icon based on weather condition
            weather_icon = get_weather_icon(weather_main, weather_id, data['weather'][0]['icon'])
            
            # Calculate dew point using Magnus formula
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            a = 17.27
            b = 237.7
            alpha = ((a * temp) / (b + temp)) + (humidity / 100.0)
            dew_point = (b * alpha) / (a - alpha)
            
            # Process 8-day forecast if available
            forecast_days = []
            try:
                # Try One Call API 3.0 (free tier includes 8 days)
                onecall_url = f'https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,alerts&appid={api_key}&units=metric&lang=vi'
                onecall_response = requests.get(onecall_url, timeout=5)
                
                if onecall_response.status_code == 200:
                    forecast_data = onecall_response.json()
                    for day in forecast_data.get('daily', [])[:8]:
                        day_desc = day['weather'][0]['description']
                        day_desc_vn = WEATHER_TRANSLATIONS.get(day_desc.lower(), day_desc.capitalize())
                        
                        forecast_days.append({
                            'date': datetime.fromtimestamp(day['dt']).strftime('%d/%m'),
                            'day_name': datetime.fromtimestamp(day['dt']).strftime('%a'),
                            'temp_min': round(day['temp']['min'], 1),
                            'temp_max': round(day['temp']['max'], 1),
                            'temp_day': round(day['temp']['day'], 1),
                            'weather': day['weather'][0]['main'],
                            'weather_description': day_desc_vn,
                            'icon': get_weather_icon(day['weather'][0]['main'], day['weather'][0]['id'], day['weather'][0]['icon']),
                            'humidity': day['humidity'],
                            'rain': day.get('rain', 0),
                        })
                else:
                    # Fallback: Use 5-day/3-hour forecast and aggregate by day
                    forecast_5day_url = f'http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=vi'
                    forecast_5day_response = requests.get(forecast_5day_url, timeout=5)
                    
                    if forecast_5day_response.status_code == 200:
                        forecast_5day_data = forecast_5day_response.json()
                        daily_data = {}
                        
                        for item in forecast_5day_data['list']:
                            date_str = datetime.fromtimestamp(item['dt']).strftime('%d/%m')
                            if date_str not in daily_data:
                                daily_data[date_str] = {
                                    'temps': [],
                                    'weather': item['weather'][0],
                                    'humidity': [],
                                    'rain': 0,
                                    'dt': item['dt']
                                }
                            daily_data[date_str]['temps'].append(item['main']['temp'])
                            daily_data[date_str]['humidity'].append(item['main']['humidity'])
                            daily_data[date_str]['rain'] += item.get('rain', {}).get('3h', 0)
                        
                        for date_str, day_data in list(daily_data.items())[:8]:
                            day_desc = day_data['weather']['description']
                            day_desc_vn = WEATHER_TRANSLATIONS.get(day_desc.lower(), day_desc.capitalize())
                            
                            forecast_days.append({
                                'date': date_str,
                                'day_name': datetime.fromtimestamp(day_data['dt']).strftime('%a'),
                                'temp_min': round(min(day_data['temps']), 1),
                                'temp_max': round(max(day_data['temps']), 1),
                                'temp_day': round(sum(day_data['temps']) / len(day_data['temps']), 1),
                                'weather': day_data['weather']['main'],
                                'weather_description': day_desc_vn,
                                'icon': get_weather_icon(day_data['weather']['main'], day_data['weather']['id'], day_data['weather']['icon']),
                                'humidity': round(sum(day_data['humidity']) / len(day_data['humidity'])),
                                'rain': round(day_data['rain'], 1),
                            })
            except Exception as e:
                print(f"Forecast error: {e}")
            
            weather_data = {
                'success': True,
                'city': city_data['name'],
                'country': city_data['country'],
                'temperature': round(data['main']['temp'], 1),
                'feels_like': round(data['main']['feels_like'], 1),
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': round(data['wind']['speed'], 1),
                'wind_direction': data['wind']['deg'],
                'clouds': data['clouds']['all'],
                'weather': weather_main,
                'weather_description': weather_desc_vn,
                'weather_icon': weather_icon,
                'visibility': round(data.get('visibility', 0) / 1000, 1),
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M'),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M'),
                'rain': data.get('rain', {}).get('1h', 0),
                'dew_point': round(dew_point, 1),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'forecast': forecast_days
            }
            
            return JsonResponse(weather_data)
        else:
            # API error - return fallback data
            return JsonResponse({
                'success': True,
                'city': city_data['name'],
                'country': city_data['country'],
                'temperature': 19,
                'feels_like': 19,
                'humidity': 66,
                'pressure': 1019,
                'wind_speed': 0.8,
                'wind_direction': 337,
                'clouds': 0,
                'weather': 'Clear',
                'weather_description': 'Trời quang đãng',
                'weather_icon': 'sun-fill',
                'visibility': 10.0,
                'sunrise': '05:52',
                'sunset': '17:16',
                'rain': 0,
                'dew_point': 13,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'forecast': [],
                'demo_mode': True
            })
            
    except Exception as e:
        print(f"Weather API error: {e}")
        # Return fallback data on error
        return JsonResponse({
            'success': True,
            'city': city_data['name'],
            'country': city_data['country'],
            'temperature': 19,
            'feels_like': 19,
            'humidity': 66,
            'pressure': 1019,
            'wind_speed': 0.8,
            'wind_direction': 337,
            'clouds': 0,
            'weather': 'Clear',
            'weather_description': 'Trời quang đãng',
            'weather_icon': 'sun-fill',
            'visibility': 10.0,
            'sunrise': '05:52',
            'sunset': '17:16',
            'rain': 0,
            'dew_point': 13,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'forecast': [],
            'demo_mode': True
        })
        
        
        condition = choice(weather_conditions)
        
        return JsonResponse({
            'success': True,
            'city': 'Hà Nội',
            'country': 'VN',
            'temperature': randint(18, 32),
            'feels_like': randint(20, 35),
            'humidity': randint(60, 90),
            'pressure': randint(1010, 1018),
            'wind_speed': round(randint(5, 20) / 10, 1),
            'wind_direction': randint(0, 359),
            'clouds': randint(10, 90),
            'weather': condition[0],
            'weather_description': condition[1],
            'weather_icon': condition[2],
            'visibility': randint(5, 10),
            'sunrise': '05:45',
            'sunset': '17:30',
            'rain': condition[3],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'demo_mode': True,
            'error_message': str(e)
        })

def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)
            if user.groups.filter(name='Farmer').exists():
                return redirect('farmer_dashboard')
            elif user.groups.filter(name='Expert').exists():
                return redirect('expert_dashboard')
            elif user.is_staff:
                return redirect('admin_dashboard')
            else:
                return redirect('home')
        else:
            messages.error(request, 'Tên đăng nhập hoặc mật khẩu không đúng.')
    elif request.method == 'GET':
        if request.GET.get('next'):
            messages.warning(request, 'Vui lòng đăng nhập để sử dụng tính năng này.')
        return render(request, 'login.html')
    return render(request, 'home.html')

def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        user_type = request.POST.get('user_type')

        if not user_type or user_type not in ['Farmer', 'Expert']:
            messages.error(request, 'Vui lòng chọn vai trò hợp lệ.')
        elif password != confirm_password:
            messages.error(request, 'Mật khẩu không khớp.')
        elif User.objects.filter(username=username).exists():
            messages.error(request, 'Tên đăng nhập đã tồn tại.')
        elif User.objects.filter(email=email).exists():
            messages.error(request, 'Email đã được sử dụng.')
        else:
            try:
                user = User.objects.create_user(username=username, email=email, password=password)
                user.save()
                UserProfile.objects.create(user=user)
                group = Group.objects.get(name=user_type)
                user.groups.add(group)
                # Thông báo cho admin khi có người dùng mới đăng ký
                admins = User.objects.filter(is_staff=True)
                for admin in admins:
                    Notification.objects.create(
                        recipient=admin,
                        message=f"Người dùng mới {username} đã đăng ký với vai trò {user_type}.",
                        link=reverse('admin_dashboard')
                    )
                messages.success(request, 'Đăng ký thành công! Vui lòng đăng nhập.')
                return redirect('login')
            except Group.DoesNotExist:
                messages.error(request, 'Nhóm không tồn tại. Vui lòng liên hệ quản trị viên.')
            except Exception as e:
                messages.error(request, f'Lỗi khi đăng ký: {str(e)}')
    elif request.method == 'GET':
        return render(request, 'register.html')
    return redirect('register')

@login_required
def prediction(request):
    """Bước 1: Upload ảnh & AI dự đoán (Gợi ý)"""
    
    # 1. KHAI BÁO CỨNG DANH SÁCH 9 CÂY CHUẨN (Dành cho trang upload)
    BASE_PLANT_TYPES = [
        'Apple', 'Cherry', 'Corn', 'Grape', 
        'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato'
    ]

    if request.method == 'POST':
        image = request.FILES.get('plant_image')
        user_plant_type = request.POST.get('plant_type', 'auto')

        if not image:
            messages.error(request, 'Vui lòng chọn một ảnh để dự đoán.')
            return render(request, 'prediction.html', {
                'PLANT_TYPE_CLASSES': BASE_PLANT_TYPES,  # ĐÃ SỬA: Chỉ truyền 9 cây
                'GROUPS': GROUPS
            })

        valid_extensions = ['.jpg', '.jpeg', '.png']
        ext = os.path.splitext(image.name)[1].lower()
        if ext not in valid_extensions:
            messages.error(request, 'Vui lòng tải file ảnh (.jpg, .jpeg, .png).')
            return render(request, 'prediction.html', {
                'PLANT_TYPE_CLASSES': BASE_PLANT_TYPES,  # ĐÃ SỬA: Chỉ truyền 9 cây
                'GROUPS': GROUPS
            })

        try:
            # Tạo record với contribution_type mặc định là CONSULTING (chỉ xem)
            plant_image = PredictionHistory(
                image=image, 
                user=request.user,
                contribution_type=PredictionHistory.ContributionType.CONSULTING,
                approval_status=PredictionHistory.ApprovalStatus.PENDING  # Mặc định PENDING
            )
            plant_image.save()

            image_path = plant_image.image.path
            image_path = image_path.replace('\\', '/').replace('rain/', 'train/')

            start_time = time.perf_counter()
            
            # AI dự đoán
            disease, disease_confidence, plant_type, plant_confidence, cropped_image_path = predict_disease(
                image_path,
                user_plant_type
            )
            
            end_time = time.perf_counter()
            latency = end_time - start_time

            plant_image.disease = disease
            plant_image.confidence = disease_confidence
            plant_image.plant_type = plant_type
            plant_image.inference_latency = latency
            if cropped_image_path and os.path.exists(cropped_image_path):
                with open(cropped_image_path, 'rb') as cropped_file:
                    plant_image.cropped_image.save(
                        os.path.basename(cropped_image_path),
                        File(cropped_file),
                        save=False
                    )
            plant_image.save()

            disease_details = get_disease_details_dict(disease, plant_type)
            if not disease_details.get('db_obj'):
                messages.warning(request, f'Không tìm thấy thông tin chi tiết trong DB cho bệnh {disease} trên {plant_type}. Hiển thị thông tin mặc định.')

            print(f"Predicted: disease={disease}, disease_confidence={disease_confidence:.2f}%, plant_type={plant_type}, plant_confidence={plant_confidence if plant_confidence else 'N/A'}")
            
            # Kiểm tra confidence thấp
            LOW_CONFIDENCE_THRESHOLD = 70.0
            is_low_confidence = disease_confidence < LOW_CONFIDENCE_THRESHOLD
            
            # Chuyển sang trang kết quả với form xác nhận
            context = {
                'prediction_id': plant_image.id,
                'image_url': plant_image.image.url,
                'cropped_image_url': plant_image.cropped_image.url if plant_image.cropped_image else None,
                'prediction': {
                    'disease': disease,
                    'disease_confidence': disease_confidence,
                    'plant_type': plant_type,
                    'plant_confidence': plant_confidence,
                    'disease_details': disease_details,
                    'is_low_confidence': is_low_confidence,
                },
                'PLANT_TYPE_CLASSES': BASE_PLANT_TYPES,  # ĐÃ SỬA: Chỉ truyền 9 cây
                'GROUPS': GROUPS,  # Dict {plant_type: [diseases]}
                'all_diseases': sorted(list(set([d for diseases in GROUPS.values() for d in diseases]))),  # Tất cả bệnh
            }
            return render(request, 'result.html', context)

        except Exception as e:
            messages.error(request, f'Lỗi khi xử lý ảnh: {str(e)}')
            traceback.print_exc()
            return render(request, 'prediction.html', {
                'PLANT_TYPE_CLASSES': BASE_PLANT_TYPES,  # ĐÃ SỬA: Chỉ truyền 9 cây
                'GROUPS': GROUPS
            })

    # GET request - hiển thị trang ban đầu
    return render(request, 'prediction.html', {
        'PLANT_TYPE_CLASSES': BASE_PLANT_TYPES,  # ĐÃ SỬA: Chỉ truyền 9 cây chuẩn
        'GROUPS': GROUPS
    })

@login_required
def result(request):
    if request.method == 'POST':
        image = request.FILES.get('plant_image')
        user_plant_type = request.POST.get('plant_type', 'auto')
        if image:
            valid_extensions = ['.jpg', '.jpeg', '.png']
            ext = os.path.splitext(image.name)[1].lower()
            if ext not in valid_extensions:
                messages.error(request, 'Vui lòng tải file ảnh (.jpg, .jpeg, .png).')
                return render(request, 'result.html', {
                    'PLANT_TYPE_CLASSES': PLANT_TYPE_CLASSES,
                    'GROUPS': GROUPS,
                    'all_diseases': sorted(list(set([d for diseases in GROUPS.values() for d in diseases])))
                })

            try:
                # Tạo record với contribution_type mặc định là CONSULTING
                plant_image = PredictionHistory(
                    image=image, 
                    user=request.user,
                    contribution_type=PredictionHistory.ContributionType.CONSULTING,
                    approval_status=PredictionHistory.ApprovalStatus.PENDING
                )
                plant_image.save()

                image_path = plant_image.image.path
                image_path = image_path.replace('\\', '/').replace('rain/', 'train/')
                disease, disease_confidence, plant_type, plant_confidence, cropped_image_path = predict_disease(
                    image_path,
                    user_plant_type
                )
                plant_image.disease = disease
                plant_image.confidence = disease_confidence
                plant_image.plant_type = plant_type
                if cropped_image_path and os.path.exists(cropped_image_path):
                    with open(cropped_image_path, 'rb') as cropped_file:
                        plant_image.cropped_image.save(
                            os.path.basename(cropped_image_path),
                            File(cropped_file),
                            save=False
                        )
                plant_image.save()

                disease_details = get_disease_details_dict(disease, plant_type)
                if not disease_details.get('db_obj'):
                    messages.warning(request, f'Không tìm thấy thông tin chi tiết trong DB cho bệnh {disease} trên {plant_type}. Hiển thị thông tin mặc định.')

                admins = User.objects.filter(is_staff=True)
                for admin in admins:
                    Notification.objects.create(
                        recipient=admin,
                        message=f"Người dùng {request.user.username} đã thực hiện dự đoán bệnh: {disease}.",
                        link=reverse('admin_dashboard')
                    )
                
                # Kiểm tra confidence thấp
                LOW_CONFIDENCE_THRESHOLD = 70.0
                is_low_confidence = disease_confidence < LOW_CONFIDENCE_THRESHOLD

                context = {
                    'prediction_id': plant_image.id,
                    'image_url': plant_image.image.url,
                    'cropped_image_url': plant_image.cropped_image.url if plant_image.cropped_image else None,
                    'prediction': {
                        'disease': disease,
                        'disease_confidence': disease_confidence,
                        'plant_type': plant_type,
                        'plant_confidence': plant_confidence,
                        'disease_details': disease_details,
                        'is_low_confidence': is_low_confidence,
                    },
                    'PLANT_TYPE_CLASSES': PLANT_TYPE_CLASSES,
                    'GROUPS': GROUPS,
                    'all_diseases': sorted(list(set([d for diseases in GROUPS.values() for d in diseases])))
                }
                return render(request, 'result.html', context)
            except Exception as e:
                messages.error(request, f'Lỗi khi xử lý ảnh: {str(e)}')
                import traceback
                traceback.print_exc()
                return render(request, 'result.html', {
                    'PLANT_TYPE_CLASSES': PLANT_TYPE_CLASSES,
                    'GROUPS': GROUPS,
                    'all_diseases': sorted(list(set([d for diseases in GROUPS.values() for d in diseases])))
                })

    elif request.method == 'GET':
        latest_image = PredictionHistory.objects.filter(user=request.user).order_by('-uploaded_at').first()
        if latest_image:
            if not latest_image.disease:
                image_path = latest_image.image.path
                image_path = image_path.replace('\\', '/').replace('rain/', 'train/')
                disease, disease_confidence, plant_type, plant_confidence, cropped_image_path = predict_disease(
                    image_path,
                    'auto'
                )
                latest_image.disease = disease
                latest_image.confidence = disease_confidence
                latest_image.plant_type = plant_type
                if cropped_image_path and os.path.exists(cropped_image_path):
                    with open(cropped_image_path, 'rb') as cropped_file:
                        latest_image.cropped_image.save(
                            os.path.basename(cropped_image_path),
                            File(cropped_file),
                            save=False
                        )
                latest_image.save()
            else:
                disease_confidence = latest_image.confidence

            disease_details = get_disease_details_dict(latest_image.disease, latest_image.plant_type)
            if not disease_details.get('db_obj'):
                messages.warning(request, f'Không tìm thấy thông tin chi tiết trong DB cho bệnh {latest_image.disease} trên {latest_image.plant_type}. Hiển thị thông tin mặc định.')
            
            # Kiểm tra confidence thấp
            LOW_CONFIDENCE_THRESHOLD = 70.0
            is_low_confidence = disease_confidence < LOW_CONFIDENCE_THRESHOLD

            context = {
                'prediction_id': latest_image.id,
                'image_url': latest_image.image.url,
                'cropped_image_url': latest_image.cropped_image.url if latest_image.cropped_image else None,
                'prediction': {
                    'disease': latest_image.disease,
                    'disease_confidence': disease_confidence,
                    'plant_type': latest_image.plant_type,
                    'plant_confidence': None,
                    'disease_details': disease_details,
                    'is_low_confidence': is_low_confidence,
                },
                'PLANT_TYPE_CLASSES': PLANT_TYPE_CLASSES,
                'GROUPS': GROUPS,
                'all_diseases': sorted(list(set([d for diseases in GROUPS.values() for d in diseases])))
            }
            return render(request, 'result.html', context)
        messages.info(request, 'Bạn chưa upload ảnh nào.')
        return redirect('prediction')
    return redirect('result')

@login_required
def user_confirm_prediction(request, prediction_id):
    """Bước 2: User xác nhận - Chọn 'Chỉ xem' hoặc 'Đóng góp' (KHÔNG BẮT BUỘC gán nhãn)"""
    if request.method != 'POST':
        return redirect('prediction')
    
    prediction = get_object_or_404(PredictionHistory, id=prediction_id, user=request.user)
    
    action = request.POST.get('action')  # 'consulting' hoặc 'contributing'
    
    if action == 'consulting':
        # User chỉ muốn xem kết quả -> Không cần admin duyệt
        prediction.contribution_type = PredictionHistory.ContributionType.CONSULTING
        prediction.approval_status = PredictionHistory.ApprovalStatus.ACCEPTED  # Tự động duyệt
        prediction.save()
        
        messages.success(request, 'Cảm ơn bạn đã sử dụng dịch vụ! Kết quả đã được lưu vào lịch sử.')
        return redirect('prediction_history')
    
    elif action == 'contributing':
        # User muốn đóng góp -> Cần gán nhãn (hoặc đồng ý với AI)
        user_disease_choice = request.POST.get('user_disease_choice')  # 'agree_with_ai' hoặc 'custom'
        
        if user_disease_choice == 'agree_with_ai':
            # User đồng ý với AI prediction
            prediction.contribution_type = PredictionHistory.ContributionType.CONTRIBUTING
            prediction.user_agreed_with_ai = True
            prediction.user_confirmed_disease = prediction.disease  # Sử dụng nhãn AI
            prediction.approval_status = PredictionHistory.ApprovalStatus.PENDING  # Chờ admin duyệt
            prediction.save()
            
            messages.success(request, 'Cảm ơn bạn đã đồng ý với kết quả AI! Ảnh của bạn đang chờ admin phê duyệt.')
            
        elif user_disease_choice == 'custom':
            # User tự gán nhãn
            user_plant_type = request.POST.get('user_plant_type', '').strip()
            user_disease = request.POST.get('user_disease', '').strip()
            
            if not user_plant_type or not user_disease:
                messages.error(request, '❌ Vui lòng chọn/nhập đầy đủ Loại cây và Tên bệnh!')
                return redirect('result')
            
            # Chuẩn hóa tên (Title Case nếu user gõ tay)
            user_plant_type = user_plant_type.title()
            user_disease = user_disease.title()
            
            prediction.contribution_type = PredictionHistory.ContributionType.CONTRIBUTING
            prediction.user_agreed_with_ai = False
            prediction.plant_type = user_plant_type  # Cập nhật plant_type theo user
            prediction.user_confirmed_disease = user_disease  # Lưu nhãn user gán
            prediction.approval_status = PredictionHistory.ApprovalStatus.PENDING  # Chờ admin duyệt
            prediction.save()
            
            messages.success(request, f'Cảm ơn bạn đã đóng góp! Ảnh với nhãn "{user_disease}" trên cây "{user_plant_type}" đang chờ admin phê duyệt.')
        
        else:
            messages.error(request, 'Vui lòng chọn cách gán nhãn (Đồng ý với AI hoặc Tự nhập)!')
            return redirect('result')
        
        # Gửi thông báo cho admin
        admins = User.objects.filter(is_staff=True)
        for admin in admins:
            Notification.objects.create(
                recipient=admin,
                message=f"{request.user.username} đã đóng góp ảnh mới (ID: {prediction.id}). Vui lòng kiểm duyệt!",
                link=reverse('admin_moderation')
            )
        
        return redirect('prediction_history')
    
    else:
        messages.error(request, 'Hành động không hợp lệ!')
        return redirect('prediction')

@login_required
def prediction_history(request):
    if request.method == 'GET':
        # CHỈ LẤY ACTIVE PREDICTIONS
        predictions = PredictionHistory.objects.filter(user=request.user, is_active=True).order_by('-uploaded_at')

        date_from = request.GET.get('date_from', '')
        date_to = request.GET.get('date_to', '')
        plant_type = request.GET.get('plant_type', '')
        disease = request.GET.get('disease', '')

        if date_from:
            try:
                date_from = timezone.datetime.strptime(date_from, '%Y-%m-%d')
                predictions = predictions.filter(uploaded_at__gte=date_from)
            except ValueError:
                messages.error(request, 'Định dạng ngày bắt đầu không hợp lệ.')
        if date_to:
            try:
                date_to = timezone.datetime.strptime(date_to, '%Y-%m-%d')
                date_to = date_to + timezone.timedelta(days=1)
                predictions = predictions.filter(uploaded_at__lte=date_to)
            except ValueError:
                messages.error(request, 'Định dạng ngày kết thúc không hợp lệ.')

        if plant_type:
            predictions = predictions.filter(plant_type__iexact=plant_type)
        if disease:
            predictions = predictions.filter(disease__icontains=disease)

        # CHỈ LẤY ACTIVE CHO DROPDOWN
        plant_types = PredictionHistory.objects.filter(user=request.user, is_active=True).values('plant_type').distinct()
        diseases = PredictionHistory.objects.filter(user=request.user, is_active=True).values('disease').distinct()

        paginator = Paginator(predictions, 9)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        return render(request, 'prediction_history.html', {
            'plant_images': page_obj,
            'plant_types': plant_types,
            'diseases': diseases,
            'date_from': date_from,
            'date_to': date_to,
            'selected_plant_type': plant_type,
            'selected_disease': disease,
            'GROUPS': list(GROUPS.keys())
        })
    return redirect('prediction_history')

@login_required
def prediction_detail(request, prediction_id):
    prediction = get_object_or_404(PredictionHistory, id=prediction_id, user=request.user)
    disease_details = get_disease_details_dict(prediction.disease, prediction.plant_type)
    if not disease_details.get('db_obj'):
        messages.warning(request, f'Không tìm thấy thông tin chi tiết trong DB cho bệnh {prediction.disease} trên {prediction.plant_type}. Hiển thị thông tin mặc định.')

    # Kiểm tra confidence thấp
    LOW_CONFIDENCE_THRESHOLD = 70.0
    is_low_confidence = prediction.confidence < LOW_CONFIDENCE_THRESHOLD

    context = {
        'prediction_id': prediction.id,
        'image_url': prediction.image.url, 
        'cropped_image_url': prediction.cropped_image.url if prediction.cropped_image else None,
        'prediction': {
            'disease': prediction.disease,
            'disease_confidence': prediction.confidence,
            'plant_type': prediction.plant_type,
            'plant_confidence': None,
            'disease_details': disease_details,
            'is_low_confidence': is_low_confidence,
        },
        'PLANT_TYPE_CLASSES': PLANT_TYPE_CLASSES,
        'GROUPS': GROUPS,
        'all_diseases': sorted(list(set([d for diseases in GROUPS.values() for d in diseases])))
    }
    return render(request, 'result.html', context)


@login_required
def plant_image_detail(request, image_id):
    """Show details for a PredictionHistory (renamed from PlantImage for backward compatibility).

    Allow viewing if:
      - the image owner (uploader),
      - staff, or
      - any expert (so experts can inspect images for questions).
    """
    prediction = get_object_or_404(PredictionHistory, id=image_id)

    # Permission: owner, staff, or expert may view
    allowed = False
    if request.user.is_authenticated:
        if prediction.user and prediction.user == request.user:
            allowed = True
        if request.user.is_staff:
            allowed = True
        if request.user.groups.filter(name='Expert').exists():
            allowed = True

    if not allowed:
        messages.error(request, 'Bạn không có quyền xem ảnh này.')
        return redirect(request.META.get('HTTP_REFERER') or reverse('home'))

    # Build context similar to prediction_detail
    disease_details = get_disease_details_dict(prediction.disease, prediction.plant_type)
    
    # Kiểm tra confidence thấp
    LOW_CONFIDENCE_THRESHOLD = 70.0
    is_low_confidence = prediction.confidence < LOW_CONFIDENCE_THRESHOLD

    context = {
        'prediction_id': prediction.id,
        'image_url': prediction.image.url,
        'prediction': {
            'disease': prediction.disease,
            'disease_confidence': prediction.confidence,
            'plant_type': prediction.plant_type,
            'plant_confidence': None,
            'disease_details': disease_details,
            'is_low_confidence': is_low_confidence,
        },
        'PLANT_TYPE_CLASSES': PLANT_TYPE_CLASSES,
        'GROUPS': GROUPS,
        'all_diseases': sorted(list(set([d for diseases in GROUPS.values() for d in diseases])))
    }
    return render(request, 'result.html', context)


@login_required
def delete_prediction_history(request, prediction_id):
    if request.method == 'POST':
        prediction = get_object_or_404(PredictionHistory, id=prediction_id, user=request.user)
        # SOFT DELETE - không xóa hẳn
        prediction.soft_delete(user=request.user)
        messages.success(request, 'Dự đoán đã được xóa thành công.')

        # Kiểm tra xem yêu cầu có phải là Ajax không
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'status': 'success', 'message': 'Dự đoán đã được xóa.'})
        else:
            return redirect('prediction_history')
    return JsonResponse({'status': 'error', 'message': 'Yêu cầu không hợp lệ.'}, status=400)


@login_required
def undo_delete(request, deleted_id):
    """Restore a previously backed-up DeletedObject. Only staff or the user who deleted may restore.

    Expects POST. Returns JSON.
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Yêu cầu không hợp lệ.'}, status=400)

    deleted = get_object_or_404(DeletedObject, id=deleted_id)
    # permission: staff or the user who deleted
    if not (request.user.is_staff or (deleted.deleted_by and deleted.deleted_by == request.user)):
        return JsonResponse({'status': 'error', 'message': 'Bạn không có quyền hoàn tác thao tác này.'}, status=403)

    try:
        objs = list(django_serializers.deserialize('json', deleted.data))
        restored = []
        for des_obj in objs:
            instance = des_obj.object
            # save will re-create the instance (uses original PK)
            instance.save()
            restored.append({'model': f"{instance._meta.app_label}.{instance._meta.model_name}", 'pk': instance.pk})
        # remove backup after successful restore
        deleted.delete()
        return JsonResponse({'status': 'success', 'message': 'Đã hoàn tác thành công.', 'restored': restored})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@login_required
@user_passes_test(is_farmer)
def farmer_dashboard(request):
    if request.method == 'GET':
        user = request.user
        
        # Lấy blog posts của user (only active)
        my_blog_posts = BlogPost.objects.filter(author=user, is_active=True).order_by('-created_at')[:5]
        
        # Biểu đồ theo loại cây: Số lượng dự đoán theo plant_type
        prediction_by_plant_type = PredictionHistory.objects.filter(user=user).values('plant_type').annotate(count=Count('id')).order_by('-count')[:10]
        plant_type_labels = [item['plant_type'] or 'Không xác định' for item in prediction_by_plant_type]
        plant_type_data = [item['count'] for item in prediction_by_plant_type]

        # Biểu đồ theo bệnh: Số lượng theo disease
        prediction_by_disease = PredictionHistory.objects.filter(user=user).values('disease').annotate(count=Count('id')).order_by('-count')[:10]
        disease_labels = [item['disease'] or 'Không xác định' for item in prediction_by_disease]
        disease_data = [item['count'] for item in prediction_by_disease]

        context = {
            'title': 'Bảng điều khiển Nông dân',
            'my_blog_posts': my_blog_posts,
            # Hai biểu đồ chính
            'plant_type_chart': {
                'labels': plant_type_labels,
                'data': plant_type_data,
                'title': 'Số lượng dự đoán theo loại cây'
            },
            'disease_chart': {
                'labels': disease_labels,
                'data': disease_data,
                'title': 'Số lượng dự đoán theo bệnh'
            },
            'total_predictions': sum(plant_type_data),  # Tổng để hiển thị
        }
        return render(request, 'farmer_dashboard.html', context)
    return redirect('farmer_dashboard')

@login_required
@user_passes_test(is_expert)
def expert_dashboard(request):
    if request.method == 'GET':
        user = request.user
        
        # Lấy blog posts của user (only active)
        my_blog_posts = BlogPost.objects.filter(author=user, is_active=True).order_by('-created_at')[:5]
        
        # Biểu đồ theo loại cây: Số lượng dự đoán theo plant_type
        prediction_by_plant_type = PredictionHistory.objects.filter(user=user).values('plant_type').annotate(count=Count('id')).order_by('-count')[:10]
        plant_type_labels = [item['plant_type'] or 'Không xác định' for item in prediction_by_plant_type]
        plant_type_data = [item['count'] for item in prediction_by_plant_type]

        # Biểu đồ theo bệnh: Số lượng theo disease
        prediction_by_disease = PredictionHistory.objects.filter(user=user).values('disease').annotate(count=Count('id')).order_by('-count')[:10]
        disease_labels = [item['disease'] or 'Không xác định' for item in prediction_by_disease]
        disease_data = [item['count'] for item in prediction_by_disease]

        context = {
            'my_blog_posts': my_blog_posts,
            # Hai biểu đồ chính
            'plant_type_chart': {
                'labels': plant_type_labels,
                'data': plant_type_data,
                'title': 'Số lượng dự đoán theo loại cây'
            },
            'disease_chart': {
                'labels': disease_labels,
                'data': disease_data,
                'title': 'Số lượng dự đoán theo bệnh'
            },
            'total_predictions': sum(plant_type_data),  # Tổng để hiển thị
            'total_blog_posts': my_blog_posts.count(),
        }
        return render(request, 'expert_dashboard.html', context)
    
    return redirect('expert_dashboard')

@login_required
def feedback_view(request):
    if request.method == 'POST':
        feedback_text = request.POST.get('feedback')
        if feedback_text:
            feedback = Feedback(feedback_text=feedback_text)
            if request.user.is_authenticated:
                feedback.user = request.user
            feedback.save()
            # Thông báo cho admin khi có feedback mới
            admins = User.objects.filter(is_staff=True)
            for admin in admins:
                Notification.objects.create(
                    recipient=admin,
                    message=f"Người dùng {request.user.username} đã gửi phản hồi: {feedback_text[:50]}...",
                    link=reverse('admin_dashboard')
                )
            messages.success(request, 'Phản hồi của bạn đã được gửi!')
            if request.user.is_authenticated:
                if request.user.groups.filter(name='Farmer').exists():  # Sửa: user -> request.user
                    return redirect('farmer_dashboard')
                elif request.user.groups.filter(name='Expert').exists():  # Sửa: user -> request.user
                    return redirect('expert_dashboard')
                elif request.user.is_staff:  # Sửa: user -> request.user
                    return redirect('admin_dashboard')
            return redirect('home')
        else:
            messages.error(request, 'Vui lòng nhập nội dung phản hồi.')
    elif request.method == 'GET':
        return render(request, 'feedback.html')
    return redirect('feedback_view')

@login_required
def logout_view(request):
    logout(request)
    messages.success(request, 'Bạn đã đăng xuất thành công.')
    return redirect('login')

login_required
@user_passes_test(lambda u: u.is_superuser)
def admin_dashboard(request):
    # Quản lý người dùng - CHỈ LẤY ACTIVE
    users_list = User.objects.filter(profile__is_active=True).order_by('-date_joined')
    for user in users_list:
        try:
            user.profile_info = user.profile
        except UserProfile.DoesNotExist:
            user.profile_info = None

    user_paginator = Paginator(users_list, 5)
    user_page_number = request.GET.get('user_page', 1)
    user_page_obj = user_paginator.get_page(user_page_number)

    # Lịch sử dự đoán - CHỈ LẤY ACTIVE
    predictions = PredictionHistory.objects.filter(is_active=True).order_by('-uploaded_at')
    prediction_paginator = Paginator(predictions, 5)
    prediction_page_number = request.GET.get('prediction_page', 1)
    prediction_page_obj = prediction_paginator.get_page(prediction_page_number)

    # Phản hồi - CHỈ LẤY ACTIVE
    feedbacks = Feedback.objects.filter(is_active=True).order_by('-created_at')
    feedback_paginator = Paginator(feedbacks, 5)
    feedback_page_number = request.GET.get('feedback_page', 1)
    feedback_page_obj = feedback_paginator.get_page(feedback_page_number)

    # Thống kê dự đoán theo loại cây - CHỈ ACTIVE
    prediction_by_plant_type = PredictionHistory.objects.filter(is_active=True).values('plant_type').annotate(count=Count('id')).order_by('-count')[:10]
    plant_type_labels = [item['plant_type'] for item in prediction_by_plant_type]
    plant_type_data = [item['count'] for item in prediction_by_plant_type]

    # Thống kê dự đoán theo bệnh - CHỈ ACTIVE
    prediction_by_disease = PredictionHistory.objects.filter(is_active=True).values('disease').annotate(count=Count('id')).order_by('-count')[:10]
    disease_labels = [item['disease'] for item in prediction_by_disease]
    disease_data = [item['count'] for item in prediction_by_disease]

    # Thống kê blog posts - CHỈ ACTIVE
    total_blog_posts = BlogPost.objects.filter(is_active=True).count()
    approved_posts = BlogPost.objects.filter(status='approved', is_active=True).count()
    pending_posts = BlogPost.objects.filter(status='pending', is_active=True).count()
    
    # Blog posts by month - CHỈ ACTIVE
    blog_posts_by_month = BlogPost.objects.filter(is_active=True).annotate(month=TruncMonth('created_at')).values('month').annotate(count=Count('id')).order_by('month')
    blog_month_labels = [item['month'].strftime('%Y-%m') for item in blog_posts_by_month]
    blog_month_data = [item['count'] for item in blog_posts_by_month]
    
    # Pending blog posts for quick approval - CHỈ ACTIVE
    pending_blog_posts = BlogPost.objects.filter(status='pending', is_active=True).order_by('-created_at')[:5]

    avg_latency_obj = PredictionHistory.objects.filter(is_active=True, inference_latency__isnull=False).aggregate(Avg('inference_latency'))
    avg_latency = avg_latency_obj['inference_latency__avg'] or 0.0

    context = {
        'users': user_page_obj,
        'prediction_history': prediction_page_obj,
        'feedbacks': feedback_page_obj,
        'current_date': timezone.now(),
        'pending_blog_posts': pending_blog_posts,
        'avg_latency': avg_latency,
        # Thống kê cho biểu đồ
        'plant_type_chart': {
            'labels': plant_type_labels,
            'data': plant_type_data,
            'title': 'Thống kê dự đoán theo loại cây'
        },
        'disease_chart': {
            'labels': disease_labels,
            'data': disease_data,
            'title': 'Thống kê dự đoán theo bệnh'
        },
        'blog_chart': {
            'labels': blog_month_labels,
            'data': blog_month_data,
            'title': 'Thống kê bài viết theo thời gian'
        },
        'blog_stats_chart': {
            'labels': ['Đã duyệt', 'Chờ duyệt', 'Tổng số'],
            'data': [approved_posts, pending_posts, total_blog_posts],
            'title': 'Thống kê trạng thái bài viết'
        }
    }
    return render(request, 'admin_dashboard.html', context)

@login_required
@user_passes_test(is_admin)
def delete_user(request, user_id):
    if request.method == 'POST':
        user = get_object_or_404(User, id=user_id)
        if user.is_superuser:
            # AJAX-aware response
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'message': 'Không thể xóa superuser.'}, status=400)
            messages.error(request, 'Không thể xóa superuser.')
        else:
            try:
                # SOFT DELETE - xóa profile của user
                if hasattr(user, 'profile'):
                    user.profile.soft_delete(user=request.user)
                if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                    return JsonResponse({'status': 'success', 'message': f'Người dùng {user.username} đã được vô hiệu hóa.'})
                messages.success(request, f'Người dùng {user.username} đã được vô hiệu hóa.')
            except Exception as e:
                if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                    return JsonResponse({'status': 'error', 'message': f'Lỗi khi xóa người dùng: {str(e)}'}, status=500)
                messages.error(request, f'Lỗi khi xóa người dùng: {str(e)}')
        return redirect('admin_dashboard')
    elif request.method == 'GET':
        user = get_object_or_404(User, id=user_id)
        return render(request, 'confirm_delete_user.html', {'user': user})
    return redirect('admin_dashboard')

@login_required
@user_passes_test(is_admin)
def delete_prediction(request, image_id):
    """Soft delete a prediction record (AJAX/POST only)."""
    if request.method == 'POST':
        prediction = get_object_or_404(PredictionHistory, id=image_id)
        try:
            # SOFT DELETE - không xóa file ảnh
            prediction.soft_delete(user=request.user)
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'status': 'success', 'message': 'Ảnh dự đoán đã được xóa.'})
            messages.success(request, 'Ảnh dự đoán đã được xóa.')
        except Exception as e:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'message': f'Lỗi khi xóa ảnh: {str(e)}'}, status=500)
            messages.error(request, f'Lỗi khi xóa ảnh: {str(e)}')
        return redirect('admin_dashboard')
    # Only POST allowed
    return redirect('admin_dashboard')

@login_required
@user_passes_test(is_admin)
def edit_feedback(request, feedback_id):
    feedback = get_object_or_404(Feedback, id=feedback_id)
    if request.method == 'POST':
        feedback_text = request.POST.get('feedback_text')
        if feedback_text:
            feedback.feedback_text = feedback_text
            feedback.save()
            messages.success(request, 'Phản hồi đã được cập nhật.')
            return redirect('admin_dashboard')
        else:
            messages.error(request, 'Vui lòng nhập nội dung phản hồi.')
    elif request.method == 'GET':
        return render(request, 'edit_feedback.html', {
            'title': 'Chỉnh sửa phản hồi',
            'feedback': feedback,
        })
    return redirect('admin_dashboard')

@login_required
@user_passes_test(is_admin)
def delete_feedback(request, feedback_id):
    if request.method == 'POST':
        feedback = get_object_or_404(Feedback, id=feedback_id)
        try:
            # SOFT DELETE
            feedback.soft_delete(user=request.user)
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'status': 'success', 'message': 'Phản hồi đã được xóa.'})
            messages.success(request, 'Phản hồi đã được xóa.')
        except Exception as e:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'message': f'Lỗi khi xóa phản hồi: {str(e)}'}, status=500)
            messages.error(request, f'Lỗi khi xóa phản hồi: {str(e)}')
        return redirect('admin_dashboard')
    elif request.method == 'GET':
        feedback = get_object_or_404(Feedback, id=feedback_id)
        return render(request, 'confirm_delete_feedback.html', {'feedback': feedback})
    return redirect('admin_dashboard')

@login_required
@user_passes_test(lambda u: u.is_superuser)
def manage(request):
    if request.method == 'GET':
        selected_table = request.GET.get('table', 'users')
        search_query = request.GET.get('search', '')

        if selected_table == 'users':
            if search_query:
                users_list = User.objects.filter(
                    Q(username__icontains=search_query) | Q(email__icontains=search_query)
                ).order_by('-date_joined')
            else:
                users_list = User.objects.all().order_by('-date_joined')
            for user in users_list:
                try:
                    user.profile_info = user.profile
                except UserProfile.DoesNotExist:
                    user.profile_info = None
            user_paginator = Paginator(users_list, 20)
            user_page_number = request.GET.get('user_page', 1)
            users_page = user_paginator.get_page(user_page_number)
            data = users_page
        elif selected_table == 'prediction_history':
            if search_query:
                prediction_history = PredictionHistory.objects.filter(
                    Q(disease__icontains=search_query) | Q(user__username__icontains=search_query)
                ).order_by('-uploaded_at')
            else:
                prediction_history = PredictionHistory.objects.all().order_by('-uploaded_at')
            prediction_paginator = Paginator(prediction_history, 20)
            prediction_page_number = request.GET.get('prediction_page', 1)
            prediction_page = prediction_paginator.get_page(prediction_page_number)
            data = prediction_page
        else:
            if search_query:
                feedbacks = Feedback.objects.filter(
                    Q(feedback_text__icontains=search_query) | Q(user__username__icontains=search_query)
                ).order_by('-created_at')
            else:
                feedbacks = Feedback.objects.all().order_by('-created_at')
            feedback_paginator = Paginator(feedbacks, 20)
            feedback_page_number = request.GET.get('feedback_page', 1)
            feedback_page = feedback_paginator.get_page(feedback_page_number)
            data = feedback_page

        return render(request, 'manage.html', {
            'title': 'Quản Lý Hệ Thống',
            'selected_table': selected_table,
            'users': users_page if selected_table == 'users' else None,
            'prediction_history': prediction_page if selected_table == 'prediction_history' else None,
            'feedbacks': feedback_page if selected_table == 'feedback' else None,
            'search_query': search_query,
        })
    return redirect('manage')

@login_required
def upload_plant_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image = request.FILES['image']
            
            # Kiểm tra định dạng file
            ext = os.path.splitext(image.name)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.gif']:
                return JsonResponse({
                    'success': False,
                    'error': 'Định dạng ảnh không hỗ trợ. Vui lòng chọn JPG, PNG hoặc GIF.'
                })

            segmentation_model = None
            if YOLO_POTATO_MODEL is not None:
                segmentation_model, _ = SegmentationModel.objects.get_or_create(
                    file_path=YOLO_POTATO_MODEL_PATH,
                    defaults={
                        'name': 'YOLO Potato Segmentation',
                        'description': 'YOLO model for potato leaf segmentation.',
                        'is_default': True,
                        'is_active': True,
                    }
                )

            # Lưu ảnh vào PredictionHistory (chưa có dự đoán)
            prediction = PredictionHistory(
                user=request.user,
                image=image,
                uploaded_at=timezone.now(),
                segmentation_model=segmentation_model
            )
            prediction.save()

            segment_data = []
            if YOLO_POTATO_MODEL is not None:
                try:
                    results = YOLO_POTATO_MODEL(prediction.image.path, verbose=False)
                    masks = getattr(results[0], 'masks', None)
                    if masks is not None and masks.xy is not None:
                        for idx, mask in enumerate(masks.xy):
                            points = [[int(x), int(y)] for x, y in mask.tolist()]
                            if len(points) >= 3:
                                segment_data.append({
                                    'id': idx,
                                    'points': points,
                                })
                except Exception as e:
                    print(f"YOLO segmentation error: {e}")

            prediction.segment_data = {'segments': segment_data}
            prediction.save(update_fields=['segment_data'])

            return JsonResponse({
                'success': True,
                'id': prediction.id,
                'image_url': prediction.image.url,
                'uploaded_at': prediction.uploaded_at.strftime('%d/%m/%Y %H:%M'),
                'segment_data': segment_data
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Lỗi server: {str(e)}'
            })
    return JsonResponse({
        'success': False,
        'error': 'Yêu cầu không hợp lệ.'
    })


@login_required
def select_segment(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Yêu cầu không hợp lệ.'}, status=400)

    try:
        image_id = request.POST.get('image_id')
        selected_segment = request.POST.get('selected_segment')
        if not image_id or not selected_segment:
            return JsonResponse({'success': False, 'error': 'Thiếu dữ liệu vùng chọn.'}, status=400)

        prediction = get_object_or_404(PredictionHistory, id=image_id, user=request.user)
        selected_payload = json.loads(selected_segment)

        segment_data = prediction.segment_data or {}
        if isinstance(segment_data, list):
            segment_data = {'segments': segment_data}
        segment_data['selected'] = selected_payload
        prediction.segment_data = segment_data
        prediction.save(update_fields=['segment_data'])

        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'success': True})
        return redirect('prediction')
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@login_required
def disease_library(request):
    # Lấy tham số từ GET
    search_query = request.GET.get('search', '')
    plant_type_filter = request.GET.get('plant_type_filter', '')
    sort_order = request.GET.get('sort_order', '-created_at')  # Mặc định sắp xếp mới nhất

    # Lọc danh sách bệnh - CHỈ LẤY ACTIVE
    diseases = DiseaseLibrary.objects.filter(is_active=True).exclude(name__iexact='healthy')
    if search_query:
        diseases = diseases.filter(Q(name__icontains=search_query) | Q(plant_type__icontains=search_query))
    if plant_type_filter:
        diseases = diseases.filter(plant_type__iexact=plant_type_filter)

    # Sắp xếp
    if sort_order not in ['created_at', '-created_at', 'name']:
        sort_order = '-created_at'  # Bảo vệ trước giá trị không hợp lệ
    diseases = diseases.order_by(sort_order)

    # Lấy danh sách loại cây duy nhất cho dropdown - CHỈ ACTIVE
    plant_types = DiseaseLibrary.objects.filter(is_active=True).exclude(plant_type__isnull=True).values_list('plant_type', flat=True).distinct()

    # Phân trang
    paginator = Paginator(diseases, 9)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Gán image_url
    for disease in page_obj:
        if disease.image:
            disease.image_url = f'{settings.MEDIA_URL}{disease.image.name}'
        else:
            disease.image_url = '/static/images/placeholder.jpg'

    return render(request, 'disease_library.html', {
        'diseases': page_obj,
        'is_expert_or_admin': is_admin_or_expert(request.user),
        'search_query': search_query,
        'plant_type_filter': plant_type_filter,
        'sort_order': sort_order,
        'plant_types': plant_types,
    })

@login_required
@user_passes_test(is_admin_or_expert)
def update_disease(request, disease_id):
    disease = get_object_or_404(DiseaseLibrary, id=disease_id)
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        symptoms = request.POST.get('symptoms')
        treatment = request.POST.get('treatment')
        image = request.FILES.get('image')
        if name and description and symptoms and treatment:
            disease.name = name
            disease.description = description
            disease.symptoms = symptoms
            disease.treatment = treatment
            if image:
                disease.image = image
            disease.save()
            messages.success(request, 'Đã cập nhật bệnh thành công.')
        else:
            messages.error(request, 'Vui lòng điền đầy đủ thông tin.')
        return redirect('disease_library')
    return render(request, 'update_disease.html', {
        'disease': disease
    })

@login_required
@user_passes_test(is_admin_or_expert)
def delete_disease(request, disease_id):
    disease = get_object_or_404(DiseaseLibrary, id=disease_id)
    if request.method == 'POST':
        disease_name = disease.name
        # SOFT DELETE
        disease.soft_delete(user=request.user)
        # AJAX-aware
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'status': 'success', 'message': f'Đã xóa bệnh "{disease_name}" thành công.'})
        messages.success(request, f'Đã xóa bệnh "{disease_name}" thành công.')
        return redirect('disease_library')
    # Nếu không phải POST, trả về lỗi 405 (Method Not Allowed)
    return redirect('disease_library')  # Hoặc có thể raise Http404 nếu cần

@login_required
@user_passes_test(is_admin_or_expert)
def add_disease(request):
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        name = request.POST.get('name')
        description = request.POST.get('description')
        symptoms = request.POST.get('symptoms')
        treatment = request.POST.get('treatment')
        image = request.FILES.get('image')
        
        # Kiểm tra các trường bắt buộc
        if name and description and symptoms and treatment:
            try:
                # Kiểm tra định dạng ảnh
                if image:
                    ext = os.path.splitext(image.name)[1].lower()
                    if ext not in ['.jpg', '.jpeg', '.png']:
                        messages.error(request, 'Vui lòng tải file ảnh (.jpg, .jpeg, .png).')
                        return render(request, 'add_disease.html', {
                            'form_data': request.POST,
                        })
                
                # Tạo bản ghi mới trong DiseaseLibrary
                disease = DiseaseLibrary.objects.create(
                    name=name,
                    description=description,
                    symptoms=symptoms,
                    treatment=treatment,
                    created_by=request.user
                )
                if image:
                    disease.image = image
                    disease.save()
                
                # Gửi thông báo cho admin
                admins = User.objects.filter(is_staff=True)
                for admin in admins:
                    Notification.objects.create(
                        recipient=admin,
                        message=f"Bệnh mới {name} đã được thêm bởi {request.user.username}.",
                        link=reverse('disease_library')
                    )
                messages.success(request, 'Đã thêm bệnh mới.')
                return redirect('disease_library')
            except Exception as e:
                messages.error(request, f'Lỗi khi lưu bệnh: {str(e)}')
        else:
            missing_fields = []
            if not name: missing_fields.append('tên bệnh')
            if not description: missing_fields.append('mô tả')
            if not symptoms: missing_fields.append('triệu chứng')
            if not treatment: missing_fields.append('cách điều trị')
            messages.error(request, f'Vui lòng điền đầy đủ thông tin. Thiếu: {", ".join(missing_fields)}')
        
        # Nếu có lỗi, trả về template với dữ liệu form
        return render(request, 'add_disease.html', {
            'form_data': request.POST,
        })
    
    # Xử lý GET: Hiển thị form thêm bệnh
    return render(request, 'add_disease.html', {
        'form_data': None,
    })

class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['email']

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['age', 'phone_number', 'nationality']

@login_required
def profile(request):
    try:
        profile = request.user.profile
    except UserProfile.DoesNotExist:
        UserProfile.objects.create(user=request.user)
        profile = request.user.profile

    is_expert = request.user.groups.filter(name='Expert').exists()
    is_farmer = request.user.groups.filter(name='Farmer').exists()
    is_staff = request.user.is_staff

    if request.method == 'POST':
        profile_form = UserProfileForm(request.POST, instance=profile)
        email_form = ProfileUpdateForm(request.POST, instance=request.user)
        if profile_form.is_valid() and email_form.is_valid():
            profile_form.save()
            email_form.save()
            messages.success(request, 'Thông tin hồ sơ đã được cập nhật thành công!')
            return redirect('profile')
        else:
            errors = {**profile_form.errors, **email_form.errors}
            messages.error(request, f'Lỗi khi cập nhật thông tin: {errors}')
    elif request.method == 'GET':
        profile_form = UserProfileForm(instance=profile)
        email_form = ProfileUpdateForm(instance=request.user)
        return render(request, 'profile.html', {
            'profile_form': profile_form,
            'email_form': email_form,
            'is_expert': is_expert,
            'is_farmer': is_farmer,
            'is_staff': is_staff,
        })
    return redirect('profile')

@login_required
def change_password(request):
    if request.method == 'POST':
        password_form = PasswordChangeForm(request.user, request.POST)
        if password_form.is_valid():
            user = password_form.save()
            update_session_auth_hash(request, user)
            messages.success(request, 'Mật khẩu đã được thay đổi thành công!')
            return redirect('change_password')
        else:
            messages.error(request, 'Lỗi khi đổi mật khẩu. Vui lòng kiểm tra lại.')
    elif request.method == 'GET':
        password_form = PasswordChangeForm(request.user)
        is_expert = request.user.groups.filter(name='Expert').exists()
        is_farmer = request.user.groups.filter(name='Farmer').exists()
        is_staff = request.user.is_staff
        return render(request, 'change_password.html', {
            'password_form': password_form,
            'is_expert': is_expert,
            'is_farmer': is_farmer,
            'is_staff': is_staff,
        })
    return redirect('change_password')

@login_required
def get_notifications(request):
    try:
        notifications = Notification.objects.filter(recipient=request.user).order_by('-created_at')[:10]
        data = [{
            'id': n.id,
            'message': n.message,
            'link': n.link or '',
            'is_read': n.is_read,
            'created_at': n.created_at.strftime('%Y-%m-%d %H:%M:%S')
        } for n in notifications]
        unread_count = Notification.objects.filter(recipient=request.user, is_read=False).count()
        return JsonResponse({'notifications': data, 'unread_count': unread_count})
    except Exception as e:
        print(f"Error in get_notifications: {str(e)}")
        return JsonResponse({'notifications': [], 'unread_count': 0, 'error': str(e)}, status=500)

@login_required
def mark_notification_read(request, notification_id):
    notification = get_object_or_404(Notification, id=notification_id, recipient=request.user)
    notification.is_read = True
    notification.save()
    return JsonResponse({'status': 'success'})

@login_required
def all_notifications(request):
    notifications = Notification.objects.filter(recipient=request.user).order_by('-created_at')
    return render(request, 'notifications.html', {'notifications': notifications})


# ==================== BLOG VIEWS ====================

@login_required
def blog_list(request):
    """Display list of approved blog posts."""
    # Get filter parameters
    visibility_filter = request.GET.get('visibility', '')
    author_filter = request.GET.get('author', '')
    
    # Base queryset - only approved and active posts
    posts = BlogPost.objects.filter(status='approved', is_active=True)
    
    # Filter by visibility
    if visibility_filter:
        posts = posts.filter(visibility=visibility_filter)
    
    # Filter by author
    if author_filter:
        posts = posts.filter(author__username__icontains=author_filter)
    
    # Filter posts user can view
    viewable_posts = []
    for post in posts:
        if post.can_view(request.user):
            viewable_posts.append(post)
    
    # Pagination
    paginator = Paginator(viewable_posts, 10)
    page_number = request.GET.get('page')
    try:
        page_obj = paginator.get_page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.get_page(1)
    except EmptyPage:
        page_obj = paginator.get_page(paginator.num_pages)
    
    return render(request, 'blog_list.html', {
        'page_obj': page_obj,
        'visibility_filter': visibility_filter,
        'author_filter': author_filter,
    })


@login_required
def blog_detail(request, post_id):
    """Display blog post detail with comments."""
    post = get_object_or_404(BlogPost, id=post_id, is_active=True)
    
    # Check permission
    if not post.can_view(request.user):
        messages.error(request, 'Bạn không có quyền xem bài viết này.')
        return redirect('blog_list')
    
    # Increment views count
    post.views_count += 1
    post.save()
    
    # Get active comments only
    comments = post.comments.filter(is_active=True)
    
    # Handle comment submission
    if request.method == 'POST':
        content = request.POST.get('content', '').strip()
        if content:
            BlogComment.objects.create(
                post=post,
                author=request.user,
                content=content
            )
            
            # Create notification for post author
            if post.author != request.user:
                Notification.objects.create(
                    recipient=post.author,
                    message=f"{request.user.username} đã bình luận vào bài viết của bạn: {post.title}",
                    link=f"/blog/{post.id}/"
                )
            
            messages.success(request, 'Bình luận đã được thêm.')
            return redirect('blog_detail', post_id=post.id)
        else:
            messages.error(request, 'Nội dung bình luận không được để trống.')
    
    return render(request, 'blog_detail.html', {
        'post': post,
        'comments': comments,
    })


@login_required
def blog_create(request):
    """Create new blog post."""
    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        visibility = request.POST.get('visibility', 'public')
        image = request.FILES.get('image')
        allowed_viewers = request.POST.getlist('allowed_viewers')
        
        if not title or not content:
            messages.error(request, 'Tiêu đề và nội dung không được để trống.')
            return render(request, 'blog_create.html', {
                'users': User.objects.filter(is_active=True).exclude(id=request.user.id),
            })
        
        post = BlogPost.objects.create(
            author=request.user,
            title=title,
            content=content,
            visibility=visibility,
            image=image,
            status='pending'
        )
        
        # Add allowed viewers if visibility is 'specific'
        if visibility == 'specific' and allowed_viewers:
            post.allowed_viewers.set(User.objects.filter(id__in=allowed_viewers))
        
        # Notify admins about new post pending approval
        admins = User.objects.filter(is_staff=True)
        for admin in admins:
            Notification.objects.create(
                recipient=admin,
                message=f"{request.user.username} đã tạo bài viết mới cần duyệt: {title}",
                link=f"/blog/pending/"
            )
        
        messages.success(request, 'Bài viết đã được tạo và đang chờ admin duyệt.')
        return redirect('my_blog_posts')
    
    # GET request
    users = User.objects.filter(is_active=True).exclude(id=request.user.id)
    return render(request, 'blog_create.html', {'users': users})


@login_required
def blog_edit(request, post_id):
    """Edit existing blog post."""
    post = get_object_or_404(BlogPost, id=post_id)
    
    # Check permission
    if post.author != request.user and not request.user.is_staff:
        messages.error(request, 'Bạn không có quyền chỉnh sửa bài viết này.')
        return redirect('blog_detail', post_id=post.id)
    
    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        visibility = request.POST.get('visibility', 'public')
        image = request.FILES.get('image')
        allowed_viewers = request.POST.getlist('allowed_viewers')
        
        if not title or not content:
            messages.error(request, 'Tiêu đề và nội dung không được để trống.')
            return render(request, 'blog_edit.html', {
                'post': post,
                'users': User.objects.filter(is_active=True).exclude(id=request.user.id),
            })
        
        post.title = title
        post.content = content
        post.visibility = visibility
        if image:
            post.image = image
        
        # Set status back to pending if content changed
        if post.author == request.user:
            post.status = 'pending'
        
        post.save()
        
        # Update allowed viewers
        if visibility == 'specific' and allowed_viewers:
            post.allowed_viewers.set(User.objects.filter(id__in=allowed_viewers))
        else:
            post.allowed_viewers.clear()
        
        messages.success(request, 'Bài viết đã được cập nhật.')
        return redirect('blog_detail', post_id=post.id)
    
    users = User.objects.filter(is_active=True).exclude(id=request.user.id)
    return render(request, 'blog_edit.html', {
        'post': post,
        'users': users,
    })


@login_required
def blog_delete(request, post_id):
    """Delete blog post (soft delete)."""
    post = get_object_or_404(BlogPost, id=post_id, is_active=True)
    
    # Check permission
    if post.author != request.user and not request.user.is_staff:
        messages.error(request, 'Bạn không có quyền xóa bài viết này.')
        return redirect('blog_detail', post_id=post.id)
    
    if request.method == 'POST':
        post.soft_delete(user=request.user)
        messages.success(request, 'Bài viết đã được xóa.')
        return redirect('blog_list')
    
    return redirect('blog_detail', post_id=post.id)


@login_required
def my_blog_posts(request):
    """Display user's own blog posts."""
    posts = BlogPost.objects.filter(author=request.user, is_active=True)
    
    # Pagination
    paginator = Paginator(posts, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'my_blog_posts.html', {'page_obj': page_obj})


@login_required
@user_passes_test(is_admin)
def blog_pending(request):
    """Admin view for pending blog posts."""
    posts = BlogPost.objects.filter(status='pending', is_active=True).order_by('-created_at')
    
    # Pagination
    paginator = Paginator(posts, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'blog_pending.html', {'page_obj': page_obj})


@login_required
@user_passes_test(is_admin)
def blog_approve(request, post_id):
    """Approve blog post."""
    post = get_object_or_404(BlogPost, id=post_id)
    
    if request.method == 'POST':
        post.status = 'approved'
        post.approved_by = request.user
        post.approved_at = timezone.now()
        post.save()
        
        # Notify author
        Notification.objects.create(
            recipient=post.author,
            message=f"Bài viết '{post.title}' của bạn đã được duyệt.",
            link=f"/blog/{post.id}/"
        )
        
        # Check if AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Content-Type') == 'application/json':
            return JsonResponse({'status': 'success', 'message': 'Bài viết đã được duyệt.'})
        
        messages.success(request, 'Bài viết đã được duyệt.')
        return redirect('blog_pending')
    
    return redirect('blog_pending')


@login_required
@user_passes_test(is_admin)
def blog_reject(request, post_id):
    """Reject blog post."""
    post = get_object_or_404(BlogPost, id=post_id)
    
    if request.method == 'POST':
        reason = request.POST.get('reason', '').strip()
        post.status = 'rejected'
        post.rejection_reason = reason if reason else None  # Save rejection reason
        post.save()
        
        # Notify author
        message = f"Bài viết '{post.title}' của bạn đã bị từ chối."
        if reason:
            message += f" Lý do: {reason}"
        
        Notification.objects.create(
            recipient=post.author,
            message=message,
            link=f"/my-blog/"
        )
        
        # Check if AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Content-Type') == 'application/json':
            return JsonResponse({'status': 'success', 'message': 'Bài viết đã bị từ chối.'})
        
        messages.success(request, 'Bài viết đã bị từ chối.')
        return redirect('blog_pending')
    
    return redirect('blog_pending')


@login_required
def blog_comment_delete(request, comment_id):
    """Delete blog comment (soft delete)."""
    comment = get_object_or_404(BlogComment, id=comment_id, is_active=True)
    post_id = comment.post.id
    
    # Check permission
    if comment.author != request.user and not request.user.is_staff:
        messages.error(request, 'Bạn không có quyền xóa bình luận này.')
        return redirect('blog_detail', post_id=post_id)
    
    if request.method == 'POST':
        comment.soft_delete(user=request.user)
        messages.success(request, 'Bình luận đã được xóa.')
    
    return redirect('blog_detail', post_id=post_id)


@login_required
@login_required
@user_passes_test(is_admin)
def admin_moderation(request):
    """Bước 3: Trang admin phê duyệt ảnh đóng góp"""
    # Lấy tất cả ảnh đang chờ duyệt
    pending_predictions = PredictionHistory.objects.filter(
        contribution_type=PredictionHistory.ContributionType.CONTRIBUTING,
        approval_status=PredictionHistory.ApprovalStatus.PENDING
    ).select_related('user', 'reviewed_by').order_by('-uploaded_at')
    
    # Phân trang
    paginator = Paginator(pending_predictions, 20)
    page = request.GET.get('page')
    
    try:
        predictions = paginator.page(page)
    except PageNotAnInteger:
        predictions = paginator.page(1)
    except EmptyPage:
        predictions = paginator.page(paginator.num_pages)
    
    context = {
        'predictions': predictions,
        'PLANT_TYPE_CLASSES': PLANT_TYPE_CLASSES,  # Danh sách loại cây cho admin
        'GROUPS': GROUPS,  # Dict {plant_type: [diseases]}
        'all_diseases': sorted(list(set([d for diseases in GROUPS.values() for d in diseases]))),
    }
    return render(request, 'admin_moderation.html', context)


@login_required
@user_passes_test(is_admin)
def approve_contribution(request, prediction_id):
    """Admin phê duyệt ảnh đóng góp - CÓ THỂ SỬA NHÃN trước khi duyệt"""
    if request.method != 'POST':
        return redirect('admin_moderation')
    
    prediction = get_object_or_404(PredictionHistory, id=prediction_id)
    
    # Admin có thể sửa nhãn trước khi approve
    admin_plant_type = request.POST.get('admin_plant_type', '').strip()
    admin_disease = request.POST.get('admin_disease', '').strip()
    
    # Lưu nhãn cũ để hiển thị thông báo
    old_plant_type = prediction.plant_type
    old_disease = prediction.user_confirmed_disease
    label_changed = False
    
    # Nếu admin sửa nhãn -> cập nhật (tự động title() để chuẩn hóa)
    if admin_plant_type:
        prediction.plant_type = admin_plant_type.title()
        label_changed = True
    if admin_disease:
        prediction.user_confirmed_disease = admin_disease.title()
        label_changed = True
    
    # Sử dụng helper method approve()
    prediction.approve(request.user)
    
    # Thông báo cho user
    if prediction.user:
        final_label = f"{prediction.plant_type} - {prediction.user_confirmed_disease}"
        if label_changed:
            # Admin đã sửa nhãn
            Notification.objects.create(
                recipient=prediction.user,
                message=f'✅ Ảnh của bạn đã được duyệt với nhãn đã chỉnh sửa: {final_label} (Admin đã cập nhật để chuẩn hóa dữ liệu)',
                link=reverse('prediction_history')
            )
        else:
            # Giữ nguyên nhãn
            Notification.objects.create(
                recipient=prediction.user,
                message=f'Ảnh của bạn ({final_label}) đã được duyệt và thêm vào hệ thống!',
                link=reverse('prediction_history')
            )
    
    # Thông báo cho admin
    if label_changed:
        messages.success(request, f'Đã duyệt ảnh ID {prediction_id}. Nhãn đã sửa: {old_plant_type}/{old_disease} → {prediction.plant_type}/{prediction.user_confirmed_disease}')
    else:
        messages.success(request, f'Đã duyệt ảnh ID {prediction_id} với nhãn: {prediction.plant_type} - {prediction.user_confirmed_disease}')
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'status': 'success', 'message': 'Ảnh đã được duyệt.'})
    
    return redirect('admin_moderation')


@login_required
@user_passes_test(is_admin)
def reject_contribution(request, prediction_id):
    """Admin từ chối ảnh đóng góp"""
    if request.method != 'POST':
        return redirect('admin_moderation')
    
    prediction = get_object_or_404(PredictionHistory, id=prediction_id)
    reason = request.POST.get('reason', 'Ảnh không đạt yêu cầu chất lượng.')
    
    # Sử dụng helper method reject()
    prediction.reject(request.user, reason)
    
    # Thông báo cho user
    if prediction.user:
        Notification.objects.create(
            recipient=prediction.user,
            message=f'Ảnh của bạn (bệnh: {prediction.user_confirmed_disease}) đã bị từ chối. Lý do: {reason}',
            link=reverse('prediction_history')
        )
    
    messages.success(request, f'Đã từ chối ảnh ID {prediction_id}.')
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'status': 'success', 'message': 'Ảnh đã bị từ chối.'})
    
    return redirect('admin_moderation')


@login_required
def mark_all_notifications_read(request):
    Notification.objects.filter(recipient=request.user, is_read=False).update(is_read=True)
    return redirect('all_notifications')


# ==================== DATASET MANAGEMENT ====================
@login_required
def manage_datasets(request):
    """View for managing training datasets."""
    if not request.user.is_staff:
        messages.error(request, 'Bạn không có quyền truy cập trang này.')
        return redirect('home')
    
    datasets = TrainingDataset.objects.all()
    
    context = {
        'datasets': datasets,
    }
    return render(request, 'manage_datasets.html', context)


@login_required
def create_dataset(request):
    """Create a new training dataset."""
    if not request.user.is_staff:
        messages.error(request, 'Bạn không có quyền truy cập.')
        return redirect('home')
    
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description', '')
        dataset_type = request.POST.get('dataset_type', 'DISEASE')
        plant_type = request.POST.get('plant_type', '')
        
        remove_duplicates = request.POST.get('remove_duplicates') == 'on'
        remove_blurry = request.POST.get('remove_blurry') == 'on'
        
        # Handle blur_threshold safely
        blur_threshold_str = request.POST.get('blur_threshold', '100').strip()
        try:
            blur_threshold = float(blur_threshold_str) if blur_threshold_str else 100.0
        except ValueError:
            blur_threshold = 100.0
        
        # Handle sample_size safely
        sample_size_str = request.POST.get('sample_size', '').strip()
        sample_size = int(sample_size_str) if sample_size_str else None
        
        # Source data options
        include_new_contributions = request.POST.get('include_new_contributions') == 'on'
        include_original_dataset = request.POST.get('include_original_dataset') == 'on'
        original_dataset_path = request.POST.get('original_dataset_path', '')
        
        # Validate plant_type for DISEASE dataset
        if dataset_type == 'DISEASE' and not plant_type:
            messages.error(request, 'Vui lòng chọn loại cây cho Dataset phân loại bệnh.')
            return redirect('create_dataset')
        
        # Create dataset
        dataset = TrainingDataset.objects.create(
            name=name,
            description=description,
            created_by=request.user,
            dataset_type=dataset_type,
            plant_type=plant_type if dataset_type == 'DISEASE' else None,
            remove_duplicates=remove_duplicates,
            remove_blurry=remove_blurry,
            blur_threshold=blur_threshold,
            sample_size=sample_size,
            include_new_contributions=include_new_contributions,
            include_original_dataset=include_original_dataset,
            original_dataset_path=original_dataset_path if include_original_dataset else None,
            status='PREPARING'
        )
        
        messages.success(request, f'Đã tạo dataset "{name}". Đang phân tích dữ liệu...')
        return redirect('process_dataset', dataset_id=dataset.id)
    
    # Get statistics for new contributions
    total_new_images = PredictionHistory.objects.filter(
        approval_status=PredictionHistory.ApprovalStatus.ACCEPTED,
        is_active=True
    ).count()
    
    # Get stats by plant_type (for Disease dataset)
    plant_types = ['Apple', 'Cherry', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato']
    plant_type_stats = {}
    for pt in plant_types:
        count = PredictionHistory.objects.filter(
            approval_status=PredictionHistory.ApprovalStatus.ACCEPTED,
            is_active=True,
            plant_type__iexact=pt
        ).count()
        plant_type_stats[pt] = count
    
    context = {
        'total_new_images': total_new_images,
        'plant_types': plant_types,
        'plant_type_stats': plant_type_stats,
    }
    return render(request, 'create_dataset.html', context)


@login_required
def process_dataset(request, dataset_id):
    """Process dataset: detect blur, duplicates, and select images from multiple sources."""
    if not request.user.is_staff:
        messages.error(request, 'Bạn không có quyền truy cập.')
        return redirect('home')
    
    import glob
    from PIL import Image as PILImage
    
    dataset = get_object_or_404(TrainingDataset, id=dataset_id)
    
    all_image_paths = []  # List of (image_path, source_type, metadata)
    
    # Source 1: New contributions from users
    if dataset.include_new_contributions:
        approved_images = PredictionHistory.objects.filter(
            approval_status=PredictionHistory.ApprovalStatus.ACCEPTED,
            is_active=True
        )
        
        # Filter by plant_type for DISEASE dataset
        if dataset.dataset_type == 'DISEASE' and dataset.plant_type:
            approved_images = approved_images.filter(plant_type__iexact=dataset.plant_type)
        
        for pred in approved_images:
            all_image_paths.append({
                'path': pred.image.path,
                'source': 'NEW_CONTRIBUTION',
                'prediction_history': pred,
                'label': pred.user_confirmed_disease or pred.disease,
                'plant_type': pred.plant_type
            })
    
    # Source 2: Original dataset (87000 ảnh)
    if dataset.include_original_dataset and dataset.original_dataset_path:
        original_path = os.path.join(settings.BASE_DIR, dataset.original_dataset_path)
        
        # Iterate through folders (each folder is a class)
        if os.path.exists(original_path):
            for class_folder in os.listdir(original_path):
                class_path = os.path.join(original_path, class_folder)
                if os.path.isdir(class_path):
                    # Extract plant_type and disease from folder name
                    # Format: "Apple___Apple_scab" or "Tomato___Tomato_mosaic_virus"
                    parts = class_folder.split('___')
                    if len(parts) == 2:
                        plant_type_orig, disease_orig = parts
                        
                        # Filter by plant_type for DISEASE dataset
                        if dataset.dataset_type == 'DISEASE' and dataset.plant_type:
                            if plant_type_orig.lower() != dataset.plant_type.lower():
                                continue
                        
                        # Get all images in this class folder
                        for img_file in glob.glob(os.path.join(class_path, '*.jpg')) + \
                                       glob.glob(os.path.join(class_path, '*.JPG')) + \
                                       glob.glob(os.path.join(class_path, '*.png')):
                            all_image_paths.append({
                                'path': img_file,
                                'source': 'ORIGINAL_DATASET',
                                'prediction_history': None,
                                'label': disease_orig,
                                'plant_type': plant_type_orig
                            })
    
    # Step 1: Random sampling from all sources
    total_available = len(all_image_paths)
    if dataset.sample_size and total_available > dataset.sample_size:
        # Random sample before processing (to save time)
        all_image_paths = random.sample(all_image_paths, dataset.sample_size)
        messages.info(request, f'Đã chọn ngẫu nhiên {dataset.sample_size} ảnh từ {total_available} ảnh.')
    
    # Step 2: Process images (blur detection, duplicate detection)
    image_hashes = {}
    processed_count = 0
    
    for img_data in all_image_paths:
        img_path = img_data['path']
        
        # For new contributions, create TrainingDatasetImage
        if img_data['source'] == 'NEW_CONTRIBUTION':
            dataset_image, created = TrainingDatasetImage.objects.get_or_create(
                dataset=dataset,
                prediction_history=img_data['prediction_history']
            )
        else:
            # For original dataset, we don't have PredictionHistory
            # Skip creating TrainingDatasetImage for now (or create a placeholder)
            dataset_image = None
            created = True
        
        if created:
            try:
                # Calculate blur score
                blur_score = calculate_blur_score(Path(img_path))
                is_blurry = False
                if dataset.remove_blurry and blur_score and blur_score < dataset.blur_threshold:
                    is_blurry = True
                
                # Calculate hash for duplicate detection
                is_duplicate = False
                if dataset.remove_duplicates:
                    img_hash = calculate_image_hash(Path(img_path))
                    if img_hash:
                        if img_hash in image_hashes:
                            is_duplicate = True
                        else:
                            image_hashes[img_hash] = img_path
                
                # Update dataset_image if exists
                if dataset_image:
                    dataset_image.blur_score = blur_score
                    dataset_image.is_blurry = is_blurry
                    dataset_image.is_duplicate = is_duplicate
                    dataset_image.included = not (is_blurry or is_duplicate)
                    dataset_image.save()
                
                # Track included images
                if not is_blurry and not is_duplicate:
                    processed_count += 1
            
            except Exception as e:
                messages.warning(request, f'Lỗi xử lý ảnh {img_path}: {e}')
                continue
    
    # Update dataset status
    dataset.total_images = processed_count
    dataset.status = 'READY'
    dataset.save()
    
    messages.success(request, f'Dataset đã xử lý xong! Tổng số ảnh: {processed_count}')
    return redirect('view_dataset', dataset_id=dataset.id)


@login_required
def view_dataset(request, dataset_id):
    """View dataset details and images."""
    if not request.user.is_staff:
        messages.error(request, 'Bạn không có quyền truy cập.')
        return redirect('home')
    
    from .models import TrainingDataset, TrainingDatasetImage
    
    dataset = get_object_or_404(TrainingDataset, id=dataset_id)
    
    # Get statistics
    total_images = TrainingDatasetImage.objects.filter(dataset=dataset).count()
    included_images = TrainingDatasetImage.objects.filter(dataset=dataset, included=True).count()
    duplicate_count = TrainingDatasetImage.objects.filter(dataset=dataset, is_duplicate=True).count()
    blurry_count = TrainingDatasetImage.objects.filter(dataset=dataset, is_blurry=True).count()
    
    # Get sample images
    dataset_images = TrainingDatasetImage.objects.filter(
        dataset=dataset,
        included=True
    ).select_related('prediction_history')[:50]
    
    context = {
        'dataset': dataset,
        'total_images': total_images,
        'included_images': included_images,
        'duplicate_count': duplicate_count,
        'blurry_count': blurry_count,
        'dataset_images': dataset_images,
    }
    return render(request, 'view_dataset.html', context)


@login_required
def delete_dataset(request, dataset_id):
    """Delete a dataset."""
    if not request.user.is_staff:
        messages.error(request, 'Bạn không có quyền truy cập.')
        return redirect('home')
    
    from .models import TrainingDataset
    
    dataset = get_object_or_404(TrainingDataset, id=dataset_id)
    dataset_name = dataset.name
    dataset.delete()
    
    messages.success(request, f'Đã xóa dataset "{dataset_name}".')
    return redirect('manage_datasets')


# ==================== MODEL MANAGEMENT ====================
@login_required
def manage_models(request):
    """View for managing model versions."""
    if not request.user.is_staff:
        messages.error(request, 'Bạn không có quyền truy cập trang này.')
        return redirect('home')
    
    from .models import PlantTypeModel, DiseaseModel, SegmentationModel
    
    # Get ALL plant type models (not just default)
    plant_type_models = PlantTypeModel.objects.filter(is_active=True).order_by('-is_default', '-created_at')
    
    # Get disease models grouped by plant_type
    disease_models_grouped = {}
    plant_types = ['Apple', 'Cherry', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato']
    
    for plant_type in plant_types:
        models = DiseaseModel.objects.filter(
            plant_type__iexact=plant_type,
            is_active=True
        ).order_by('-is_default', '-created_at')
        
        # Only add to dict if there are models for this plant type
        if models.exists():
            disease_models_grouped[plant_type] = models

    segmentation_models = SegmentationModel.objects.filter(is_active=True).order_by('-is_default', '-created_at')
    
    context = {
        'plant_type_models': plant_type_models,
        'disease_models_grouped': disease_models_grouped,
        'plant_types': plant_types,
        'segmentation_models': segmentation_models,
    }
    return render(request, 'manage_models.html', context)


@login_required
def add_model(request):
    """Add a new model version."""
    if not request.user.is_staff:
        messages.error(request, 'Bạn không có quyền truy cập.')
        return redirect('home')
    
    from .models import PlantTypeModel, DiseaseModel, SegmentationModel, TrainingDataset
    from django.utils import timezone # Đảm bảo timezone được import
    
    if request.method == 'POST':
        model_type = request.POST.get('model_type')
        name = request.POST.get('name')
        description = request.POST.get('description', '')
        architecture = request.POST.get('architecture')
        optimizer = request.POST.get('optimizer')
        file_path = request.POST.get('file_path')
        dataset_id = request.POST.get('dataset')
        training_accuracy = request.POST.get('training_accuracy')
        validation_accuracy = request.POST.get('validation_accuracy')
        is_default = request.POST.get('is_default') == 'on'
        
        # THÊM TRƯỜNG LẤY TỪ FORM (chỉ dùng cho Disease Model)
        label_group = request.POST.get('label_group')
        
        # Create appropriate model based on type
        if model_type == 'PLANT_TYPE':
            num_classes = request.POST.get('num_classes', 9)
            model = PlantTypeModel.objects.create(
                name=name,
                description=description,
                architecture=architecture,
                optimizer=optimizer,
                file_path=file_path,
                dataset_id=dataset_id if dataset_id else None,
                num_classes=int(num_classes) if num_classes else 9,
                training_accuracy=float(training_accuracy) if training_accuracy else None,
                validation_accuracy=float(validation_accuracy) if validation_accuracy else None,
                is_default=is_default,
                created_by=request.user,
                training_date=timezone.now()
            )
        elif model_type == 'DISEASE':
            plant_type = request.POST.get('plant_type')
            num_classes = request.POST.get('num_classes')
            
            if not plant_type:
                messages.error(request, 'Vui lòng chọn loại cây cho Disease Model.')
                return redirect('add_model')
            
            model = DiseaseModel.objects.create(
                name=name,
                description=description,
                plant_type=plant_type,
                label_group=label_group,  # THÊM VÀO ĐÂY ĐỂ LƯU XUỐNG DB
                architecture=architecture,
                optimizer=optimizer,
                file_path=file_path,
                dataset_id=dataset_id if dataset_id else None,
                num_classes=int(num_classes) if num_classes else None,
                training_accuracy=float(training_accuracy) if training_accuracy else None,
                validation_accuracy=float(validation_accuracy) if validation_accuracy else None,
                is_default=is_default,
                created_by=request.user,
                training_date=timezone.now()
            )
        elif model_type == 'SEGMENTATION':
            model = SegmentationModel.objects.create(
                name=name,
                description=description,
                file_path=file_path,
                is_default=is_default,
                created_by=request.user,
            )
        else:
            messages.error(request, 'Vui lòng chọn loại model hợp lệ.')
            return redirect('add_model')
        
        messages.success(request, f'Đã thêm mô hình "{name}".')
        
        # Send notification to all admins
        admins = User.objects.filter(is_staff=True)
        for admin in admins:
            Notification.objects.create(
                recipient=admin,
                message=f'Mô hình mới "{name}" đã được thêm vào hệ thống.',
                link=reverse('manage_models')
            )
        
        return redirect('manage_models')
    
    datasets = TrainingDataset.objects.filter(status='READY')
    
    # Plant types for disease models
    plant_types = [
        'Apple', 'Cherry', 'Corn', 'Grape', 
        'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato'
    ]
    
    context = {
        'datasets': datasets,
        'plant_types': plant_types,
    }
    return render(request, 'add_model.html', context)


@login_required
def set_default_model(request, model_type, model_id):
    """Set a model as default for web predictions.
    
    Args:
        model_type: 'plant_type', 'disease', or 'segmentation'
        model_id: ID of the model to set as default
    """
    if not request.user.is_staff:
        messages.error(request, 'Bạn không có quyền truy cập.')
        return redirect('home')
    
    from .models import PlantTypeModel, DiseaseModel, SegmentationModel
    
    if model_type == 'plant_type':
        model = get_object_or_404(PlantTypeModel, id=model_id)
        # Unset all other plant type defaults
        PlantTypeModel.objects.filter(is_default=True).update(is_default=False)
        model.is_default = True
        model.save()
        model_name = f"Plant Type - {model.name}"
    elif model_type == 'disease':
        model = get_object_or_404(DiseaseModel, id=model_id)
        # Unset other defaults for this plant_type only
        DiseaseModel.objects.filter(
            plant_type=model.plant_type,
            is_default=True
        ).update(is_default=False)
        model.is_default = True
        model.save()
        model_name = f"{model.plant_type} Disease - {model.name}"
    else:  # segmentation
        model = get_object_or_404(SegmentationModel, id=model_id)
        SegmentationModel.objects.filter(is_default=True).update(is_default=False)
        model.is_default = True
        model.save()
        model_name = f"Segmentation - {model.name}"
    
    messages.success(request, f'Đã chọn mô hình "{model_name}" làm mô hình mặc định.')
    
    # Send notification to all admins
    admins = User.objects.filter(is_staff=True)
    for admin in admins:
        Notification.objects.create(
            recipient=admin,
            message=f'Mô hình "{model_name}" đã được chọn làm mô hình mặc định.',
            link=reverse('manage_models')
        )
    
    return redirect('manage_models')


@login_required
def delete_model(request, model_type, model_id):
    """Delete (deactivate) a model version.
    
    Args:
        model_type: 'plant_type', 'disease', or 'segmentation'
        model_id: ID of the model to delete
    """
    if not request.user.is_staff:
        messages.error(request, 'Bạn không có quyền truy cập.')
        return redirect('home')
    
    from .models import PlantTypeModel, DiseaseModel, SegmentationModel
    
    if model_type == 'plant_type':
        model = get_object_or_404(PlantTypeModel, id=model_id)
        model_name = f"Plant Type - {model.name}"
    elif model_type == 'disease':
        model = get_object_or_404(DiseaseModel, id=model_id)
        model_name = f"{model.plant_type} Disease - {model.name}"
    else:  # segmentation
        model = get_object_or_404(SegmentationModel, id=model_id)
        model_name = f"Segmentation - {model.name}"
    
    if model.is_default:
        messages.error(request, 'Không thể xóa mô hình đang được sử dụng. Vui lòng chọn mô hình khác làm mặc định trước.')
        return redirect('manage_models')
    
    model.is_active = False
    model.save()
    
    messages.success(request, f'Đã xóa mô hình "{model_name}".')
    return redirect('manage_models')


@login_required
def edit_model(request, model_type, model_id):
    """Edit a model version.
    
    Args:
        model_type: 'plant_type', 'disease', or 'segmentation'
        model_id: ID of the model to edit
    """
    if not request.user.is_staff:
        messages.error(request, 'Bạn không có quyền truy cập.')
        return redirect('home')
    
    from .models import PlantTypeModel, DiseaseModel, SegmentationModel, TrainingDataset
    
    if model_type == 'plant_type':
        model = get_object_or_404(PlantTypeModel, id=model_id, is_active=True)
    elif model_type == 'disease':
        model = get_object_or_404(DiseaseModel, id=model_id, is_active=True)
    else:  # segmentation
        model = get_object_or_404(SegmentationModel, id=model_id, is_active=True)
    
    if request.method == 'POST':
        # Update model fields
        model.name = request.POST.get('name')
        model.description = request.POST.get('description', '')
        model.file_path = request.POST.get('file_path')

        if model_type in ['plant_type', 'disease']:
            model.architecture = request.POST.get('architecture')
            model.optimizer = request.POST.get('optimizer')

            dataset_id = request.POST.get('dataset')
            if dataset_id:
                model.dataset_id = int(dataset_id)
            else:
                model.dataset = None

            training_accuracy = request.POST.get('training_accuracy')
            validation_accuracy = request.POST.get('validation_accuracy')

            if training_accuracy:
                model.training_accuracy = float(training_accuracy)
            if validation_accuracy:
                model.validation_accuracy = float(validation_accuracy)

            # Update model-specific fields
            if model_type == 'plant_type':
                num_classes = request.POST.get('num_classes')
                if num_classes:
                    model.num_classes = int(num_classes)
            else:  # disease
                plant_type = request.POST.get('plant_type')
                if plant_type:
                    model.plant_type = plant_type
                num_classes = request.POST.get('num_classes')
                if num_classes:
                    model.num_classes = int(num_classes)
                
                # THÊM LOGIC LƯU LABEL GROUP
                label_group = request.POST.get('label_group')
                if label_group is not None:
                    model.label_group = label_group
        
        model.save()
        
        messages.success(request, f'Đã cập nhật mô hình "{model.name}".')
        return redirect('manage_models')
    
    # GET request - show edit form
    datasets = TrainingDataset.objects.filter(status='READY')
    plant_types = [
        'Apple', 'Cherry', 'Corn', 'Grape', 
        'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato'
    ]
    
    context = {
        'model': model,
        'model_type': model_type,
        'datasets': datasets,
        'plant_types': plant_types,
    }
    return render(request, 'edit_model.html', context)


# ==================== DATASET EXPORT ====================

def _generate_dataset_zip(task_id):
    """Background function to generate dataset ZIP file.
    Updates ExportTask progress in database.
    """
    from .models import TrainingDataset, TrainingDatasetImage, ExportTask
    import zipfile
    import json
    import os
    import random
    import csv
    from django.conf import settings
    from django.utils import timezone
    import traceback
    import django
    import re
    
    # Setup Django for thread
    django.setup()
    
    def normalize_label(plant_type, disease):
        """Normalize label to consistent format: PlantType___disease_name
        Ensures old and new images are grouped in same folder.
        """
        if not disease:
            return plant_type.replace(' ', '_').replace('/', '_').replace('\\', '_')
        
        # Normalize plant type and disease to lowercase with underscores
        plant = plant_type.replace(' ', '_').replace('/', '_').replace('\\', '_')
        disease_normalized = disease.replace(' ', '_').replace('/', '_').replace('\\', '_').lower()
        
        return f"{plant}___{disease_normalized}"
    
    try:
        task = ExportTask.objects.get(id=task_id)
        task.status = 'PROCESSING'
        task.started_at = timezone.now()
        task.progress = 0
        task.current_step = 'Khởi tạo...'
        task.save()
        
        print(f"[EXPORT] Starting export for task {task_id}")
        
        dataset = task.dataset
        
        # Create temp ZIP file
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_exports')
        os.makedirs(temp_dir, exist_ok=True)
        zip_filename = f"{dataset.name}_{task.id}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            train_data = []
            val_data = []
            total_processed = 0
            
            # Estimate total images
            estimated_total = dataset.sample_size if dataset.sample_size else 10000
            
            task.progress = 5
            task.current_step = 'Đang đọc dataset gốc...'
            task.save()
            
            # ==================== PART 1: Process Original Dataset ====================
            if dataset.include_original_dataset and dataset.original_dataset_path:
                try:
                    csv_path = os.path.join(settings.BASE_DIR, dataset.original_dataset_path, 'plant_data.csv')
                    
                    if os.path.exists(csv_path):
                        original_images_added = 0
                        max_original_images = dataset.sample_size if dataset.sample_size else None
                        
                        with open(csv_path, 'r', encoding='utf-8') as csvfile:
                            reader = csv.DictReader(csvfile)
                            
                            for row in reader:
                                if max_original_images and original_images_added >= max_original_images:
                                    break
                                
                                image_rel_path = row.get('image', '')
                                disease = row.get('disease', '')
                                plant_type = row.get('plant_type', '')
                                split_type = row.get('dataset_type', 'train')
                                
                                if dataset.plant_type and plant_type != dataset.plant_type:
                                    continue
                                
                                image_rel_path = image_rel_path.replace('/', os.sep).replace('\\', os.sep)
                                
                                if image_rel_path.startswith(f'plant_images{os.sep}'):
                                    image_rel_path = image_rel_path[len(f'plant_images{os.sep}'):]
                                
                                image_abs_path = os.path.join(settings.BASE_DIR, dataset.original_dataset_path, image_rel_path)
                                
                                if os.path.exists(image_abs_path):
                                    # Extract folder name from original path to preserve naming convention
                                    # e.g., "images/train/Apple___Apple_scab/img.jpg" → "Apple___Apple_scab"
                                    path_parts = image_rel_path.replace('/', os.sep).split(os.sep)
                                    if len(path_parts) >= 2:
                                        # Get the class folder name from path
                                        original_folder_name = path_parts[-2]
                                    else:
                                        # Fallback to normalized label if path structure is unexpected
                                        original_folder_name = normalize_label(plant_type, disease)
                                    
                                    if split_type == 'valid' or split_type == 'val':
                                        filename = f"val/{original_folder_name}/{os.path.basename(image_abs_path)}"
                                        zip_file.write(image_abs_path, filename)
                                        val_data.append({
                                            'filename': filename,
                                            'label': original_folder_name,
                                            'disease': disease,
                                            'plant_type': plant_type,
                                            'source': 'original_dataset',
                                        })
                                    else:
                                        filename = f"train/{original_folder_name}/{os.path.basename(image_abs_path)}"
                                        zip_file.write(image_abs_path, filename)
                                        train_data.append({
                                            'filename': filename,
                                            'label': original_folder_name,
                                            'disease': disease,
                                            'plant_type': plant_type,
                                            'source': 'original_dataset',
                                        })
                                    
                                    original_images_added += 1
                                    total_processed += 1
                                    
                                    # Update progress every 100 images
                                    if total_processed % 100 == 0:
                                        progress = min(5 + int((total_processed / estimated_total) * 60), 65)
                                        task.progress = progress
                                        task.current_step = f'Đã xử lý {total_processed} ảnh từ dataset gốc...'
                                        task.save()
                                        
                except Exception as e:
                    task.error_message = f'Lỗi xử lý dataset gốc: {str(e)}'
                    task.save()
            
            task.progress = 70
            task.current_step = 'Đang xử lý ảnh mới đóng góp...'
            task.save()
            
            # Build mapping from disease names to folder names from original dataset
            disease_to_folder = {}
            for item in train_data + val_data:
                disease = item.get('disease', '').strip().lower()
                folder = item.get('label', '')
                if disease and folder:
                    disease_to_folder[disease] = folder
            
            # ==================== PART 2: Process New Contributions ====================
            if dataset.include_new_contributions:
                new_images = list(TrainingDatasetImage.objects.filter(dataset=dataset))
                random.shuffle(new_images)
                
                split_index = int(len(new_images) * 0.8)
                train_new = new_images[:split_index]
                val_new = new_images[split_index:]
                
                for img in train_new:
                    pred = img.prediction_history
                    img_path = pred.image.path
                    
                    if os.path.exists(img_path):
                        # Match folder name from original dataset or use normalized label
                        plant_type = pred.plant_type or 'Unknown'
                        disease = pred.disease or ''
                        disease_key = disease.strip().lower()
                        
                        # Use existing folder name from original dataset if available
                        folder_name = disease_to_folder.get(disease_key)
                        if not folder_name:
                            # Fallback to normalized label for new diseases
                            folder_name = normalize_label(plant_type, disease)
                        
                        filename = f"train/{folder_name}/{os.path.basename(img_path)}"
                        
                        zip_file.write(img_path, filename)
                        train_data.append({
                            'filename': filename,
                            'label': folder_name,
                            'disease': pred.disease,
                            'plant_type': pred.plant_type,
                            'confidence': float(pred.confidence) if pred.confidence else None,
                            'source': 'new_contribution',
                        })
                
                for img in val_new:
                    pred = img.prediction_history
                    img_path = pred.image.path
                    
                    if os.path.exists(img_path):
                        # Match folder name from original dataset or use normalized label
                        plant_type = pred.plant_type or 'Unknown'
                        disease = pred.disease or ''
                        disease_key = disease.strip().lower()
                        
                        # Use existing folder name from original dataset if available
                        folder_name = disease_to_folder.get(disease_key)
                        if not folder_name:
                            # Fallback to normalized label for new diseases
                            folder_name = normalize_label(plant_type, disease)
                        
                        filename = f"val/{folder_name}/{os.path.basename(img_path)}"
                        
                        zip_file.write(img_path, filename)
                        val_data.append({
                            'filename': filename,
                            'label': folder_name,
                            'disease': pred.disease,
                            'plant_type': pred.plant_type,
                            'confidence': float(pred.confidence) if pred.confidence else None,
                            'source': 'new_contribution',
                        })
            
            task.progress = 85
            task.current_step = 'Đang tạo metadata...'
            task.save()
            
            # ==================== PART 3: Create Metadata ====================
            train_original = sum(1 for d in train_data if d.get('source') == 'original_dataset')
            train_new = sum(1 for d in train_data if d.get('source') == 'new_contribution')
            val_original = sum(1 for d in val_data if d.get('source') == 'original_dataset')
            val_new = sum(1 for d in val_data if d.get('source') == 'new_contribution')
            
            metadata = {
                'dataset_id': dataset.id,
                'dataset_name': dataset.name,
                'description': dataset.description,
                'dataset_type': dataset.dataset_type,
                'plant_type': dataset.plant_type,
                'total_images': len(train_data) + len(val_data),
                'train_images': len(train_data),
                'val_images': len(val_data),
                'split_ratio': {
                    'train': round(len(train_data) / (len(train_data) + len(val_data)), 2) if (len(train_data) + len(val_data)) > 0 else 0,
                    'val': round(len(val_data) / (len(train_data) + len(val_data)), 2) if (len(train_data) + len(val_data)) > 0 else 0,
                },
                'sources': {
                    'train': {'original_dataset': train_original, 'new_contributions': train_new, 'total': len(train_data)},
                    'val': {'original_dataset': val_original, 'new_contributions': val_new, 'total': len(val_data)},
                },
                'export_date': timezone.now().isoformat(),
                'exported_by': task.created_by.username,
            }
            
            zip_file.writestr('metadata.json', json.dumps(metadata, indent=2, ensure_ascii=False))
            
            # Create README
            readme_content = f"""# {dataset.name} Dataset Export

## Dataset Information
- **Type**: {dataset.dataset_type}
- **Plant Type**: {dataset.plant_type or 'All'}
- **Total Images**: {len(train_data) + len(val_data)}
- **Train Images**: {len(train_data)} ({round(len(train_data) / (len(train_data) + len(val_data)) * 100, 1) if (len(train_data) + len(val_data)) > 0 else 0}%)
- **Validation Images**: {len(val_data)} ({round(len(val_data) / (len(train_data) + len(val_data)) * 100, 1) if (len(train_data) + len(val_data)) > 0 else 0}%)

## Data Sources
### Training Set
- Original Dataset: {train_original} images
- New Contributions: {train_new} images

### Validation Set
- Original Dataset: {val_original} images
- New Contributions: {val_new} images

## Directory Structure
```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
├── metadata.json
└── README.md
```

## Usage
1. Unzip this file
2. Upload to Google Colab or your training environment
3. Use the train/ and val/ folders for model training
4. Check metadata.json for detailed information

Exported on: {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}
Exported by: {task.created_by.username}
"""
            zip_file.writestr('README.md', readme_content)
        
        print(f"[EXPORT] ZIP file created successfully: {zip_path}")
        print(f"[EXPORT] Total images: {len(train_data) + len(val_data)}")
        
        # Update task with results
        task.status = 'COMPLETED'
        task.progress = 100
        task.current_step = 'Hoàn thành!'
        task.file_path = zip_path
        task.file_size = os.path.getsize(zip_path)
        task.total_images = len(train_data) + len(val_data)
        task.completed_at = timezone.now()
        task.save()
        
        print(f"[EXPORT] Task completed successfully")
        
    except Exception as e:
        print(f"[EXPORT ERROR] {str(e)}")
        print(traceback.format_exc())
        
        try:
            task = ExportTask.objects.get(id=task_id)
            task.status = 'FAILED'
            task.error_message = f"{str(e)}\n\n{traceback.format_exc()}"
            task.completed_at = timezone.now()
            task.save()
        except Exception as save_error:
            print(f"[EXPORT ERROR] Could not save error to task: {save_error}")


@login_required
def export_dataset(request, dataset_id):
    """Start background export task and return task_id"""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Unauthorized'}, status=403)
    
    from .models import TrainingDataset, ExportTask
    import threading
    
    dataset = get_object_or_404(TrainingDataset, id=dataset_id)
    
    # Create export task
    task = ExportTask.objects.create(
        dataset=dataset,
        created_by=request.user,
        status='PENDING',
        progress=0
    )
    
    # Start background thread
    thread = threading.Thread(target=_generate_dataset_zip, args=(task.id,))
    thread.daemon = True
    thread.start()
    
    # Return task_id for polling
    return JsonResponse({
        'task_id': str(task.id),
        'status': 'started'
    })


@login_required
def export_progress(request, dataset_id, task_id):
    """API endpoint to check export progress"""
    from .models import ExportTask
    
    try:
        task = ExportTask.objects.get(id=task_id, dataset_id=dataset_id)
        
        response_data = {
            'status': task.status,
            'progress': task.progress,
            'current_step': task.current_step,
            'total_images': task.total_images,
        }
        
        if task.status == 'COMPLETED':
            response_data['download_url'] = f'/admin/datasets/{task.dataset.id}/download/{task.id}/'
            response_data['file_size'] = task.file_size
        elif task.status == 'FAILED':
            response_data['error'] = task.error_message
            
        return JsonResponse(response_data)
        
    except ExportTask.DoesNotExist:
        return JsonResponse({'error': 'Task not found'}, status=404)


@login_required
def download_export(request, dataset_id, task_id):
    """Download completed export file"""
    from .models import ExportTask
    import mimetypes
    
    if not request.user.is_staff:
        messages.error(request, 'Bạn không có quyền truy cập.')
        return redirect('home')
    
    task = get_object_or_404(ExportTask, id=task_id, dataset_id=dataset_id)
    
    if task.status != 'COMPLETED' or not task.file_path:
        messages.error(request, 'File export chưa sẵn sàng.')
        return redirect('view_dataset', dataset_id=dataset_id)
    
    if not os.path.exists(task.file_path):
        messages.error(request, 'File không tồn tại.')
        return redirect('view_dataset', dataset_id=dataset_id)
    
    # Stream file to download
    with open(task.file_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/zip')
        response['Content-Disposition'] = f'attachment; filename="{task.dataset.name}_export.zip"'
        return response