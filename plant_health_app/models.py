import os
import uuid
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.utils.text import slugify
from django.conf import settings
from .utils import PLANT_TYPE_MAPPING  # Import PLANT_TYPE_MAPPING

def get_image_path(instance, filename):
    """
    Tạo đường dẫn lưu ảnh với tên tệp duy nhất dựa trên tên bệnh.
    """
    # Lấy tên bệnh hoặc mặc định là 'unknown'
    disease = getattr(instance, 'disease', None) or getattr(instance, 'name', 'unknown')
    disease_folder = slugify(disease.lower().replace(" ", "_"))  # Chuyển thành slug an toàn (Tomato Late Blight -> tomato_late_blight)

    # Lấy phần mở rộng của tệp
    ext = os.path.splitext(filename)[1].lower()  # .jpg, .png, v.v.
    
    # Tạo tên tệp duy nhất với định dạng tương tự ảnh cũ
    new_filename = f"{uuid.uuid4()}___RS_{disease_folder}{ext}"  # Ví dụ: 005a2c1f-4e15-49e4-9e5c-61dc3ecf9708___RS_tomato_late_blight.jpg
    
    # Tạo đường dẫn
    relative_path = os.path.join('plant_images', 'images', 'valid', disease_folder, new_filename)
    
    # Đảm bảo thư mục tồn tại
    full_path = os.path.join(settings.MEDIA_ROOT, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    return relative_path

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    age = models.PositiveIntegerField(null=True, blank=True)
    phone_number = models.CharField(max_length=15, null=True, blank=True)
    nationality = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return f"Profile of {self.user.username}"

class Plant(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class PlantImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    plant = models.ForeignKey('Plant', on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to=get_image_path, max_length=255)
    disease = models.CharField(max_length=100, blank=True)  # Tương ứng với class_name
    plant_type = models.CharField(max_length=100, blank=True)
    dataset_type = models.CharField(max_length=50, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    uploaded_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Image {self.id} by {self.user.username if self.user else 'Anonymous'}"
    
class PlantModel(models.Model):
    image = models.ImageField(upload_to=get_image_path, max_length=255)
    disease = models.CharField(max_length=100, blank=True)  # Tương ứng với class_name
    plant_type = models.CharField(max_length=100, blank=True)
    dataset_type = models.CharField(max_length=50, blank=True)

    def __str__(self):
        return f"Image {self.id} by {self.user.username if self.user else 'Anonymous'}"

class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    feedback_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback from {self.user.username} at {self.created_at}"

class Question(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='questions')
    plant_image = models.ForeignKey(PlantImage, on_delete=models.CASCADE, null=True, blank=True)
    question_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_public = models.BooleanField(default=True)  # True: Công khai, False: Riêng tư
    designated_expert = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='designated_questions')  # Chuyên gia được chỉ định

    def __str__(self):
        return f"{self.user.username} - {self.question_text[:50]}"

class Answer(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='answers')
    expert = models.ForeignKey(User, on_delete=models.CASCADE)
    answer_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Answer by {self.expert.username} for {self.question}"

class DiseaseLibrary(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    symptoms = models.TextField()
    treatment = models.TextField()
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to=get_image_path, null=True, blank=True, max_length=1000)  # Tăng max_length

    class Meta:
        db_table = 'plant_health_app_diseaselibrary'

    def __str__(self):
        return self.name

class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    plant = models.ForeignKey('Plant', on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to=get_image_path, max_length=255)
    disease = models.CharField(max_length=100, blank=True)  # Tương ứng với class_name
    plant_type = models.CharField(max_length=100, blank=True)
    dataset_type = models.CharField(max_length=50, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    uploaded_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Image {self.id} by {self.user.username if self.user else 'Anonymous'}"

class Notification(models.Model):
    recipient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    message = models.TextField()
    link = models.CharField(max_length=255, blank=True, null=True)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Notification for {self.recipient.username}: {self.message}"