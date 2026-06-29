import os
import uuid
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.conf import settings

def get_image_path(instance, filename):
    """
    Tạo đường dẫn lưu ảnh với tên tệp duy nhất dựa trên tên bệnh.
    """
    # Lấy phần mở rộng của tệp
    ext = os.path.splitext(filename)[1].lower()  # .jpg, .png, v.v.
    # Tạo tên tệp duy nhất
    new_filename = f"{uuid.uuid4()}{ext}"
    # Đường dẫn mới cho pending uploads
    relative_path = os.path.join('plant_images', 'pending_uploads', new_filename)
    # Đảm bảo thư mục tồn tại
    full_path = os.path.join(settings.MEDIA_ROOT, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    return relative_path

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    age = models.PositiveIntegerField(null=True, blank=True)
    phone_number = models.CharField(max_length=15, null=True, blank=True)
    nationality = models.CharField(max_length=100, null=True, blank=True)
    
    # Soft delete fields
    is_active = models.BooleanField(default=True, db_index=True)
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='deleted_user_profiles'
    )

    def __str__(self):
        return f"Profile of {self.user.username}"
    
    def soft_delete(self, user=None):
        """Soft delete: mark as inactive"""
        self.is_active = False
        self.deleted_at = timezone.now()
        self.deleted_by = user
        self.save()
    
    def restore(self):
        """Restore soft deleted object"""
        self.is_active = True
        self.deleted_at = None
        self.deleted_by = None
        self.save()

class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    feedback_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Soft delete fields
    is_active = models.BooleanField(default=True, db_index=True)
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='deleted_feedbacks'
    )

    def __str__(self):
        return f"Feedback from {self.user.username} at {self.created_at}"
    
    def soft_delete(self, user=None):
        """Soft delete: mark as inactive"""
        self.is_active = False
        self.deleted_at = timezone.now()
        self.deleted_by = user
        self.save()
    
    def restore(self):
        """Restore soft deleted object"""
        self.is_active = True
        self.deleted_at = None
        self.deleted_by = None
        self.save()


class DiseaseLibrary(models.Model):
    name = models.CharField(max_length=100)
    plant_type = models.CharField(max_length=100, blank=True, null=True)  # Thêm trường plant_type
    description = models.TextField()
    symptoms = models.TextField()
    treatment = models.TextField()
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to=get_image_path, null=True, blank=True, max_length=1000)  # Tăng max_length
    
    # Soft delete fields
    is_active = models.BooleanField(default=True, db_index=True)
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='deleted_diseases'
    )

    class Meta:
        db_table = 'plant_health_app_diseaselibrary'

    def __str__(self):
        return self.name
    
    def soft_delete(self, user=None):
        """Soft delete: mark as inactive"""
        self.is_active = False
        self.deleted_at = timezone.now()
        self.deleted_by = user
        self.save()
    
    def restore(self):
        """Restore soft deleted object"""
        self.is_active = True
        self.deleted_at = None
        self.deleted_by = None
        self.save()


class SegmentationModel(models.Model):
    """Segmentation model registry (e.g. YOLO)."""
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    file_path = models.CharField(max_length=500, unique=True)

    is_active = models.BooleanField(default=True, db_index=True)
    is_default = models.BooleanField(default=False, help_text="Model đang được sử dụng cho segmentation")
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Segmentation Model'
        verbose_name_plural = 'Segmentation Models'

    def __str__(self):
        status = " ✓ Đang dùng" if self.is_default else ""
        return f"Segmentation - {self.name}{status}"

    def save(self, *args, **kwargs):
        if self.is_default:
            SegmentationModel.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)
        super().save(*args, **kwargs)


class PredictionHistory(models.Model):
    """Model lưu lịch sử dự đoán với workflow kiểm duyệt"""
    
    # Định nghĩa các trạng thái kiểm duyệt
    class ApprovalStatus(models.TextChoices):
        PENDING = 'PENDING', 'Chờ duyệt'      # Mặc định khi mới upload
        ACCEPTED = 'ACCEPTED', 'Đã duyệt'     # Admin đã chấp nhận
        REJECTED = 'REJECTED', 'Đã từ chối'   # Admin từ chối
    
    # Định nghĩa mục đích sử dụng
    class ContributionType(models.TextChoices):
        CONSULTING = 'CONSULTING', 'Chỉ xem (Hỏi đường)'      # User chỉ muốn biết kết quả
        CONTRIBUTING = 'CONTRIBUTING', 'Đóng góp cho AI'      # User muốn đóng góp ảnh vào training
    
    # Thông tin người dùng và ảnh
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to=get_image_path, max_length=255)
    cropped_image = models.ImageField(upload_to='plant_images/cropped/', null=True, blank=True, max_length=255)
    segmentation_model = models.ForeignKey(
        'SegmentationModel',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='predictions'
    )
    segment_data = models.JSONField(null=True, blank=True)
    
    # Kết quả AI prediction
    disease = models.CharField(max_length=100, blank=True)  # Tương ứng với class_name
    plant_type = models.CharField(max_length=100, blank=True)
    dataset_type = models.CharField(max_length=50, blank=True)
    confidence = models.FloatField(null=True, blank=True)

    # ---- [TRƯỜNG MỚI ĐỂ ĐO ĐỘ TRỄ] ----
    inference_latency = models.FloatField(
        null=True, 
        blank=True,
        help_text="Thời gian suy luận toàn trình đo bằng giây (End-to-End Inference Latency)"
    )
    # -----------------------------------
    
    # Thông tin đóng góp của user
    contribution_type = models.CharField(
        max_length=20,
        choices=ContributionType.choices,
        default=ContributionType.CONSULTING,
        help_text="Mục đích upload ảnh: Chỉ xem hay đóng góp cho AI"
    )
    user_confirmed_disease = models.CharField(
        max_length=100, 
        blank=True, 
        null=True,
        help_text="Nhãn bệnh do user xác nhận (có thể khác với AI prediction)"
    )
    user_agreed_with_ai = models.BooleanField(
        default=False,
        help_text="User đồng ý với dự đoán của AI hay tự sửa nhãn"
    )
    
    # Workflow kiểm duyệt
    approval_status = models.CharField(
        max_length=20,
        choices=ApprovalStatus.choices,
        default=ApprovalStatus.PENDING,
        db_index=True,  # Index để filter nhanh
        help_text="Trạng thái kiểm duyệt của ảnh"
    )
    rejection_reason = models.TextField(
        blank=True, 
        null=True, 
        help_text="Lý do từ chối (nếu approval_status = REJECTED)"
    )
    reviewed_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='reviewed_predictions',
        help_text="Admin đã kiểm duyệt"
    )
    reviewed_at = models.DateTimeField(
        null=True, 
        blank=True,
        help_text="Thời gian kiểm duyệt"
    )
    
    # Timestamps
    uploaded_at = models.DateTimeField(default=timezone.now)
    
    # Soft delete fields
    is_active = models.BooleanField(default=True, db_index=True)
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='deleted_predictions'
    )

    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Lịch sử dự đoán'
        verbose_name_plural = 'Lịch sử dự đoán'

    def __str__(self):
        return f"Image {self.id} by {self.user.username if self.user else 'Anonymous'} - {self.get_approval_status_display()}"
    
    def soft_delete(self, user=None):
        """Soft delete: mark as inactive"""
        self.is_active = False
        self.deleted_at = timezone.now()
        self.deleted_by = user
        self.save()
    
    def restore(self):
        """Restore soft deleted object"""
        self.is_active = True
        self.deleted_at = None
        self.deleted_by = None
        self.save()
    
    def approve(self, admin_user):
        """Duyệt ảnh"""
        self.approval_status = self.ApprovalStatus.ACCEPTED
        self.reviewed_by = admin_user
        self.reviewed_at = timezone.now()
        self.rejection_reason = None  # Xóa lý do từ chối cũ (nếu có)
        self.save()
    
    def reject(self, admin_user, reason=''):
        """Từ chối ảnh"""
        self.approval_status = self.ApprovalStatus.REJECTED
        self.reviewed_by = admin_user
        self.reviewed_at = timezone.now()
        self.rejection_reason = reason
        self.save()
    
    def is_pending(self):
        """Kiểm tra xem ảnh có đang chờ duyệt không"""
        return self.approval_status == self.ApprovalStatus.PENDING
    
    def is_accepted(self):
        """Kiểm tra xem ảnh đã được duyệt chưa"""
        return self.approval_status == self.ApprovalStatus.ACCEPTED
    
    def is_rejected(self):
        """Kiểm tra xem ảnh có bị từ chối không"""
        return self.approval_status == self.ApprovalStatus.REJECTED

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


class DeletedObject(models.Model):
    """Store a JSON backup of a deleted object to allow undo/restore.

    Fields:
      model_label: e.g. 'plant_health_app.predictionhistory'
      object_pk: original primary key as string
      data: serialized JSON from django.core.serializers
      deleted_by: user who performed deletion (optional)
      created_at: timestamp when stored
    """
    model_label = models.CharField(max_length=200)
    object_pk = models.CharField(max_length=100)
    data = models.TextField()  # serialized JSON
    deleted_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Deleted {self.model_label} pk={self.object_pk} at {self.created_at}"


class BlogPost(models.Model):
    """Blog post model for sharing experiences and asking questions."""
    VISIBILITY_CHOICES = [
        ('public', 'Công khai'),
        ('private', 'Riêng tư'),
        ('specific', 'Chỉ định người xem'),
    ]
    
    STATUS_CHOICES = [
        ('draft', 'Nháp'),
        ('pending', 'Chờ duyệt'),
        ('approved', 'Đã duyệt'),
        ('rejected', 'Từ chối'),
    ]
    
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='blog_posts')
    title = models.CharField(max_length=255)
    content = models.TextField()
    image = models.ImageField(upload_to='blog_images/', null=True, blank=True, max_length=500)
    visibility = models.CharField(max_length=10, choices=VISIBILITY_CHOICES, default='public')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    allowed_viewers = models.ManyToManyField(User, related_name='allowed_blog_posts', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    approved_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='approved_posts')
    approved_at = models.DateTimeField(null=True, blank=True)
    rejection_reason = models.TextField(null=True, blank=True, help_text='Lý do từ chối bài viết')
    views_count = models.IntegerField(default=0)
    
    # Soft delete fields
    is_active = models.BooleanField(default=True, db_index=True)
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='deleted_blog_posts'
    )
    
    class Meta:
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.title} by {self.author.username}"
    
    def soft_delete(self, user=None):
        """Soft delete: mark as inactive"""
        self.is_active = False
        self.deleted_at = timezone.now()
        self.deleted_by = user
        self.save()
    
    def restore(self):
        """Restore soft deleted object"""
        self.is_active = True
        self.deleted_at = None
        self.deleted_by = None
        self.save()
    
    def can_view(self, user):
        """Check if user can view this post."""
        if self.status != 'approved':
            return user == self.author or user.is_staff
        
        if self.visibility == 'public':
            return True
        elif self.visibility == 'private':
            return user == self.author or user.is_staff
        elif self.visibility == 'specific':
            return user == self.author or user in self.allowed_viewers.all() or user.is_staff
        return False


class BlogComment(models.Model):
    """Comments on blog posts."""
    post = models.ForeignKey(BlogPost, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='blog_comments')
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Soft delete fields
    is_active = models.BooleanField(default=True, db_index=True)
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='deleted_blog_comments'
    )
    
    class Meta:
        ordering = ['created_at']
        
    def __str__(self):
        return f"Comment by {self.author.username} on {self.post.title}"
    
    def soft_delete(self, user=None):
        """Soft delete: mark as inactive"""
        self.is_active = False
        self.deleted_at = timezone.now()
        self.deleted_by = user
        self.save()
    
    def restore(self):
        """Restore soft deleted object"""
        self.is_active = True
        self.deleted_at = None
        self.deleted_by = None
        self.save()
    
    def get_author_role(self):
        """Get the author's role (farmer, expert, admin)."""
        if self.author.is_staff:
            return 'Admin'
        elif self.author.groups.filter(name='Expert').exists():
            return 'Expert'
        elif self.author.groups.filter(name='Farmer').exists():
            return 'Farmer'
        return 'User'


class TrainingDataset(models.Model):
    """Model for managing training datasets."""
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Dataset type and filters
    DATASET_TYPE_CHOICES = [
        ('PLANT_TYPE', 'Phân loại cây'),
        ('DISEASE', 'Phân loại bệnh'),
    ]
    dataset_type = models.CharField(max_length=20, choices=DATASET_TYPE_CHOICES, default='DISEASE')
    plant_type = models.CharField(max_length=100, blank=True, null=True, help_text='Loại cây (chỉ cho Disease dataset)')
    
    # Filter settings
    remove_duplicates = models.BooleanField(default=True)
    remove_blurry = models.BooleanField(default=True)
    blur_threshold = models.FloatField(default=100.0)  # Laplacian variance threshold
    
    # Source data settings
    include_new_contributions = models.BooleanField(default=True, help_text='Bao gồm ảnh mới từ user đóng góp')
    include_original_dataset = models.BooleanField(default=False, help_text='Bao gồm ảnh từ dataset gốc (87000 ảnh)')
    original_dataset_path = models.CharField(max_length=500, blank=True, null=True, 
                                            help_text='Đường dẫn đến dataset gốc (plant_health_app/data/plant_images/)')
    
    # Sample settings
    total_images = models.IntegerField(default=0, help_text='Tổng số ảnh sau khi filter')
    sample_size = models.IntegerField(null=True, blank=True, help_text='Số ảnh cần lấy (để trống = lấy tất cả)')
    
    # Status
    # PREPARING: Đang tạo dataset, chưa xử lý
    # READY: Đã xử lý xong, có thể dùng để train
    # USED: Đã được sử dụng để train model
    STATUS_CHOICES = [
        ('PREPARING', 'Đang chuẩn bị'),
        ('READY', 'Sẵn sàng'),
        ('USED', 'Đã sử dụng'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PREPARING')
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Training Dataset'
        verbose_name_plural = 'Training Datasets'
    
    def __str__(self):
        type_label = dict(self.DATASET_TYPE_CHOICES).get(self.dataset_type, '')
        plant_info = f" - {self.plant_type}" if self.plant_type else ""
        return f"{self.name} ({type_label}{plant_info}, {self.total_images} ảnh)"


class TrainingDatasetImage(models.Model):
    """Images selected for a training dataset."""
    dataset = models.ForeignKey(TrainingDataset, on_delete=models.CASCADE, related_name='images')
    prediction_history = models.ForeignKey('PredictionHistory', on_delete=models.CASCADE)
    is_duplicate = models.BooleanField(default=False)
    is_blurry = models.BooleanField(default=False)
    blur_score = models.FloatField(null=True, blank=True)
    included = models.BooleanField(default=True)  # Whether included in final dataset
    
    class Meta:
        unique_together = ['dataset', 'prediction_history']
        verbose_name = 'Dataset Image'
        verbose_name_plural = 'Dataset Images'
    
    def __str__(self):
        return f"{self.dataset.name} - {self.prediction_history.id}"


class PlantTypeModel(models.Model):
    """Model for Plant Type Classification (Tầng 1 - Dự đoán loại cây)."""
    name = models.CharField(max_length=200, help_text="Tên phiên bản model (VD: EfficientNet-B0 AdamW v1)")
    description = models.TextField(blank=True)
    
    # Model info
    architecture = models.CharField(max_length=100, default='EfficientNet-B0')  # ResNet50, MobileNetV2, EfficientNet-B0
    optimizer = models.CharField(max_length=50)  # SGD, Adam, AdamW
    file_path = models.CharField(max_length=500, help_text="Đường dẫn file .pth (VD: models/plant_type_v1.pth)")
    
    # Training info
    dataset = models.ForeignKey(TrainingDataset, on_delete=models.SET_NULL, null=True, blank=True)
    training_accuracy = models.FloatField(null=True, blank=True)
    validation_accuracy = models.FloatField(null=True, blank=True)
    num_classes = models.IntegerField(default=9, help_text="Số lượng loại cây (mặc định 9)")
    training_date = models.DateTimeField(null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=True, db_index=True)
    is_default = models.BooleanField(default=False, help_text="Model đang được sử dụng cho web")
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Plant Type Model'
        verbose_name_plural = 'Plant Type Models'
    
    def __str__(self):
        status = " ✓ Đang dùng" if self.is_default else ""
        return f"Plant Type - {self.name}{status}"
    
    def save(self, *args, **kwargs):
        # Only one default plant type model
        if self.is_default:
            PlantTypeModel.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)
        super().save(*args, **kwargs)


class DiseaseModel(models.Model):
    """Model for Disease Classification (Tầng 2 - Dự đoán bệnh theo từng loại cây)."""
    name = models.CharField(max_length=200, help_text="Tên phiên bản model")
    description = models.TextField(blank=True)
    
    # Plant type specific
    plant_type = models.CharField(
        max_length=100,
        db_index=True,
        help_text="Loại cây (Tomato, Potato, Apple, Cherry, Corn, Grape, Peach, Pepper, Strawberry)"
    )
    
    # Model info
    architecture = models.CharField(max_length=100, default='EfficientNet-B0')
    optimizer = models.CharField(max_length=50)  # SGD, Adam, AdamW
    file_path = models.CharField(max_length=500, help_text="Đường dẫn file .pth (VD: models/tomato_disease_v1.pth)")
    label_group = models.CharField(max_length=100, blank=True, null=True, 
                                   help_text="Ví dụ: Potato_3, Potato_7, Apple")
    # Training info
    dataset = models.ForeignKey(TrainingDataset, on_delete=models.SET_NULL, null=True, blank=True)
    training_accuracy = models.FloatField(null=True, blank=True)
    validation_accuracy = models.FloatField(null=True, blank=True)
    num_classes = models.IntegerField(null=True, blank=True, help_text="Số lượng bệnh cho loại cây này")
    training_date = models.DateTimeField(null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=True, db_index=True)
    is_default = models.BooleanField(default=False, help_text="Model đang được sử dụng cho loại cây này")
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['plant_type', '-created_at']
        verbose_name = 'Disease Model'
        verbose_name_plural = 'Disease Models'
        indexes = [
            models.Index(fields=['plant_type', 'is_default']),
        ]
    
    def __str__(self):
        status = " ✓ Đang dùng" if self.is_default else ""
        return f"{self.plant_type} Disease - {self.name}{status}"
    
    def save(self, *args, **kwargs):
        # Only one default per plant_type
        if self.is_default:
            DiseaseModel.objects.filter(
                plant_type=self.plant_type,
                is_default=True
            ).exclude(pk=self.pk).update(is_default=False)
        super().save(*args, **kwargs)


class ExportTask(models.Model):
    """Track export progress for background tasks"""
    STATUS_CHOICES = [
        ('PENDING', 'Chờ xử lý'),
        ('PROCESSING', 'Đang xử lý'),
        ('COMPLETED', 'Hoàn thành'),
        ('FAILED', 'Thất bại'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(TrainingDataset, on_delete=models.CASCADE)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    progress = models.IntegerField(default=0)  # 0-100
    current_step = models.CharField(max_length=200, blank=True)
    
    # Results
    file_path = models.CharField(max_length=500, blank=True)  # Path to generated ZIP
    file_size = models.BigIntegerField(null=True, blank=True)  # Size in bytes
    total_images = models.IntegerField(default=0)
    
    # Error tracking
    error_message = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'created_at']),
        ]
    
    def __str__(self):
        return f"Export {self.dataset.name} - {self.status} ({self.progress}%)"