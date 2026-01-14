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
    
    # Kết quả AI prediction
    disease = models.CharField(max_length=100, blank=True)  # Tương ứng với class_name
    plant_type = models.CharField(max_length=100, blank=True)
    dataset_type = models.CharField(max_length=50, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    
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