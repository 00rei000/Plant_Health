from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login as auth_login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User, Group
from django.contrib.auth.forms import PasswordChangeForm
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
from django.conf import settings
from django.db.models import Q

from django.urls import reverse  # Thêm import reverse để sử dụng hàm reverse
from django import forms

from .models import PlantImage, Feedback, Question, DiseaseLibrary, Answer, UserProfile, PredictionHistory, Notification
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from .utils import PLANT_TYPE_MAPPING

# Định nghĩa danh sách nhãn lớp (21 lớp)
CLASSES = [
    "apple scab", "bacterial spot", "black rot", "cedar apple rust", 
    "cercospora leaf spot gray leaf spot", "common rust", "early blight", 
    "esca (black measles)", "haunglongbing (citrus greening)", "healthy", 
    "late blight", "leaf blight (isariopsis leaf spot)", "leaf mold", 
    "leaf scorch", "northern leaf blight", "powdery mildew", 
    "septoria leaf spot", "spider mites two-spotted spider mite", 
    "target spot", "tomato mosaic virus", "tomato yellow leaf curl virus"
]

# Định nghĩa biến đổi ảnh
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Biến toàn cục để lưu mô hình
model = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    global model
    if model is None:
        model = models.resnet18(weights=None)
        num_classes = len(CLASSES)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
    return model

def predict_disease(image_path):
    try:
        model = load_model()
        image = Image.open(image_path).convert('RGB')
        image = data_transforms(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            image = image.to(device)
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            disease = CLASSES[predicted.item()]
            confidence = confidence.item() * 100
        plant_type = PLANT_TYPE_MAPPING.get(disease.lower(), 'Unknown')
        return disease, confidence, plant_type
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
    return redirect('login')

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
    if request.method == 'POST':
        image = request.FILES.get('plant_image')
        if image:
            valid_extensions = ['.jpg', '.jpeg', '.png']
            ext = os.path.splitext(image.name)[1].lower()
            if ext not in valid_extensions:
                messages.error(request, 'Vui lòng tải file ảnh (.jpg, .jpeg, .png).')
                return render(request, 'prediction.html')

            try:
                plant_image = PredictionHistory(image=image, user=request.user)
                plant_image.save()

                image_path = plant_image.image.path
                disease, confidence, plant_type = predict_disease(image_path)
                plant_image.disease = disease
                plant_image.confidence = confidence
                plant_image.plant_type = plant_type
                plant_image.save()

                # Thông báo cho admin khi có dự đoán mới
                admins = User.objects.filter(is_staff=True)
                for admin in admins:
                    Notification.objects.create(
                        recipient=admin,
                        message=f"Người dùng {request.user.username} đã thực hiện dự đoán bệnh: {disease}.",
                        link=reverse('admin_dashboard')
                    )

                return render(request, 'result.html', {
                    'image_url': plant_image.image.url,
                    'prediction': {'disease': disease, 'confidence': confidence, 'plant_type': plant_type}
                })
            except Exception as e:
                messages.error(request, f'Lỗi khi xử lý ảnh: {str(e)}')
                return render(request, 'prediction.html')
        else:
            messages.error(request, 'Vui lòng chọn một ảnh để dự đoán.')
    elif request.method == 'GET':
        return render(request, 'prediction.html')
    return redirect('prediction')

@login_required
def result(request):
    if request.method == 'POST':
        image = request.FILES.get('plant_image')
        if image:
            valid_extensions = ['.jpg', '.jpeg', '.png']
            ext = os.path.splitext(image.name)[1].lower()
            if ext not in valid_extensions:
                messages.error(request, 'Vui lòng tải file ảnh (.jpg, .jpeg, .png).')
                return render(request, 'result.html')

            try:
                plant_image = PredictionHistory(image=image, user=request.user)
                plant_image.save()

                image_path = plant_image.image.path
                disease, confidence, plant_type = predict_disease(image_path)
                plant_image.disease = disease
                plant_image.confidence = confidence
                plant_image.plant_type = plant_type
                plant_image.save()

                # Thông báo cho admin khi có dự đoán mới
                admins = User.objects.filter(is_staff=True)
                for admin in admins:
                    Notification.objects.create(
                        recipient=admin,
                        message=f"Người dùng {request.user.username} đã thực hiện dự đoán bệnh: {disease}.",
                        link=reverse('admin_dashboard')
                    )

                return render(request, 'result.html', {
                    'image_url': plant_image.image.url,
                    'prediction': {'disease': disease, 'confidence': confidence, 'plant_type': plant_type}
                })
            except Exception as e:
                messages.error(request, f'Lỗi khi xử lý ảnh: {str(e)}')
                return render(request, 'result.html')
        else:
            messages.error(request, 'Vui lòng chọn một ảnh để dự đoán.')
    elif request.method == 'GET':
        latest_image = PredictionHistory.objects.filter(user=request.user).order_by('-uploaded_at').first()
        if latest_image:
            if not latest_image.disease:
                disease, confidence, plant_type = predict_disease(latest_image.image.path)
                latest_image.disease = disease
                latest_image.confidence = confidence
                latest_image.plant_type = plant_type
                latest_image.save()
            return render(request, 'result.html', {
                'image_url': latest_image.image.url,
                'prediction': {'disease': latest_image.disease, 'confidence': latest_image.confidence, 'plant_type': latest_image.plant_type}
            })
        messages.info(request, 'Bạn chưa upload ảnh nào.')
        return redirect('prediction')
    return redirect('result')

@login_required
@user_passes_test(is_farmer)
def farmer_dashboard(request):
    if request.method == 'GET':
        answered_questions = Question.objects.filter(
            user=request.user,
            answers__isnull=False
        ).distinct().order_by('-created_at')[:5]

        question_data = []
        for question in answered_questions:
            latest_answer = question.answers.order_by('-created_at').first()
            if latest_answer:
                question_data.append({
                    'question': question,
                    'answer': latest_answer
                })

        return render(request, 'farmer_dashboard.html', {
            'title': 'Bảng điều khiển Nông dân',
            'question_data': question_data
        })
    return redirect('farmer_dashboard')

@login_required
@user_passes_test(is_expert)
def expert_dashboard(request):
    if request.method == 'GET':
        return render(request, 'expert_dashboard.html')
    return redirect('expert_dashboard')

@login_required
@user_passes_test(is_expert)
def expert_questions(request):
    if request.method == 'GET':
        questions = Question.objects.filter(Q(is_public=True) | Q(designated_expert=request.user)).order_by('-created_at')
        question_data = []
        for question in questions:
            latest_answer = question.answers.order_by('-created_at').first()
            user_answer = question.answers.filter(expert=request.user).first()
            question_data.append({
                'question': question,
                'latest_answer': latest_answer,
                'answer_count': question.answers.count(),
                'user_has_answered': bool(user_answer)
            })
        paginator = Paginator(question_data, 10)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        return render(request, 'expert_questions.html', {
            'question_data': page_obj
        })
    return redirect('expert_questions')
@login_required
def prediction_history(request):
    if request.method == 'GET':
        predictions = PredictionHistory.objects.filter(user=request.user).order_by('-uploaded_at')
        paginator = Paginator(predictions, 9)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        return render(request, 'prediction_history.html', {
            'title': 'Lịch sử dự đoán',
            'plant_images': page_obj
        })
    return redirect('prediction_history')

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

@login_required
@user_passes_test(lambda u: u.is_superuser)
def admin_dashboard(request):
    if request.method == 'GET':
        users_list = User.objects.all().order_by('-date_joined')
        for user in users_list:
            try:
                user.profile_info = user.profile
            except UserProfile.DoesNotExist:
                user.profile_info = None

        user_paginator = Paginator(users_list, 5)
        user_page_number = request.GET.get('user_page', 1)
        try:
            user_page_obj = user_paginator.page(user_page_number)
        except PageNotAnInteger:
            user_page_obj = user_paginator.page(1)
        except EmptyPage:
            user_page_obj = user_paginator.page(user_paginator.num_pages)

        predictions = PredictionHistory.objects.all().order_by('-uploaded_at')
        prediction_paginator = Paginator(predictions, 5)
        prediction_page_number = request.GET.get('prediction_page', 1)
        try:
            prediction_page_obj = prediction_paginator.page(prediction_page_number)
        except PageNotAnInteger:
            prediction_page_obj = prediction_paginator.page(1)
        except EmptyPage:
            prediction_page_obj = prediction_paginator.page(prediction_paginator.num_pages)

        feedbacks = Feedback.objects.all().order_by('-created_at')
        feedback_paginator = Paginator(feedbacks, 5)
        feedback_page_number = request.GET.get('feedback_page', 1)
        try:
            feedback_page_obj = feedback_paginator.page(feedback_page_number)
        except PageNotAnInteger:
            feedback_page_obj = feedback_paginator.page(1)
        except EmptyPage:
            feedback_page_obj = feedback_paginator.page(feedback_paginator.num_pages)

        context = {
            'users': user_page_obj,
            'prediction_history': prediction_page_obj,
            'feedbacks': feedback_page_obj,
            'current_date': timezone.now(),
        }
        return render(request, 'admin_dashboard.html', context)
    return redirect('admin_dashboard')

@login_required
@user_passes_test(is_admin)
def delete_user(request, user_id):
    if request.method == 'POST':
        user = get_object_or_404(User, id=user_id)
        if user.is_superuser:
            messages.error(request, 'Không thể xóa superuser.')
        else:
            try:
                user.delete()
                messages.success(request, f'Người dùng {user.username} đã được xóa.')
            except Exception as e:
                messages.error(request, f'Lỗi khi xóa người dùng: {str(e)}')
        return redirect('admin_dashboard')
    elif request.method == 'GET':
        user = get_object_or_404(User, id=user_id)
        return render(request, 'confirm_delete_user.html', {'user': user})
    return redirect('admin_dashboard')

@login_required
@user_passes_test(is_admin)
def delete_prediction(request, image_id):
    if request.method == 'POST':
        plant_image = get_object_or_404(PlantImage, id=image_id)
        try:
            if os.path.exists(plant_image.image.path):
                os.remove(plant_image.image.path)
            plant_image.delete()
            messages.success(request, 'Ảnh dự đoán đã được xóa.')
        except Exception as e:
            messages.error(request, f'Lỗi khi xóa ảnh: {str(e)}')
        return redirect('admin_dashboard')
    elif request.method == 'GET':
        plant_image = get_object_or_404(PlantImage, id=image_id)
        return render(request, 'confirm_delete_prediction.html', {'plant_image': plant_image})
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
            feedback.delete()
            messages.success(request, 'Phản hồi đã được xóa.')
        except Exception as e:
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
@user_passes_test(is_farmer)
def ask_expert(request):
    if request.method == 'POST':
        plant_image_id = request.POST.get('plant_image_id')
        question_text = request.POST.get('question_text')
        is_public = request.POST.get('is_public') == 'public'
        expert_identifier = request.POST.get('expert_identifier') if not is_public else None

        if question_text:
            plant_image = get_object_or_404(PlantImage, id=plant_image_id, user=request.user) if plant_image_id else None
            designated_expert = None

            if not is_public and expert_identifier:
                try:
                    designated_expert = User.objects.get(Q(username=expert_identifier) | Q(email=expert_identifier))
                    if not designated_expert.groups.filter(name='Expert').exists():
                        messages.error(request, 'Người dùng được chỉ định không phải là chuyên gia.')
                        return render(request, 'ask_expert.html', {'plant_images': PlantImage.objects.filter(user=request.user)})
                except User.DoesNotExist:
                    messages.error(request, 'Không tìm thấy chuyên gia với tên người dùng hoặc email này.')
                    return render(request, 'ask_expert.html', {'plant_images': PlantImage.objects.filter(user=request.user)})

            question = Question.objects.create(
                user=request.user,
                plant_image=plant_image,
                question_text=question_text,
                is_public=is_public,
                designated_expert=designated_expert
            )

            if is_public:
                experts = User.objects.filter(groups__name='Expert')
                for expert in experts:
                    Notification.objects.create(
                        recipient=expert,
                        message=f"Nông dân {request.user.username} đã đặt câu hỏi công khai: {question_text[:50]}...",
                        link=reverse('expert_questions')
                    )
            else:
                Notification.objects.create(
                    recipient=designated_expert,
                    message=f"Nông dân {request.user.username} đã đặt câu hỏi riêng tư: {question_text[:50]}...",
                    link=reverse('expert_questions')
                )

            messages.success(request, 'Câu hỏi đã được gửi.')
            return redirect('my_questions')
        else:
            messages.error(request, 'Vui lòng nhập câu hỏi.')
    elif request.method == 'GET':
        plant_images = PlantImage.objects.filter(user=request.user)
        return render(request, 'ask_expert.html', {'plant_images': plant_images})
    return redirect('ask_expert')

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

            # Lưu ảnh
            plant_image = PlantImage(
                user=request.user,
                image=image,
                uploaded_at=timezone.now()
            )
            plant_image.save()

            return JsonResponse({
                'success': True,
                'id': plant_image.id,
                'image_url': plant_image.image.url,
                'uploaded_at': plant_image.uploaded_at.strftime('%d/%m/%Y %H:%M')
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
@user_passes_test(is_expert)
def answer_question(request, question_id):
    question = get_object_or_404(Question, id=question_id)
    if not question.is_public and question.designated_expert != request.user:
        messages.error(request, 'Bạn không có quyền trả lời câu hỏi riêng tư này.')
        return redirect('expert_questions')

    answer = Answer.objects.filter(question=question, expert=request.user).first()
    is_editing = bool(answer)

    if request.method == 'POST':
        answer_text = request.POST.get('answer_text')
        if answer_text:
            if is_editing:
                answer.answer_text = answer_text
                answer.save()
                messages.success(request, 'Câu trả lời đã được cập nhật.')
            else:
                answer = Answer.objects.create(
                    question=question,
                    expert=request.user,
                    answer_text=answer_text
                )
                Notification.objects.create(
                    recipient=question.user,
                    message=f"Chuyên gia {request.user.username} đã trả lời câu hỏi của bạn: {question.question_text[:50]}...",
                    link=reverse('my_questions')
                )
                messages.success(request, 'Câu trả lời đã được gửi.')
            return redirect('expert_questions')
        else:
            messages.error(request, 'Vui lòng nhập câu trả lời.')
    elif request.method == 'GET':
        return render(request, 'answer_question.html', {
            'question': question,
            'answer': answer,
            'is_editing': is_editing
        })
    return redirect('expert_questions')

@login_required
@user_passes_test(is_expert)
def delete_answer(request, answer_id):
    answer = get_object_or_404(Answer, id=answer_id)
    if answer.expert != request.user:
        messages.error(request, 'Bạn không có quyền xóa câu trả lời này.')
        return redirect('expert_questions')
    if request.method == 'POST':
        answer.delete()
        messages.success(request, 'Câu trả lời đã được xóa.')
    elif request.method == 'GET':
        return render(request, 'confirm_delete_answer.html', {'answer': answer}) 
    return redirect('expert_questions')

@login_required
def disease_library(request):
    if request.method == 'POST':
        if not is_admin_or_expert(request.user):
            messages.error(request, 'Bạn không có quyền thêm bệnh.')  # Sửa lỗi thiếu dấu nháy
            return redirect('disease_list')
        
        name = request.POST.get('name')
        description = request.POST.get('description')
        symptoms = request.POST.get('symptoms')
        treatment = request.POST.get('treatment')
        image = request.FILES.get('image')
        
        if name and description and symptoms and treatment:
            try:
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
                # Thông báo cho admin khi có bệnh mới được thêm
                admins = User.objects.filter(is_staff=True)
                for admin in admins:
                    Notification.objects.create(
                        recipient=admin,
                        message=f"Bệnh mới {name} đã được thêm bởi {request.user.username}.",
                        link=reverse('disease_list')
                    )
                messages.success(request, 'Đã thêm bệnh mới.')
                return redirect('disease_list')
            except Exception as e:
                messages.error(request, f'Lỗi khi lưu bệnh: {str(e)}')
        else:
            missing_fields = []
            if not name: missing_fields.append('tên bệnh')
            if not description: missing_fields.append('mô tả')
            if not symptoms: missing_fields.append('triệu chứng')
            if not treatment: missing_fields.append('cách điều trị')
            messages.error(request, f'Vui lòng điền đầy đủ thông tin. Thiếu: {", ".join(missing_fields)}')
        
        diseases = DiseaseLibrary.objects.exclude(name__iexact='healthy')
        return render(request, 'disease_library.html', {
            'diseases': diseases,
            'is_expert_or_admin': is_admin_or_expert(request.user),
            'form_data': request.POST,
        })
    
    diseases = DiseaseLibrary.objects.exclude(name__iexact='healthy')
    return render(request, 'disease_library.html', {
        'diseases': diseases,
        'is_expert_or_admin': is_admin_or_expert(request.user),
    })

@login_required
def disease_list(request):
    if request.method == 'POST':
        return redirect('disease_library')
    
    diseases = DiseaseLibrary.objects.exclude(name__iexact='healthy').order_by('name')
    paginator = Paginator(diseases, 9)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    for disease in page_obj:
        if disease.image:
            disease.image_url = f'{settings.MEDIA_URL}{disease.image.name}'
        else:
            disease.image_url = '/static/images/placeholder.jpg'
    
    return render(request, 'disease_library.html', {
        'diseases': page_obj,
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
        return redirect('disease_list')
    return render(request, 'update_disease.html', {
        'disease': disease
    })

@login_required
@user_passes_test(is_admin_or_expert)
def delete_disease(request, disease_id):
    disease = get_object_or_404(DiseaseLibrary, id=disease_id)
    if request.method == 'POST':
        disease.delete()
        messages.success(request, 'Đã xóa bệnh thành công.')
        return redirect('disease_list')
    return render(request, 'confirm_delete_disease.html', {
        'disease': disease
    })

@login_required
def my_questions(request):
    if request.method == 'GET':
        questions = Question.objects.filter(user=request.user).order_by('-created_at')
        question_data = []
        for question in questions:
            answers = question.answers.order_by('created_at')  # Sớm nhất lên đầu
            recent_answers = question.answers.order_by('-created_at')[:3]  # Lấy 3 câu trả lời gần nhất
            # Phân trang câu trả lời
            answers_paginator = Paginator(answers, 20)  
            answer_page_number = request.GET.get(f'answer_page_{question.id}', 1)
            answers_page = answers_paginator.get_page(answer_page_number)
            question_data.append({
                'question': question,
                'recent_answers': recent_answers,  # Thay latest_answer bằng recent_answers
                'answer_count': answers.count(),
                'answers': answers_page,  # Trang câu trả lời
                'answers_paginator': answers_paginator  # Để hiển thị phân trang
            })
        paginator = Paginator(question_data, 9)  # Hiển thị 9 câu hỏi mỗi trang
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        return render(request, 'my_questions.html', {
            'title': 'Câu hỏi của tôi',
            'question_data': page_obj
        })
    return redirect('my_questions')

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
    notifications = Notification.objects.filter(recipient=request.user).order_by('-created_at')[:10]
    data = [{
        'id': n.id,
        'message': n.message,
        'link': n.link,
        'is_read': n.is_read,
        'created_at': n.created_at.strftime('%Y-%m-%d %H:%M:%S')
    } for n in notifications]
    unread_count = Notification.objects.filter(recipient=request.user, is_read=False).count()
    return JsonResponse({'notifications': data, 'unread_count': unread_count})

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

@login_required
def mark_all_notifications_read(request):
    Notification.objects.filter(recipient=request.user, is_read=False).update(is_read=True)
    return redirect('all_notifications')