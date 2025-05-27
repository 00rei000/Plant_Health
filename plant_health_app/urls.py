from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login, name='login'),
    path('register/', views.register, name='register'),
    path('prediction/', views.prediction, name='prediction'),
    path('result/', views.result, name='result'),
    path('prediction_history/', views.prediction_history, name='prediction_history'),
    path('feedback/', views.feedback_view, name='feedback'),
    path('logout/', views.logout_view, name='logout'),
    path('admin/', views.admin_dashboard, name='admin_dashboard'),
    path('admin/delete_user/<int:user_id>/', views.delete_user, name='delete_user'),
    path('admin/delete_prediction/<int:image_id>/', views.delete_prediction, name='delete_prediction'),
    path('admin/edit_feedback/<int:feedback_id>/', views.edit_feedback, name='edit_feedback'),
    path('admin/delete_feedback/<int:feedback_id>/', views.delete_feedback, name='delete_feedback'),
    path('manage/', views.manage, name='manage'),
    path('farmer_dashboard/', views.farmer_dashboard, name='farmer_dashboard'),
    path('expert_dashboard/', views.expert_dashboard, name='expert_dashboard'),
    path('ask_expert/', views.ask_expert, name='ask_expert'),
    path('expert_questions/', views.expert_questions, name='expert_questions'),
    path('answer_question/<int:question_id>/', views.answer_question, name='answer_question'),
    path('my_questions/', views.my_questions, name='my_questions'),
    path('delete_answer/<int:question_id>/', views.delete_answer, name='delete_answer'),
    path('profile/', views.profile, name='profile'),  
    path('change_password/', views.change_password, name='change_password'),
    path('disease_library/', views.disease_library, name='disease_library'),
    path('disease_list/', views.disease_list, name='disease_list'),
    path('update_disease/<int:disease_id>/', views.update_disease, name='update_disease'),
    path('delete_disease/<int:disease_id>/', views.delete_disease, name='delete_disease'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)