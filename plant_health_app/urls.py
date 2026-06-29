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
    path('profile/', views.profile, name='profile'),
    path('change_password/', views.change_password, name='change_password'),
    path('disease_library/', views.disease_library, name='disease_library'),
    path('update_disease/<int:disease_id>/', views.update_disease, name='update_disease'),
    path('delete_disease/<int:disease_id>/', views.delete_disease, name='delete_disease'),
    path('notifications/get/', views.get_notifications, name='get_notifications'),
    path('notifications/mark-read/<int:notification_id>/', views.mark_notification_read, name='mark_notification_read'),
    path('notifications/all/', views.all_notifications, name='all_notifications'),
    path('notifications/mark-all-read/', views.mark_all_notifications_read, name='mark_all_notifications_read'),
    path('weather/', views.get_weather, name='get_weather'),
    path('upload-plant-image/', views.upload_plant_image, name='upload_plant_image'),
    path('add_disease/', views.add_disease, name='add_disease'),
    path('prediction_detail/<int:prediction_id>/', views.prediction_detail, name='prediction_detail'),
    path('prediction_history/delete/<int:prediction_id>/', views.delete_prediction_history, name='delete_prediction_history'),
    path('plant_image/<int:image_id>/', views.plant_image_detail, name='plant_image_detail'),
    path('undo_delete/<int:deleted_id>/', views.undo_delete, name='undo_delete'),
    
    # Contribution workflow URLs
    path('user_confirm/<int:prediction_id>/', views.user_confirm_prediction, name='user_confirm_prediction'),
    path('admin/moderation/', views.admin_moderation, name='admin_moderation'),
    path('admin/approve/<int:prediction_id>/', views.approve_contribution, name='approve_contribution'),
    path('admin/reject/<int:prediction_id>/', views.reject_contribution, name='reject_contribution'),
    
    # Blog URLs
    path('blog/', views.blog_list, name='blog_list'),
    path('blog/<int:post_id>/', views.blog_detail, name='blog_detail'),
    path('blog/create/', views.blog_create, name='blog_create'),
    path('blog/<int:post_id>/edit/', views.blog_edit, name='blog_edit'),
    path('blog/<int:post_id>/delete/', views.blog_delete, name='blog_delete'),
    path('my-blog/', views.my_blog_posts, name='my_blog_posts'),
    path('blog/pending/', views.blog_pending, name='blog_pending'),
    path('blog/<int:post_id>/approve/', views.blog_approve, name='blog_approve'),
    path('blog/<int:post_id>/reject/', views.blog_reject, name='blog_reject'),
    path('blog/comment/<int:comment_id>/delete/', views.blog_comment_delete, name='blog_comment_delete'),
    
    # Dataset Management URLs
    path('admin/datasets/', views.manage_datasets, name='manage_datasets'),
    path('admin/datasets/create/', views.create_dataset, name='create_dataset'),
    path('admin/datasets/<int:dataset_id>/process/', views.process_dataset, name='process_dataset'),
    path('admin/datasets/<int:dataset_id>/', views.view_dataset, name='view_dataset'),
    path('admin/datasets/<int:dataset_id>/delete/', views.delete_dataset, name='delete_dataset'),
    path('admin/datasets/<int:dataset_id>/export/', views.export_dataset, name='export_dataset'),
    path('admin/datasets/<int:dataset_id>/export/progress/<uuid:task_id>/', views.export_progress, name='export_progress'),
    path('admin/datasets/<int:dataset_id>/download/<uuid:task_id>/', views.download_export, name='download_export'),
    
    # Model Management URLs
    path('admin/models/', views.manage_models, name='manage_models'),
    path('admin/models/add/', views.add_model, name='add_model'),
    path('admin/models/<str:model_type>/<int:model_id>/edit/', views.edit_model, name='edit_model'),
    path('admin/models/<str:model_type>/<int:model_id>/set-default/', views.set_default_model, name='set_default_model'),
    path('admin/models/<str:model_type>/<int:model_id>/delete/', views.delete_model, name='delete_model'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)