from django.contrib.auth.models import Group

def user_roles(request):
    return {
        'is_farmer': request.user.is_authenticated and request.user.groups.filter(name='Farmer').exists(),
        'is_expert': request.user.is_authenticated and request.user.groups.filter(name='Expert').exists(),
        'is_admin': request.user.is_authenticated and request.user.is_staff,
    }