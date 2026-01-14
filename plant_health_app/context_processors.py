from .models import BlogPost

def pending_posts_count(request):
    """Context processor to add pending blog posts count to all templates."""
    if request.user.is_authenticated and request.user.is_staff:
        count = BlogPost.objects.filter(status='pending').count()
        return {'pending_posts_count': count}
    return {'pending_posts_count': 0}
