// static/js/upload.js
document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('upload-form');
    if (!uploadForm) {
        return; // Thoát sớm nếu không có upload-form
    }

    const imageInput = document.getElementById('plant-image');
    const imagePreview = document.getElementById('image-preview');

    if (imageInput && imagePreview) {
        imageInput.addEventListener('change', function () {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function (e) {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            } else {
                imagePreview.style.display = 'none';
            }
        });
    } else {
        console.warn('Missing elements: imageInput or imagePreview in upload form');
    }
});