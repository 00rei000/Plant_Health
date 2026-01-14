# Plant Health App

A Django web application that uses **ResNet18** to diagnose plant diseases from images. Features role-based access for Farmers, Experts, and Admins with blog system and Q&A functionality.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.2.13-green.svg)](https://www.djangoproject.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)

## Overview

**Tech Stack**: Django 4.2 + PyTorch ResNet18 + SQL Server + Bootstrap 5  
**AI Model**: ResNet18 trained on 20 plant disease classes  
**User Roles**: Farmer (upload & ask) | Expert (answer & contribute) | Admin (manage & moderate)

## Features

- **Disease Detection**: Upload plant images → ResNet18 inference → Display disease name, confidence score, treatment recommendations
- **Blog System**: Create/approve posts, rejection reasons, comment system with role badges
- **Q&A Platform**: Farmers ask questions, Experts answer with notifications
- **Disease Library**: Searchable database with symptoms & treatments
- **Dashboards**: Role-specific analytics with Chart.js, 8 quick action shortcuts
- **Admin Tools**: User management, blog moderation, batch approval/rejection

**Recent Updates (Nov 2025)**:
- Approval/rejection modals with reason field
- Single-click upload fix
- Multi-table pagination fix
- Quick actions menu (8 buttons per role)

## Tech Stack

- **Backend**: Django 4.2, PyTorch 2.0, torchvision, mssql-django, Pillow, OpenCV
- **Frontend**: Bootstrap 5.3, Chart.js 4.4, JavaScript ES6
- **Database**: Microsoft SQL Server / SQLite
- **AI**: ResNet18 (21 disease classes)

## Quick Start

```bash
# 1. Clone & setup environment
git clone https://github.com/00rei000/Plant_Health.git
cd Plant_Health
pip install -r requirements.txt

# 2. Configure .env file
SECRET_KEY=your-key
DATABASE_URL=sqlite:///db.sqlite3  # or SQL Server connection string

# 3. Download AI models (place in plant_health_app/notebook/plant_disease/checkpoints/)
# best_model.pth and plant_type_model.pth

# 4. Setup database
python manage.py migrate
python manage.py createsuperuser
python manage.py collectstatic --noinput

# 5. Run server
python manage.py runserver
# Access: http://127.0.0.1:8000/
```

**Requirements**: Python 3.8+, SQL Server (or SQLite for dev)

## Usage

**Farmers**: Upload images → Get diagnosis → Ask experts → View history  
**Experts**: Answer questions → Add disease info → Create blog posts  
**Admins**: Moderate blogs (approve/reject with reasons) → Manage users → View analytics

## Project Structure

```
Plant_Health/
├── mysite/                    # Django settings, URLs
├── plant_health_app/
│   ├── models.py              # BlogPost, DiseaseLibrary, Notification, etc.
│   ├── views.py               # prediction(), blog_approve(), admin_dashboard()
│   ├── templates/             # HTML files (prediction, blog, dashboards)
│   ├── static/                # CSS, JS, images
│   ├── notebook/              # Model training (best_model.pth)
│   └── migrations/            # DB schema changes
├── scripts/                   # import_disease.py, check_inference.py
├── media/                     # User uploads (gitignored)
└── requirements.txt           # Dependencies
```

## Key Components

**Views** ([`views.py`](plant_health_app/views.py)): `prediction()`, `predict_disease()`, `blog_approve()`, `blog_reject()`, `admin_dashboard()`  
**Models** ([`models.py`](plant_health_app/models.py)): `BlogPost`, `DiseaseLibrary`, `PredictionHistory`, `Notification`  
**URLs**: `/prediction/`, `/blog/`, `/blog/pending/`, `/my-blog/`, `/admin_dashboard/`  
**Training**: [`notebook/plant_disease_model.ipynb`](plant_health_app/notebook/plant_disease/train_model.ipynb)

## Production Notes

**Performance**: Inference runs synchronously (blocking) → Consider Celery/separate service for production  
**Security**: File validation (jpg/png), HTTPS, env variables, CSRF enabled  
**Scalability**: Move media to S3, use CDN for static files, add DB indexes  
**Testing**: Add unit tests, mock inference, E2E tests

## Interview Tips

**Architecture**: ResNet18 (accuracy/speed balance), SQL Server (transactions), Django MVT, role-based access (`@user_passes_test`)  
**Challenges Solved**: Async inference (Celery proposal), double-click bug (event delegation), pagination (query params), model caching (singleton)  
**Future Work**: Microservices, mobile app, WebSockets, model versioning, API layer (DRF), CI/CD

**Key Metrics**: 20 disease classes, 3 roles, 20+ views, 30+ templates

## Author

**00rei000** - [GitHub](https://github.com/00rei000)