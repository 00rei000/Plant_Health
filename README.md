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
Plant Health App is a web application that uses **Computer Vision** to analyze plant health through images. Built with **Django** for the back-end and **Bootstrap**, **HTML**, **CSS**, and **JavaScript** for the front-end, it integrates the **ResNet18** deep learning model to identify 21 types of plant diseases. The app supports **Farmers**, **Experts**, and **Admins** in managing plant health, with **Microsoft SQL Server** as the database and a **Jupyter Notebook** for model training documentation.

## Table of Contents
- [English Version](#english-version)
- [Japanese Version](#japanese-version)

## English Version <a id="english-version"></a>

### Overview
This project leverages **Computer Vision** to detect plant diseases, enabling farmers to monitor crop health and seek expert advice efficiently. It addresses challenges like delayed disease identification and lack of accessible expert consultation in agriculture.

### Key Features
- **Plant Disease Prediction**: Upload images to predict 21 diseases using **ResNet18**, view results (disease name, confidence score, plant type), and store prediction history.
- **Farmer-Expert Interaction**: Farmers submit questions with images; Experts provide, edit, or delete responses.
- **Disease Library**: Experts/Admins manage disease information (name, symptoms, treatment, images).
- **User Management**: Supports **Farmer**, **Expert**, and **Admin** roles with registration, login, and profile updates.
- **System Management (Admin)**: Manage users, predictions, and feedback with search and pagination.
- **User-Friendly Interface**: Responsive design with **Bootstrap 5.3.0**, dark mode, and role-based navigation.

### Technologies Used
- **Back-end**: Django 4.2.13, PyTorch 2.0.1 (ResNet18), mssql-django 1.5, pyodbc 5.2.0, django-environ 0.12.0, Pillow 10.0.0, opencv-python 4.8.0.76, djangorestframework 3.16.0, djangorestframework-simplejwt 5.5.0, django-cors-headers 4.7.0.
- **Front-end**: HTML, CSS, JavaScript, Bootstrap 5.3.0, django-bootstrap-v5 1.0.11.
- **Database**: Microsoft SQL Server.
- **Environment**: Anaconda, Jupyter Notebook for model training.
- **File System**: Django Media for image storage.

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/00rei000/Plant_Health.git
   cd Plant_Health
   ```

2. **Set up Anaconda environment**:
   ```bash
   conda create -n plant_health_app python=3.8
   conda activate plant_health_app
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download AI models**:
   - Ensure `class_names.json` is in `plant_health_app/data/`.

5. **Configure Microsoft SQL Server**:
   - Create database `plant_health_db` and install **ODBC Driver 17 for SQL Server** from [Microsoft](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server).
   - Update `settings.py` or create `.env`:
     ```plaintext
     SECRET_KEY=your-secret-key
     DATABASE_URL=sqlserver://your_username:your_password@localhost:1433/plant_health_db?driver=ODBC+Driver+17+for+SQL+Server
     ```

6. **Set up media and static files**:
   - Create `media/` directory with write permissions (`chmod 755 media/` on Linux).
   - Run:
     ```bash
     python manage.py collectstatic
     python manage.py migrate
     python manage.py createsuperuser
     ```

7. **Run the server**:
   ```bash
   python manage.py runserver
   ```
   Access at `http://localhost:8000`.

### Usage
- **Register/Login**: Create an account at `/register` and log in at `/login`.
- **Predict Disease**: Upload images at `/prediction` and view history at `/prediction_history`.
- **Ask Experts**: Farmers submit questions at `/ask_expert`; Experts respond at `/expert_questions`.
- **Manage Diseases**: Experts/Admins update disease info at `/disease_library` or `/disease_list`.
- **Admin Tasks**: Manage users and data at `/admin_dashboard` or `/manage`.
- **Feedback**: Submit feedback at `/feedback`.

## Japanese Version <a id="japanese-version"></a>
日本語版

### 概要
このプロジェクトは、**コンピュータビジョン**を活用して植物の病気を検出し、農家が作物の健康を監視し、専門家に効率的に相談できるようにします。農業における病気特定の遅延や専門家へのアクセスの不足といった課題を解決します。

### 主な機能
- **植物病気の予測**: **ResNet18**を使用して21種類の病気を予測し、結果（病名、信頼スコア、植物の種類）を表示、予測履歴を保存。
- **農家と専門家の対話**: 農家は画像付きで質問を送信、専門家は回答を追加、編集、削除可能。
- **病気ライブラリ**: 専門家/管理者が病気の情報（名前、症状、治療法、画像）を管理。
- **ユーザー管理**: **農家**、**専門家**、**管理者**の役割をサポートし、登録、ログイン、プロフィール更新が可能。
- **システム管理（管理者）**: ユーザー、予測、フィードバックを検索およびページネーションで管理。
- **ユーザーフレンドリーなインターフェース**: **Bootstrap 5.3.0**を使用したレスポンシブデザイン、ダークモード、役割ベースのナビゲーション。

### 使用技術
- **バックエンド**: Django 4.2.13、PyTorch 2.0.1（ResNet18）、mssql-django 1.5、pyodbc 5.2.0、django-environ 0.12.0、Pillow 10.0.0、opencv-python 4.8.0.76、djangorestframework 3.16.0、djangorestframework-simplejwt 5.5.0、django-cors-headers 4.7.0。
- **フロントエンド**: HTML、CSS、JavaScript、Bootstrap 5.3.0、django-bootstrap-v5 1.0.11。
- **データベース**: Microsoft SQL Server。
- **環境**: Anaconda、モデルトレーニング用のJupyter Notebook。
- **ファイルシステム**: 画像保存用のDjango Media。

### インストール
1. **リポジトリのクローン**:
   ```bash
   git clone https://github.com/00rei000/Plant_Health.git
   cd Plant_Health
   ```

2. **Anaconda環境の設定**:
   ```bash
   conda create -n plant_health_app python=3.8
   conda activate plant_health_app
   ```

3. **依存関係のインストール**:
   ```bash
   pip install -r requirements.txt
   ```

4. **AIモデルのダウンロード**:
   - `class_names.json`が`plant_health_app/data/`にあることを確認。

5. **Microsoft SQL Serverの設定**:
   - `plant_health_db`データベースを作成し、[Microsoft](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)から**ODBC Driver 17 for SQL Server**をインストール。
   - `settings.py`を更新または`.env`ファイルを作成:
     ```plaintext
     SECRET_KEY=your-secret-key
     DATABASE_URL=sqlserver://your_username:your_password@localhost:1433/plant_health_db?driver=ODBC+Driver+17+for+SQL+Server
     ```

6. **メディアと静的ファイルの設定**:
   - `media/`ディレクトリを作成し、書き込み権限を付与（Linuxでは`chmod 755 media/`）。
   - 以下を実行:
     ```bash
     python manage.py collectstatic
     python manage.py migrate
     python manage.py createsuperuser
     ```

7. **サーバーの実行**:
   ```bash
   python manage.py runserver
   ```
   `http://localhost:8000`でアクセス。

### 使用方法
- **登録/ログイン**: `/register`でアカウントを作成し、`/login`でログイン。
- **病気予測**: `/prediction`で画像をアップロードし、`/prediction_history`で履歴を確認。
- **専門家への質問**: 農家は`/ask_expert`で質問を送信、専門家は`/expert_questions`で回答。
- **病気ライブラリの管理**: 専門家/管理者は`/disease_library`または`/disease_list`で情報を更新。
- **管理者タスク**: `/admin_dashboard`または`/manage`でユーザーとデータを管理。
- **フィードバック**: `/feedback`でフィードバックを送信。
