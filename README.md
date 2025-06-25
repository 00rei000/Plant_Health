# Plant Health App

Plant Health App is a web application that uses **Computer Vision** to analyze plant health through images. Built with **Django** for the back-end and **Bootstrap**, **HTML**, **CSS**, and **JavaScript** for the front-end, it integrates the **ResNet18** deep learning model to identify 21 types of plant diseases. The app supports **Farmers**, **Experts**, and **Admins** in managing plant health, with **Microsoft SQL Server** as the database and a **Jupyter Notebook** for model training documentation.

## Table of Contents
- [English Version](#english-version)
- [Japanese Version](#japanese-version)

## English Version

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

   Sample `requirements.txt`:
   ```plaintext
   Django==4.2.13
   django-bootstrap-v5==1.0.11
   django-cors-headers==4.7.0
   django-environ==0.12.0
   djangorestframework==3.16.0
   djangorestframework-simplejwt==5.5.0
   Pillow==10.0.0
   opencv-python==4.8.0.76
   torch==2.0.1
   torchvision==0.15.2
   pyodbc==5.2.0
   mssql-django==1.5
   ```

4. **Download AI models**:
   - Download `best_model.pth` and `plant_disease_model.pth` from [Google Drive](https://drive.google.com/your-link) and place in `plant_health_app/models/`.
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

### Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit changes (`git commit -m 'Add feature XYZ'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Create a Pull Request.

## Japanese Version / 日本語版

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

   サンプル `requirements.txt`:
   ```plaintext
   Django==4.2.13
   django-bootstrap-v5==1.0.11
   django-cors-headers==4.7.0
   django-environ==0.12.0
   djangorestframework==3.16.0
   djangorestframework-simplejwt==5.5.0
   Pillow==10.0.0
   opencv-python==4.8.0.76
   torch==2.0.1
   torchvision==0.15.2
   pyodbc==5.2.0
   mssql-django==1.5
   ```

4. **AIモデルのダウンロード**:
   - [Google Drive](https://drive.google.com/your-link)から`best_model.pth`と`plant_disease_model.pth`をダウンロードし、`plant_health_app/models/`に配置。
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

### 貢献
貢献を歓迎します！貢献するには：
1. リポジトリをフォーク。
2. 新しいブランチを作成（`git checkout -b feature/your-feature-name`）。
3. 変更をコミット（`git commit -m 'Add feature XYZ'`）。
4. ブランチにプッシュ（`git push origin feature/your-feature-name`）。
5. プルリクエストを作成。
