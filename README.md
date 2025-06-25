# Smart Grocery Management System / スマート食料管理システム

![Smart Grocery Management System](https://github.com/user-attachments/assets/placeholder-image.jpg)

The **Smart Grocery Management System** is a web application designed to help households manage food inventory, plan meals, and track consumption efficiently. It supports three user roles: **Home Cook** (manages meal plans, food inventory, and shopping lists), **Family Member** (views consumption reports and collaborates on shopping lists), and **Admin** (manages users and system performance). Built with **Django** for the back-end and **React (Vite)** for the front-end, it uses **Microsoft SQL Server** as the database to streamline grocery management and reduce food waste.

## Table of Contents
- [English Version](#english-version)
  - [Features](#features-en)
  - [Technologies Used](#technologies-used-en)
  - [Installation](#installation-en)
  - [Project Structure](#project-structure-en)
  - [Usage](#usage-en)
  - [Contributing](#contributing-en)
  - [License](#license-en)
- [Japanese Version](#japanese-version)
  - [特徴](#特徴-ja)
  - [使用技術](#使用技術-ja)
  - [インストール](#インストール-ja)
  - [プロジェクト構造](#プロジェクト構造-ja)
  - [使用方法](#使用方法-ja)
  - [貢献](#貢献-ja)
  - [ライセンス](#ライセンス-ja)

## English Version

### Features <a name="features-en"></a>
- **Meal Planning & Food Management**:
  - Home Cooks can create meal plans, manage refrigerator inventory (input, categorize, track expiration dates), and receive notifications for expiring food.
  - Store and search food inventory history.
- **Shopping List Collaboration**:
  - Create, categorize (e.g., vegetables, meat), and share shopping lists among family members.
- **Consumption Tracking**:
  - Family Members can view reports on food consumption, expired or wasted food, and analyze consumption trends.
- **System Management (Admin)**:
  - Manage user accounts, data categories (e.g., food groups, units), user-generated content, and monitor system performance with search and pagination.
- **User Management**:
  - Supports **Home Cook**, **Family Member**, and **Admin** roles with registration, login, and profile updates.
- **User-Friendly Interface**:
  - Responsive design with **React** and **Tailwind CSS** (optional for styling).
  - Role-based navigation and dynamic dashboards.

### Technologies Used <a name="technologies-used-en"></a>
- **Back-end**:
  - **Django 4.2.13**: Python web framework for logic, authentication, and database management.
  - **mssql-django 1.5**: Django database backend for Microsoft SQL Server.
  - **pyodbc 5.2.0**: ODBC driver for SQL Server connectivity.
  - **django-environ 0.12.0**: Environment variable management.
- **Front-end**:
  - **React 18**: JavaScript library for building user interfaces.
  - **Vite**: Front-end build tool for fast development.
  - **Tailwind CSS** (optional): Utility-first CSS framework for styling.
  - **Axios**: HTTP client for API requests.
- **Database**: Microsoft SQL Server.
- **Environment**: **Anaconda** for virtual environment management.

### Installation <a name="installation-en"></a>
#### Requirements
- Python 3.8+
- Node.js 18+
- Anaconda
- Microsoft SQL Server (with a running instance and database created)
- ODBC Driver 17 for SQL Server
- Modern browser (Chrome, Firefox, etc.)

#### Installation Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/smart-grocery-management-system.git
   cd smart-grocery-management-system
   ```

2. **Create an Anaconda virtual environment**:
   ```bash
   conda create -n food-management python=3.8
   conda activate food-management
   ```

3. **Install backend dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   pip install pyodbc
   ```

   Sample `backend/requirements.txt`:
   ```plaintext
   Django==4.2.13
   mssql-django==1.5
   pyodbc==5.2.0
   django-environ==0.12.0
   ```

4. **Configure Microsoft SQL Server**:
   - Ensure a SQL Server instance is running and a database is created (e.g., `food_management`).
   - Install the **ODBC Driver 17 for SQL Server** from [Microsoft](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server).
   - Create database:
     ```sql
     CREATE DATABASE food_management;
     GO
     ```
   - Update `backend/settings.py`:
     ```python
     DATABASES = {
         'default': {
             'ENGINE': 'mssql',
             'NAME': 'food_management',
             'USER': 'your-username',
             'PASSWORD': 'your-password',
             'HOST': 'localhost',
             'PORT': '',
             'OPTIONS': {
                 'driver': 'ODBC Driver 17 for SQL Server',
             },
         }
     }
     ```
   - Alternatively, create a `.env` file in `backend/`:
     ```plaintext
     SECRET_KEY=your-django-secret-key
     DEBUG=True
     DATABASE_URL=sqlserver://your-username:your-password@localhost:1433/food_management?driver=ODBC+Driver+17+for+SQL+Server
     EMAIL_HOST=smtp.gmail.com
     EMAIL_PORT=587
     EMAIL_HOST_USER=your-email@gmail.com
     EMAIL_HOST_PASSWORD=your-email-password
     ```

5. **Run backend migrations**:
   ```bash
   cd backend
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Create a superuser (Admin)**:
   ```bash
   python manage.py createsuperuser
   ```

7. **Install frontend dependencies**:
   ```bash
   cd ../frontend
   npm install
   ```

8. **Configure frontend environment**:
   - Create a `.env` file in `frontend/`:
     ```plaintext
     VITE_API_URL=http://localhost:8000/api
     ```

9. **Run the application**:
   - Start backend:
     ```bash
     cd backend
     conda activate food-management
     python manage.py runserver
     ```
     Backend runs at: `http://localhost:8000`
   - Start frontend:
     ```bash
     cd frontend
     npm run dev
     ```
     Frontend runs at: `http://localhost:3000`

### Project Structure <a name="project-structure-en"></a>
```plaintext
smart-grocery-management-system/
├── backend/
│   ├── static/               # Static files (CSS, JS, images)
│   ├── templates/            # HTML templates (optional for Django-rendered views)
│   ├── __init__.py
│   ├── settings.py           # Django configuration
│   ├── urls.py               # URL routing
│   ├── views.py              # API endpoints
│   ├── models.py             # Data models (User, FoodInventory, ShoppingList, etc.)
│   └── requirements.txt      # Backend dependencies
├── frontend/
│   ├── src/                  # React source code
│   ├── public/               # Static assets
│   ├── .env                  # Frontend environment variables
│   ├── package.json          # Frontend dependencies
│   └── vite.config.js        # Vite configuration
├── manage.py                 # Django management script
├── .gitignore                # Excludes .conda/, __pycache__, media/, etc.
├── .env.example              # Sample environment variables
└── README.md                 # This file
```

### Usage <a name="usage-en"></a>
1. **Register/Login**:
   - Access `/register` to create an account (Home Cook or Family Member).
   - Log in at `/login` to access role-based dashboards.
2. **Manage Food Inventory**:
   - Home Cooks: Add, categorize, and track food at `/inventory`.
   - View expiration notifications at `/notifications`.
3. **Create Meal Plans**:
   - Home Cooks: Plan meals at `/meal-plans`.
4. **Collaborate on Shopping Lists**:
   - Home Cooks/Family Members: Create and share lists at `/shopping-lists`.
5. **View Consumption Reports**:
   - Family Members: Access reports at `/reports`.
6. **System Management (Admin)**:
   - Access `/admin` to manage users, data, and performance.

### Contributing <a name="contributing-en"></a>
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit changes (`git commit -m 'Add feature XYZ'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Create a Pull Request.

---

## Japanese Version / 日本語版

### スマート食料管理システム
**スマート食料管理システム**は、家庭での食品在庫管理、食事計画、消費追跡を効率的に行うためのウェブアプリケーションです。食品の無駄を減らし、食料計画を効率化したい家庭に最適です。3つの役割をサポートします：**ホームクック**（食事計画、食品在庫、買い物リストの管理）、**家族メンバー**（消費レポートの閲覧、買い物リストの共有）、**管理者**（ユーザーおよびシステムパフォーマンスの管理）。**Django**をバックエンドに、**React (Vite)**をフロントエンドに使用し、**Microsoft SQL Server**をデータベースとして採用しています。

### 特徴 <a name="特徴-ja"></a>
- **食事計画と食品管理**:
  - ホームクックは食事計画を作成し、冷蔵庫の在庫を管理（入力、カテゴリ分け、賞味期限追跡）し、期限切れの通知を受け取れます。
  - 在庫履歴の保存と検索。
- **買い物リストの共同作業**:
  - 買い物リストを作成、カテゴリ分け（例：野菜、肉）、家族間で共有。
- **消費追跡**:
  - 家族メンバーは食品消費、期限切れ、または無駄になった食品のレポートを閲覧し、消費傾向を分析。
- **システム管理（管理者）**:
  - ユーザーアカウント、データカテゴリ（例：食品グループ、単位）、ユーザー生成コンテンツを管理し、検索とページネーションでシステムパフォーマンスを監視。
- **ユーザー管理**:
  - **ホームクック**、**家族メンバー**、**管理者**の役割をサポートし、登録、ログイン、プロフィール更新を提供。
- **ユーザーフレンドリーなインターフェース**:
  - **React**と**Tailwind CSS**（オプション）によるレスポンシブデザイン。
  - 役割ベースのナビゲーションと動的ダッシュボード。

### 使用技術 <a name="使用技術-ja"></a>
- **バックエンド**:
  - **Django 4.2.13**: ロジック、認証、データベース管理のためのPythonウェブフレームワーク。
  - **mssql-django 1.5**: Microsoft SQL Server用のDjangoデータベースバックエンド。
  - **pyodbc 5.2.0**: SQL Server接続用のODBCドライバ。
  - **django-environ 0.12.0**: 環境変数管理。
- **フロントエンド**:
  - **React 18**: ユーザーインターフェース構築用のJavaScriptライブラリ。
  - **Vite**: 高速開発用のフロントエンドビルドツール。
  - **Tailwind CSS**（オプション）：ユーティリティ優先のCSSフレームワーク。
  - **Axios**: APIリクエスト用のHTTPクライアント。
- **データベース**: Microsoft SQL Server。
- **環境**: **Anaconda**による仮想環境管理。

### インストール <a name="インストール-ja"></a>
#### 要件
- Python 3.8+
- Node.js 18+
- Anaconda
- Microsoft SQL Server（動作中のインスタンスと作成済みのデータベース）
- ODBC Driver 17 for SQL Server
- 最新のブラウザ（Chrome、Firefoxなど）

#### インストール手順
1. **リポジトリのクローン**:
   ```bash
   git clone https://github.com/your-username/smart-grocery-management-system.git
   cd smart-grocery-management-system
   ```

2. **Anaconda仮想環境の作成**:
   ```bash
   conda create -n food-management python=3.8
   conda activate food-management
   ```

3. **バックエンド依存関係のインストール**:
   ```bash
   pip install -r backend/requirements.txt
   pip install pyodbc
   ```

   サンプル `backend/requirements.txt`:
   ```plaintext
   Django==4.2.13
   mssql-django==1.5
   pyodbc==5.2.0
   django-environ==0.12.0
   ```

4. **Microsoft SQL Serverの設定**:
   - SQL Serverインスタンスが動作中で、データベース（例：`food_management`）が作成されていることを確認。
   - [Microsoft](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)から**ODBC Driver 17 for SQL Server**をインストール。
   - データベースを作成:
     ```sql
     CREATE DATABASE food_management;
     GO
     ```
   - `backend/settings.py`を更新:
     ```python
     DATABASES = {
         'default': {
             'ENGINE': 'mssql',
             'NAME': 'food_management',
             'USER': 'your-username',
             'PASSWORD': 'your-password',
             'HOST': 'localhost',
             'PORT': '',
             'OPTIONS': {
                 'driver': 'ODBC Driver 17 for SQL Server',
             },
         }
     }
     ```
   - または、`backend/` に `.env` ファイルを作成:
     ```plaintext
     SECRET_KEY=your-django-secret-key
     DEBUG=True
     DATABASE_URL=sqlserver://your-username:your-password@localhost:1433/food_management?driver=ODBC+Driver+17+for+SQL+Server
     EMAIL_HOST=smtp.gmail.com
     EMAIL_PORT=587
     EMAIL_HOST_USER=your-email@gmail.com
     EMAIL_HOST_PASSWORD=your-email-password
     ```

5. **バックエンドマイグレーションの実行**:
   ```bash
   cd backend
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **スーパーユーザー（管理者）の作成**:
   ```bash
   python manage.py createsuperuser
   ```

7. **フロントエンド依存関係のインストール**:
   ```bash
   cd ../frontend
   npm install
   ```

8. **フロントエンド環境の設定**:
   - `frontend/` に `.env` ファイルを作成:
     ```plaintext
     VITE_API_URL=http://localhost:8000/api
     ```

9. **アプリケーションの実行**:
   - バックエンドの起動:
     ```bash
     cd backend
     conda activate food-management
     python manage.py runserver
     ```
     バックエンドは `http://localhost:8000` で動作。
   - フロントエンドの起動:
     ```bash
     cd frontend
     npm run dev
     ```
     フロントエンドは `http://localhost:3000` で動作。

### プロジェクト構造 <a name="プロジェクト構造-ja"></a>
```plaintext
smart-grocery-management-system/
├── backend/
│   ├── static/               # 静的ファイル（CSS、JS、画像）
│   ├── templates/            # HTMLテンプレート（Djangoレンダリングビュー用、オプション）
│   ├── __init__.py
│   ├── settings.py           # Django設定
│   ├── urls.py               # URLルーティング
│   ├── views.py              # APIエンドポイント
│   ├── models.py             # データモデル（User、FoodInventory、ShoppingListなど）
│   └── requirements.txt      # バックエンド依存関係
├── frontend/
│   ├── src/                  # Reactソースコード
│   ├── public/               # 静的アセット
│   ├── .env                  # フロントエンド環境変数
│   ├── package.json          # フロントエンド依存関係
│   └── vite.config.js        # Vite設定
├── manage.py                 # Django管理スクリプト
├── .gitignore                # .conda/、__pycache__、media/などを除外
├── .env.example              # サンプル環境変数
└── README.md                 # このファイル
```

### 使用方法 <a name="使用方法-ja"></a>
1. **登録/ログイン**:
   - `/register` でアカウント（ホームクックまたは家族メンバー）を作成。
   - `/login` でログインし、役割ベースのダッシュボードにアクセス。
2. **食品在庫の管理**:
   - ホームクック：`/inventory` で食品を追加、カテゴリ分け、追跡。
   - `/notifications` で賞味期限通知を確認。
3. **食事計画の作成**:
   - ホームクック：`/meal-plans` で食事計画を作成。
4. **買い物リストの共同作業**:
   - ホームクック/家族メンバー：`/shopping-lists` でリストを作成、共有。
5. **消費レポートの閲覧**:
   - 家族メンバー：`/reports` でレポートにアクセス。
6. **システム管理（管理者）**:
   - `/admin` でユーザー、データ、パフォーマンスを管理。

### 貢献 <a name="貢献-ja"></a>
貢献を歓迎します！貢献するには：
1. リポジトリをフォーク。
2. 新しいブランチを作成（`git checkout -b feature/your-feature-name`）。
3. 変更をコミット（`git commit -m 'Add feature XYZ'`）。
4. ブランチをプッシュ（`git push origin feature/your-feature-name`）。
5. プルリクエストを作成。
