# Plant Health App

Plant Health App is a web application that leverages **Computer Vision** to analyze plant health through images. It is built using **Django** for the back-end and **Bootstrap**, **HTML**, **CSS**, and **JavaScript** for the front-end. The project integrates the **ResNet18** deep learning model to identify 21 types of plant diseases, supporting **Farmers**, **Experts**, and **Admins** in managing and advising on plant health. The application uses **Microsoft SQL Server** as the database back-end and includes a **Jupyter Notebook** for model training documentation.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Plant Disease Prediction**:
  - Upload plant images (jpg, jpeg, png) and use the **ResNet18** model to predict 21 types of diseases (e.g., apple scab, bacterial spot, healthy).
  - Display disease name, confidence score (%), and plant type.
  - Store prediction history for users.
- **User Management**:
  - Supports three roles: **Farmer**, **Expert**, and **Admin** with detailed permissions.
  - Features for registration, login, password change, and profile updates.
- **Farmer-Expert Interaction**:
  - Farmers can submit questions with optional plant images to Experts.
  - Experts can answer, edit, or delete their responses.
- **Disease Library**:
  - Experts and Admins can add, edit, or delete plant disease information (name, description, symptoms, treatment, image).
  - Display a list of diseases with illustrative images.
- **System Management (Admin)**:
  - Manage users, prediction history, and feedback with search and pagination features.
  - Delete users (except superusers), prediction images, or feedback.
- **User-Friendly Interface**:
  - Responsive design with **Bootstrap 5.3.0**.
  - Supports **dark mode** with dynamic toggling.
  - Fixed navbar with role-based links.

## Technologies Used
- **Back-end**:
  - **Django 4.2.13**: Python web framework for logic, authentication, and database management.
  - **PyTorch 2.0.1**: **ResNet18** model for plant disease classification.
  - **Django ORM**: Manages data models (User, PlantImage, Feedback, Question, Answer, DiseaseLibrary, UserProfile).
  - **mssql-django 1.5**: Django database backend for Microsoft SQL Server.
  - **pyodbc 5.2.0**: ODBC driver for SQL Server connectivity.
  - **django-cors-headers 4.7.0**: Handles CORS for cross-origin requests.
  - **djangorestframework 3.16.0**: Supports RESTful API development.
  - **djangorestframework-simplejwt 5.5.0**: JWT-based authentication.
  - **django-environ 0.12.0**: Environment variable management.
  - **Pillow 10.0.0**: Image processing.
  - **opencv-python 4.8.0.76**: Image handling for computer vision tasks.
- **Front-end**:
  - **HTML/CSS/JavaScript**: Client-side interface and interactions.
  - **Bootstrap 5.3.0**: Responsive design with navbar, modals, dropdowns, and pagination.
  - **django-bootstrap-v5 1.0.11**: Bootstrap integration for Django templates.
- **Database**: Microsoft SQL Server.
- **File System**: Django Media for storing uploaded images in the `media/` directory.
- **Environment**: **Anaconda** for virtual environment management.
- **Jupyter Notebook**: Used for model training and evaluation (`plant_disease_model.ipynb`).

## Installation
### Requirements
- Python 3.8+
- Anaconda
- Microsoft SQL Server (with a running instance and database created)
- ODBC Driver 17 for SQL Server
- Modern browser (Chrome, Firefox, etc.)

### Installation Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/00rei000/Plant_Health.git
   cd Plant_Health
   ```

2. **Create an Anaconda virtual environment**:
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

   Alternatively, install dependencies with Conda:
   ```bash
   conda install django=4.2.13
   pip install django-bootstrap-v5==1.0.11 django-cors-headers==4.7.0 django-environ==0.12.0 djangorestframework==3.16.0 djangorestframework-simplejwt==5.5.0 Pillow==10.0.0 opencv-python==4.8.0.76 torch==2.0.1 torchvision==0.15.2 pyodbc==5.2.0 mssql-django==1.5
   ```

4. **Download AI models**:
   - Download the pre-trained models (`best_model.pth` and `plant_disease_model.pth`) from [this Google Drive link](https://drive.google.com/your-link) and place them in `plant_health_app/models/`.
   - Ensure `class_names.json` is in `plant_health_app/data/` (included in the repository).

5. **Configure Microsoft SQL Server**:
   - Ensure a SQL Server instance is running and a database is created (e.g., `plant_health_db`).
   - Install the **ODBC Driver 17 for SQL Server** from [Microsoft](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server).
   - Update `settings.py` in `plant_health_app/` with SQL Server configuration:
     ```python
     DATABASES = {
         'default': {
             'ENGINE': 'mssql',
             'NAME': 'plant_health_db',
             'USER': 'your_username',
             'PASSWORD': 'your_password',
             'HOST': 'localhost',
             'PORT': '',
             'OPTIONS': {
                 'driver': 'ODBC Driver 17 for SQL Server',
             },
         }
     }
     ```
   - Alternatively, use `django-environ` to store database credentials in a `.env` file:
     ```plaintext
     SECRET_KEY=your-secret-key
     DATABASE_URL=sqlserver://your_username:your_password@localhost:1433/plant_health_db?driver=ODBC+Driver+17+for+SQL+Server
     ```

6. **Configure the environment**:
   - Create a `.env` file to store environment variables (e.g., `SECRET_KEY`, `DATABASE_URL`).
   - Create a `media/` directory in the project root and ensure it has write permissions (`chmod 755 media/` on Linux).
   - Optionally, download sample images for `media/` from [this Google Drive link](https://drive.google.com/your-link).
   - Run `collectstatic` to generate static files:
     ```bash
     python manage.py collectstatic
     ```
   - Verify the default image path (`plant_images/images/train/disease/img.jpg`) or use the placeholder (`/static/images/placeholder.jpg`) included in the repository.

7. **Run migrations**:
   ```bash
   python manage.py migrate
   ```

8. **Create a superuser (Admin)**:
   ```bash
   python manage.py createsuperuser
   ```

9. **Run the server**:
   ```bash
   python manage.py runserver
   ```
   Access the application at `http://localhost:8000`.

### Notes
- For GPU support, ensure **CUDA** is installed and use a compatible PyTorch version:
  ```bash
  conda install pytorch==2.0.1 torchvision==0.15.2 -c pytorch
  ```
- The `plant_disease_model.ipynb` file in `plant_health_app/notebooks/` documents the model training process. It is included for reference but not required to run the application.
- The `media/` directory is excluded from the repository. Create it manually or download sample images from the provided link.
- The `staticfiles/` directory (generated by `collectstatic`) is excluded from the repository. Run `collectstatic` to regenerate it.
- The `best_model.pth` and `plant_disease_model.pth` files are large and excluded from the repository. Download them from the provided link and place them in `plant_health_app/models/`.
- The `.ipynb_checkpoints/` directories are temporary files created by Jupyter Notebook and are excluded via `.gitignore`.

## Project Structure
```plaintext
Plant_Health/
├── plant_health_app/
│   ├── models/               # AI models (best_model.pth, plant_disease_model.pth, excluded)
│   ├── notebooks/            # Jupyter Notebook (plant_disease_model.ipynb)
│   ├── data/                 # Class names (class_names.json)
│   ├── static/               # Custom static files (style.css, upload.js, favicon, logo, etc.)
│   ├── templates/            # HTML templates (base.html, prediction.html, etc.)
│   ├── media/                # Uploaded images (excluded)
│   ├── __init__.py
│   ├── settings.py           # Django configuration
│   ├── urls.py               # URL routing
│   ├── views.py              # Request handling logic
│   ├── models.py             # Data models
│   └── utils.py              # PLANT_TYPE_MAPPING
├── staticfiles/              # Collected static files (excluded)
├── manage.py                 # Django management script
├── requirements.txt          # Dependency list
├── .gitignore                # Excludes .conda/, __pycache__, media/, staticfiles/, etc.
├── .env.example              # Sample environment variables
└── README.md                 # This file
```

## Usage
1. **Register/Login**:
   - Access `/register` to create an account (Farmer or Expert).
   - Log in at `/login` to access the respective dashboard (Farmer, Expert, or Admin).
2. **Predict Plant Disease**:
   - Farmers: Go to `/prediction`, upload a plant image, and view results at `/result`.
   - View prediction history at `/prediction_history`.
3. **Farmer-Expert Interaction**:
   - Farmers: Submit questions at `/ask_expert` and view them at `/my_questions`.
   - Experts: Answer questions at `/expert_questions`.
4. **Manage Disease Library**:
   - Experts/Admins: Add, edit, or delete diseases at `/disease_library` or `/disease_list`.
5. **System Management (Admin)**:
   - Access `/admin_dashboard` or `/manage` to manage users, predictions, and feedback.
6. **Feedback**:
   - Submit feedback at `/feedback`.

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit changes (`git commit -m 'Add feature XYZ'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Create a Pull Request.


