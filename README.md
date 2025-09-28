# 🪞 Mirror Mirror AI - Smart Clothing Recommendation System

An intelligent AI system for clothing recommendations based on facial recognition, clothing detection, weather data, and user feedback learning. Now featuring both **terminal interface** and **modern web UI** for enhanced user experience.

## 🚀 Live Demo & Links

**[🌐 View Live Website](https://aviad1486.github.io/Machine-Learning/)**

### 📋 Project Resources
- **🎬 [Demo Presentation](https://www.canva.com/design/DAGzsCh04XU/pAElWKISa-OeqFAqpbcSkQ/edit?utm_content=DAGzsCh04XU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)**
- **📹 [Marketing Video](https://www.canva.com/design/DAGzsPHfhyU/LOvqWDIlEEccAweBjKIE0w/edit?utm_content=DAGzsPHfhyU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)**
- **📊 [Explanatory Presentation](https://www.canva.com/design/DAGzV3dyELE/FtfjOwmDhIZZoSNEObPtIQ/edit?utm_content=DAGzV3dyELE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)**
- **📄 [Summary Document](https://docs.google.com/document/d/1W4IK6ipGdWQxoeqbFvuVZluwZ9Uwtt_v5Hicuhv_gq4/edit?usp=sharing)**
- **💻 [Source Code](https://github.com/iLihiS/git-dinamic-page)**

## 👥 Development Team

**Developed by:**
- **Lihi Saar** 
- **Aviad Zer**
- **Yael Zini**

**Mentored by:**
- **Eyal Zinger**
- **Lior Noy**

**Institution:** Ono Academic College

## 🌟 Key Features

### 🎯 Smart Recognition & Learning
- **Advanced Face Recognition**: Automatic user identification using DeepFace
- **Clothing Detection**: Automatic clothing item detection using YOLO
- **Machine Learning**: Random Forest model that learns from user feedback
- **Temporal Adaptation**: Time-weighted system that adapts to changing preferences

### 🌤️ Weather Integration
- **Real-time Data**: Integration with OpenWeatherMap API
- **Advanced Metrics**: Temperature, humidity, and wind speed
- **Accurate Predictions**: Weather-aware recommendations

### 🌐 Dual Interface Options
1. **Web Interface** (New!) - Modern, interactive browser-based UI
2. **Terminal Interface** - Command-line based interaction

### 🗣️ Interactive Features
- **Voice Recognition**: Voice command recognition
- **Voice Responses**: Text-to-Speech system
- **Visual Interface**: Real-time camera display
- **Web Controls**: Interactive buttons and real-time feedback

## � Technology Stack

### 🤖 Computer Vision & AI Models
- **YOLO (You Only Look Once)**: State-of-the-art real-time object detection for clothing item recognition. Uses YOLOv8 from Ultralytics for accurate and fast detection of various clothing categories including shirts, jackets, pants, and accessories.

- **DeepFace**: Advanced facial recognition library built on top of TensorFlow. Utilizes the Facenet512 model for generating high-dimensional face embeddings, enabling robust user identification and re-identification across sessions.

- **Random Forest**: Ensemble learning algorithm chosen for its reliability and interpretability. Trained on user feedback data to predict clothing comfort levels and temperature adjustments, replacing more complex neural networks for better stability.

### 📊 Data Processing & Machine Learning
- **scikit-learn**: Core machine learning library providing Random Forest implementation, data preprocessing tools (StandardScaler), and model evaluation metrics.

- **NumPy & Pandas**: Essential data manipulation libraries for feature vector processing, numerical computations, and structured data handling.

- **Time-Weighted Learning**: Custom algorithm that applies exponential decay to historical user data, giving higher importance to recent feedback while maintaining learning from past preferences.

### 🏷️ Data Annotation & Training
- **Label Studio**: Professional data labeling platform used for annotating clothing images and creating training datasets. Enables precise bounding box creation and category labeling for YOLO model training.

- **Custom Training Pipeline**: Automated data collection system that captures user interactions, weather conditions, and feedback to continuously improve model performance.

### 🌐 External APIs & Services
- **OpenWeatherMap API**: Real-time weather data integration providing temperature, humidity, wind speed, and atmospheric conditions for weather-aware recommendations.

- **Speech Recognition**: Python library for voice command processing, enabling hands-free interaction with the system.

### 🎙️ Audio & Interaction
- **pyttsx3**: Text-to-Speech engine for providing verbal feedback and recommendations to users.

- **OpenCV**: Computer vision library for camera capture, image processing, and real-time video stream handling.

### 🔧 Infrastructure & Storage
- **JSON Storage**: Lightweight data persistence for user profiles, preferences, and historical interactions.

- **JSONL Format**: Structured logging for AI training data, enabling efficient data streaming and model retraining.

- **Environment Management**: Secure API key management using python-dotenv for configuration handling.

## �📋 System Requirements

### Hardware Requirements
- Webcam
- Microphone (optional for voice commands)
- Speakers or headphones

### Software Requirements
- Python 3.8 or higher
- Windows / macOS / Linux
- Internet connection (for weather data)

## 🚀 Installation

### Quick Setup (Web Interface)
```bash
python setup_web.py
```

### Manual Setup

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd "Machine Learning"
```

### Step 2: Create Virtual Environment
```bash
python -m venv mirror_ai_env
```

**Windows:**
```bash
mirror_ai_env\Scripts\activate
```

**macOS/Linux:**
```bash
source mirror_ai_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Setup API Key
1. Create an account at [OpenWeatherMap](https://openweathermap.org/api)
2. Get your API Key
3. Create a `.env` file in the project directory:
```
OPENWEATHER_API_KEY=your_api_key_here
```

### Step 5: Download YOLO Model
The system will automatically download the YOLO model on first run.

## 🎮 Usage

### Option 1: Web Interface (Recommended)

1. **Launch Web App**:
```bash
python app.py
```

2. **Open Browser**: Navigate to `http://localhost:5000`

3. **Web Interface Features**:
   - 📹 **Camera Control**: Start/stop camera with interactive buttons
   - 👤 **User Management**: Face recognition and new user enrollment
   - 👕 **Clothing Detection**: Real-time scanning with visual feedback
   - 🧠 **AI Recommendations**: Smart suggestions with confidence scores
   - 💬 **Interactive Feedback**: Click-based feedback system
   - 📊 **Session History**: View past recommendations and trends
   - 📱 **Responsive Design**: Works on desktop, tablet, and mobile

### Option 2: Terminal Interface

```bash
python RecommendationSystemMirror.py
```

### Train AI Model
```bash
python simple_ai_engine.py
```

## 📁 Project Structure

```
Machine Learning/
├── 🌐 Web Interface
│   ├── app.py                         # Flask web application
│   ├── setup_web.py                   # Web interface setup script
│   ├── templates/
│   │   ├── index.html                 # Main web interface
│   │   ├── 404.html                   # Error pages
│   │   └── 500.html
│   └── static/                        # CSS, JS, images
├── 🖥️ Core System
│   ├── RecommendationSystemMirror.py  # Main terminal program
│   ├── face_id.py                     # Face recognition module
│   ├── simple_ai_engine.py            # AI engine
│   ├── enhanced_ai_engine.py          # Enhanced AI with time weighting
│   ├── ai_training_collector.py       # Training data collection
│   ├── model_analysis.py              # Performance analysis tool
│   └── preference_trend_analyzer.py   # Preference trend analysis
├── 📊 Visualizations
│   ├── create_landing_page_graphs.py  # GitHub visualization generator
│   ├── model_accuracy_evolution.png   # Performance graphs
│   ├── user_preference_adaptation.png
│   ├── accuracy_by_type.png
│   ├── response_time_breakdown.png
│   ├── user_satisfaction_growth.png
│   └── feature_importance.png
├── 🗃️ Data & Config
│   ├── users.json                     # User data
│   ├── ai_training_data.jsonl         # AI training data
│   ├── classes.txt                    # Clothing categories
│   ├── .env                           # Environment settings
│   └── requirements.txt               # Dependencies
└── README.md                          # This guide
```

## 🎯 How to Use

### Web Interface Workflow
1. **Launch Web App**: `python app.py` → Open `http://localhost:5000`
2. **Start Camera**: Click "Start Camera" button
3. **User Recognition**: 
   - Click "Recognize Me" for existing users
   - Click "New User" to enroll your face
4. **Scan Clothes**: Click "Start Scanning" and show your outfit
5. **Get Recommendations**: Click "Get Smart Recommendations"
6. **Provide Feedback**: Rate recommendations (Good/Too Cold/Too Hot)
7. **View History**: Check past sessions and trends

### Terminal Interface Workflow
1. **Launch**: Run the main program
2. **Recognition**: Stand in front of the camera for face recognition
3. **Registration**: If you're a new user - enter your name
4. **Clothing Detection**: Show your clothes to the camera
5. **Recommendation**: Get AI recommendation on outfit suitability
6. **Feedback**: Provide feedback (good/cold/hot) for system learning

### Voice Commands (Terminal Only)
- **"analyze"** - Start analysis
- **"good"** - Good recommendation
- **"cold"** - I'm cold
- **"hot"** - I'm hot
- **"exit"** - Exit program

## 🎯 Project Goals & Vision

Mirror Mirror AI demonstrates how artificial intelligence can be meaningfully integrated into daily life, turning a simple mirror into an evolving AI companion that blends practicality, creativity, and style. The project bridges the gap between fashion, accessibility, and technology.

### Key Objectives:
- **Accessibility**: Making fashion advice accessible to people with visual impairments
- **Personalization**: Learning individual preferences and adapting over time  
- **Real-time Intelligence**: Providing instant, contextual outfit recommendations
- **User Experience**: Creating an intuitive, voice-enabled interface

## 📊 Results & Achievements

Our findings show that the system successfully:
- **High Accuracy Detection**: Identifies garments in real-time with high precision using custom YOLO model
- **Robust User Recognition**: Maintains personalized profiles with DeepFace facial recognition
- **Adaptive Learning**: Provides increasingly accurate outfit suggestions with each user interaction
- **Accessibility Impact**: Offers significant value to everyday users as well as people with visual impairments
- **Weather Integration**: Delivers context-aware recommendations based on real-time weather data

## 🏆 About This Project

This project represents the culmination of our work in artificial intelligence and computer vision, showcasing how technology can enhance everyday experiences while maintaining accessibility and user-friendliness.

**© 2025 Lihi Saar, Aviad Zer, Yael Zini. All rights reserved.**
