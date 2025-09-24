# 🪞 Mirror Mirror AI - Smart Clothing Recommendation System

An intelligent AI system for clothing recommendations based on facial recognition, clothing detection, weather data, and user feedback learning.

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

### 🗣️ Interactive Interface
- **Voice Recognition**: Voice command recognition
- **Voice Responses**: Text-to-Speech system
- **Visual Interface**: Real-time camera display

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

### Basic Usage
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
├── RecommendationSystemMirror.py  # Main program
├── face_id.py                     # Face recognition module
├── simple_ai_engine.py           # AI engine
├── enhanced_ai_engine.py         # Enhanced AI with time weighting
├── ai_training_collector.py      # Training data collection
├── model_analysis.py             # Performance analysis tool
├── preference_trend_analyzer.py  # Preference trend analysis
├── users.json                    # User data
├── ai_training_data.jsonl        # AI training data
├── classes.txt                   # Clothing categories
├── .env                          # Environment settings
├── requirements.txt              # Dependencies
└── README.md                     # This guide
```

## 🎯 How to Use

### Basic Workflow
1. **Launch**: Run the main program
2. **Recognition**: Stand in front of the camera for face recognition
3. **Registration**: If you're a new user - enter your name
4. **Clothing Detection**: Show your clothes to the camera
5. **Recommendation**: Get AI recommendation on outfit suitability
6. **Feedback**: Provide feedback (good/cold/hot) for system learning

### Voice Commands
- **"analyze"** - Start analysis
- **"good"** - Good recommendation
- **"cold"** - I'm cold
- **"hot"** - I'm hot
- **"exit"** - Exit program
