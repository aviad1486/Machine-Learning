#!/usr/bin/env python3
"""
Installation script for Mirror Mirror AI Web Interface
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please install manually.")
        return False

def check_camera():
    """Check if camera is available"""
    print("📹 Checking camera availability...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera is available!")
            cap.release()
            return True
        else:
            print("⚠️  Camera not detected. Please connect a camera.")
            return False
    except ImportError:
        print("❌ OpenCV not installed properly.")
        return False

def setup_environment():
    """Setup environment file"""
    env_file = ".env"
    if not os.path.exists(env_file):
        print("🌐 Creating environment file...")
        with open(env_file, "w") as f:
            f.write("# Mirror Mirror AI Configuration\n")
            f.write("OPENWEATHER_API_KEY=your_api_key_here\n")
        
        print("✅ Created .env file. Please add your OpenWeather API key!")
        print("   Get your free API key at: https://openweathermap.org/api")
    else:
        print("✅ Environment file already exists!")

def main():
    print("🪞 Mirror Mirror AI - Web UI Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    
    print(f"✅ Python {sys.version} detected")
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check camera
    check_camera()
    
    # Setup environment
    setup_environment()
    
    print("\n🎉 Setup complete!")
    print("\nTo start the web interface:")
    print("   python app.py")
    print("\nThen open your browser to: http://localhost:5000")
    print("\n📝 Don't forget to:")
    print("   1. Add your OpenWeather API key to .env file")
    print("   2. Make sure your camera is connected")
    print("   3. Check that your YOLO model (my_model.pt) is in the current directory")

if __name__ == "__main__":
    main()