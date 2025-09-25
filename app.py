from flask import Flask, render_template, request, jsonify, Response
import cv2
import os
import sys
import json
from datetime import datetime
import base64
import numpy as np
from threading import Thread, Lock
import time
from io import BytesIO
from PIL import Image
import requests

# Import your existing modules
import face_id
from RecommendationSystemMirror import (
    model, labels, get_city_and_temperature, recommend_for_user, 
    apply_user_delta_to_ranges, BASE_RANGES, OPENWEATHER_API_KEY,
    analyze_user_history, calculate_smart_delta
)
from simple_ai_engine import get_simple_ai_recommendation, simple_ai_engine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mirror-mirror-ai-2025'

# Global variables for camera handling
camera = None
camera_lock = Lock()
current_frame = None
is_scanning = False
detected_clothes = []
current_user = None

# Camera management
class Camera:
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.is_active = False
        
    def start(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_active = True
        return self.cap.isOpened()
    
    def stop(self):
        self.is_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def get_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

# Initialize camera
camera = Camera()

@app.route('/')
def index():
    """Main page with camera interface"""
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start the camera feed"""
    global camera
    try:
        if camera.start():
            return jsonify({'status': 'success', 'message': 'Camera started successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start camera'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Camera error: {str(e)}'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop the camera feed"""
    global camera, is_scanning
    try:
        camera.stop()
        is_scanning = False
        return jsonify({'status': 'success', 'message': 'Camera stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error stopping camera: {str(e)}'})

@app.route('/toggle_scanning', methods=['POST'])
def toggle_scanning():
    """Toggle clothing detection scanning"""
    global is_scanning, detected_clothes
    is_scanning = not is_scanning
    if is_scanning:
        detected_clothes = []  # Reset detected clothes
        return jsonify({'status': 'success', 'scanning': True, 'message': 'Started scanning for clothes'})
    else:
        return jsonify({'status': 'success', 'scanning': False, 'message': 'Stopped scanning'})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Generate video frames with clothing detection"""
    global current_frame, is_scanning, detected_clothes, camera
    
    while True:
        if not camera.is_active:
            time.sleep(0.1)
            continue
            
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
            
        # Store current frame for face recognition
        current_frame = frame.copy()
        
        # Run clothing detection
        try:
            results = model(frame, verbose=False)
            detections = results[0].boxes
            
            # Process detections
            for box in detections:
                conf = float(box.conf.item())
                if conf < 0.5:  # minimum threshold
                    continue
                    
                xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                class_id = int(box.cls.item())
                class_name = labels[class_id]
                
                # Color mapping for bounding boxes
                colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133),
                         (88,159,106), (96,202,231), (159,124,168), (169,162,241)]
                color = colors[class_id % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                # Draw label
                label = f"{class_name}: {int(conf*100)}%"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                            (xmin + label_size[0] + 4, label_ymin + 5), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                
                # Add to detected clothes if scanning
                if is_scanning and class_name not in detected_clothes:
                    detected_clothes.append(class_name)
                    
        except Exception as e:
            print(f"Detection error: {e}")
        
        # Add scanning status overlay
        if is_scanning:
            cv2.putText(frame, "SCANNING...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/get_detected_clothes', methods=['GET'])
def get_detected_clothes():
    """Get currently detected clothing items"""
    global detected_clothes
    return jsonify({'clothes': list(set(detected_clothes))})

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    """Recognize face from current frame"""
    global current_frame, current_user
    
    if current_frame is None:
        return jsonify({'status': 'error', 'message': 'No camera frame available'})
    
    try:
        # Try to match user
        user_profile, distance = face_id.match_user(current_frame)
        
        if user_profile:
            current_user = user_profile
            return jsonify({
                'status': 'success',
                'user_found': True,
                'user': {
                    'name': user_profile['name'],
                    'user_id': user_profile['user_id'],
                    'distance': distance
                },
                'message': f"Welcome back, {user_profile['name']}!"
            })
        else:
            current_user = None
            return jsonify({
                'status': 'success',
                'user_found': False,
                'message': 'No matching user found'
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Face recognition error: {str(e)}'})

@app.route('/enroll_user', methods=['POST'])
def enroll_user():
    """Enroll a new user"""
    global current_frame, current_user
    
    data = request.json
    name = data.get('name', '').strip()
    
    if not name:
        return jsonify({'status': 'error', 'message': 'Please provide a name'})
    
    if current_frame is None:
        return jsonify({'status': 'error', 'message': 'No camera frame available'})
    
    try:
        # Check if user with this name already exists
        users = face_id._load_db()
        existing_user = None
        for user in users:
            if user.get('name', '').lower() == name.lower():
                existing_user = user
                break
        
        if existing_user:
            return jsonify({
                'status': 'error', 
                'message': f'User "{name}" already exists in the system! Please use "Recognize Me" instead or choose a different name.'
            })
        
        # Proceed with enrollment if name is unique
        enrolled_user = face_id.enroll_user(current_frame, name=name)
        if enrolled_user:
            current_user = enrolled_user
            return jsonify({
                'status': 'success',
                'message': f'Successfully enrolled {name}!',
                'user': {
                    'name': enrolled_user['name'],
                    'user_id': enrolled_user['user_id']
                }
            })
        else:
            return jsonify({'status': 'error', 'message': 'Could not detect face for enrollment'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Enrollment error: {str(e)}'})

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Get clothing recommendations based on detected items and weather"""
    global detected_clothes, current_user
    
    unique_clothes = set(detected_clothes)
    
    if not unique_clothes:
        return jsonify({'status': 'error', 'message': 'No clothing items detected. Please scan first.'})
    
    try:
        # Get weather data
        city, current_temp, humidity, wind_speed = get_city_and_temperature(OPENWEATHER_API_KEY)
        
        if city is None or current_temp is None:
            return jsonify({'status': 'error', 'message': 'Could not retrieve weather data'})
        
        # === TRY AI PREDICTION FIRST ===
        ai_delta = None
        ai_explanation = ""
        ai_confidence = 0.0
        
        if simple_ai_engine.is_available():
            current_conditions = {
                'temperature': current_temp,
                'humidity': humidity if humidity is not None else 50,
                'wind_speed': wind_speed if wind_speed is not None else 0
            }
            
            ai_delta, ai_explanation, ai_confidence = get_simple_ai_recommendation(
                current_user, current_conditions, list(unique_clothes)
            )
        
        # === FALLBACK TO RULE-BASED SYSTEM ===
        rule_delta, rule_explanation, rule_confidence = calculate_smart_delta(current_user, current_temp)
        
        # === CHOOSE BEST RECOMMENDATION ===
        if ai_delta is not None and ai_confidence > 0.7:
            # Use AI prediction when confident
            final_delta = ai_delta
            system_explanation = f"🤖 AI: {ai_explanation}"
            confidence = ai_confidence
            recommendation_source = "ai"
        else:
            # Use rule-based system
            final_delta = rule_delta
            system_explanation = f"📋 Rules: {rule_explanation}"
            confidence = rule_confidence
            recommendation_source = "rules"
            
            if ai_delta is not None:
                system_explanation += f" (AI available but low confidence: {ai_confidence:.2f})"
        
        # Apply the chosen delta to clothing ranges
        ranges = apply_user_delta_to_ranges(BASE_RANGES, final_delta)
        
        # Analyze each clothing item
        recommendations = []
        for item in unique_clothes:
            rng = ranges.get(item)
            if not rng:
                continue
                
            if current_temp < rng["min_temp"]:
                status = "too_light"
                message = f"{item} is too light for {current_temp:.1f}°C. Consider dressing warmer."
            elif current_temp > rng["max_temp"]:
                status = "too_warm" 
                message = f"{item} may be too warm for {current_temp:.1f}°C. Consider dressing lighter."
            else:
                status = "appropriate"
                message = f"{item} is appropriate for {current_temp:.1f}°C."
                
            recommendations.append({
                'item': item,
                'status': status,
                'message': message
            })
        
        return jsonify({
            'status': 'success',
            'weather': {
                'city': city,
                'temperature': round(current_temp, 1),
                'humidity': humidity,
                'wind_speed': wind_speed
            },
            'user': {
                'name': current_user['name'] if current_user else 'Guest',
                'delta': final_delta,
                'explanation': system_explanation,
                'confidence': confidence,
                'recommendation_source': recommendation_source  # Add this new field
            },
            'recommendations': recommendations,
            'detected_clothes': list(unique_clothes)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Recommendation error: {str(e)}'})

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback on recommendations"""
    global current_user, detected_clothes
    
    data = request.json
    feedback = data.get('feedback')
    temperature_feeling = data.get('temperature_feeling')  # 'cold', 'hot', or None
    
    if not feedback:
        return jsonify({'status': 'error', 'message': 'No feedback provided'})
    
    try:
        # Process feedback
        if feedback == 'bad' and temperature_feeling:
            delta_change = +2 if temperature_feeling == 'cold' else -2
            
            if current_user:
                current_delta = int(current_user.get("prefs", {}).get("delta_temp", 0))
                new_delta = current_delta + delta_change
                face_id.update_user_prefs(current_user["user_id"], {"delta_temp": new_delta})
                
                # Update current user data
                current_user["prefs"]["delta_temp"] = new_delta
                
                feedback_str = f"bad-{temperature_feeling}"
            else:
                feedback_str = f"bad-{temperature_feeling}"
                new_delta = delta_change
        else:
            feedback_str = feedback
            new_delta = None
        
        # Save to history if user exists
        if current_user and detected_clothes:
            city, current_temp, _, _ = get_city_and_temperature(OPENWEATHER_API_KEY)
            
            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "clothes_detected": list(set(detected_clothes)),
                "city": city,
                "temperature": current_temp,
                "feedback": feedback_str
            }
            
            current_history = current_user.get("prefs", {}).get("history", [])
            face_id.update_user_prefs(current_user["user_id"], {
                "history": current_history + [entry]
            })
        
        return jsonify({
            'status': 'success',
            'message': 'Thank you for your feedback!',
            'delta_change': delta_change if 'delta_change' in locals() else None,
            'new_delta': new_delta
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Feedback error: {str(e)}'})

@app.route('/user_history', methods=['GET'])
def get_user_history():
    """Get user's session history"""
    global current_user
    
    if not current_user:
        return jsonify({'status': 'error', 'message': 'No user logged in'})
    
    try:
        history = current_user.get("prefs", {}).get("history", [])
        
        # Limit to last 10 sessions
        recent_history = history[-10:] if len(history) > 10 else history
        
        return jsonify({
            'status': 'success',
            'history': recent_history,
            'total_sessions': len(history)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'History error: {str(e)}'})

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset current session"""
    global detected_clothes, current_user, is_scanning
    
    detected_clothes = []
    current_user = None
    is_scanning = False
    
    return jsonify({'status': 'success', 'message': 'Session reset successfully'})

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    
    print("🌟 Starting Mirror Mirror AI Web Interface...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔥 Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)