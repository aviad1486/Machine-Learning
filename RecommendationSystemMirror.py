import os
import sys
import glob
import cv2
import numpy as np
import math
import requests
from ultralytics import YOLO
import pyttsx3
from time import sleep, time
import face_id  
from dotenv import load_dotenv
import speech_recognition as sr
from ai_training_collector import AITrainingDataCollector
from simple_ai_engine import get_simple_ai_recommendation, simple_ai_engine

load_dotenv()

# ===================== AI TRAINING SETUP =====================
ai_collector = AITrainingDataCollector()

# ===================== USER CONFIG =====================
model_path = "my_model.pt"
img_source = "usb0"
min_thresh = 0.5
user_res = "640x480"
record = False
USE_TTS = True  # turn off if you don't want voice feedback

# OpenWeather API key
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not OPENWEATHER_API_KEY:
    print("❌ ERROR: Missing OPENWEATHER_API_KEY in .env")
    sys.exit(1)


# ===================== INIT TTS =====================
def say(txt: str):
    engine = pyttsx3.init()
    if USE_TTS:
        engine.say(txt)
        engine.runAndWait()
    engine.stop()

print('Hello! and welcome to Mirror Mirror AI!')
say("Hello! and welcome to Mirror Mirror AI!")
sleep(1.0)

# ===================== LOAD MODEL =====================
if not os.path.exists(model_path):
    print("❌ ERROR: YOLO model not found:", model_path)
    sys.exit(1)

model = YOLO(model_path, task='detect')
labels = model.names

# ===================== SOURCE DETECTION =====================
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mp4', '.mov', '.mkv']

source_type = ''
usb_idx = 0

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    ext = os.path.splitext(img_source)[1].lower()
    source_type = 'image' if ext in img_ext_list else 'video' if ext in vid_ext_list else ''
elif isinstance(img_source, str) and img_source.startswith("usb"):
    source_type = 'usb'
    try:
        usb_idx = int(img_source[3:])
    except:
        usb_idx = 0
else:
    print("❌ Invalid source:", img_source)
    sys.exit(1)

# ===================== RESOLUTION =====================
resize = False
if user_res:
    try:
        resW, resH = map(int, user_res.lower().split('x'))
        resize = True
    except:
        print("❌ Invalid resolution format. Use 'WIDTHxHEIGHT', e.g., 640x480")
        sys.exit(1)

# ===================== CAMERA/VIDEO SETUP =====================
cap = None
recorder = None

def camera_setup():
    """Open the camera/video for reading."""
    global cap
    if source_type in ['video', 'usb']:
        if source_type == 'usb':
            cap = cv2.VideoCapture(usb_idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(img_source)

        if resize:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

        if not cap.isOpened():
            print("❌ Failed to open video/camera source.")
            sys.exit(1)
        else:
            print('✅ Camera connected')
            say("camera connected")
            sleep(0.2)

if record:
    if source_type not in ['video', 'usb']:
        print("❌ Recording only supported for video/camera.")
        sys.exit(1)
    if not resize:
        print("❌ Please set 'user_res' to enable recording.")
        sys.exit(1)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# ===================== VISUALS =====================
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133),
               (88,159,106), (96,202,231), (159,124,168), (169,162,241)]

# ===================== FILE LISTS =====================
imgs_list = []
if source_type == 'folder':
    imgs_list = [f for f in glob.glob(os.path.join(img_source, '*'))
                 if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type == 'image':
    imgs_list = [img_source]

img_count = 0
fps_log = []

# ===================== CLOTHING RANGES (base, global) =====================
BASE_RANGES = {
    "Jacket":  {"min_temp": -1 * math.inf, "max_temp": 15},
    "Jeans":   {"min_temp": 5,  "max_temp": 30},
    "Jogger":  {"min_temp": 5,  "max_temp": 25},
    "Polo":    {"min_temp": 18, "max_temp": 32},
    "Shirt":   {"min_temp": 16, "max_temp": 28},
    "Short":   {"min_temp": 22, "max_temp": math.inf},
    "T-Shirt": {"min_temp": 18, "max_temp": math.inf},
    "Trouser": {"min_temp": 5,  "max_temp": 30}
}

def apply_user_delta_to_ranges(base_ranges: dict, delta: int) -> dict:
    """Return a copy of ranges shifted by user's delta."""
    out = {}
    for k, v in base_ranges.items():
        out[k] = {
            "min_temp": v["min_temp"] + (delta if v["min_temp"] != -1 * math.inf else 0),
            "max_temp": v["max_temp"] + (delta if v["max_temp"] != math.inf else 0),
        }
    return out

# ===================== LOCATION + WEATHER =====================
def get_city_and_temperature(openweather_api_key: str):
    """
    Resolve user's approximate city and weather data using ipinfo (lat,lon) + OpenWeather.
    Returns: (city_name:str, current_temp:float, humidity:int, wind_speed:float) or (None, None, None, None) on hard failure
    """
    default_city = "Tel Aviv"
    default_lat = "32.0853"
    default_lon = "34.7818"

    try:
        loc_response = requests.get("https://ipinfo.io/json", timeout=5)
        loc_response.raise_for_status()
        loc_data = loc_response.json()

        if "loc" in loc_data and isinstance(loc_data["loc"], str) and "," in loc_data["loc"]:
            lat, lon = loc_data["loc"].split(",", 1)
        else:
            lat, lon = default_lat, default_lon

        city_name = loc_data.get("city", default_city)

    except Exception as e:
        print(f"⚠️ ipinfo lookup failed, using defaults: {e}")
        lat, lon = default_lat, default_lon
        city_name = default_city

    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"lat={lat}&lon={lon}&appid={openweather_api_key}&units=metric"
        )
        weather_response = requests.get(url, timeout=7)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        main = weather_data.get("main", {})
        wind = weather_data.get("wind", {})
        
        if "temp" not in main:
            raise ValueError("Missing 'temp' in OpenWeather response")

        current_temp = float(main["temp"])
        humidity = int(main.get("humidity", 50))  # Default to 50% if missing
        wind_speed = float(wind.get("speed", 0.0))  # Default to 0 if missing (m/s)

        return city_name, current_temp, humidity, wind_speed

    except Exception as e:
        print(f"⚠️ OpenWeather request failed: {e}")
        return None, None, None, None

# ===================== RECOMMENDER =====================
def analyze_user_history(user_profile: dict | None) -> dict:
    """
    Analyze user's feedback history to extract learning patterns.
    Returns analysis results including recommended adjustments.
    """
    if not user_profile or "prefs" not in user_profile:
        return {
            "total_sessions": 0,
            "feedback_sessions": 0,
            "has_sufficient_data": False, 
            "confidence": 0.0, 
            "suggested_delta": 0,
            "patterns": {
                "temperature_ranges": {},
                "clothing_preferences": {},
                "seasonal_patterns": {}
            }
        }
    
    history = user_profile["prefs"].get("history", [])
    
    # Filter out sessions without feedback
    feedback_sessions = [s for s in history if s.get("feedback") and s["feedback"] != "good"]
    
    analysis = {
        "total_sessions": len(history),
        "feedback_sessions": len(feedback_sessions),
        "has_sufficient_data": len(feedback_sessions) >= 3,  # Need at least 3 feedback sessions
        "confidence": min(len(feedback_sessions) / 5.0, 1.0),  # Max confidence at 5+ sessions
        "suggested_delta": 0,
        "patterns": {
            "temperature_ranges": {},
            "clothing_preferences": {},
            "seasonal_patterns": {}
        }
    }
    
    if len(feedback_sessions) == 0:
        return analysis
    
    # Analyze temperature-based feedback patterns
    cold_feedback_temps = []
    hot_feedback_temps = []
    
    for session in feedback_sessions:
        temp = session.get("temperature")
        feedback = session.get("feedback", "")
        
        if temp is not None:
            if "cold" in feedback:
                cold_feedback_temps.append(temp)
            elif "hot" in feedback:
                hot_feedback_temps.append(temp)
    
    # Calculate suggested delta based on patterns
    suggested_delta = 0
    
    if cold_feedback_temps:
        # User felt cold at these temperatures - they prefer warmer
        avg_cold_temp = sum(cold_feedback_temps) / len(cold_feedback_temps)
        # Suggest positive delta (warmer clothing recommendations)
        suggested_delta += 2 * len(cold_feedback_temps)
    
    if hot_feedback_temps:
        # User felt hot at these temperatures - they prefer cooler
        avg_hot_temp = sum(hot_feedback_temps) / len(hot_feedback_temps)
        # Suggest negative delta (cooler clothing recommendations)
        suggested_delta -= 2 * len(hot_feedback_temps)
    
    # Weight the suggestion by confidence
    analysis["suggested_delta"] = int(suggested_delta * analysis["confidence"])
    
    # Store patterns for future reference
    analysis["patterns"]["cold_temperatures"] = cold_feedback_temps
    analysis["patterns"]["hot_temperatures"] = hot_feedback_temps
    
    return analysis

def get_fallback_recommendations() -> dict:
    """
    Provide fallback recommendations for users without sufficient history.
    Uses general population preferences and seasonal adjustments.
    """
    from datetime import datetime
    current_month = datetime.now().month
    
    # Seasonal adjustments (Northern Hemisphere bias, can be customized)
    seasonal_delta = 0
    if current_month in [12, 1, 2]:  # Winter
        seasonal_delta = +1  # Slightly warmer recommendations
    elif current_month in [6, 7, 8]:  # Summer
        seasonal_delta = -1  # Slightly cooler recommendations
    
    return {
        "type": "fallback",
        "seasonal_delta": seasonal_delta,
        "confidence": 0.3,  # Low confidence for fallback
        "explanation": f"Using seasonal adjustment for month {current_month}"
    }

def calculate_smart_delta(user_profile: dict | None, current_temp: float) -> tuple[int, str, float]:
    """
    Calculate intelligent delta based on user history, current conditions, and fallbacks.
    Returns: (final_delta, explanation, confidence)
    """
    base_delta = user_profile.get("prefs", {}).get("delta_temp", 0) if user_profile else 0
    
    # Analyze user history
    history_analysis = analyze_user_history(user_profile)
    
    if history_analysis["has_sufficient_data"]:
        # Use learning from history
        learned_delta = history_analysis["suggested_delta"]
        final_delta = base_delta + learned_delta
        confidence = history_analysis["confidence"]
        
        explanation = f"Learned from {history_analysis['feedback_sessions']} feedback sessions"
        if learned_delta > 0:
            explanation += f" (you tend to feel cold, +{learned_delta}°C adjustment)"
        elif learned_delta < 0:
            explanation += f" (you tend to feel warm, {learned_delta}°C adjustment)"
        else:
            explanation += " (your preferences are well-calibrated)"
            
    elif history_analysis["feedback_sessions"] > 0:
        # Partial learning with reduced confidence
        learned_delta = int(history_analysis["suggested_delta"] * 0.5)  # Reduce impact
        final_delta = base_delta + learned_delta
        confidence = history_analysis["confidence"]
        
        explanation = f"Learning from {history_analysis['feedback_sessions']} sessions (limited data)"
        
    else:
        # Use fallback for new users or no feedback
        fallback = get_fallback_recommendations()
        final_delta = base_delta + fallback["seasonal_delta"]
        confidence = fallback["confidence"]
        explanation = fallback["explanation"]
        
        if not user_profile:
            explanation += " (new user)"
        else:
            explanation += " (no feedback history yet)"
    
    return final_delta, explanation, confidence

def recommend_for_user(unique_clothes: set, user_profile: dict | None, city: str = None, current_temp: float = None,
                      humidity: int = None, wind_speed: float = None):
    """
    Computes and speaks recommendations based on current weather, user history, and smart learning.
    Now includes AI predictions alongside rule-based learning.
    """
    # Get weather data if not provided
    if city is None or current_temp is None:
        city, current_temp, humidity, wind_speed = get_city_and_temperature(OPENWEATHER_API_KEY)
        if city is None or current_temp is None:
            print("⚠️ Could not retrieve weather; skipping outfit check.")
            return

    # === TRY AI PREDICTION FIRST ===
    ai_delta = None
    ai_explanation = ""
    ai_confidence = 0.0
    
    if simple_ai_engine.is_available():
        current_conditions = {
            'temperature': current_temp,
            'humidity': humidity if humidity is not None else 50,  # Use real data or default
            'wind_speed': wind_speed if wind_speed is not None else 0   # Use real data or default
        }
        
        ai_delta, ai_explanation, ai_confidence = get_simple_ai_recommendation(
            user_profile, current_conditions, list(unique_clothes)
        )
    
    # === FALLBACK TO RULE-BASED SYSTEM ===
    rule_delta, rule_explanation, rule_confidence = calculate_smart_delta(user_profile, current_temp)
    
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

    # Enhanced greeting with AI info
    if user_profile:
        say(f"Hi {user_profile.get('name', 'there')}!")
        print(f"\n👤 Welcome back, {user_profile.get('name', 'there')}!")
    else:
        print(f"\n👋 Hello! New user detected.")
        say("Hello! I'm analyzing your style preferences.")

    print(f"🌡️ Current temperature in {city}: {current_temp:.1f}°C")
    print(f"🧠 {system_explanation}")
    print(f"⚙️ Applied adjustment: {final_delta:+d}°C (confidence: {confidence:.2f}, source: {recommendation_source})")
    
    # Speak the temperature and AI status
    say(f"The current temperature in {city} is {int(round(current_temp))} degrees.")
    
    if recommendation_source == "ai":
        say("Using AI neural network for recommendations.")
    
    # Speak confidence level
    if confidence > 0.7:
        if recommendation_source == "ai":
            say("AI is confident about this recommendation.")
        else:
            say("I'm confident about your preferences.")
    elif confidence > 0.3:
        say("I'm learning your preferences.")
    else:
        say("I'm still learning about your style preferences.")

    # Analyze each clothing item
    recommendations = []
    for item in unique_clothes:
        rng = ranges.get(item)
        if not rng:
            continue

        if current_temp < rng["min_temp"]:
            message = f"⚠️ {item} is too light for {current_temp:.1f}°C. Consider dressing warmer."
            recommendations.append(("too_light", item, current_temp))
            print(message)
            say(f"{item} is too light for {int(round(current_temp))} degrees. Consider dressing warmer.")
        elif current_temp > rng["max_temp"]:
            message = f"⚠️ {item} may be too warm for {current_temp:.1f}°C. Consider dressing lighter."
            recommendations.append(("too_warm", item, current_temp))
            print(message)
            say(f"{item} may be too warm for {int(round(current_temp))} degrees. Consider dressing lighter.")
        else:
            message = f"✅ {item} is appropriate for {current_temp:.1f}°C."
            recommendations.append(("appropriate", item, current_temp))
            print(message)
            say(f"{item} is appropriate for {int(round(current_temp))} degrees.")
    
    # Provide learning encouragement for new users
    if not user_profile or confidence < 0.5:
        print("\n💡 Tip: Your feedback helps me learn your preferences better!")
        say("Your feedback helps me learn your preferences better!")
    elif len(recommendations) > 0 and all(rec[0] == "appropriate" for rec in recommendations):
        print("\n🎯 Perfect! Your outfit matches your learned preferences.")
        say("Perfect! Your outfit matches your learned preferences.")

# ===================== FEEDBACK (TERMINAL) =====================
def listen_for_speech(prompt_text: str, timeout: int = 5) -> str | None:
    """
    Attempts to capture speech input from microphone and convert to text.
    Returns the recognized text (lowercase) or None if failed/timeout.
    """
    try:
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        print(f"{prompt_text}")
        say("Please speak your answer now.")
        print("🎤 Listening...")
        
        # Adjust for ambient noise
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        # Listen for speech
        with microphone as source:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=3)
        
        # Recognize speech using Google's free service
        print("🔄 Processing speech...")
        text = recognizer.recognize_google(audio).lower().strip()
        print(f"🗣️ You said: '{text}'")
        return text
        
    except sr.WaitTimeoutError:
        print("⏰ No speech detected within timeout period.")
        return None
    except sr.UnknownValueError:
        print("❓ Could not understand the speech.")
        return None
    except sr.RequestError as e:
        print(f"❌ Speech recognition service error: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Microphone error: {e}")
        return None

def prompt_feedback_and_update_user(user_profile: dict | None) -> str | None:
    """
    Terminal-based feedback updates the user's delta_temp (personalization).
    Supports both speech and keyboard input.
    Returns the feedback string ('good', 'bad-cold', 'bad-hot') or None if skipped.
    """
    print("\nReady for feedback!")
    say('I am ready to get your feedback!')
    
    # Ask for input method preference
    print("How would you like to give feedback?")
    print("1. Speech - press '1'")
    print("2. Keyboard press '2'")
    print("3. Skip (press Enter)")
    
    method_choice = input("> ").strip().lower()
    
    # Determine input method
    use_speech = method_choice in ['1', 'speech', 'voice', 'mic', 'microphone']
    use_keyboard = method_choice in ['2', 'keyboard', 'type', 'text']
    
    if method_choice == "":
        print("Skipping feedback.")
        return None
    
    # Default to keyboard if unclear
    if not use_speech and not use_keyboard:
        print("Using keyboard input as default.")
        use_keyboard = True
    
    ans = None
    
    if use_speech:
        # Try speech input first
        ans = listen_for_speech("Say 'good' if the recommendations felt right, or 'bad' if they felt off:")
        
        # Fallback to keyboard if speech failed
        if ans is None:
            print("Speech input failed. Falling back to keyboard input.")
            print("Type 'good' if it felt right, 'bad' if it felt off (or press Enter to skip):")
            ans = input("> ").strip().lower()
    else:
        # Direct keyboard input
        print("Type 'good' if it felt right, 'bad' if it felt off (or press Enter to skip):")
        ans = input("> ").strip().lower()

    if not ans or ans not in ["good", "bad","bed"]:
        if ans and ans not in ["good", "bad"]:
            print("Unrecognized input; skipping feedback.")
        return None

    if ans == "good":
        print("Perfect! Good to know!")
        say("Perfect! Good to know!")
        return "good"

    # Handle "bad" feedback - ask for specifics
    print("Was it too 'cold' or too 'hot'?")
    
    ans2 = None
    if use_speech:
        # Try speech for the follow-up question
        ans2 = listen_for_speech("Say 'cold' if you felt cold, or 'hot' if you felt too warm:")
        
        # Fallback to keyboard if speech failed
        if ans2 is None:
            print("Speech input failed. Please type your answer:")
            ans2 = input("> ").strip().lower()
    else:
        # Direct keyboard input for follow-up
        ans2 = input("> ").strip().lower()
    
    if ans2 not in ["cold", "hot"]:
        print("Unknown answer; skipping adjustments.")
        return "bad"

    delta_change = +2 if ans2 == "cold" else -2
    say('Thanks! Updating your preferences.')
    print(f"Applying delta_temp change {delta_change:+d}°C")

    if user_profile:
        current_delta = int(user_profile.get("prefs", {}).get("delta_temp", 0))
        new_delta = current_delta + delta_change
        face_id.update_user_prefs(user_profile["user_id"], {"delta_temp": new_delta})
        print(f"New personal delta_temp: {new_delta:+d}°C")

    return f"bad-{ans2}"

# ===================== MAIN LOOP =====================
def run_cycle():
    global cap, img_count, fps_log, recorder

    clothes_detected = []
    camera_active = False
    last_frame_for_face = None  # we'll keep the latest frame for face recognition at the end

    # Prepare capture for video/cam
    if source_type in ['video', 'usb']:
        camera_setup()

    print("\nControls (focus the video window): 's' = start/stop scanning, 'q' = quit.")
    say("Press S to start scanning, press S again to stop. Press Q to quit.")

    while True:
        key = cv2.waitKey(5 if source_type in ['video', 'usb'] else 0) & 0xFF
        if key == ord('q'):
            print("Quitting…")
            break
        if key == ord('s'):
            camera_active = not camera_active
            if camera_active:
                print("Camera started scanning…")
                say("Camera started scanning")
            else:
                print("Camera stopped scanning.")
                say("Camera stopped scanning")

        t_start = time()

        # Acquire next frame
        if source_type in ['image', 'folder']:
            if len(imgs_list) == 0:
                print("❌ No images found.")
                break
            frame = cv2.imread(imgs_list[img_count])
            img_count = (img_count + 1) % len(imgs_list)
        elif source_type in ['video', 'usb']:
            ret, frame = cap.read()
            if not ret:
                print("✅ Done with video/camera stream.")
                break
        else:
            print("❌ Unsupported source type.")
            break

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        last_frame_for_face = frame.copy()  # keep latest frame for face recognition later

        # Inference (clothes) – always run to draw boxes
        results = model(frame, verbose=False)
        detections = results[0].boxes

        for box in detections:
            conf = float(box.conf.item())
            if conf < min_thresh:
                continue

            xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            class_id = int(box.cls.item())
            class_name = labels[class_id]
            color = bbox_colors[class_id % len(bbox_colors)]

            # Draw box + label (always visible)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{class_name}: {int(conf*100)}%"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                          (xmin + label_size[0] + 4, label_ymin + 5), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            # Only record detections if scanning is active
            if camera_active:
                clothes_detected.append(class_name)

        # FPS overlay
        fps = 1.0 / max(1e-6, (time() - t_start))
        fps_log.append(fps)
        if len(fps_log) > 100:
            fps_log.pop(0)
        cv2.putText(frame, f"FPS: {np.mean(fps_log):.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Clothing Detection", frame)
        if record and recorder is not None:
            recorder.write(frame)

    # Cleanup
    if cap is not None and source_type in ['video', 'usb']:
        cap.release()
        cap = None
    if record and recorder is not None:
        recorder.release()
        recorder = None
    cv2.destroyAllWindows()

    unique_clothes = set(clothes_detected)
    print("\nDetected items this run:", unique_clothes)

    # === Face recognition / enrollment, then recommendation + feedback ===
    user_profile = None
    if last_frame_for_face is not None:
        user_profile, dist = face_id.match_user(last_frame_for_face)
        if user_profile:
            print(f"👤 Recognized user: {user_profile['name']} (distance={dist:.3f})")
            say(f"Welcome back {user_profile['name']}")
        else:
            print("🆕 New user detected (no match).")
            say("Hello! Looks like you're new here, what's your name?")
            name = input("What's your name? ").strip() or "User"
            enrolled = face_id.enroll_user(last_frame_for_face, name=name)
            if enrolled:
                print(f"Enrolled new user: {enrolled['name']} ({enrolled['user_id']})")
                say(f"Nice to meet you {enrolled['name']}")
                user_profile = enrolled
            else:
                print("Could not detect a face for enrollment. Continuing anonymously.")
                say("I couldn't capture your face this time. I'll continue without a profile.")

        if unique_clothes:
            # === GET WEATHER DATA FOR AI TRAINING ===
            city, current_temp, humidity, wind_speed = get_city_and_temperature(OPENWEATHER_API_KEY)
            
            # === START AI TRAINING DATA COLLECTION ===
            session_id = ai_collector.start_session(
                user_profile=user_profile,
                detected_clothes=list(unique_clothes),
                weather_data={
                    "temperature": current_temp,
                    "city": city,
                    "humidity": humidity if humidity is not None else 50,  # Use real humidity or default
                    "wind_speed": wind_speed if wind_speed is not None else 0   # Use real wind speed or default
                }
            )
            
            # Make the recommendation to user (this now uses hybrid AI + rules)
            # The recommend_for_user function will choose between AI and rules automatically
            recommend_for_user(unique_clothes, user_profile, city, current_temp, humidity, wind_speed)
            
            # For training data collection, record both AI and rule-based recommendations
            rule_delta, rule_explanation, rule_confidence = calculate_smart_delta(user_profile, current_temp)
            ai_collector.record_recommendation(session_id, rule_delta, rule_confidence, "hybrid_ai_rules")
            
            # Get user feedback
            feedback = prompt_feedback_and_update_user(user_profile)
            
            # Record feedback for AI training (this creates a complete training example!)
            if feedback:
                ai_collector.record_feedback(session_id, feedback)
            
            # Show AI training progress
            stats = ai_collector.get_training_stats()
            if stats["total_examples"] > 0:
                print(f"\n🤖 AI Training Progress: {stats['total_examples']} examples collected")
                print(f"   📊 Feedback distribution: {stats['feedback_distribution']}")
                if stats["total_examples"] >= 50:
                    print(f"   🎯 Ready for AI model training soon!")
            # === END AI TRAINING DATA COLLECTION ===
        else:
            feedback = None
            city, current_temp = None, None

        # === Save history if we have a user profile ===
        if user_profile and unique_clothes:
            from datetime import datetime
            # Use the weather data we already retrieved above
            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "clothes_detected": list(unique_clothes),
                "city": city,
                "temperature": current_temp,
                "feedback": feedback
            }
            face_id.update_user_prefs(user_profile["user_id"], {
                "history": user_profile.get("prefs", {}).get("history", []) + [entry]
            })
            print("📒 Session saved to history.")


def main_loop():
    while True:
        run_cycle()
        print("Run again? (y/n)")
        ans = input("> ").strip().lower()
        if ans != 'y':
            break

if __name__ == "__main__":
    main_loop()
