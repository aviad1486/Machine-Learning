#!/usr/bin/env python3
"""
Data Collection Module for AI Training
Integrates with your existing recommendation system to collect training data
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

class AITrainingDataCollector:
    """
    Collects training data from your existing recommendation system
    to train the AI model later.
    """
    
    def __init__(self, data_file: str = "ai_training_data.jsonl"):
        self.data_file = data_file
        self.session_data = {}
    
    def start_session(self, user_profile: Dict | None, detected_clothes: List[str], 
                     weather_data: Dict):
        """
        Start collecting data for a recommendation session.
        Call this when you make a recommendation.
        """
        session_id = datetime.now().isoformat()
        
        self.session_data[session_id] = {
            "session_id": session_id,
            "timestamp": session_id,
            "user_id": user_profile.get("user_id", "anonymous") if user_profile else "anonymous",
            "user_name": user_profile.get("name", "unknown") if user_profile else "unknown",
            
            # Weather features
            "temperature": weather_data.get("temperature"),
            "city": weather_data.get("city"),
            "humidity": weather_data.get("humidity", 50),  # Default if not available
            "wind_speed": weather_data.get("wind_speed", 0),  # Default if not available
            
            # Clothing features (from your YOLO detection)
            "detected_clothes": detected_clothes,
            "clothing_count": len(detected_clothes),
            "has_jacket": "Jacket" in detected_clothes,
            "has_short_sleeves": any(item in detected_clothes for item in ["T-Shirt", "Polo"]),
            "has_long_sleeves": any(item in detected_clothes for item in ["Shirt", "Jacket"]),
            "has_shorts": "Short" in detected_clothes,
            "has_long_pants": any(item in detected_clothes for item in ["Jeans", "Trouser", "Jogger"]),
            
            # User features (from your existing system)
            "user_delta_temp": user_profile.get("prefs", {}).get("delta_temp", 0) if user_profile else 0,
            "user_history_count": len(user_profile.get("prefs", {}).get("history", [])) if user_profile else 0,
            
            # Seasonal features
            "month": datetime.now().month,
            "season": self._get_season(datetime.now().month),
            "hour": datetime.now().hour,
            "is_weekend": datetime.now().weekday() >= 5,
            
            # Recommendation made
            "recommended_delta": None,  # To be filled by your current system
            "recommendation_confidence": None,  # To be filled by your current system
            "recommendation_source": "rule_based",  # vs "ai" later
            
            # Feedback (to be filled when user gives feedback)
            "user_feedback": None,
            "feedback_received": False
        }
        
        return session_id
    
    def record_recommendation(self, session_id: str, recommended_delta: int, 
                            confidence: float = 1.0, source: str = "rule_based"):
        """
        Record the recommendation that was made.
        """
        if session_id in self.session_data:
            self.session_data[session_id].update({
                "recommended_delta": recommended_delta,
                "recommendation_confidence": confidence,
                "recommendation_source": source
            })
    
    def record_feedback(self, session_id: str, feedback: str):
        """
        Record user feedback for this session.
        This creates a complete training example!
        """
        if session_id in self.session_data:
            self.session_data[session_id].update({
                "user_feedback": feedback,
                "feedback_received": True,
                "feedback_timestamp": datetime.now().isoformat()
            })
            
            # Save this complete training example
            self._save_training_example(self.session_data[session_id])
            
            # Clean up session data
            del self.session_data[session_id]
    
    def _save_training_example(self, example: Dict):
        """Save complete training example to file."""
        try:
            # Convert to AI training format
            ai_example = self._convert_to_ai_format(example)
            
            # Append to training file
            with open(self.data_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(ai_example) + '\n')
            
            print(f"✅ Saved training example: {example['user_feedback']} feedback")
            
        except Exception as e:
            print(f"❌ Failed to save training example: {e}")
    
    def _convert_to_ai_format(self, session: Dict) -> Dict:
        """
        Convert session data to AI training format.
        
        Features: [temperature, clothing_encoded, user_features, seasonal_features]
        Target: [comfort_score, ideal_delta]
        """
        # Encode clothing as features
        all_clothing = ["Jacket", "Jeans", "Jogger", "Polo", "Shirt", "Short", "T-Shirt", "Trouser"]
        clothing_features = [1 if item in session["detected_clothes"] else 0 for item in all_clothing]
        
        # Create feature vector
        features = [
            session["temperature"] or 20.0,
            session["humidity"],
            session["wind_speed"],
            session["month"] / 12.0,  # Normalize month
            session["hour"] / 24.0,   # Normalize hour
            1 if session["is_weekend"] else 0,
            session["user_delta_temp"],
            session["user_history_count"],
            session["clothing_count"]
        ] + clothing_features
        
        # Convert feedback to target values
        comfort_score, ideal_delta = self._feedback_to_targets(session["user_feedback"])
        
        return {
            "features": features,
            "targets": [comfort_score, ideal_delta],
            "metadata": {
                "session_id": session["session_id"],
                "user_id": session["user_id"],
                "feedback": session["user_feedback"],
                "temperature": session["temperature"],
                "clothes": session["detected_clothes"]
            }
        }
    
    def _feedback_to_targets(self, feedback: str) -> tuple[float, int]:
        """
        Convert user feedback to AI training targets.
        
        Returns: (comfort_score, ideal_delta_adjustment)
        """
        if feedback == "good":
            return 1.0, 0  # Perfect comfort, no adjustment needed
        elif feedback == "bad-cold":
            return 0.2, +3  # Low comfort, need warmer clothes
        elif feedback == "bad-hot":
            return 0.2, -3  # Low comfort, need cooler clothes
        else:
            return 0.5, 0  # Unknown feedback
    
    def _get_season(self, month: int) -> str:
        """Get season from month."""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"
    
    def get_training_stats(self) -> Dict:
        """Get statistics about collected training data."""
        if not os.path.exists(self.data_file):
            return {"total_examples": 0}
        
        examples = []
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    examples.append(json.loads(line.strip()))
        except Exception as e:
            print(f"Error reading training data: {e}")
            return {"error": str(e)}
        
        if not examples:
            return {"total_examples": 0}
        
        # Calculate statistics
        feedbacks = [ex["metadata"]["feedback"] for ex in examples]
        temperatures = [ex["metadata"]["temperature"] for ex in examples if ex["metadata"]["temperature"]]
        
        stats = {
            "total_examples": len(examples),
            "feedback_distribution": {
                "good": feedbacks.count("good"),
                "bad-cold": feedbacks.count("bad-cold"),
                "bad-hot": feedbacks.count("bad-hot")
            },
            "temperature_range": {
                "min": min(temperatures) if temperatures else 0,
                "max": max(temperatures) if temperatures else 0,
                "avg": sum(temperatures) / len(temperatures) if temperatures else 0
            },
            "unique_users": len(set(ex["metadata"]["user_id"] for ex in examples))
        }
        
        return stats


# ===== INTEGRATION EXAMPLE =====

def show_integration_example():
    """
    Show how to integrate with your existing recommendation system.
    """
    
    integration_code = '''
    # ADD TO YOUR RecommendationSystemMirror.py:
    
    from ai_training_collector import AITrainingDataCollector
    
    # Initialize collector
    ai_collector = AITrainingDataCollector()
    
    # In your run_cycle() function, BEFORE calling recommend_for_user():
    session_id = ai_collector.start_session(
        user_profile=user_profile,
        detected_clothes=list(unique_clothes),
        weather_data={
            "temperature": current_temp,
            "city": city
        }
    )
    
    # AFTER your recommendation logic:
    smart_delta, explanation, confidence = calculate_smart_delta(user_profile, current_temp)
    ai_collector.record_recommendation(session_id, smart_delta, confidence)
    
    # AFTER getting user feedback:
    feedback = prompt_feedback_and_update_user(user_profile)
    if feedback:
        ai_collector.record_feedback(session_id, feedback)
    
    # Check training data progress:
    stats = ai_collector.get_training_stats()
    print(f"AI Training Progress: {stats}")
    '''
    
    return integration_code


if __name__ == "__main__":
    print("=== AI Training Data Collector ===")
    print("\nThis module collects training data from your existing system!")
    print("\n📊 What it collects:")
    print("✓ Weather conditions (temperature, humidity)")
    print("✓ Detected clothing items (from your YOLO model)")
    print("✓ User preferences (from your face recognition)")
    print("✓ Recommendations made (from your current logic)")
    print("✓ User feedback (good/bad-cold/bad-hot)")
    
    print("\n🔗 Integration:")
    print(show_integration_example())
    
    print("\n📈 After 2-3 weeks of data collection:")
    print("✓ Train neural network on collected examples")
    print("✓ Compare AI vs rule-based accuracy")
    print("✓ Deploy hybrid system")
    
    # Demo the collector
    collector = AITrainingDataCollector("demo_training_data.jsonl")
    
    # Simulate a session
    session_id = collector.start_session(
        user_profile={"user_id": "demo_user", "name": "Demo", "prefs": {"delta_temp": 2, "history": []}},
        detected_clothes=["T-Shirt", "Jeans"],
        weather_data={"temperature": 22.0, "city": "Tel Aviv"}
    )
    
    collector.record_recommendation(session_id, recommended_delta=2, confidence=0.8)
    collector.record_feedback(session_id, "bad-cold")
    
    stats = collector.get_training_stats()
    print(f"\n📊 Demo Stats: {stats}")