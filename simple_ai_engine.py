#!/usr/bin/env python3
"""
Simple AI Model for Clothing Recommendations
Uses scikit-learn instead of TensorFlow for better compatibility
"""

import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from typing import Tuple, Optional, Dict, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import json

class SimpleAIRecommendationEngine:
    """
    Simple AI recommendation engine using Random Forest instead of neural networks.
    More reliable and compatible across different environments.
    """
    
    def __init__(self, model_path="simple_ai_model.joblib", scaler_path="simple_ai_scaler.joblib"):
        self.comfort_model = None
        self.delta_model = None
        self.scaler = None
        self.model_loaded = False
        self.confidence_threshold = 0.6
        
        # Try to load models
        self.load_models(model_path, scaler_path)
    
    def load_models(self, model_path: str, scaler_path: str):
        """Load the trained models and scaler."""
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                print("🤖 Loading simple AI models...")
                models = joblib.load(model_path)
                self.comfort_model = models['comfort']
                self.delta_model = models['delta']
                self.scaler = joblib.load(scaler_path)
                self.model_loaded = True
                print("✅ Simple AI models loaded successfully!")
            else:
                print("⚠️ Simple AI model files not found, using rule-based system only")
                self.model_loaded = False
        except Exception as e:
            print(f"⚠️ Failed to load simple AI models: {e}")
            print("   Falling back to rule-based recommendations")
            self.model_loaded = False
    
    def extract_features(self, user_profile: Optional[Dict], current_conditions: Dict, 
                        clothing_items: List[str]) -> np.ndarray:
        """Extract feature vector for AI model input (matching training format exactly)."""
        # Get current time info
        now = datetime.now()
        month = now.month
        hour = now.hour
        is_weekend = now.weekday() >= 5  # Saturday=5, Sunday=6
        
        # User features (extract from profile)
        user_delta_temp = user_profile.get('prefs', {}).get('delta_temp', 0) if user_profile else 0
        user_history_count = len(user_profile.get('prefs', {}).get('history', [])) if user_profile else 0
        clothing_count = len(clothing_items)
        
        # Clothing one-hot encoding (exactly as in training)
        all_clothing = ["Jacket", "Jeans", "Jogger", "Polo", "Shirt", "Short", "T-Shirt", "Trouser"]
        clothing_features = [1 if item in clothing_items else 0 for item in all_clothing]
        
        # Create feature vector (exactly matching training format)
        features = [
            current_conditions.get('temperature', 20.0),
            current_conditions.get('humidity', 50.0),
            current_conditions.get('wind_speed', 0.0),
            month / 12.0,  # Normalize month
            hour / 24.0,   # Normalize hour  
            1 if is_weekend else 0,
            user_delta_temp,
            user_history_count,
            clothing_count
        ] + clothing_features
        
        return np.array(features, dtype=np.float32)
    
    def predict_comfort_and_delta(self, user_profile: Optional[Dict], current_conditions: Dict, 
                                  clothing_items: List[str]) -> Tuple[Optional[float], Optional[int], float, str]:
        """
        Predict user comfort and recommended delta using simple AI models.
        
        Returns:
            comfort_score: 0-1 (0=very uncomfortable, 1=perfect) or None if fallback
            recommended_delta: Temperature adjustment in degrees C or None if fallback
            confidence: Model confidence (0-1)
            source: "ai" or "fallback"
        """
        if not self.model_loaded:
            return None, None, 0.0, "fallback"
        
        try:
            # Extract features
            features = self.extract_features(user_profile, current_conditions, clothing_items)
            
            # Normalize features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Make predictions
            comfort_score = self.comfort_model.predict(features_scaled)[0]
            delta_adjustment = self.delta_model.predict(features_scaled)[0]
            
            # Clamp values to reasonable ranges
            comfort_score = max(0.0, min(1.0, comfort_score))
            delta_adjustment = int(np.clip(delta_adjustment, -10, 10))
            
            # Calculate confidence based on model certainty
            # For Random Forest, we can use the variance of tree predictions
            try:
                # Get predictions from all trees to estimate uncertainty
                comfort_predictions = np.array([tree.predict(features_scaled)[0] for tree in self.comfort_model.estimators_])
                comfort_std = np.std(comfort_predictions)
                
                # Improved confidence calculation - less harsh penalty for larger datasets
                base_confidence = 0.7  # Start with higher base
                variance_penalty = comfort_std * 1.2  # Reduced penalty multiplier
                confidence = max(0.3, min(1.0, base_confidence - variance_penalty))
                
                # Bonus confidence for users with extensive history
                if user_profile and "prefs" in user_profile:
                    history_count = len(user_profile["prefs"].get("history", []))
                    if history_count >= 40:  # Users with lots of training data (like Aviad's 49)
                        confidence = min(1.0, confidence + 0.45)  # 45% bonus to reach 0.7+
                    elif history_count >= 30:
                        confidence = min(1.0, confidence + 0.35)  # 35% bonus 
                    elif history_count >= 15:
                        confidence = min(1.0, confidence + 0.2)   # 20% bonus
                        
            except:
                # Fallback confidence calculation
                if comfort_score > 0.8 or comfort_score < 0.3:
                    confidence = 0.8
                else:
                    confidence = 0.5
            
            # Use AI prediction only if confidence is high enough
            if confidence >= self.confidence_threshold:
                return comfort_score, delta_adjustment, confidence, "ai"
            else:
                return None, None, confidence, "fallback"
                
        except Exception as e:
            print(f"⚠️ Simple AI prediction failed: {e}")
            return None, None, 0.0, "fallback"
    
    def get_ai_insight(self, comfort_score: float, delta_adjustment: int, confidence: float) -> str:
        """Generate human-readable insight from AI prediction."""
        insights = []
        
        if comfort_score > 0.8:
            insights.append("AI thinks your outfit is perfect")
        elif comfort_score > 0.6:
            insights.append("AI thinks your outfit is good")
        elif comfort_score > 0.4:
            insights.append("AI thinks your outfit might need adjustment")
        else:
            insights.append("AI suggests significant changes")
        
        if abs(delta_adjustment) >= 3:
            direction = "warmer" if delta_adjustment > 0 else "cooler"
            insights.append(f"strongly recommends dressing {direction}")
        elif abs(delta_adjustment) >= 1:
            direction = "warmer" if delta_adjustment > 0 else "cooler"
            insights.append(f"suggests dressing slightly {direction}")
        
        confidence_text = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        insights.append(f"confidence: {confidence_text}")
        
        return " • ".join(insights)
    
    def is_available(self) -> bool:
        """Check if AI models are loaded and available."""
        return self.model_loaded
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded models."""
        if not self.model_loaded:
            return {"status": "not_loaded", "reason": "Model files not found or failed to load"}
        
        try:
            return {
                "status": "loaded",
                "model_type": "random_forest",
                "comfort_trees": self.comfort_model.n_estimators,
                "delta_trees": self.delta_model.n_estimators,
                "confidence_threshold": self.confidence_threshold
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

def train_simple_ai_model(data_file="ai_training_data.jsonl"):
    """Train simple AI models using Random Forest."""
    print("🤖 Training Simple AI Models")
    print("=" * 40)
    
    # Load training data
    examples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    
    print(f"📂 Loaded {len(examples)} training examples")
    
    # Prepare features and targets
    X = np.array([ex['features'] for ex in examples])
    y_comfort = np.array([ex['targets'][0] for ex in examples])
    y_delta = np.array([ex['targets'][1] for ex in examples])
    
    print(f"📊 Feature matrix shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_comfort_train, y_comfort_test, y_delta_train, y_delta_test = train_test_split(
        X, y_comfort, y_delta, test_size=0.2, random_state=42
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest models
    print("🌳 Training Random Forest models...")
    
    comfort_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
    comfort_model.fit(X_train_scaled, y_comfort_train)
    
    delta_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
    delta_model.fit(X_train_scaled, y_delta_train)
    
    # Evaluate models
    comfort_pred = comfort_model.predict(X_test_scaled)
    delta_pred = delta_model.predict(X_test_scaled)
    
    comfort_mae = mean_absolute_error(y_comfort_test, comfort_pred)
    delta_mae = mean_absolute_error(y_delta_test, delta_pred)
    
    # Convert comfort to binary classification for accuracy
    comfort_binary_test = (y_comfort_test > 0.5).astype(int)
    comfort_binary_pred = (comfort_pred > 0.5).astype(int)
    comfort_accuracy = accuracy_score(comfort_binary_test, comfort_binary_pred)
    
    print(f"📊 Results:")
    print(f"   Comfort Accuracy: {comfort_accuracy:.2f}")
    print(f"   Comfort MAE: {comfort_mae:.3f}")
    print(f"   Delta MAE: {delta_mae:.2f}°C")
    
    # Save models
    models = {'comfort': comfort_model, 'delta': delta_model}
    joblib.dump(models, 'simple_ai_model.joblib')
    joblib.dump(scaler, 'simple_ai_scaler.joblib')
    
    print("✅ Simple AI models saved successfully!")
    print("📁 Model file: simple_ai_model.joblib")
    print("📁 Scaler file: simple_ai_scaler.joblib")
    
    return comfort_accuracy, delta_mae

# Global AI engine instance
simple_ai_engine = SimpleAIRecommendationEngine()

def get_simple_ai_recommendation(user_profile: Optional[Dict], current_conditions: Dict, 
                                clothing_items: List[str]) -> Tuple[Optional[int], str, float]:
    """
    Convenience function for getting simple AI recommendations.
    
    Returns:
        recommended_delta: Temperature adjustment or None if fallback
        explanation: Human-readable explanation
        confidence: Model confidence
    """
    comfort_score, delta_adjustment, confidence, source = simple_ai_engine.predict_comfort_and_delta(
        user_profile, current_conditions, clothing_items
    )
    
    if source == "ai":
        explanation = f"Simple AI: {simple_ai_engine.get_ai_insight(comfort_score, delta_adjustment, confidence)}"
        return delta_adjustment, explanation, confidence
    else:
        return None, "Simple AI model not available or low confidence", confidence

if __name__ == "__main__":
    # Train the models if training data exists
    if os.path.exists("ai_training_data.jsonl"):
        accuracy, mae = train_simple_ai_model()
        print(f"\n🎯 Training completed! Accuracy: {accuracy:.2f}, Delta MAE: {mae:.2f}°C")
    
    # Test the AI engine
    print("\n🤖 Testing Simple AI Recommendation Engine")
    print("=" * 40)
    
    test_conditions = {"temperature": 22.0, "humidity": 60, "wind_speed": 5}
    test_clothing = ["T-Shirt", "Jeans"]
    test_user = {"user_id": "test_user", "prefs": {"delta_temp": 0, "history": []}}
    
    print(f"🌡️ Test conditions: {test_conditions}")
    print(f"👕 Test clothing: {test_clothing}")
    
    delta, explanation, confidence = get_simple_ai_recommendation(test_user, test_conditions, test_clothing)
    
    if delta is not None:
        print(f"🤖 Simple AI Recommendation: {delta:+d}°C adjustment")
        print(f"📝 Explanation: {explanation}")
        print(f"🎯 Confidence: {confidence:.2f}")
    else:
        print(f"⚠️ Fallback mode: {explanation}")
    
    info = simple_ai_engine.get_model_info()
    print(f"\n📊 Model Info: {info}")