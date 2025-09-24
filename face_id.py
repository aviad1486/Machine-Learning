# face_id.py
import os
import json
import uuid
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from deepface import DeepFace

DB_PATH = "users.json"

MODEL_NAME = "Facenet512"          
DETECTOR_BACKEND = "opencv"      
DISTANCE_METRIC = "cosine"         
MATCH_THRESHOLD = 0.35        


# ------------ Storage ------------
def _load_db() -> List[Dict[str, Any]]:
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def _save_db(users: List[Dict[str, Any]]):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


# ------------ Embeddings ------------
def extract_embedding_bgr(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Given a BGR frame (OpenCV), detect the most prominent face and return its embedding as np.array.
    Returns None if no face found or representation failed.
    """
    try:
        # First try with strict detection
        reps = DeepFace.represent(
            img_path=frame_bgr,                 # can pass np array directly
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,             # raise if no face
            align=True
        )
        # DeepFace.represent returns a list of dicts (one per face); take the first
        if isinstance(reps, list) and len(reps) > 0 and "embedding" in reps[0]:
            emb = np.array(reps[0]["embedding"], dtype=np.float32)
            return emb
        return None
    except Exception as e:
        # If strict detection fails, try with more lenient settings
        try:
            reps = DeepFace.represent(
                img_path=frame_bgr,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,        # Don't raise if no face
                align=True
            )
            if isinstance(reps, list) and len(reps) > 0 and "embedding" in reps[0]:
                emb = np.array(reps[0]["embedding"], dtype=np.float32)
                print("✅ Face detected with lenient settings!")
                return emb
            print("❌ No face detected even with lenient settings")
            return None
        except Exception as e2:
            print(f"❌ Face detection failed completely: {e2}")
            return None


# ------------ Similarity ------------
def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # both are 1D vectors
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    # cosine similarity -> distance
    return float(1.0 - np.dot(a_norm, b_norm))

def _min_distance_to_user(emb: np.ndarray, user: Dict[str, Any]) -> float:
    dists = []
    for e in user.get("face_embeddings", []):
        e = np.array(e, dtype=np.float32)
        dists.append(_cosine_distance(emb, e))
    return min(dists) if dists else 999.0


# ------------ Public API ------------
def match_user(frame_bgr: np.ndarray, threshold: float = MATCH_THRESHOLD) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """
    Tries to extract an embedding from frame, then finds the closest user in DB.
    Returns (user_dict, distance) if matched under threshold; else (None, None).
    """
    emb = extract_embedding_bgr(frame_bgr)
    if emb is None:
        return None, None

    users = _load_db()
    if not users:
        return None, None

    best_user = None
    best_dist = 999.0
    for u in users:
        d = _min_distance_to_user(emb, u)
        if d < best_dist:
            best_dist = d
            best_user = u

    if best_user is not None and best_dist <= threshold:
        return best_user, best_dist
    return None, None


def enroll_user(frame_bgr: np.ndarray, name: str, base_prefs: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Creates a new user with one embedding from the given frame. Returns the user dict or None on failure.
    """
    emb = extract_embedding_bgr(frame_bgr)
    if emb is None:
        return None

    users = _load_db()
    new_user = {
        "user_id": str(uuid.uuid4()),
        "name": name,
        "face_embeddings": [emb.tolist()],
        "prefs": base_prefs or {
            "delta_temp": 0,
            "history": []  # you can log sessions here later
        }
    }
    users.append(new_user)
    _save_db(users)
    return new_user


def add_embedding_to_user(user_id: str, frame_bgr: np.ndarray) -> bool:
    """
    Adds another embedding to an existing user (helps robustness). Returns True on success.
    """
    emb = extract_embedding_bgr(frame_bgr)
    if emb is None:
        return False

    users = _load_db()
    for u in users:
        if u.get("user_id") == user_id:
            u.setdefault("face_embeddings", []).append(emb.tolist())
            _save_db(users)
            return True
    return False


def update_user_prefs(user_id: str, patch: Dict[str, Any]) -> bool:
    """
    Patch user prefs (e.g., update delta_temp, per-item preferences, etc.). Returns True on success.
    """
    users = _load_db()
    for u in users:
        if u.get("user_id") == user_id:
            prefs = u.setdefault("prefs", {})
            prefs.update(patch)
            _save_db(users)
            return True
    return False
