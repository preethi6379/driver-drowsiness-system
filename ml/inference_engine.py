import joblib
import numpy as np
import pandas as pd
from collections import deque

class DrowsinessInferenceEngine:
    def __init__(self, model_path='ml/models/drowsiness_model.pkl'):
        # 1. Load the frozen model into RAM
        print("Loading trained AI model...")
        self.model = joblib.load(model_path)
        
        # 2. Self-cleaning sliding windows for the last 60 frames (2 seconds)
        self.ear_history = deque(maxlen=60)
        self.mar_history = deque(maxlen=60)
        self.pitch_history = deque(maxlen=60)
        
        # 3. Stopwatch for continuous closed eyes
        self.closed_frames = 0
        self.fps = 30.0
        
        # Risk score weights for smooth 0 - 100 gauge
        self.risk_weights = {
            'ALERT': 0.0,
            'WARNING': 35.0,
            'DROWSY': 75.0,
            'CRITICAL': 100.0
        }

    def predict(self, ear, mar, head_pitch):
        # Step A: Push new frame data into sliding queues
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        self.pitch_history.append(head_pitch)

        # Step B: Update continuous eye closure stopwatch
        if ear < 0.20:
            self.closed_frames += 1
        else:
            self.closed_frames = 0
            
        closure_duration = self.closed_frames / self.fps

        # Step C: Calculate rolling statistics on the fly
        rolling_ear_mean = float(np.mean(self.ear_history))
        rolling_mar_max = float(np.max(self.mar_history))
        rolling_head_pitch = float(np.mean(self.pitch_history))

        # Step D: Package into 1-row table for model
        input_features = pd.DataFrame([{
            'rolling_ear_mean': rolling_ear_mean,
            'rolling_mar_max': rolling_mar_max,
            'rolling_head_pitch': rolling_head_pitch,
            'closure_duration': closure_duration
        }])

        # Step E: Predict State and Continuous Risk Score (0 - 100)
        state = self.model.predict(input_features)[0]
        
        # Calculate smooth risk score using prediction probabilities
        probabilities = self.model.predict_proba(input_features)[0]
        risk_score = sum(
            prob * self.risk_weights.get(cls, 0.0) 
            for cls, prob in zip(self.model.classes_, probabilities)
        )

        # Step F: Return clean dictionary for Arduino & Backend
        return {
            "state": state,
            "risk_score": round(float(risk_score), 1),
            "closure_duration": round(float(closure_duration), 2),
            "metrics": {
                "rolling_ear_mean": round(rolling_ear_mean, 3),
                "rolling_mar_max": round(rolling_mar_max, 3),
                "rolling_head_pitch": round(rolling_head_pitch, 1)
            }
        }

# ==========================================
# TEST HARNESS: Simulate live incoming frames
# ==========================================
if __name__ == "__main__":
    engine = DrowsinessInferenceEngine()
    print("\n--- Simulation 1: Driver is awake (10 normal frames) ---")
    for _ in range(10):
        result = engine.predict(ear=0.31, mar=0.15, head_pitch=0.0)
    print(f"Result: State = {result['state']} | Risk Score = {result['risk_score']} | Closure = {result['closure_duration']}s")

    print("\n--- Simulation 2: Driver falls asleep (eyes closed for 60 frames / 2 seconds) ---")
    for frame in range(60):
        result = engine.predict(ear=0.12, mar=0.15, head_pitch=-18.0)
        # Print progress every 15 frames
        if (frame + 1) % 15 == 0:
            print(f"Frame {frame+1}/60: State = {result['state']} | Risk Score = {result['risk_score']} | Eyes Closed = {result['closure_duration']}s")