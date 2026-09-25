import os
import numpy as np
import pandas as pd

def generate_driver_dataset(filename="ml/dataset/driving_features.csv", fps=30, duration_minutes=15):
    total_frames = fps * 60 * duration_minutes
    dt = 1.0 / fps
    current_time = 0.0
    
    print(f"Generating {total_frames} frames ({duration_minutes} minutes) of driving data...")

    # Lists to hold data in memory
    timestamps = []
    ears = []
    mars = []
    head_pitches = []
    labels = []

    frame = 0
    while frame < total_frames:
        # Step A: Pick what state the driver enters next
        # (70% Alert, 15% Warning/Yawn, 10% Drowsy, 5% Critical)
        scenario = np.random.choice(
            ["ALERT", "WARNING", "DROWSY", "CRITICAL"],
            p=[0.70, 0.15, 0.10, 0.05]
        )

        # Step B: Pick duration of this event in seconds
        if scenario == "ALERT":
            event_duration = np.random.uniform(5.0, 15.0)
        elif scenario == "WARNING":
            event_duration = np.random.uniform(2.0, 5.0)
        elif scenario == "DROWSY":
            event_duration = np.random.uniform(2.0, 4.0)
        else: # CRITICAL
            event_duration = np.random.uniform(3.0, 6.0)

        # Step C: Convert seconds to camera frame count
        event_frames = int(event_duration * fps)

        # Step D: Generate each frame's biological values
        for _ in range(event_frames):
            if frame >= total_frames:
                break

            current_time += dt
            timestamps.append(round(current_time, 3))

            # Scenario 1: Alert (eyes open ~0.30 with occasional 0.15s blinks)
            if scenario == "ALERT":
                if np.random.rand() < 0.03:
                    ear = np.random.uniform(0.12, 0.18)  # quick normal blink
                else:
                    ear = np.random.uniform(0.28, 0.34)  # open eyes
                
                mar = np.random.uniform(0.08, 0.22)
                pitch = np.random.uniform(-4.0, 4.0)
                label = "ALERT"

            # Scenario 2: Warning (yawning or heavy eyelids)
            elif scenario == "WARNING":
                is_yawn = np.random.rand() < 0.60
                if is_yawn:
                    mar = np.random.uniform(0.60, 0.85)  # yawn
                    ear = np.random.uniform(0.20, 0.26)
                else:
                    mar = np.random.uniform(0.15, 0.30)
                    ear = np.random.uniform(0.19, 0.23)
                
                pitch = np.random.uniform(-8.0, 0.0)
                label = "WARNING"

            # Scenario 3: Drowsy (eyes closed, head drooping)
            elif scenario == "DROWSY":
                ear = np.random.uniform(0.10, 0.17)
                mar = np.random.uniform(0.15, 0.40)
                pitch = np.random.uniform(-18.0, -10.0)
                label = "DROWSY"

            # Scenario 4: Critical (deep microsleep)
            else:
                ear = np.random.uniform(0.08, 0.14)
                mar = np.random.uniform(0.10, 0.30)
                pitch = np.random.uniform(-25.0, -15.0)
                label = "CRITICAL"

            # Append current frame values to memory lists
            ears.append(round(float(ear), 4))
            mars.append(round(float(mar), 4))
            head_pitches.append(round(float(pitch), 2))
            labels.append(label)
            
            frame += 1

    # Step E: Convert lists to DataFrame table
    df = pd.DataFrame({
        "timestamp": timestamps,
        "ear": ears,
        "mar": mars,
        "head_pitch": head_pitches,
        "label": labels
    })

    # Step F: Ensure folder exists and save to CSV
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    
    print(f"Dataset successfully created at: {filename}")
    print(f"Total rows created: {len(df)}")
    print("\nClass distribution (Frame counts):")
    print(df["label"].value_counts())

if __name__ == "__main__":
    generate_driver_dataset()