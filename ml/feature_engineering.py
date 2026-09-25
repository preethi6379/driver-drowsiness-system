import pandas as pd
import numpy as np
dataset = pd.read_csv('ml/dataset/driving_features.csv')
dataset['rolling_ear_mean'] = dataset['ear'].rolling(window=60).mean()
dataset['rolling_mar_max'] = dataset['mar'].rolling(window=60).max()
dataset['rolling_head_pitch'] = dataset['head_pitch'].rolling(window=60).mean()
closed_frames = 0
durations = []

for i in dataset['ear']:
    if i < 0.20:
        closed_frames += 1
    else:
        closed_frames = 0
    seconds = closed_frames / 30.0
    durations.append(seconds)

dataset['closure_duration'] = durations
dataset = dataset.dropna()
dataset.to_csv('ml/dataset/engineered_features.csv')
print("Features successfully created!")
print(f"Total clean rows: {len(dataset)}")
print(dataset[['rolling_ear_mean', 'rolling_mar_max', 'closure_duration', 'label']].head())
print(dataset['label'].value_counts())