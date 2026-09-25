import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
dataset = pd.read_csv('ml/dataset/engineered_features.csv')
features = ['rolling_ear_mean','rolling_mar_max','rolling_head_pitch','closure_duration']
X = dataset[features]
y = dataset['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Predict on test data
y_pred = model.predict(X_test)

# Print Report Card
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Detailed Performance Report:")
print(classification_report(y_test, y_pred))
os.makedirs('ml/models', exist_ok=True)
joblib.dump(model, 'ml/models/drowsiness_model.pkl')
print("\nSuccess! Trained model saved to: ml/models/drowsiness_model.pkl")