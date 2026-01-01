import os
import numpy as np
import itertools
import copy
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATASET_PATH = "isl_landmarks"  
MODEL_NAME = "isl_rf_model.pkl"

def pre_process_landmark(landmark_array):
    """
    Takes a numpy array of shape (21, 3) or (21, 2) or flattened equivalent
    and converts it to the relative, normalized 42-float list used by app.py.
    """

    temp_array = np.array(landmark_array).flatten()

    if len(temp_array) == 63: 

        reshaped = temp_array.reshape(21, 3)
        points = reshaped[:, :2] 
    elif len(temp_array) == 42:
        points = temp_array.reshape(21, 2)
    else:
        return None

    base_x, base_y = points[0][0], points[0][1]
    
    relative_points = []
    for i in range(len(points)):
        x = points[i][0] - base_x
        y = points[i][1] - base_y
        relative_points.append([x, y])

    flat = list(itertools.chain.from_iterable(relative_points))

    max_value = max(list(map(abs, flat)))
    if max_value == 0:
        max_value = 1
    
    normalized = [float(n) / max_value for n in flat]

    return normalized

data = []
labels = []

print(f"📂 Scanning dataset at: {DATASET_PATH}...")

if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: Folder '{DATASET_PATH}' not found. Please unzip your data here.")
    exit()

classes = sorted(os.listdir(DATASET_PATH))
print(f"Found {len(classes)} classes: {classes}")

for label in classes:
    class_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_path): continue

    npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
    
    print(f"   Processing class '{label}' ({len(npy_files)} files)...", end="\r")
    
    for npy_file in npy_files:
        file_path = os.path.join(class_path, npy_file)
        
        try:
            lm_array = np.load(file_path)
            
            processed_features = pre_process_landmark(lm_array)
            
            if processed_features is not None:
                data.append(processed_features)
                labels.append(label)
        except Exception as e:
            continue

print(f"\n✅ Loading Complete! Processed {len(data)} samples.")

print("🧠 Training Random Forest Model...")

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {acc * 100:.2f}%")

joblib.dump(rf, MODEL_NAME)
print(f"💾 Saved '{MODEL_NAME}'. Move this file to your app folder!")