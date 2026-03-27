import os
import numpy as np

# Path to dataset
DATA_PATH = "data"

X = []
y = []

# Loop through each gesture folder
for label in os.listdir(DATA_PATH):
    gesture_path = os.path.join(DATA_PATH, label)

    if os.path.isdir(gesture_path):
        for file in os.listdir(gesture_path):
            file_path = os.path.join(gesture_path, file)

            # Load .npy file
            data = np.load(file_path)

            X.append(data)
            y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Sample labels:", y[:10])