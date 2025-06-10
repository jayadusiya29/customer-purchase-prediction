# main.py
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

warnings.filterwarnings('ignore')

# Generate synthetic data
np.random.seed(0)
data_size = 200
features = np.random.rand(data_size, 2)
labels = (features[:, 0] + features[:, 1] > 1).astype(int)

df = pd.DataFrame(features, columns=['VisitDuration', 'PagesVisited'])
df['Purchase'] = labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[['VisitDuration', 'PagesVisited']],
    df['Purchase'],
    test_size=0.2,
    random_state=42
)

# Define and compile model
model = Sequential([
    Dense(10, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=10)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
