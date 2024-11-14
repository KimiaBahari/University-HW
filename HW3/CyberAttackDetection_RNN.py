import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 1. Data Collection
# Load the dataset (example with KDD Cup 99)
url = "https://raw.githubusercontent.com/ramzi-abbas/cyberattack-classification/master/KDDTrain+.csv"
data = pd.read_csv(url)

# Display first few rows of the dataset
print(data.head())

# 2. Data Preprocessing
# Select features and labels
# Example: Only a few relevant features (adjust depending on your dataset)
data = data[['duration', 'protocol_type', 'service', 'src_bytes', 'dst_bytes', 'flag', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'attack']]

# Convert categorical features into numerical using label encoding
data['protocol_type'] = data['protocol_type'].astype('category').cat.codes
data['service'] = data['service'].astype('category').cat.codes
data['flag'] = data['flag'].astype('category').cat.codes

# Map attack types to 0 (normal) and 1 (attack)
data['attack'] = data['attack'].apply(lambda x: 1 if x != 'normal' else 0)

# Split data into features (X) and target (y)
X = data.drop('attack', axis=1)
y = data['attack']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Reshaping data for RNN input (sequence-based input)
# LSTM expects input with shape [samples, time steps, features]
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# 4. RNN Model Design (LSTM)
model = Sequential()

# First LSTM layer
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

# Second LSTM layer
model.add(LSTM(units=32))
model.add(Dropout(0.2))

# Output layer (binary classification: normal or attack)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 5. Model Training
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# 6. Model Evaluation
# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"AUC-ROC: {auc_roc}")

# 7. Visualize the results (Actual vs Predicted)
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='red')
plt.title('Actual vs Predicted Attack/Normal')
plt.xlabel('Sample Index')
plt.ylabel('Attack/Normal (1/0)')
plt.legend()
plt.show()

# 8. Conclusion and Feature Analysis
# Analyze important features
# In this case, analyzing which features are most predictive could be done using SHAP or permutation importance techniques.
