# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Data Collection
# Downloading the dataset from UCI Machine Learning in Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00375/household_power_consumption.txt"
# Assuming the file is in CSV format
data = pd.read_csv(url, sep=';', low_memory=False)

# Display the first few rows to understand the structure
print(data.head())

# 2. Data Preprocessing
# Dropping rows with missing values
data = data.dropna()

# Converting Date and Time columns to appropriate datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.hour

# Converting 'Global_active_power' to numeric (energy consumption)
data['Global_active_power'] = pd.to_numeric(data['Global_active_power'], errors='coerce')

# Dropping unnecessary columns (Date and Time are not needed as features for prediction)
data = data.drop(columns=['Date', 'Time'])

# Dropping rows that became NaN after conversion
data = data.dropna()

# Selecting features (X) and target variable (y)
X = data[['Global_active_power']]  # Features (for simplicity, we use only one feature)
y = data['Global_active_power']   # Target variable (energy consumption)

# 3. Data Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Creating the Machine Learning Model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# 6. Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R^2: {r2}")

# 7. Visualizing the Results
# Plotting Actual vs Predicted Energy Consumption
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual Energy Consumption')
plt.ylabel('Predicted Energy Consumption')
plt.title('Actual vs Predicted Energy Consumption')
plt.show()

# 8. Conclusion
print("The Linear Regression model has performed reasonably well.")
