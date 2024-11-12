import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('household_power_consumption.csv')

# Check for missing values
print(data.isnull().sum())

# Replace missing values with the mean of each column
data.fillna(data.mean(), inplace=True)

# Select relevant features (e.g., time, temperature, and energy consumption)
features = data[['Time', 'Temperature', 'Humidity']]
target = data['Energy_Consumption']

# Normalize the features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)


# Step 1: Data Collection available datasets such as: Household Energy Consumption Dataset, Electricity Load Diagrams Dataset, Solar Energy Prediction Dataset
# Step 2: Data Preprocessing:
# Checking for missing data: We first load the data and check for any missing or incorrect data.
# Handling missing data: If there are missing values, we can either drop them or replace them with the mean or median values.
# Feature selection: We remove unnecessary features that do not add value to the prediction.
# Normalization: For better model performance, we might normalize the data using techniques such as MinMaxScaler or StandardScaler.
