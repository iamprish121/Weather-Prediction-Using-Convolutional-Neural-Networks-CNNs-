import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data from CSV file
data = pd.read_csv(r"C:\Users\PRIYANSHU ANAND\Desktop\sem6\AI\weather.csv")

# Convert the date column to datetime
data['Date.Full'] = pd.to_datetime(data['Date.Full'])

# Drop non-numeric columns
data = data.select_dtypes(include=[np.number])

# Split features and target variable
X = data.drop('Data.Precipitation', axis=1)
y = data['Data.Precipitation']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

# Reshape data for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Design CNN model
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate model
y_pred = model.predict(X_test)

# Transform back the scaled predictions to original scale
y_pred = scaler_y.inverse_transform(y_pred)

# Calculate evaluation metrics
mae = mean_absolute_error(scaler_y.inverse_transform(y_test), y_pred)
mse = mean_squared_error(scaler_y.inverse_transform(y_test), y_pred)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", np.sqrt(mse))
