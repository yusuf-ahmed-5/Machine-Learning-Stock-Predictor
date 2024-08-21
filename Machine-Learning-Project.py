%matplotlib inline

# Import the libraries
import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries

plt.style.use('fivethirtyeight')

# Insert Alpha Vantage API key
api_key = '---------------'
ts = TimeSeries(key=api_key, output_format='pandas')

# Fetch historical data from Alpha Vantage
df, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')
df = df.sort_index()

# Define your desired time frame
start_date = '2020-09-01'
end_date = '2024-08-06'

# Filter the DataFrame for the desired time frame
df_filtered = df.loc[start_date:end_date].copy()

# Rename the column '4. close' to 'close'
df_filtered.rename(columns={'4. close': 'close'}, inplace=True)

# Show the filtered data
print(df_filtered.head())
print(df_filtered.columns)

# Get the number of rows and columns in the data set
print(f"Data shape: {df_filtered.shape}")

# Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Stock Close Price History (Apple Inc.)')
plt.plot(df_filtered['close'], linewidth=1)  # Default color for visual clarity
plt.xlabel('Year', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# Create a new dataframe with only the close column 
data = df_filtered.filter(['close'])
# Convert the dataframe to a numpy array
dataset = data.values
print(f"Data range: Min price = {dataset.min()}, Max price = {dataset.max()}")

# Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)
print(f"Training data length: {training_data_len}")

# Separate the training data and testing data
train_data = dataset[:training_data_len]
test_data = dataset[training_data_len - 60:]           # Include look-back period for test data

# Scale the training data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_data = scaler.fit_transform(train_data)

# Create the training data set
x_train = []
y_train = []

for i in range(60, len(scaled_train_data)):
    x_train.append(scaled_train_data[i-60:i, 0])         # Append a sequence of 60 elements
    y_train.append(scaled_train_data[i, 0])              # Append the corresponding y value

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Scale the test data
scaled_test_data = scaler.transform(test_data)  # Scale using the scaler fitted on training data

# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:]

for i in range(60, len(scaled_test_data)):
    x_test.append(scaled_test_data[i-60:i, 0])

# Convert the x_test to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted closing price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Unscaling predictions

# Get the root mean squared error (RMSE) to evaluate the model
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print(f"RMSE: {rmse}")

# Prepare data for plotting
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Plot the data
plt.figure(figsize=(16,8))
plt.title('AI Model of Stock Close Price History (Apple Inc.)')
plt.xlabel('Year', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['close'], linewidth=1, label='Training Data')  # Default color for training data
plt.plot(valid['close'], linewidth=1, color='blue', label='Actual Data')  # Red for actual data
plt.plot(valid.index, valid['Predictions'], linewidth=1, color='red', label='Predictions')  # Green for predictions
plt.legend(loc='lower right')
plt.show()
