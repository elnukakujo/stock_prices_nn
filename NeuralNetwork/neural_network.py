import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

stock_symbol = 'AAPL'
start_date = '2019-01-01'
end_date = '2021-01-01'
data = yf.download(stock_symbol, start=start_date, end=end_date)

close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)

def create_lstm_data(data, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(x), np.array(y)

time_steps = 10
x, y = create_lstm_data(close_prices_scaled, time_steps)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Split data into training and validation sets
train_size = int(len(x) * 0.8)  # 80% training data, 20% validation data

x_train, x_val = np.split(x, [train_size])
y_train, y_val = np.split(y, [train_size])

# Generate future dates for prediction
future_dates = pd.date_range(start=end_date, periods=60)

# Predict future prices
last_prices = close_prices[-time_steps:]
last_prices_scaled = scaler.transform(last_prices.reshape(-1, 1))

predicted_prices_scaled=list()
for training in range(20):
    print("Training:", training)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping])
    
    x_pred = np.array([last_prices_scaled[-time_steps:, 0]])
    x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))
    training_predicted_prices_scaled=list()
    for i in range(future_dates.shape[0]):
        prediction=model.predict(x_pred)
        training_predicted_prices_scaled.append(prediction[0].tolist())
        x_pred = np.append(x_pred, prediction)
        x_pred = np.array([x_pred[1:]])
        x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))
    predicted_prices_scaled.append(training_predicted_prices_scaled)

predicted_prices_scaled=np.array(predicted_prices_scaled).mean(axis=0)
predicted_prices = scaler.inverse_transform(np.array(predicted_prices_scaled)).reshape(-1)
actual_prices = yf.download(stock_symbol, start=end_date, end=future_dates[-1]).reindex(future_dates)['Close'].values

# Create DataFrame with future dates and predicted prices and more
future_data = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices, 'Actual Price': actual_prices, 'Difference': actual_prices - predicted_prices})
print(future_data)
