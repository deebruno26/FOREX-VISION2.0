import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------
# 1. Load dataset
# -----------------------

df = pd.read_csv("forex_dataset_clean2.csv")

data = df[["EURUSD"]].dropna()

print("Data shape:", data.shape)

# -----------------------
# 2. Normalize (VERY IMPORTANT)
# -----------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# -----------------------
# 3. Create sequences
# -----------------------
X = []
y = []


lookback = 30

for i in range(lookback, len(scaled)):
    X.append(scaled[i-lookback:i, 0])
    y.append(scaled[i, 0])

X = np.array(X)
y = np.array(y)

print("Length of scaled:", len(scaled))
print("Length of X:", len(X))
print("Length of y:", len(y))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

print(X_train.shape)
print(X_test.shape)

# reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

print("X shape:", X.shape)

# -----------------------
# 4. Build LSTM model
# -----------------------
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

# -----------------------
# 5. Train model
# -----------------------
#model.fit(X, y, epochs=10, batch_size=16)
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

import matplotlib.pyplot as plt

# Predict on test set
predictions = model.predict(X_test)

# Convert back to real prices
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1,1))

plt.figure(figsize=(12,6))
plt.plot(actual, label="Actual Gold Price")
plt.plot(predictions, label="Predicted Gold Price")

plt.title("LSTM Gold Price Prediction")
plt.xlabel("Time")
plt.ylabel("Gold Price")
plt.legend()

plt.show()

# -----------------------
# 6. Predict next value
# -----------------------
last_30 = scaled[-lookback:]
input_data = last_30.reshape(1, lookback, 1)

pred = model.predict(input_data)
pred_price = scaler.inverse_transform(pred)

print("\n Predicted GOLD Price:", pred_price[0][0])

# -----------------------
# 7. Direction (simple logic)
# -----------------------
current = data.iloc[-1].values[0]

if pred_price[0][0] > current:
    print(" BULLISH")
else:
    print(" BEARISH")

