import yfinance as yf
import pandas as pd
import numpy as np

df = yf.download("SPY", period = "8d", interval = "1m", auto_adjust = True, progress = False)
df.columns = df.columns.get_level_values(0)
close = df["Close"].values
print(df.shape)
print("number of bars: ", len(close))

returns = np.diff(close) / close[:-1]
WINDOW = 20
X = []
for i in range(len(returns) - WINDOW):
    X.append(returns[i : i + WINDOW])
X = np.array(X)
print("X shape:", X.shape)
y = []
for i in range(len(returns) - WINDOW):
    next_return = returns[i + WINDOW]
    y.append(1 if next_return > 0 else 0)
y = np.array(y)
print("y shape:", y.shape, "| up fraction:", y.mean().round(3))

split = int(len(X) * 0.8)
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]
print("train:", X_train.shape, "test:", X_test.shape)

np.random.seed(0)

n_inputs = X.shape[1]
n_hidden = 16
n_outputs = 2

W1 = np.random.randn(n_inputs, n_hidden) * 0.1
b1 = np.zeros(n_hidden)
W2 = np.random.randn(n_hidden, n_outputs) * 0.1
b2 = np.zeros(n_outputs)

print("W1:", W1.shape, "b1:", b1.shape, "W2:", W2.shape, "b2:", b2.shape)

def relu(z):
    return np.maximum(0, z)

def forward(x):
    z1 = x @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    return z1, a1, z2

z1, a1, scores = forward(X_test)
preds = scores.argmax(axis=1)
print("untrained test accuracy:", (preds == y_test).mean().round(3))


