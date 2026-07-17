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

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test  = (X_test - mean) / std

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

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)

Y_train = np.zeros((len(y_train), n_outputs))
Y_train[np.arange(len(y_train)), y_train] = 1

lr = 0.1       
epochs = 200

for epoch in range(epochs):
    z1 = X_train @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    probs = softmax(z2)

    loss = -np.mean(np.sum(Y_train * np.log(probs + 1e-9), axis=1))

    dz2 = (probs - Y_train) / len(X_train)
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)       
    dW1 = X_train.T @ dz1
    db1 = dz1.sum(axis=0)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 20 == 0:
        print(f"epoch {epoch:3d} | loss {loss:.4f}")

z1, a1, scores = forward(X_test)
preds = scores.argmax(axis=1)
print("trained test accuracy:", (preds == y_test).mean().round(3))

SCALE = 64  

def quantize(arr):
    q = np.round(arr * SCALE)     
    q = np.clip(q, -128, 127)       
    return q.astype(np.int8)

print("weight ranges (float):")
print("  W1:", W1.min().round(3), "to", W1.max().round(3))
print("  W2:", W2.min().round(3), "to", W2.max().round(3))
print("  max abs value:", round(max(np.abs(W1).max(), np.abs(W2).max()), 3), "(clips above", round(127/SCALE, 3), ")")

W1_q = quantize(W1)
b1_q = quantize(b1)
W2_q = quantize(W2)
b2_q = quantize(b2)

import os
os.makedirs("weights", exist_ok=True)

def save_int_matrix(filename, arr):
    np.savetxt(filename, arr, fmt="%d")

save_int_matrix("weights/W1.txt", W1_q)
save_int_matrix("weights/b1.txt", b1_q.reshape(1, -1))
save_int_matrix("weights/W2.txt", W2_q)
save_int_matrix("weights/b2.txt", b2_q.reshape(1, -1))

X_test_q = quantize(X_test[:5])
save_int_matrix("weights/X_test.txt", X_test_q)

z1_int = X_test_q.astype(np.int32) @ W1_q.astype(np.int32)
z1_int = z1_int + (b1_q.astype(np.int32) * SCALE)   # bias scaled to match accumulation
a1_int = np.maximum(0, z1_int)
save_int_matrix("weights/z1_reference.txt", a1_int)

print("\nExported quantized weights and reference to weights/ folder")
print("Quantized W1 sample (first row):", W1_q[0])
