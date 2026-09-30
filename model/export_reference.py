"""Build the full golden reference and $readmemh hex files from the committed weights.

Reads only model/weights/*.txt (the frozen int8 model), so it never retrains and
never changes the weights. Safe to re-run any time.

Integer math, exactly as the hardware does it:
  z1   = X @ W1 + b1 * 64            (products are scaled 64*64, so bias gets one more 64)
  a1   = relu(z1)                    (must equal z1_reference.txt)
  a1_q = min((a1 + 32) >> 6, 127)    (requant back to int8, round-half-up, saturate)
  z2   = a1_q @ W2 + b2 * 64
  pred = argmax(z2, axis=1)          (0 = down, 1 = up; tie goes to index 0)

Hex files (weights/hex/): one value per line, row-major, two's complement.
  8-bit  (2 hex digits): X_test, W1, b1, W2, b2, a1_q
  32-bit (8 hex digits): z1_reference, z2_reference
"""
from pathlib import Path

import numpy as np

SCALE = 64
SHIFT = 6  # log2(SCALE)

WEIGHTS = Path(__file__).parent / "weights"
HEX = WEIGHTS / "hex"


def load(name):
    return np.loadtxt(WEIGHTS / name, dtype=np.int64, ndmin=2)


def save_int_matrix(name, arr):
    np.savetxt(WEIGHTS / name, arr, fmt="%d")


def save_hex(name, arr, bits):
    mask = (1 << bits) - 1
    digits = bits // 4
    lines = [f"{int(v) & mask:0{digits}x}" for v in arr.flatten()]
    (HEX / name).write_text("\n".join(lines) + "\n")


X = load("X_test.txt")    # 5 x 20
W1 = load("W1.txt")       # 20 x 16
b1 = load("b1.txt")       # 1 x 16
W2 = load("W2.txt")       # 16 x 2
b2 = load("b2.txt")       # 1 x 2

for name, arr in [("X_test", X), ("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
    assert arr.min() >= -128 and arr.max() <= 127, f"{name} out of int8 range"

z1 = X @ W1 + b1 * SCALE
a1 = np.maximum(0, z1)
assert np.array_equal(a1, load("z1_reference.txt")), "layer 1 disagrees with z1_reference.txt"

a1_q = np.minimum((a1 + (1 << (SHIFT - 1))) >> SHIFT, 127)
z2 = a1_q @ W2 + b2 * SCALE
pred = np.argmax(z2, axis=1)

save_int_matrix("a1_q.txt", a1_q)
save_int_matrix("z2_reference.txt", z2)
save_int_matrix("predictions.txt", pred.reshape(1, -1))

HEX.mkdir(exist_ok=True)
for name, arr in [("X_test", X), ("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2), ("a1_q", a1_q)]:
    save_hex(f"{name}.hex", arr, 8)
save_hex("z1_reference.hex", a1, 32)
save_hex("z2_reference.hex", z2, 32)

print("layer 1 matches z1_reference.txt")
print("a1_q range:", a1_q.min(), "to", a1_q.max(), f"({(a1_q == 127).sum()} saturated)")
print("z2:\n", z2)
print("predictions:", *pred)
