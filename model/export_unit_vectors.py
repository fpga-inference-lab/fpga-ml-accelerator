"""Build golden test vectors for the unit-level testbenches (mac_unit_tb, pe_tb, systolic_array_tb).

--- cycle-by-cycle vectors, shared by mac_unit_tb and pe_tb ---------------------------
One line per clock cycle. The testbench drives rst/en/a/b for that cycle, and after the
clock edge compares the outputs with the expected values on the same line.
  unit_rst, unit_en  (1 hex digit)   unit_a, unit_b, unit_aout, unit_bout  (8-bit)
  unit_acc           (32-bit)        unit_n = number of cycles
Cycle model, exactly what mac_unit.sv / pe.sv do on a clock edge:
  rst       -> acc = 0, a_out = 0, b_out = 0
  else en   -> acc += a * b, a_out = a, b_out = b
  else      -> everything holds
(mac_unit has no a_out/b_out; mac_unit_tb ignores those two files.)
The stimulus is built from segments, each followed by a check against NumPy:
  zeros, ones, mixed signs, extreme values, enable low with live inputs, reset in the
  middle of a run, 80 real dot products (X_test rows x W1 columns), and seeded random.

--- matrix-product vectors, for systolic_array_tb ---------------------------------------
  arr_A, arr_B (8-bit, 4x4 row-major per case)   arr_C (32-bit, 4x4)   arr_n = cases
C = A @ B in int64, checked to fit in int32.
Cases: zeros, identity on each side, mixed signs, extremes (-128 and 127 only),
B all ones (the original hand test), two tiles of the real X_test @ W1, seeded random.
"""
from pathlib import Path

import numpy as np

N = 4
SEED = 1

WEIGHTS = Path(__file__).parent / "weights"
HEX = WEIGHTS / "hex"


def load(name):
    return np.loadtxt(WEIGHTS / name, dtype=np.int64, ndmin=2)


def save_hex(name, values, bits):
    mask = (1 << bits) - 1
    digits = max(1, bits // 4)
    lines = [f"{int(v) & mask:0{digits}x}" for v in values]
    (HEX / name).write_text("\n".join(lines) + "\n")


def fits(v, bits):
    return -(1 << (bits - 1)) <= v < (1 << (bits - 1))


rng = np.random.default_rng(SEED)
X = load("X_test.txt")   # 5 x 20
W1 = load("W1.txt")      # 20 x 16
assert X.min() >= -128 and X.max() <= 127 and W1.min() >= -128 and W1.max() <= 127

# ---------------------------------------------------------------- MAC / PE vectors
cycles = []   # (rst, en, a, b) per cycle


def cyc(rst, en, a, b):
    assert fits(a, 8) and fits(b, 8)
    cycles.append((rst, en, a, b))


expected_totals = []   # (cycle index of the last cycle of the segment, NumPy dot product)


def run(a_list, b_list):
    """One reset cycle, then len(a_list) enabled cycles; the final acc must be dot(a, b)."""
    cyc(1, 0, 0, 0)
    for a, b in zip(a_list, b_list):
        cyc(0, 1, int(a), int(b))
    expected_totals.append((len(cycles) - 1, int(np.dot(np.array(a_list, dtype=np.int64),
                                                         np.array(b_list, dtype=np.int64)))))


run([0] * 8, [0] * 8)                                        # zeros
run([1] * 8, list(range(-4, 4)))                             # a = 1: acc is a plain sum
run([3, -2, 5, -7, 1, 0, 9, -4], [-6, 8, -3, 2, 7, 4, -1, 5])  # mixed signs
run([-128] * 20, [-128] * 20)                                # most-positive products
run([127] * 20, [-128] * 20)                                 # most-negative products
run([-128, 127] * 10, [127, -128] * 10)                      # extremes, alternating
run([-128] * 20, [127] * 20)

# enable low with live inputs: acc and the forwarding registers must hold
cyc(1, 0, 0, 0)
for a, b in [(5, 5), (-3, 4)]:
    cyc(0, 1, a, b)
for a, b in [(100, 100), (-128, -128), (127, 127), (1, -1)]:
    cyc(0, 0, a, b)
for a, b in [(2, 2), (-7, 3)]:
    cyc(0, 1, a, b)
expected_totals.append((len(cycles) - 1, 5 * 5 + -3 * 4 + 2 * 2 + -7 * 3))

# reset in the middle of a run: everything clears, then a new run starts from zero
cyc(1, 0, 0, 0)
for a, b in [(9, 9), (8, 8), (7, 7)]:
    cyc(0, 1, a, b)
cyc(1, 1, 50, 50)                  # rst wins over en
for a, b in [(1, 2), (3, 4)]:
    cyc(0, 1, a, b)
expected_totals.append((len(cycles) - 1, 1 * 2 + 3 * 4))

# real data: every X_test row against every W1 column (this is X @ W1, layer 1 before bias)
for r in range(X.shape[0]):
    for c in range(W1.shape[1]):
        run(X[r], W1[:, c])

# seeded random, full int8 range, length 20 like a layer-1 dot product
for _ in range(30):
    run(rng.integers(-128, 128, 20), rng.integers(-128, 128, 20))

# random cycles with rst/en toggling, to check hold and forwarding behavior
for _ in range(300):
    cyc(int(rng.random() < 0.05), int(rng.random() < 0.7),
        int(rng.integers(-128, 128)), int(rng.integers(-128, 128)))

# cycle model, as the RTL does it
acc, a_out, b_out = 0, 0, 0
exp_acc, exp_aout, exp_bout = [], [], []
for rst, en, a, b in cycles:
    if rst:
        acc, a_out, b_out = 0, 0, 0
    elif en:
        acc += a * b
        a_out, b_out = a, b
    assert fits(acc, 32)
    exp_acc.append(acc)
    exp_aout.append(a_out)
    exp_bout.append(b_out)

# the cycle model must agree with NumPy at the end of every run
for idx, total in expected_totals:
    assert exp_acc[idx] == total, f"cycle model disagrees with NumPy at cycle {idx}"

# the real-data dot products must equal X @ W1, and layer 1 must still match the frozen reference
z1_ref = load("z1_reference.txt")
b1 = load("b1.txt")[0]
pre = X @ W1
pos = z1_ref > 0
assert np.array_equal((pre + b1 * 64)[pos], z1_ref[pos]), "X @ W1 disagrees with z1_reference.txt"

HEX.mkdir(exist_ok=True)
save_hex("unit_n.hex", [len(cycles)], 32)
save_hex("unit_rst.hex", [c[0] for c in cycles], 4)
save_hex("unit_en.hex", [c[1] for c in cycles], 4)
save_hex("unit_a.hex", [c[2] for c in cycles], 8)
save_hex("unit_b.hex", [c[3] for c in cycles], 8)
save_hex("unit_acc.hex", exp_acc, 32)
save_hex("unit_aout.hex", exp_aout, 8)
save_hex("unit_bout.hex", exp_bout, 8)

# ---------------------------------------------------------------- array vectors
cases = []   # (A, B)


def mat(lo, hi):
    return rng.integers(lo, hi + 1, (N, N))


eye = np.eye(N, dtype=np.int64)
cases.append((np.zeros((N, N), dtype=np.int64), mat(-128, 127)))          # zeros
cases.append((np.zeros((N, N), dtype=np.int64), np.zeros((N, N), dtype=np.int64)))
cases.append((eye, mat(-128, 127)))                                      # identity on the left: C = B
cases.append((mat(-128, 127), eye))                                      # identity on the right: C = A
cases.append((mat(-9, 9), mat(-9, 9)))                                   # mixed signs, small
cases.append((np.full((N, N), -128), np.full((N, N), -128)))             # most-positive products
cases.append((np.full((N, N), 127), np.full((N, N), -128)))              # most-negative products
cases.append((rng.choice([-128, 127], (N, N)), rng.choice([-128, 127], (N, N))))   # extremes only
cases.append((np.array([[2, 3, 1, 0], [1, 4, 2, 3], [0, 2, 5, 1], [3, 1, 0, 4]]),
              np.ones((N, N), dtype=np.int64)))                          # the original hand test
cases.append((X[0:4, 0:4], W1[0:4, 0:4]))                                # real data, first tile
cases.append((X[0:4, 16:20], W1[16:20, 12:16]))                          # real data, last tile
for _ in range(12):
    cases.append((mat(-128, 127), mat(-128, 127)))

A_all, B_all, C_all = [], [], []
for A, B in cases:
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    assert A.shape == (N, N) and B.shape == (N, N)
    assert A.min() >= -128 and A.max() <= 127 and B.min() >= -128 and B.max() <= 127
    C = A @ B
    assert all(fits(int(v), 32) for v in C.flatten())
    A_all.append(A.flatten())
    B_all.append(B.flatten())
    C_all.append(C.flatten())

save_hex("arr_n.hex", [len(cases)], 32)
save_hex("arr_A.hex", np.concatenate(A_all), 8)
save_hex("arr_B.hex", np.concatenate(B_all), 8)
save_hex("arr_C.hex", np.concatenate(C_all), 32)

print(f"unit vectors: {len(cycles)} cycles, {len(expected_totals)} dot products checked against NumPy")
print(f"array vectors: {len(cases)} cases of {N}x{N}")
print("extreme accumulator values seen:", min(exp_acc), "to", max(exp_acc))
