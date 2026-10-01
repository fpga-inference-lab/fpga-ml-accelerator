"""Send the test examples to the Basys 3 over USB-UART and check its answers.

Usage:
    python host/run.py              # lists serial ports if it can't pick one
    python host/run.py COM5         # use a specific port

Needs pyserial:  pip install pyserial

Protocol (see rtl/top.sv): send 20 int8 bytes, get back 14 bytes:
    header, prediction, z2[0] (int32 LE), z2[1] (int32 LE), cycles (uint32 LE)
where the header says which engine the bitstream was built with.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports

BAUD = 115200
CLOCK_NS = 10          # Basys 3 runs at 100 MHz
RESP_LEN = 14
ENGINES = {0xA5: "systolic", 0xA6: "fast"}

WEIGHTS = Path(__file__).resolve().parent.parent / "model" / "weights"


def pick_port():
    if len(sys.argv) > 1:
        return sys.argv[1]
    ports = list(list_ports.comports())
    # the Basys 3 shows up as an FTDI dual-port device; the UART is usually the higher COM number
    ftdi = [p.device for p in ports if "FTDI" in (p.manufacturer or "") or "USB Serial" in p.description]
    if len(ftdi) >= 1:
        return sorted(ftdi)[-1]
    print("Couldn't find the board. Serial ports:")
    for p in ports:
        print(f"  {p.device}: {p.description}")
    sys.exit("Run again with the port, e.g.  python host/run.py COM5")


def infer(ser, x_row):
    ser.reset_input_buffer()
    ser.write(bytes(int(v) & 0xFF for v in x_row))
    resp = ser.read(RESP_LEN)
    if len(resp) != RESP_LEN:
        raise RuntimeError(f"expected {RESP_LEN} bytes, got {len(resp)}: {resp.hex()}")
    header, pred, z_down, z_up, cycles = struct.unpack("<BBiiI", resp)
    if header not in ENGINES:
        raise RuntimeError(f"bad reply header {header:#04x}: {resp.hex()}")
    return pred, (z_down, z_up), cycles, ENGINES[header]


def main():
    X = np.loadtxt(WEIGHTS / "X_test.txt", dtype=np.int64, ndmin=2)
    z2_ref = np.loadtxt(WEIGHTS / "z2_reference.txt", dtype=np.int64, ndmin=2)

    port = pick_port()
    print(f"using {port} at {BAUD} baud")
    errors = 0
    with serial.Serial(port, BAUD, timeout=1) as ser:
        for m, x_row in enumerate(X):
            pred, z2, cycles, engine = infer(ser, x_row)
            if m == 0:
                print(f"engine: {engine}")
            exp_z2 = tuple(int(v) for v in z2_ref[m])
            exp_pred = int(np.argmax(z2_ref[m]))
            ok = z2 == exp_z2 and pred == exp_pred
            errors += not ok
            print(f"example {m}: pred {pred}  z2 = {z2[0]:6d} {z2[1]:6d}  "
                  f"{cycles} cycles = {cycles * CLOCK_NS} ns  {'OK' if ok else 'MISMATCH'}")
            if not ok:
                print(f"    expected pred {exp_pred}  z2 = {exp_z2}")

    print("BOARD PASS: matches the Python reference" if errors == 0 else f"BOARD FAIL: {errors} mismatches")


if __name__ == "__main__":
    main()
