CPU: Intel(R) Core(TM) Ultra 9 285H. One inference = 352 multiply-accumulates.

| Implementation | median | p99 | p99.9 | worst seen | worst / median | median vs fast FPGA |
|---|---:|---:|---:|---:|---:|---:|
| FPGA fully unrolled (100 MHz) | 70 ns | 70 ns | 70 ns | 70 ns | 1.0x | — |
| FPGA systolic array (100 MHz) | 2.21 µs | 2.21 µs | 2.21 µs | 2.21 µs | 1.0x | 32x slower |
| C -O2 | 43 ns | 66 ns | 167 ns | 44.35 µs | 1,034.7x | 1.6x faster |
| C -O3 -march=native | 42 ns | 90 ns | 150 ns | 115.52 µs | 2,729.6x | 1.7x faster |
| numpy float32 (MKL BLAS) | 4.20 µs | 11.10 µs | 18.00 µs | 647.70 µs | 154.2x | 60x slower |
| numpy int64 | 4.80 µs | 12.70 µs | 20.60 µs | 1.25 ms | 260.3x | 69x slower |
| pure Python loops | 17.80 µs | 48.10 µs | 70.70 µs | 249.10 µs | 14.0x | 254x slower |

- FPGA fully unrolled (100 MHz): 7 cycles, measured on the board (this run). On-chip power (Vivado estimate, default activity): 0.095 W -> about 6.7 nJ per inference.
- FPGA systolic array (100 MHz): 221 cycles, measured on the board. On-chip power (Vivado estimate, default activity): 0.096 W -> about 212.2 nJ per inference.

USB-UART round trip to the board (for context, not the accelerator's latency): median 15.98 ms, worst 16.69 ms.
