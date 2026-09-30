CPU: Intel(R) Core(TM) Ultra 9 285H. One inference = 352 multiply-accumulates.

| Implementation | median | p99 | p99.9 | worst seen | worst / median |
|---|---:|---:|---:|---:|---:|
| FPGA (Basys 3, 100 MHz) | 2.21 µs | 2.21 µs | 2.21 µs | 2.21 µs | 1.0x |
| C -O2 | 43 ns | 134 ns | 139 ns | 44.33 µs | 1,034.2x |
| C -O3 -march=native | 42 ns | 44 ns | 45 ns | 32.83 µs | 775.8x |
| numpy float32 (MKL BLAS) | 4.20 µs | 11.00 µs | 19.70 µs | 2.11 ms | 503.0x |
| numpy int64 | 4.70 µs | 12.50 µs | 28.90 µs | 712.20 µs | 151.5x |
| pure Python loops | 18.00 µs | 47.50 µs | 100.51 µs | 1.37 ms | 76.0x |

FPGA on-chip power (Vivado estimate, default activity): 0.096 W (0.072 W static + 0.024 W dynamic) -> about 0.21 µJ per inference.
