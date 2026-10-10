# FPGA ML Inference Accelerator

A neural network trained in Python, quantized to 8-bit integers, and run on custom
hardware on a Xilinx Artix-7 FPGA (Digilent Basys 3). The hardware gives the same
answers as the software model, bit for bit, both in simulation and on the board. It is
benchmarked against the same network running on a CPU in C, NumPy and plain Python, and
on an NVIDIA GPU in CUDA.

There are two hardware designs for the same network:

| Engine | Idea | Latency (one inference, 100 MHz) | Logic used |
|---|---|---|---|
| **Systolic array** (`rtl/mlp.sv`) | 4×4 grid of multiply-accumulate units, reused across the network in tiles | 221 cycles = **2.21 µs** | 1,025 LUTs, 26 DSPs |
| **Fully unrolled** (`rtl/fast_mlp.sv`) | Every multiply has its own hardware, weights built in as constants, pipelined adder trees | 7 cycles = **70 ns** | 6,484 LUTs, 0 DSPs |

Both meet timing at 100 MHz on the `xc7a35tcpg236-1` and have been measured on the board.

## Results

![Latency of one inference: FPGA, CPU and GPU](results/latency.png)

The hardware for each set of numbers:

- **CPU:** an Intel Core Ultra 9 285H laptop.
- **GPU:** an NVIDIA Tesla T4 on Google Colab.
- **FPGA:** the Basys 3 board. These numbers are clock cycles counted on the chip itself,
  times the 10 ns clock period.

In the chart, the GPU rows are batch 1. Their solid line runs to the 99th percentile
rather than the 99.9th, because that is what the GPU run recorded.

| Implementation | Median | Worst seen | Fast FPGA compared |
|---|---:|---:|---|
| **FPGA, fully unrolled** | **70 ns** | **70 ns** | — |
| FPGA, systolic array | 2.21 µs | 2.21 µs | 32× faster |
| C (`-O3 -march=native`) | 42 ns | 116 µs | C is 1.7× faster on the median |
| NumPy float32 (Intel MKL) | 4.2 µs | 648 µs | 60-90x faster |
| Pure Python | 17.8 µs | 249 µs | 254× faster |
| CUDA GPU (T4), kernel only | 8.22 µs | 36.86 µs | 117× faster |
| CUDA GPU (T4), with copies | 28.54 µs | 44.83 µs | 407× faster |

CPU and FPGA percentiles are in [results/benchmark.md](results/benchmark.md). GPU details
are in the section below and in [results/gpu_benchmark.json](results/gpu_benchmark.json).
For the GPU rows, "Worst seen" is the slowest of 10,000 batch-1 runs, and "with copies"
includes the host-to-device copy of the input and the device-to-host copy of the result.

What the numbers say:

- **Compiled C on a CPU still wins the median, narrowly.** The CPU runs at about 5 GHz,
  50× the FPGA's 100 MHz, and a network this small (352 multiply-adds, under 1 KB of
  weights) fits entirely in its fastest cache. Counted in clock cycles, the two are close.
- **The FPGA never varies.** It takes exactly 7 cycles on every input. The CPU's worst case
  was hundreds to thousands of times its median, whenever the operating system
  interrupted it. For a latency-critical system, the worst case is what matters.
- **The FPGA is far ahead of Python and NumPy**, where interpreter and per-call overhead
  dominate at this size.
- **Power:** Vivado estimates about 0.095 W on-chip, or roughly 7 nJ per inference for the
  fully unrolled engine. This is an estimate from the tool, not a measurement. CPU power
  was not measured, so there is no energy comparison here.
- **The 70 ns is compute only.** The USB-UART round trip to the laptop is about 16 ms, mostly
  the Windows serial driver's buffering. That is why latency is counted on the chip.
  In a real low-latency system, data would come straight into the FPGA's network pins.

### GPU: CUDA on a Google Colab NVIDIA Tesla T4

`host/gpu/mlp.cu` runs the same integer network on the GPU. Each thread computes one whole
inference, and the weights are held in `__constant__` memory.

The run reported the GPU as a Tesla T4 (compute capability 7.5, 40 SMs), with CUDA
runtime and driver 13000 (CUDA 13.0). Times were measured with `cudaEvent`, after warmup
runs, using pinned host memory. Raw results are in
[results/gpu_benchmark.json](results/gpu_benchmark.json).

**Batch-1 latency (10,000 runs):**

| | Median | p99 | Min | Max |
|---|---:|---:|---:|---:|
| Kernel only | 8.22 µs | 12.19 µs | 7.10 µs | 36.86 µs |
| With host↔device copies | 28.54 µs | 29.63 µs | 13.41 µs | 44.83 µs |

**Batch sweep (500 runs per size).** Throughput uses the median time.

| Batch | Kernel median | Kernel p99 | End-to-end median | End-to-end p99 | Kernel throughput (inf/s) | End-to-end throughput (inf/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.01 µs | 12.51 µs | 28.64 µs | 29.09 µs | 1.427e+05 | 3.492e+04 |
| 16 | 7.01 µs | 12.10 µs | 28.67 µs | 29.34 µs | 2.283e+06 | 5.58e+05 |
| 256 | 8.19 µs | 9.63 µs | 28.61 µs | 29.31 µs | 3.125e+07 | 8.949e+06 |
| 4,096 | 9.06 µs | 10.59 µs | 30.69 µs | 39.49 µs | 4.523e+08 | 1.335e+08 |
| 65,536 | 22.88 µs | 26.78 µs | 184.61 µs | 192.77 µs | 2.864e+09 | 3.55e+08 |
| 1,048,576 | 182.27 µs | 196.61 µs | 2515.74 µs | 2552.32 µs | 5.753e+09 | 4.168e+08 |

The sweep's batch-1 row comes from its own 500 runs, which is why it differs slightly from
the 10,000-run latency table above.

At batch 1, the GPU's median kernel time of 8.22 µs for 352 multiply-adds is dominated by
the cost of launching a kernel, so the FPGA's 70 ns is 117× faster, or 407× once the copies
to and from the GPU are included. At 1,048,576 inputs the GPU kernel alone reaches
5.753 × 10⁹ inferences/s, but end to end only 4.168 × 10⁸/s, because the PCIe copies take
most of the 2,516 µs total and the kernel just 182 µs. The FPGA's fully unrolled pipeline
could in principle accept a new input every clock cycle (10⁸ inferences/s at 100 MHz), but
the board design handles one request at a time over UART, so that figure has not been
measured. The CPU, FPGA (in simulation and on the board) and GPU all produce bit-identical
outputs, and the GPU also matched the CPU reference on 1,048,576 random inputs.

## The model

- **Data:** SPY (S&P 500 ETF) 1-minute bars. The inputs are the last 20 one-minute
  returns, and the label is whether the next return is up or down.
- **Network:** an MLP with 20 inputs → 16 hidden (ReLU) → 2 outputs (down/up score).
  It was trained from scratch in NumPy (`model/train.py`).
- **Quantization:** weights and inputs are int8 with a fixed-point scale of 64. Accumulation
  is 32-bit. Between layers, activations are converted back to int8 as
  `min((relu(z) + 32) >> 6, 127)`. Biases are scaled by an extra 64 to match the scale of the products.
- **Accuracy is about 50%**, which is expected: the direction of the next minute's price is
  close to random. The goal of the project is the hardware, which is judged by matching
  the software model exactly, not by prediction accuracy.

## How it works

### Systolic array (`rtl/pe.sv`, `rtl/systolic_array.sv`, `rtl/dense.sv`, `rtl/mlp.sv`)

- **`pe.sv`:** one processing element: an 8×8-bit multiply into a 32-bit accumulator,
  passing its inputs on to its right and lower neighbours. Each one maps onto a single
  DSP48 block.
- **`systolic_array.sv`:** a 4×4 grid of these. It is output-stationary: inputs flow in from
  the left, weights flow in from the top, and each PE builds up one entry of the result.
- **`dense.sv`:** one parametrized fully connected layer. It splits the output into 4×4
  tiles, streams the inputs through the array with the diagonal skew, then adds bias and
  optionally applies ReLU and requantization. The write-back step is pipelined to meet
  100 MHz.
- **`mlp.sv`:** two `dense` layers chained together.

### Fully unrolled (`rtl/fast_mlp.sv`, `rtl/adder_tree.sv`)

- **Constant multipliers.** The trained weights are loaded at synthesis time and become
  constants in the circuit, so each of the 352 multiplies is a small shift-and-add
  circuit, and all of them run in the same cycle.
- **Adder trees.** Each neuron's products are summed by a balanced adder tree
  (`adder_tree.sv`), 5 levels deep instead of a 20-long chain.
- **Pipeline depth.** A register sits after every 3 tree levels. That value came from
  synthesizing several pipeline depths:

| Registers in adder tree | Cycles | Meets 100 MHz? |
|---|---|---|
| none | 5 | no, misses by 2.6 ns |
| every 3 levels | **7** | **yes, 0.8 ns slack** |
| every 2 levels | 8 | yes |
| every level | 14 | yes |

### Board top (`rtl/top.sv`)

`top.sv` connects either engine (parameter `FAST`) to a UART link at 115200 baud:

- **Request:** the laptop sends 20 input bytes (int8).
- **Reply:** the FPGA sends 14 bytes:
  - a header that names the engine (`0xA5` systolic, `0xA6` fast)
  - the prediction
  - both scores as int32
  - the number of clock cycles the computation took, counted on the chip
- **Partial requests:** if the laptop stops partway through, the FPGA drops the partial
  request after 100 ms.
- **LEDs:** they show the last prediction, a busy light, the number of bytes received, and
  a heartbeat that blinks while the bitstream is running.

## Verification

Each step was tested against a bit-exact integer reference before moving on.

1. `model/export_reference.py` recomputes the network from the committed weights using
   exactly the hardware's integer math. It checks itself against the layer-1 reference,
   then writes the expected outputs and `$readmemh` hex files.
2. Self-checking xsim testbenches:

| Testbench | What it checks |
|---|---|
| `sim/systolic_array_tb.sv` | 4×4 array on a hand-made matrix |
| `sim/dense_tb.sv` | each layer of the systolic engine: all 80 layer-1 and 10 layer-2 values |
| `sim/mlp_tb.sv` | whole systolic network, run twice to check it restarts cleanly |
| `sim/fast_mlp_tb.sv` | fully unrolled network on all 5 test examples |
| `sim/uart_tb.sv` | UART loopback of all 256 byte values at the real baud rate |
| `sim/top_tb.sv` | full system through the UART pins, for either engine, including the timeout |

3. On the board, `host/run.py` sends the test examples and checks every reply.
4. On the GPU, `host/gpu/mlp.cu` checks itself before benchmarking. It checks:
   - the 5 test examples
   - 1,048,576 repeated copies of them
   - 1,048,576 random int8 inputs, against a CPU copy of `bench_cpu.c`'s math

   It exits with an error on any mismatch.

## Repository layout

```
rtl/          SystemVerilog design (both engines, UART, board top)
sim/          self-checking testbenches
constraints/  Basys 3 pin assignments
scripts/      build.tcl: one-command bitstream build
model/        training, reference generation, frozen int8 weights and hex files
host/         laptop side: run.py (board test), benchmark.py, bench_cpu.c
host/gpu/     CUDA version: mlp.cu, gen_weights.py (writes weights.h), colab_run.md
results/      benchmark tables, JSON and chart
```

## Running it

Requirements: Vivado 2025.2 (or similar), Python 3 with `numpy`, `matplotlib` and
`pyserial`, and gcc for the C baseline. A Basys 3 board is needed for the on-board steps,
and an NVIDIA GPU with `nvcc` for the GPU benchmark (a free Google Colab T4 works).

**Regenerate the reference files** (optional, they are committed):

```
python model/export_reference.py
```

Do not re-run `model/train.py` unless you mean to retrain. It downloads fresh market
data, so it would produce a different model from the one the hardware is verified against.

**Simulate.** Run from the repo root in a Vivado Tcl shell. The testbenches load the hex
files by relative path. `--timescale` is needed because the RTL files do not declare one.

```
exec xvlog -sv rtl/adder_tree.sv rtl/fast_mlp.sv sim/fast_mlp_tb.sv
exec xelab fast_mlp_tb -s fast --timescale 1ns/1ps
exec xsim fast -R
```

Expect `FAST MLP PASS`. The other testbenches work the same way; `top_tb` needs every
file in `rtl/`.

**Build a bitstream.** This takes a few minutes. Output goes to `build/<engine>/top.bit`.

```
vivado -mode batch -source scripts/build.tcl                     # fully unrolled (default)
vivado -mode batch -source scripts/build.tcl -tclargs systolic   # systolic array
```

**Run on the board:**

1. Program `build/fast/top.bit` or `build/systolic/top.bit` from Vivado's Hardware Manager.
2. Close Hardware Manager, then run:

```
python host/run.py               # prints the engine, checks all 5 examples, expect BOARD PASS
python host/benchmark.py COM5    # full benchmark, using the board's live cycle counts
```

`benchmark.py` without a port runs only the CPU side and uses the last known FPGA cycle
counts. `python host/benchmark.py --replot` redraws `results/latency.png` from the saved
results without measuring anything. It includes the GPU rows when
`results/gpu_benchmark.json` is present.

**Run the GPU benchmark.** Run this on a machine with an NVIDIA GPU, from the repo root.
Step-by-step Colab instructions are in [host/gpu/colab_run.md](host/gpu/colab_run.md).

```
python3 host/gpu/gen_weights.py                 # optional: regenerate weights.h from model/weights/
nvcc -O3 -arch=sm_75 -o mlp host/gpu/mlp.cu      # sm_75 = T4; drop -arch for other GPUs
./mlp                                           # checks correctness, then writes results/gpu_benchmark.json
```

## Next steps

The project is complete as it stands. These would extend it:

- **Throughput test.** The fully unrolled engine is a pipeline and can accept a new input
  every clock cycle, up to 100 million inferences per second. Streaming inputs back to back
  on the chip and counting completions would turn that into a measured number, compared
  against a single CPU core.
- **Lower-precision weights.** Retraining with ternary (−1/0/+1) or 4-bit weights turns each
  multiply into an add, a subtract or nothing. The logic gets shallower and smaller, which
  could reduce the latency to about 3–4 cycles (30–40 ns), close to or below the C median.
- **Higher clock.** The board's 100 MHz could be multiplied to 150–200 MHz with the FPGA's
  clock manager and a deeper pipeline. This is likely a modest gain.
- **A recognized benchmark dataset.** The LHC jet-tagging dataset (16 features, 5 classes)
  is the standard benchmark for low-latency FPGA neural networks. A model of the same size
  would fit the unrolled engine and give a task with meaningful accuracy.
- **Bigger models.** At about 18 LUTs per multiply-add, the unrolled design tops out around
  1,000 multiply-adds on this chip. Larger models would need a hybrid of the two engines, or a larger FPGA.
- **Energy comparison.** Measure board and CPU power under load, so ops-per-watt can be
  compared with real numbers.

## Authors

Griffin Charlson and Jackson France.
