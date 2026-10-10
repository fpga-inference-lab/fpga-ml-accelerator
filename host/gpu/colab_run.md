# Running the GPU benchmark on Google Colab

`host/gpu/mlp.cu` runs the same int8 20-16-2 network on an NVIDIA GPU. These steps run
it on a free Colab T4.

## Before you start

Commit and push `host/gpu/`, so Colab can clone it from GitHub.

## 1. Get a GPU runtime

Open a new notebook at <https://colab.research.google.com>, then choose
**Runtime → Change runtime type → T4 GPU → Save**.

## 2. Run these cells

Paste each block into its own cell and run it.

```
!nvidia-smi
```

```
!git clone https://github.com/fpga-inference-lab/fpga-ml-accelerator.git
%cd fpga-ml-accelerator
```

Optional: regenerate the weights header from the committed model files and confirm it
matches the committed header:

```
!python3 host/gpu/gen_weights.py && git diff --exit-code host/gpu/weights.h && echo "weights.h matches model/weights"
```

Compile. `sm_75` is the T4's architecture; if Colab gives you a different GPU, delete
`-arch=sm_75`.

```
!nvcc --version
!nvcc -O3 -arch=sm_75 -o mlp host/gpu/mlp.cu
```

Run. This takes well under a minute.

```
!./mlp
```

## 3. What to expect

The program prints, in order:

1. The GPU name.
2. The 5 test examples and a correctness check. You should see `5 test examples: PASS`,
   `replicated batch ... PASS` and `random batch ... PASS`. If anything fails, it stops
   there with exit status 1 and does not benchmark.
3. A batch-1 latency table: the kernel alone, and with the copies to and from the GPU.
4. A batch-sweep table for 1, 16, 256, 4096, 65536 and 1048576 inputs.
5. `benchmark outputs checked: PASS` and `wrote results/gpu_benchmark.json`.

## 4. Bring the results back

Copy everything the `!./mlp` cell printed and paste it back into the chat. Then download
the JSON:

```
from google.colab import files
files.download('results/gpu_benchmark.json')
```

Save it as `results/gpu_benchmark.json` in your local repo.
