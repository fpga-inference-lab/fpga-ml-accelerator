"""Phase 7 benchmark: FPGA vs CPU latency for one inference of the int8 20-16-2 MLP.

Usage:
    python host/benchmark.py            # CPU baselines + the FPGA's on-chip cycle count (221)
    python host/benchmark.py COM5       # also measure the live board over UART

Every CPU version computes the exact same integer math as the hardware and is
checked against model/weights/z2_reference.txt before it is timed.

Writes results/benchmark.json, results/benchmark.md and results/latency.png.

What is compared:
  FPGA       on-chip compute cycles x 10 ns (measured by the FPGA itself, not over USB)
  C -O2      the same loops in C, one inference timed at a time with rdtsc
  C -O3      same, with -march=native auto-vectorization
  numpy      float32 matmuls through Intel MKL (BLAS) and int64 matmuls, timed per call from Python
  Python     plain nested loops
The CPU numbers are wall-clock under Windows, so their tails include OS interruptions.
That is the point of the comparison, not noise to remove.
"""
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "model" / "weights"
BUILD = REPO / "build"
RESULTS = REPO / "results"

FPGA_CLOCK_MHZ = 100
FPGA_CYCLES = 221            # measured on the board by rtl/top.sv; overwritten if a port is given
MACS = 20 * 16 + 16 * 2      # multiply-accumulates per inference

X = np.loadtxt(WEIGHTS / "X_test.txt", dtype=np.int64, ndmin=2)
W1 = np.loadtxt(WEIGHTS / "W1.txt", dtype=np.int64, ndmin=2)
b1 = np.loadtxt(WEIGHTS / "b1.txt", dtype=np.int64, ndmin=2)[0]
W2 = np.loadtxt(WEIGHTS / "W2.txt", dtype=np.int64, ndmin=2)
b2 = np.loadtxt(WEIGHTS / "b2.txt", dtype=np.int64, ndmin=2)[0]
Z2_REF = np.loadtxt(WEIGHTS / "z2_reference.txt", dtype=np.int64, ndmin=2)


# ---------------------------------------------------------------- CPU versions

W1_l, W2_l = W1.tolist(), W2.tolist()
b1_l, b2_l = (b1 * 64).tolist(), (b2 * 64).tolist()


def infer_python(x):
    a1 = []
    for j in range(16):
        acc = b1_l[j]
        for k in range(20):
            acc += x[k] * W1_l[k][j]
        acc = max(acc, 0)
        a1.append(min((acc + 32) >> 6, 127))
    return [b2_l[o] + sum(a1[j] * W2_l[j][o] for j in range(16)) for o in range(2)]


def infer_numpy_int(x):
    z1 = np.maximum(x @ W1 + b1 * 64, 0)
    a1 = np.minimum((z1 + 32) >> 6, 127)
    return a1 @ W2 + b2 * 64


W1_f, W2_f = W1.astype(np.float32), W2.astype(np.float32)
b1_f, b2_f = (b1 * 64).astype(np.float32), (b2 * 64).astype(np.float32)


def infer_numpy_blas(x):
    # float32 goes through MKL's sgemv; every value here is a small integer, so it is exact
    z1 = np.maximum(x @ W1_f + b1_f, 0)
    a1 = np.minimum(np.floor((z1 + 32) / 64), 127)
    return a1 @ W2_f + b2_f


def check(fn, inputs):
    for m, x in enumerate(inputs):
        got = [int(v) for v in fn(x)]
        if got != Z2_REF[m].tolist():
            sys.exit(f"{fn.__name__} gave {got} for example {m}, expected {Z2_REF[m].tolist()}")


def time_per_call(fn, inputs, iters):
    for i in range(min(iters, 2000)):          # warm up
        fn(inputs[i % len(inputs)])
    ns = np.empty(iters)
    clock = time.perf_counter_ns
    for i in range(iters):
        x = inputs[i % len(inputs)]
        t0 = clock()
        fn(x)
        ns[i] = clock() - t0
    return summarize(ns)


def summarize(ns):
    p = np.percentile(ns, [0, 50, 90, 99, 99.9, 100])
    return dict(zip(["min_ns", "p50_ns", "p90_ns", "p99_ns", "p999_ns", "max_ns"], p.round(2).tolist()))


def run_c(flags, name):
    exe = BUILD / f"bench_cpu_{name}.exe"
    src = REPO / "host" / "bench_cpu.c"
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        gcc = shutil.which("gcc") or "C:/msys64/ucrt64/bin/gcc.exe"
        if not Path(gcc).exists() and not shutil.which("gcc"):
            print(f"  (skipping C {name}: gcc not found)")
            return None
        BUILD.mkdir(exist_ok=True)
        subprocess.run([gcc, *flags, str(src), "-o", str(exe)], check=True)
    out = subprocess.run([str(exe), str(WEIGHTS)], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def cpu_name():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or platform.processor()
    except Exception:
        return platform.processor()


# ---------------------------------------------------------------- FPGA

def measure_board(port, iters=200):
    sys.path.insert(0, str(REPO / "host"))
    import serial
    from run import BAUD, infer

    cycles, round_trip = set(), []
    with serial.Serial(port, BAUD, timeout=1) as ser:
        for i in range(iters):
            m = i % len(X)
            t0 = time.perf_counter_ns()
            pred, z2, c = infer(ser, X[m])
            round_trip.append(time.perf_counter_ns() - t0)
            if list(z2) != Z2_REF[m].tolist():
                sys.exit(f"board gave {z2} for example {m}, expected {Z2_REF[m].tolist()}")
            cycles.add(c)
    return sorted(cycles), summarize(np.array(round_trip))


def fpga_power_w():
    rpt = BUILD / "power.rpt"
    if not rpt.exists():
        return None
    text = rpt.read_text()
    total = re.search(r"\| Total On-Chip Power \(W\)\s*\|\s*([\d.]+)", text)
    dynamic = re.search(r"\| Dynamic \(W\)\s*\|\s*([\d.]+)", text)
    static = re.search(r"\| Device Static \(W\)\s*\|\s*([\d.]+)", text)
    if not total:
        return None
    return {"total_w": float(total.group(1)),
            "dynamic_w": float(dynamic.group(1)) if dynamic else None,
            "static_w": float(static.group(1)) if static else None}


# ---------------------------------------------------------------- main

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    port = sys.argv[1] if len(sys.argv) > 1 else None
    results = {"cpu": cpu_name(), "macs_per_inference": MACS, "rows": []}

    fpga_cycles = [FPGA_CYCLES]
    if port:
        print(f"measuring the board on {port} ...")
        fpga_cycles, rt = measure_board(port)
        results["uart_round_trip"] = rt
        print(f"  on-chip cycles seen: {fpga_cycles}   USB round trip median {rt['p50_ns'] / 1e6:.2f} ms")
    fpga_ns = [c * 1000 / FPGA_CLOCK_MHZ for c in fpga_cycles]
    results["rows"].append({
        "name": f"FPGA (Basys 3, {FPGA_CLOCK_MHZ} MHz)", "kind": "fpga", "cycles": fpga_cycles,
        **summarize(np.array(fpga_ns)), "note": "on-chip cycle count; identical every run"})
    results["fpga_power"] = fpga_power_w()

    print("C baselines ...")
    for flags, name, label in [(["-O2"], "O2", "C -O2"),
                               (["-O3", "-march=native"], "O3native", "C -O3 -march=native")]:
        r = run_c(flags, name)
        if r:
            results["rows"].append({"name": label, "kind": "cpu", "iters": r["iters"],
                                    **{k: r[k] for k in ["min_ns", "p50_ns", "p90_ns", "p99_ns", "p999_ns", "max_ns"]}})

    print("Python baselines ...")
    X_f = X.astype(np.float32)
    X_l = X.tolist()
    for label, fn, inputs, iters in [
        ("numpy float32 (MKL BLAS)", infer_numpy_blas, X_f, 200_000),
        ("numpy int64", infer_numpy_int, X, 200_000),
        ("pure Python loops", infer_python, X_l, 20_000),
    ]:
        check(fn, inputs)
        results["rows"].append({"name": label, "kind": "cpu", "iters": iters, **time_per_call(fn, inputs, iters)})

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results)
    plot(results)

    print()
    print((RESULTS / "benchmark.md").read_text(encoding="utf-8"))


def fmt_ns(ns):
    if ns >= 1e6:
        return f"{ns / 1e6:.2f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.2f} µs"
    return f"{ns:.0f} ns"


def write_markdown(results):
    lines = [
        f"CPU: {results['cpu']}. One inference = {results['macs_per_inference']} multiply-accumulates.",
        "",
        "| Implementation | median | p99 | p99.9 | worst seen | worst / median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results["rows"]:
        lines.append(f"| {r['name']} | {fmt_ns(r['p50_ns'])} | {fmt_ns(r['p99_ns'])} | "
                     f"{fmt_ns(r['p999_ns'])} | {fmt_ns(r['max_ns'])} | {r['max_ns'] / r['p50_ns']:,.1f}x |")
    p = results.get("fpga_power")
    if p:
        fpga = results["rows"][0]
        uj = p["total_w"] * fpga["p50_ns"] / 1e3
        lines += ["", f"FPGA on-chip power (Vivado estimate, default activity): {p['total_w']:.3f} W "
                      f"({p['static_w']:.3f} W static + {p['dynamic_w']:.3f} W dynamic) "
                      f"-> about {uj:.2f} µJ per inference."]
    rt = results.get("uart_round_trip")
    if rt:
        lines += ["", f"USB-UART round trip to the board (for context, not the accelerator's latency): "
                      f"median {fmt_ns(rt['p50_ns'])}, worst {fmt_ns(rt['max_ns'])}."]
    (RESULTS / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
    fpga_color, cpu_color = "#2a78d6", "#52514e"

    rows = results["rows"][::-1]            # FPGA ends up on top
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(rows) + 1.6), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    for y, r in enumerate(rows):
        color = fpga_color if r["kind"] == "fpga" else cpu_color
        ax.plot([r["p50_ns"], r["p999_ns"]], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([r["p999_ns"], r["max_ns"]], [y, y], color=color, lw=1, ls=(0, (2, 2)), zorder=1)
        varies = r["max_ns"] > r["p50_ns"] * 1.01
        if varies:
            ax.scatter(r["max_ns"], y, s=46, facecolor=surface, edgecolor=color, linewidth=1.6, zorder=3)
        ax.scatter(r["p50_ns"], y, s=70, color=color, edgecolor=surface, linewidth=2, zorder=4)
        label = fmt_ns(r["p50_ns"]) + ("" if varies else ", every run")
        ax.annotate(label, (r["p50_ns"], y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=ink2)
        if r["max_ns"] > r["p50_ns"] * 1.5:
            ax.annotate(fmt_ns(r["max_ns"]), (r["max_ns"], y), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8, color=ink2)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] for r in rows], fontsize=9, color=ink)
    ax.set_xscale("log")
    ax.set_xlabel("latency of one inference (log scale)", fontsize=9, color=ink2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v / 1e6:g} ms" if v >= 1e6 else f"{v / 1e3:g} µs" if v >= 1e3 else f"{v:g} ns"))
    ax.tick_params(axis="x", colors=muted, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.set_ylim(-0.6, len(rows) - 0.4)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="", color=cpu_color, markeredgecolor=surface, markersize=8, label="median"),
        Line2D([], [], color=cpu_color, lw=2, label="median to 99.9th percentile"),
        Line2D([], [], marker="o", ls=(0, (2, 2)), color=cpu_color, markerfacecolor=surface, markersize=7,
               label="worst of all runs"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
              fontsize=8, labelcolor=ink2)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.text(0.02, 0.965, "One inference, FPGA vs CPU: the CPU is faster on average, the FPGA never varies",
             ha="left", va="top", fontsize=11, color=ink, fontweight="bold")
    fig.text(0.02, 0.915, f"int8 20-16-2 MLP, {MACS} MACs. CPU: {results['cpu']}",
             ha="left", va="top", fontsize=8.5, color=ink2)
    fig.savefig(RESULTS / "latency.png", facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    main()
