import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

THREAD_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]
METRIC = "Dense x Dense / FlatMatrix"
M_PER_THREAD = 16  # weak scaling: M = threads * M_PER_THREAD

CONFIGS = [
    ("rows", "", "rows (default)"),
    ("tiled", "", "tiled (default)"),
    ("tiled", "_close", "tiled (close)"),
    ("tiled", "_spread", "tiled (spread)"),
]

COLORS = [
    "#D73027",  # rows default  - red
    "#2166AC",  # tiled default - blue
    "#1A9850",  # tiled close   - green
    "#8856A7",  # tiled spread  - purple
]

marker_kw = dict(lw=1.8, ms=6, markerfacecolor="white", markeredgewidth=1.8)


def parse_time(filepath):
    if not os.path.exists(filepath):
        return None
    pattern = re.compile(
        r"\[" + re.escape(METRIC) + r"\s*\]\s+Average Time:\s+([\d.]+) ms"
    )
    with open(filepath) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    return None


def plain_formatter():
    fmt = ticker.ScalarFormatter()
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    return fmt


def load_strong(folder):
    baseline = parse_time(os.path.join(folder, "baseline_no_omp.txt"))
    data = {}
    for variant, suffix, _ in CONFIGS:
        data[(variant, suffix)] = [
            parse_time(os.path.join(folder, f"{variant}_t{t}{suffix}.txt"))
            for t in THREAD_COUNTS
        ]
    return baseline, data


def weak_filename(folder, variant, suffix, t):
    m = t * M_PER_THREAD
    for fname in [
        f"{variant}{suffix}_t{t}_m{m}.txt",  # e.g. tiled_close_t1_m16.txt
        f"{variant}_t{t}_m{m}{suffix}.txt",  # e.g. tiled_t1_m16_close.txt
    ]:
        if os.path.exists(os.path.join(folder, fname)):
            return os.path.join(folder, fname)
    return os.path.join(folder, f"{variant}{suffix}_t{t}_m{m}.txt")


def load_weak(folder):
    baseline = parse_time(os.path.join(folder, "baseline_no_omp_m16.txt"))
    data = {}
    for variant, suffix, _ in CONFIGS:
        data[(variant, suffix)] = [
            parse_time(weak_filename(folder, variant, suffix, t)) for t in THREAD_COUNTS
        ]
    return baseline, data


def plot_strong(ax, folder, title):
    if not os.path.isdir(folder):
        ax.text(
            0.5,
            0.5,
            "no data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="gray",
        )
        ax.set_title(title)
        return

    baseline, data = load_strong(folder)
    if baseline is None:
        ax.text(
            0.5,
            0.5,
            "no baseline",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="gray",
        )
        ax.set_title(title)
        return

    threads = np.array(THREAD_COUNTS, dtype=float)
    ax.plot(threads, threads, ":", color="gray", lw=1.2, alpha=0.7, label="ideal")

    for (variant, suffix, label), color in zip(CONFIGS, COLORS):
        times = data[(variant, suffix)]
        if all(t is None for t in times):
            continue
        su = [baseline / t if t else None for t in times]
        ax.plot(threads, su, "o-", color=color, label=label, **marker_kw)

    ax.set_title(title)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(THREAD_COUNTS)
    ax.xaxis.set_major_formatter(plain_formatter())
    ax.yaxis.set_major_formatter(plain_formatter())


def plot_weak(ax, folder, title):
    if not os.path.isdir(folder):
        ax.text(
            0.5,
            0.5,
            "no data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="gray",
        )
        ax.set_title(title)
        return

    baseline, data = load_weak(folder)
    if baseline is None:
        ax.text(
            0.5,
            0.5,
            "no baseline",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="gray",
        )
        ax.set_title(title)
        return

    threads = np.array(THREAD_COUNTS, dtype=float)
    ax.plot(threads, threads, ":", color="gray", lw=1.2, alpha=0.7, label="ideal")

    for (variant, suffix, label), color in zip(CONFIGS, COLORS):
        times = data[(variant, suffix)]
        if all(t is None for t in times):

            continue
        scaled_su = [
            n * baseline / t if t else None for n, t in zip(THREAD_COUNTS, times)
        ]
        ax.plot(threads, scaled_su, "o-", color=color, label=label, **marker_kw)

    ax.set_title(title)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Scaled speedup")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(THREAD_COUNTS)
    ax.xaxis.set_major_formatter(plain_formatter())
    ax.yaxis.set_major_formatter(plain_formatter())


script_dir = os.path.dirname(os.path.abspath(__file__))

fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 2, figure=fig)
fig.subplots_adjust(left=0.07, bottom=0.10, right=0.97, top=0.94, wspace=0.30, hspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

plot_strong(
    ax1,
    os.path.join(script_dir, "school_cluster_strong_scaling"),
    "Strong scaling - School Cluster",
)
plot_strong(
    ax2,
    os.path.join(script_dir, "dardel_shared_strong_scaling"),
    "Strong scaling - Dardel",
)
plot_weak(
    ax3,
    os.path.join(script_dir, "school_cluster_weak_scaling"),
    "Weak scaling - School Cluster",
)
plot_weak(
    ax4, os.path.join(script_dir, "dardel_shared_weak_scaling"), "Weak scaling - Dardel"
)

# fig.suptitle(
#     f"Scaling - Matrix Multiply ({METRIC}, K=4096, N=4096)",
#     fontsize=12,
#     y=0.98,
# )

handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=len(labels),
    fontsize=9,
    framealpha=0.85,
    bbox_to_anchor=(0.5, 0.01),
)

out = os.path.join(script_dir, "strong_scaling.png")
plt.savefig(out)
print(f"Saved: {out}")
plt.show()
