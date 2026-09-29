import csv
from pathlib import Path

import matplotlib.pyplot as plt


input_path = Path(__file__).resolve().parent / "processed_data" / "train-00000-of-00005.csv"
output_path = input_path.with_name("word_count_distribution.png")

word_counts = []

with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        word_counts.append(len((row.get("text") or "").split()))

# Keep only datapoints up to 500 words
clamped_counts = [count for count in word_counts if 0 < count <= 500]

# 20 bins: 0-25, 25-50, ..., 475-500
bins = range(0, 501, 25)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5),
    constrained_layout=True
)

# Normal distribution
axes[0].hist(
    clamped_counts,
    bins=bins,
    color="#2878b5",
    edgecolor="white",
    linewidth=0.4
)

axes[0].axvline(
    500,
    color="#c43d3d",
    linestyle="--",
    linewidth=1.8,
    label="500 words"
)

axes[0].set_xlim(0, 500)
axes[0].set_title("Word-count distribution")
axes[0].set_xlabel("Word count")
axes[0].set_ylabel("Number of datapoints")
axes[0].legend()


# Log-scale distribution
axes[1].hist(
    clamped_counts,
    bins=bins,
    color="#5a9e54",
    edgecolor="white",
    linewidth=0.4
)

axes[1].set_xscale("log")
axes[1].set_xlim(1, 500)

axes[1].axvline(
    500,
    color="#c43d3d",
    linestyle="--",
    linewidth=1.8,
    label="500 words"
)

axes[1].set_title("Word-count distribution (log x-axis)")
axes[1].set_xlabel("Word count, log scale")
axes[1].set_ylabel("Number of datapoints")
axes[1].legend()


fig.suptitle(
    f"{input_path.name}: {len(clamped_counts):,} datapoints ≤ 500 words",
    fontsize=14,
)

fig.savefig(
    output_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)

print(output_path)
print(f"original_datapoints={len(word_counts)}")
print(f"datapoints_0_to_500={len(clamped_counts)}")
print(f"excluded_over_500={len(word_counts) - len(clamped_counts)}")