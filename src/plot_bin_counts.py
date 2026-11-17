import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

COMBOS = [("normal", "fixed_time"), ("normal", "rule_based"),
          ("disrupted", "fixed_time"), ("disrupted", "rule_based")]


def load(path):
    """{(condition, controller): {(feature, label): [((lo, hi), [c0, c1, c2]), ...]}}"""
    data = defaultdict(lambda: defaultdict(list))
    row_order = []  # CSV에 나온 순서대로 행 배치
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["feature"], r["label"])
            if key not in row_order:
                row_order.append(key)
            counts = [int(r[f"bin_{i}_count"]) for i in range(3)]
            edges = (int(r["low_edge"]), int(r["high_edge"]))
            data[(r["condition"], r["controller"])][key].append((edges, counts))
    return data, row_order


def norm_entropy(counts):
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(3))


def main(csv_path, out_path="validation/bin_balance_heatmap.png"):
    data, rows = load(csv_path)
    n_cand = len(data[COMBOS[0]][rows[0]])

    fig, axes = plt.subplots(1, len(COMBOS), figsize=(17, 11), sharey=True)
    for ax, combo in zip(axes, COMBOS):
        cells = [data[combo][key] for key in rows]
        ent = np.array([[norm_entropy(c) for _, c in cand] for cand in cells])
        im = ax.imshow(ent, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")

        for i, cand in enumerate(cells):
            for j, ((lo, hi), counts) in enumerate(cand):
                share = np.array(counts) / sum(counts) * 100
                color = "white" if ent[i, j] < 0.8 else "black"
                ax.text(j, i - 0.27, f"({lo}, {hi})", ha="center", va="center",
                        fontsize=8, color=color)
                ax.text(j, i, f"{ent[i, j]:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)
                ax.text(j, i + 0.27, "/".join(f"{s:.0f}" for s in share),
                        ha="center", va="center", fontsize=7.5, color=color)

        ax.set_xticks(range(n_cand))
        ax.set_xticklabels([f"cand {k + 1}" for k in range(n_cand)], fontsize=9)
        ax.set_title(f"{combo[0]} / {combo[1]}", fontsize=11)
        for i in range(1, len(rows)):
            if rows[i][0] != rows[i - 1][0]:
                ax.axhline(i - 0.5, color="white", linewidth=3)

    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([f"{f} {lab}" for f, lab in rows], fontsize=10)

    fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02,
                 label="normalized entropy (1 = even split)")
    fig.suptitle("Bin balance per threshold candidate\n"
                 "(cell: bin edges / entropy / low·mid·high share %)", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main("validation/bin_counts.csv")