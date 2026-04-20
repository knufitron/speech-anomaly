#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stats_heatmaps.py <collected_results_wo_top20.tsv>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1], sep="\t")

    # normalize booleans
    df["actor_zscore"] = df["actor_zscore"].astype(str).str.lower().isin(["true", "1"])

    def plot_heatmap(data, title):
        fig, ax = plt.subplots()
        cax = ax.imshow(data.values)

        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_yticks(np.arange(len(data.index)))

        ax.set_xticklabels(data.columns)
        ax.set_yticklabels(data.index)

        plt.setp(ax.get_xticklabels(), rotation=45)

        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                ax.text(j, i, f"{data.values[i, j]:.2f}",
                        ha="center", va="center")

        ax.set_title(title)
        fig.colorbar(cax)
        plt.tight_layout()
        plt.show()

    # ================================
    # HEATMAP: scaler vs actor z-score
    # ================================

    for dataset in ["ravdess", "savee"]:
        subset = df[df["dataset"] == dataset]

        pivot = subset.pivot_table(
            index="scaler",
            columns="actor_zscore",
            values="mcc",
            aggfunc="mean"
        ).fillna(0)

        plot_heatmap(pivot, f"{dataset.upper()} - Scaler vs Actor Z-Score")

    # ===============================
    # HEATMAP: feature_group vs model
    # ===============================

    for dataset in ["ravdess", "savee"]:
        subset = df[df["dataset"] == dataset]

        pivot = subset.pivot_table(
            index="model",
            columns="feature_group",
            values="mcc",
            aggfunc="mean"
        ).fillna(0)

        plot_heatmap(pivot, f"{dataset.upper()} - Feature Group vs Model")
