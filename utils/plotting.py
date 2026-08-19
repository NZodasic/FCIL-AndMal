"""
Academic Plotting and Visualization Suite.
Generates publication-quality figures: Partition Heatmaps, Incremental Learning Curves,
Catastrophic Forgetting Matrices, and Per-Family Performance Breakdowns.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def plot_partition_heatmap(
    partition_matrix: np.ndarray,
    client_ids: List[int],
    class_names: List[str],
    task_boundaries: List[int],
    output_path: str,
    title: str = "Client-Class Sample Distribution (Non-IID Dirichlet)",
    log_scale: bool = True
) -> None:
    """
    Generate publication-grade heatmap showing sample counts per client across all classes,
    with distinct visual vertical separators demarcating task boundaries.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(14, 8), dpi=300)

    data_to_plot = np.copy(partition_matrix)
    if log_scale:
        data_to_plot = np.log10(data_to_plot + 1.0)
        cbar_label = r"$\log_{10}(\text{Sample Count} + 1)$"
    else:
        cbar_label = "Sample Count"

    cmap = sns.color_palette("mako", as_cmap=True)
    ax = sns.heatmap(
        data_to_plot,
        cmap=cmap,
        annot=False,
        cbar_kws={"label": cbar_label},
        xticklabels=class_names,
        yticklabels=[f"Client {cid:02d}" for cid in client_ids],
    )

    # Draw vertical task boundary lines
    for boundary in task_boundaries:
        ax.axvline(boundary, color="crimson", linestyle="--", linewidth=2.0, alpha=0.85)

    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Class / Family (Sequential Tasks demarcated by dashed lines)", fontsize=12, labelpad=10)
    plt.ylabel("Federated Clients", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fcil_learning_curves(
    task_results: List[Dict[str, Any]],
    output_path: str,
    exp_name: str = "FCIL"
) -> None:
    """
    Plot Macro-F1, Accuracy, and Average Forgetting across incremental tasks.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tasks = [res["task_id"] + 1 for res in task_results]
    macro_f1s = [res.get("macro_f1", 0.0) * 100 for res in task_results]
    accuracies = [res.get("accuracy", 0.0) * 100 for res in task_results]
    forgettings = [res.get("average_forgetting", 0.0) * 100 for res in task_results]
    malware_f1s = [res.get("f1_malware_avg", 0.0) * 100 for res in task_results]

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(tasks, macro_f1s, marker="o", linewidth=2.5, color="#1f77b4", label="Macro-F1 (%)")
    plt.plot(tasks, accuracies, marker="s", linewidth=2.0, color="#2ca02c", linestyle="--", label="Overall Accuracy (%)")
    plt.plot(tasks, malware_f1s, marker="^", linewidth=2.0, color="#ff7f0e", linestyle="-.", label="Malware Families F1 (%)")
    plt.plot(tasks, forgettings, marker="x", linewidth=2.0, color="#d62728", linestyle=":", label="Catastrophic Forgetting (%)")

    plt.title(f"Class-Incremental Evaluation Across 5 Tasks ({exp_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Task ID (1 to 5)", fontsize=11)
    plt.ylabel("Performance Metric (%)", fontsize=11)
    plt.xticks(tasks, [f"Task {t}" for t in tasks])
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_forgetting_matrix(
    R_matrix: np.ndarray,
    output_path: str,
    title: str = "Catastrophic Forgetting Evaluation Matrix ($R_{i,j}$)"
) -> None:
    """
    Plot heat matrix of test performance on Task j after completing Task i.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)

    mask = np.isnan(R_matrix)
    ax = sns.heatmap(
        R_matrix * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        mask=mask,
        cbar_kws={"label": "Accuracy (%)"},
        xticklabels=[f"Task {j+1}" for j in range(R_matrix.shape[1])],
        yticklabels=[f"After T{i+1}" for i in range(R_matrix.shape[0])],
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Evaluated Task", fontsize=11)
    plt.ylabel("Completed Incremental Step", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_per_family_f1_bar(
    per_class_f1: Dict[str, float],
    output_path: str,
    title: str = "Per-Family Final Macro-F1 Breakdown"
) -> None:
    """
    Bar plot of final F1 score for each of the 15 classes (Benign + 14 Malware families).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labels = list(per_class_f1.keys())
    scores = [per_class_f1[k] * 100 for k in labels]
    colors = ["#2ca02c" if l == "Benign" else "#1f77b4" for l in labels]

    plt.figure(figsize=(12, 6), dpi=300)
    bars = plt.bar(range(len(labels)), scores, color=colors, edgecolor="black", alpha=0.85)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=10)
    plt.ylabel("F1 Score (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)

    for bar, score in zip(bars, scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{score:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
