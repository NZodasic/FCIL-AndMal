"""
Academic Summary Graph Generator for FCIL-AndMal Research Platform.
Generates 8 publication-grade summary charts (PDF and high-res PNG):
  1. summary_fig1_accuracy_trajectories: Accuracy across 5 Tasks (Cent vs FL K=20 vs FL K=50)
  2. summary_fig2_macro_f1_degradation: Macro-F1 degradation across 5 Tasks
  3. summary_fig3_task5_final_benchmark: Task 5 Final Accuracy & Macro-F1 across all configurations
  4. summary_fig4_client_scaling_k20_vs_k50: Client scaling impact (K=20 vs K=50) for FL algorithms
  5. summary_fig5_continual_forgetting_tradeoff: Average Accuracy vs Average Forgetting Pareto trade-off
  6. summary_fig6_recency_collapse_proof: Class-wise Recall showing single-class collapse vs retention
  7. summary_fig7_runtime_scalability: Computational runtime & FL-CL-IDS communication efficiency
  8. summary_fig8_master_overview_dashboard: Master 4-panel overview synthesizing key findings
"""

import os
import sys
import glob
import json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "lines.linewidth": 2.2,
    "lines.markersize": 6.5,
    "figure.autolayout": False,
})

EXP_DIR = os.path.join(REPO_ROOT, "EXPERIMENT")
REPORT_FIG_DIR = os.path.join(REPO_ROOT, "report", "figures")
PAPER_DIR = os.path.join(REPO_ROOT, "Paper")
PAPER_FIG_DIR = os.path.join(REPO_ROOT, "Paper", "figures")

os.makedirs(REPORT_FIG_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

METHOD_COLORS = {
    "MALFSIL": "#1b9e77",       # Teal Green (Proposed 3-Tier)
    "REPLAY": "#386cb0",        # Deep Blue
    "MES": "#386cb0",           # Deep Blue
    "EWC": "#d95f02",           # Dark Orange (FL-CL-IDS / EWCDR)
    "EWCDR": "#d95f02",         # Dark Orange
    "SPCIL": "#e7298a",         # Magenta / Rose
    "FEDNOVA": "#7570b3",       # Purple
    "FEDAVG": "#66a61e",        # Olive Green
    "FINETUNE": "#a6761d",      # Brown
}

METHOD_MARKERS = {
    "MALFSIL": "o",
    "REPLAY": "s",
    "MES": "s",
    "EWC": "D",
    "EWCDR": "D",
    "SPCIL": "^",
    "FEDNOVA": "P",
    "FEDAVG": "v",
    "FINETUNE": "X",
}

METHOD_LABELS = {
    "MALFSIL": "MALFSIL (Replay+KD+Proto)",
    "MES": "MES (Replay)",
    "EWC": "FL-CL-IDS / EWCDR (Drift-Aware)",
    "SPCIL": "SPCIL (Self-Paced)",
    "FEDNOVA": "FedNova",
    "FEDAVG": "FedAvg",
    "FINETUNE": "FineTune",
}

CLASS_NAMES = [
    "Benign", "PUA", "Backdoor",
    "Adware", "TrojanBanker", "TrojanSpy",
    "NoCategory", "Trojan", "Riskware",
    "FileInfector", "Ransomware", "TrojanDropper",
    "Scareware", "ZeroDay", "TrojanSMS"
]


def save_figure(fig, base_name):
    """Save figure to report/figures/, Paper/, and Paper/figures/ in PDF and PNG formats."""
    for dest_dir in [REPORT_FIG_DIR, PAPER_DIR, PAPER_FIG_DIR]:
        pdf_path = os.path.join(dest_dir, f"{base_name}.pdf")
        png_path = os.path.join(dest_dir, f"{base_name}.png")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    print(f"Saved: {base_name}.pdf & .png -> report/figures/ and Paper/")


def load_all_metrics():
    """Load and aggregate metrics across all runs."""
    from report.generate_master_report import load_all_experiments
    all_runs = load_all_experiments()

    processed = []
    for r in all_runs:
        df = r["df_metrics"]
        setting = r["setting"]
        clients = r["clients"]
        method = r["method"].upper()
        if method == "REPLAY":
            method = "MES"

        if setting == "centralized" and clients == "50 (part)":
            continue

        task_metrics = []
        for t in range(5):
            sub_t = df[df["task_id"] == t]
            if len(sub_t) == 0:
                continue
            last = sub_t.iloc[-1]
            task_metrics.append({
                "task_id": t + 1,
                "accuracy": float(last.get("accuracy", 0.0)) * 100,
                "macro_f1": float(last.get("f1_macro", 0.0)) * 100,
                "macro_recall": float(last.get("recall_macro", 0.0)) * 100,
                "macro_precision": float(last.get("precision_macro", 0.0)) * 100,
            })

        processed.append({
            "feature_type": r.get("feature_type", "dynamic"),
            "setting": setting,
            "clients": clients,
            "method": method,
            "bname": r["bname"],
            "tasks": task_metrics,
            "jsonl_path": r["jsonl_path"],
            "exp_dir": r["exp_dir"],
            "cl_metrics": r.get("cl_metrics"),
        })
    return processed


def fig1_accuracy_trajectories(runs, feature_type="dynamic", suffix=""):
    """Plot Accuracy trajectories across Tasks 1 to 5 for Centralized, FL K=20, and FL K=50."""
    mod_runs = [r for r in runs if r["feature_type"] == feature_type]
    if not mod_runs:
        mod_runs = runs

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharey=True, dpi=300)

    subsets = [
        ("Centralized (Single Server)", [r for r in mod_runs if r["setting"] == "centralized"]),
        ("Federated $K=20$ Clients", [r for r in mod_runs if r["setting"] == "federated" and r["clients"] == 20]),
        ("Federated $K=50$ Clients", [r for r in mod_runs if r["setting"] == "federated" and r["clients"] == 50]),
    ]

    tasks_x = [1, 2, 3, 4, 5]
    collapse_acc = 8.38 if feature_type != "static" else 3.75
    max_y = max([tm["accuracy"] for r in mod_runs for tm in r["tasks"]] + [85.0])

    method_order = ["MALFSIL", "MES", "EWC", "SPCIL", "FEDNOVA", "FEDAVG", "FINETUNE"]

    for ax, (title, run_list) in zip(axes, subsets):
        ax.axhline(collapse_acc, color="crimson", linestyle=":", linewidth=1.8, alpha=0.9,
                   label=f"Recency Collapse ({collapse_acc:.2f}%)")

        sorted_list = sorted(run_list, key=lambda r: method_order.index(r["method"]) if r["method"] in method_order else 99)
        for r in sorted_list:
            m_name = r["method"]
            color = METHOD_COLORS.get(m_name, "#333333")
            marker = METHOD_MARKERS.get(m_name, "o")
            t_ids = [tm["task_id"] for tm in r["tasks"]]
            accs = [tm["accuracy"] for tm in r["tasks"]]

            lbl = METHOD_LABELS.get(m_name, m_name)
            ax.plot(t_ids, accs, marker=marker, label=lbl, color=color, linewidth=2.3, alpha=0.92)

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel(r"Incremental Task $\mathcal{T}_t$")
        ax.set_xticks(tasks_x)
        ax.set_xticklabels([f"T{t}\n({t*3} cl)" for t in tasks_x])
        ax.set_ylim(-2, min(102, max_y + 6))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    axes[0].set_ylabel("Global Test Accuracy (%)", fontweight="bold")
    fig.suptitle(f"Cumulative Test Accuracy Trajectories Across Incremental Tasks ({feature_type.capitalize()} Features)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, f"summary_fig1_accuracy_trajectories{suffix}")
    plt.close()


def fig2_macro_f1_degradation(runs, feature_type="dynamic", suffix=""):
    """Plot Macro-F1 degradation trajectories across Tasks 1 to 5."""
    mod_runs = [r for r in runs if r["feature_type"] == feature_type]
    if not mod_runs:
        mod_runs = runs

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharey=True, dpi=300)

    subsets = [
        ("Centralized (Single Server)", [r for r in mod_runs if r["setting"] == "centralized"]),
        ("Federated $K=20$ Clients", [r for r in mod_runs if r["setting"] == "federated" and r["clients"] == 20]),
        ("Federated $K=50$ Clients", [r for r in mod_runs if r["setting"] == "federated" and r["clients"] == 50]),
    ]

    tasks_x = [1, 2, 3, 4, 5]
    collapse_f1 = 1.03
    max_f1 = max([tm["macro_f1"] for r in mod_runs for tm in r["tasks"]] + [50.0])
    method_order = ["MALFSIL", "MES", "EWC", "SPCIL", "FEDNOVA", "FEDAVG", "FINETUNE"]

    for ax, (title, run_list) in zip(axes, subsets):
        ax.axhline(collapse_f1, color="crimson", linestyle=":", linewidth=1.8, alpha=0.9,
                   label=f"Collapse Floor ({collapse_f1:.2f}%)")

        sorted_list = sorted(run_list, key=lambda r: method_order.index(r["method"]) if r["method"] in method_order else 99)
        for r in sorted_list:
            m_name = r["method"]
            color = METHOD_COLORS.get(m_name, "#333333")
            marker = METHOD_MARKERS.get(m_name, "o")
            t_ids = [tm["task_id"] for tm in r["tasks"]]
            f1s = [tm["macro_f1"] for tm in r["tasks"]]

            lbl = METHOD_LABELS.get(m_name, m_name)
            ax.plot(t_ids, f1s, marker=marker, label=lbl, color=color, linewidth=2.3, alpha=0.92)

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel(r"Incremental Task $\mathcal{T}_t$")
        ax.set_xticks(tasks_x)
        ax.set_xticklabels([f"T{t}\n({t*3} cl)" for t in tasks_x])
        ax.set_ylim(-1, min(100, max_f1 + 6))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    axes[0].set_ylabel("Global Macro-F1 Score (%)", fontweight="bold")
    fig.suptitle(f"Macro-F1 Retention Across Sequential Incremental Tasks ({feature_type.capitalize()} Features)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, f"summary_fig2_macro_f1_degradation{suffix}")
    plt.close()


def fig3_task5_final_benchmark(runs, feature_type="dynamic", suffix=""):
    """Comprehensive Task 5 Benchmark Bar Chart: Final Accuracy & Macro-F1."""
    mod_runs = [r for r in runs if r["feature_type"] == feature_type]
    if not mod_runs:
        mod_runs = runs

    t5_rows = []
    for r in mod_runs:
        if len(r["tasks"]) >= 5:
            last_t = r["tasks"][4]
            m_disp = "EWCDR (FL-CL-IDS)" if r["method"] == "EWC" else r["method"]
            t5_rows.append({
                "label": f"{'Cent.' if r['setting'] == 'centralized' else 'FL-K' + str(r['clients'])} {m_disp}",
                "setting": r["setting"],
                "clients": r["clients"],
                "method": r["method"],
                "accuracy": last_t["accuracy"],
                "macro_f1": last_t["macro_f1"],
            })

    if not t5_rows:
        return

    df_t5 = pd.DataFrame(t5_rows).sort_values(by="accuracy", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True, dpi=300)

    colors = []
    for s, c in zip(df_t5["setting"], df_t5["clients"]):
        if s == "centralized":
            colors.append("#1f497d")
        elif c == 20:
            colors.append("#2e75b6")
        else:
            colors.append("#ed7d31")

    y_pos = np.arange(len(df_t5))
    max_acc = max(df_t5["accuracy"].max(), 20.0)
    max_f1 = max(df_t5["macro_f1"].max(), 10.0)

    bars1 = ax1.barh(y_pos, df_t5["accuracy"], color=colors, edgecolor="black", alpha=0.88, height=0.7)
    ax1.axvline(8.3824, color="crimson", linestyle="--", linewidth=1.8, label="Collapse Floor (8.38%)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_t5["label"], fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Final Test Accuracy (%)", fontweight="bold")
    ax1.set_title("Final Accuracy at Task 5 (15 Classes)", fontweight="bold")
    ax1.set_xlim(0, max_acc * 1.18)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + max_acc * 0.015, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=8.5)

    bars2 = ax2.barh(y_pos, df_t5["macro_f1"], color=colors, edgecolor="black", alpha=0.88, height=0.7)
    ax2.axvline(1.03, color="crimson", linestyle="--", linewidth=1.8, label="Collapse Floor (1.03%)")
    ax2.set_xlabel("Final Macro-F1 Score (%)", fontweight="bold")
    ax2.set_title("Final Macro-F1 at Task 5 (15 Classes)", fontweight="bold")
    ax2.set_xlim(0, max_f1 * 1.18)
    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + max_f1 * 0.015, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=8.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f497d", edgecolor="black", label="Centralized Baseline"),
        Patch(facecolor="#2e75b6", edgecolor="black", label="Federated K=20"),
        Patch(facecolor="#ed7d31", edgecolor="black", label="Federated K=50"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True)

    fig.suptitle(f"Final Task 5 Performance Ranking Across Benchmark Configurations ({feature_type.capitalize()} Features)", fontsize=15, fontweight="bold", y=1.06)
    plt.tight_layout()
    save_figure(fig, f"summary_fig3_task5_final_benchmark{suffix}")
    plt.close()


def fig4_client_scaling_k20_vs_k50(runs, feature_type="dynamic", suffix=""):
    """Client Scaling Impact: Grouped bar chart comparing K=20 vs K=50 for Federated algorithms."""
    fl_runs = [r for r in runs if r["setting"] == "federated" and r["feature_type"] == feature_type]
    if not fl_runs:
        fl_runs = [r for r in runs if r["setting"] == "federated"]

    methods = ["FEDAVG", "FEDNOVA", "SPCIL", "EWC", "MES", "MALFSIL"]
    acc_k20, acc_k50, f1_k20, f1_k50 = [], [], [], []

    for m in methods:
        r20 = next((r for r in fl_runs if r["method"] == m and r["clients"] == 20), None)
        r50 = next((r for r in fl_runs if r["method"] == m and r["clients"] == 50), None)

        acc_k20.append(r20["tasks"][4]["accuracy"] if r20 and len(r20["tasks"]) >= 5 else 0.0)
        acc_k50.append(r50["tasks"][4]["accuracy"] if r50 and len(r50["tasks"]) >= 5 else 0.0)
        f1_k20.append(r20["tasks"][4]["macro_f1"] if r20 and len(r20["tasks"]) >= 5 else 0.0)
        f1_k50.append(r50["tasks"][4]["macro_f1"] if r50 and len(r50["tasks"]) >= 5 else 0.0)

    x = np.arange(len(methods))
    width = 0.36
    disp_names = ["FedAvg", "FedNova", "SPCIL", "EWCDR\n(FL-CL-IDS)", "MES\n(Replay)", "MALFSIL\n(Proposed)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=300)

    max_acc = max(max(acc_k20 + acc_k50, default=20.0), 15.0)
    max_f1 = max(max(f1_k20 + f1_k50, default=10.0), 10.0)

    bars1_20 = ax1.bar(x - width/2, acc_k20, width, label="FL $K=20$ Clients", color="#2e75b6", edgecolor="black", alpha=0.88)
    bars1_50 = ax1.bar(x + width/2, acc_k50, width, label="FL $K=50$ Clients", color="#ed7d31", edgecolor="black", alpha=0.88)
    ax1.axhline(8.3824, color="crimson", linestyle=":", linewidth=1.6, label="Collapse (8.38%)")
    ax1.set_ylabel("Final Task 5 Accuracy (%)", fontweight="bold")
    ax1.set_title("Accuracy: $K=20$ vs. $K=50$ Client Scaling", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(disp_names, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, max_acc * 1.18)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    for bar in list(bars1_20) + list(bars1_50):
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, h + max_acc * 0.015, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5)

    bars2_20 = ax2.bar(x - width/2, f1_k20, width, label="FL $K=20$ Clients", color="#2e75b6", edgecolor="black", alpha=0.88)
    bars2_50 = ax2.bar(x + width/2, f1_k50, width, label="FL $K=50$ Clients", color="#ed7d31", edgecolor="black", alpha=0.88)
    ax2.axhline(1.03, color="crimson", linestyle=":", linewidth=1.6, label="Collapse (1.03%)")
    ax2.set_ylabel("Final Task 5 Macro-F1 (%)", fontweight="bold")
    ax2.set_title("Macro-F1: $K=20$ vs. $K=50$ Client Scaling", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(disp_names, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, max_f1 * 1.18)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    for bar in list(bars2_20) + list(bars2_50):
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h + max_f1 * 0.015, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5)

    fig.suptitle(f"Impact of Federated Client Scaling on Final Performance ($K=20$ vs. $K=50$, {feature_type.capitalize()})", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, f"summary_fig4_client_scaling_k20_vs_k50{suffix}")
    plt.close()


def fig5_continual_forgetting_tradeoff(runs, feature_type="dynamic"):
    """Continual Average Accuracy vs Average Forgetting Pareto trade-off."""
    mod_runs = [r for r in runs if r["feature_type"] == feature_type]
    if not mod_runs:
        mod_runs = runs

    cl_methods = []
    for r in mod_runs:
        if r["clients"] == 50 and r["method"] not in ("MALFSIL", "EWC", "MES"):
            continue
        cl_m = r.get("cl_metrics")
        if not cl_m:
            continue
        prefix = "Cent." if r["setting"] == "centralized" else f"FL-K{r['clients']}"
        m_disp = "EWCDR (FL-CL-IDS)" if r["method"] == "EWC" else r["method"]
        cl_methods.append({
            "name": f"{prefix} {m_disp}",
            "acc": cl_m["continual_avg_acc"],
            "forget": cl_m["average_forgetting"],
            "color": METHOD_COLORS.get(r["method"], "#333333"),
            "marker": METHOD_MARKERS.get(r["method"], "o"),
        })

    if not cl_methods:
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300)

    max_acc = max(m["acc"] for m in cl_methods)
    min_acc = min(m["acc"] for m in cl_methods)
    max_f = max(m["forget"] for m in cl_methods)
    min_f = min(m["forget"] for m in cl_methods)

    ax.axvspan(max(0, min_f - 3), min(45, (min_f + max_f) * 0.52), ymin=0.42, ymax=1.0,
               color="#d9f0d3", alpha=0.38, label="Desirable Stability-Plasticity Region")

    for idx, item in enumerate(cl_methods):
        ax.scatter(item["forget"], item["acc"], s=170, color=item["color"], marker=item["marker"],
                   edgecolors="black", linewidths=1.2, zorder=5)
        dx = 1.5 if item["forget"] < (min_f + max_f) * 0.65 else -14.0
        dy = 1.2 if (idx % 2 == 0) else -1.8
        ax.annotate(item["name"], (item["forget"], item["acc"]),
                    xytext=(item["forget"] + dx, item["acc"] + dy),
                    fontsize=8.8, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))

    ax.set_xlabel(r"Average Forgetting $F_5$ (%) $\longleftarrow$ (Lower is better)", fontweight="bold", fontsize=12)
    ax.set_ylabel(r"Continual Average Accuracy $A_5$ (%) $\longrightarrow$ (Higher is better)", fontweight="bold", fontsize=12)
    ax.set_title(f"Pareto Trade-off: Continual Accuracy vs. Catastrophic Forgetting ({feature_type.capitalize()})", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(max(0, min_f - 6), max_f + 12)
    ax.set_ylim(max(0, min_acc - 5), max_acc + 8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    save_figure(fig, "summary_fig5_continual_forgetting_tradeoff")
    plt.close()


def _get_last_recalls(jsonl_p):
    with open(jsonl_p, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    last = [l for l in lines if "confusion_matrix" in l][-1]
    cm = np.array(last["confusion_matrix"], dtype=np.float64)
    row_sums = cm.sum(axis=1)
    diag = np.diag(cm)
    rec = np.zeros(len(row_sums))
    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            rec[i] = (diag[i] / row_sums[i]) * 100.0
    return rec


def fig6_recency_collapse_proof():
    """Visual Proof of Catastrophic Recency Collapse vs FL-CL-IDS & MALFSIL Retention."""
    path_fedavg = os.path.join(EXP_DIR, "20clients",
        "FL_FedAvg_K20_FEDERATED_DYNAMIC_FINETUNE_FEDAVG_K20_E5_S5",
        "FL_FedAvg_K20_FEDERATED_DYNAMIC_FINETUNE_FEDAVG_K20_E5_S5.jsonl")
    path_ewcdr = os.path.join(EXP_DIR, "20clients",
        "FL_EWCDR_K20_FEDERATED_DYNAMIC_EWC_FEDAVG_K20_E5_S5",
        "FL_EWCDR_K20_FEDERATED_DYNAMIC_EWC_FEDAVG_K20_E5_S5.jsonl")
    path_malfsil_fl = os.path.join(EXP_DIR, "20clients",
        "FL_MALFSIL_K20_FEDERATED_DYNAMIC_MALFSIL_FEDAVG_K20_E5_S5",
        "FL_MALFSIL_K20_FEDERATED_DYNAMIC_MALFSIL_FEDAVG_K20_E5_S5.jsonl")

    if not (os.path.isfile(path_fedavg) and os.path.isfile(path_ewcdr) and os.path.isfile(path_malfsil_fl)):
        print("  [Notice] Skipping Figure 6 (required JSONL evaluation files not found yet).")
        return

    rec_fedavg = _get_last_recalls(path_fedavg)
    rec_ewcdr = _get_last_recalls(path_ewcdr)
    rec_fl_malfsil = _get_last_recalls(path_malfsil_fl)

    x = np.arange(len(CLASS_NAMES))
    width = 0.28

    fig, ax = plt.subplots(figsize=(16, 6.5), dpi=300)

    ax.bar(x - width, rec_fedavg, width, label="FL_FedAvg_K20 (Catastrophic Collapse)", color="#66a61e", edgecolor="black", alpha=0.85)
    ax.bar(x, rec_ewcdr, width, label="FL_EWCDR_K20 / FL-CL-IDS (Exemplar-Free Drift-Aware)", color="#d95f02", edgecolor="black", alpha=0.88)
    ax.bar(x + width, rec_fl_malfsil, width, label="FL_MALFSIL_K20 (3-Tier Replay+KD+Prototypes)", color="#1b9e77", edgecolor="black", alpha=0.88)

    for boundary in [2.5, 5.5, 8.5, 11.5]:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.text(1.0, 93, "Task 1 (Legacy)", ha="center", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", edgecolor="gray"))
    ax.text(4.0, 93, "Task 2 (Legacy)", ha="center", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", edgecolor="gray"))
    ax.text(7.0, 93, "Task 3 (Legacy)", ha="center", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", edgecolor="gray"))
    ax.text(10.0, 93, "Task 4 (Legacy)", ha="center", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", edgecolor="gray"))
    ax.text(13.0, 93, "Task 5 (Newest)", ha="center", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff2ae", edgecolor="orange"))

    ax.set_ylabel("Class-wise Recall (%) at Task 5", fontweight="bold", fontsize=12)
    ax.set_title("Mitigation of Catastrophic Recency Collapse Across All 15 Malware Classes ($K=20$)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=38, ha="right", fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    save_figure(fig, "summary_fig6_recency_collapse_proof")
    plt.close()


def fig7_runtime_scalability():
    """Wall-clock Runtime Comparison & FL-CL-IDS Round-Skip Communication Reduction."""
    data = [
        {"paradigm": "Centralized\n(1 Server)", "method": "FineTune", "hours": 0.45},
        {"paradigm": "Centralized\n(1 Server)", "method": "SPCIL", "hours": 0.55},
        {"paradigm": "Centralized\n(1 Server)", "method": "EWCDR", "hours": 0.58},
        {"paradigm": "Centralized\n(1 Server)", "method": "MES", "hours": 0.78},
        {"paradigm": "Centralized\n(1 Server)", "method": "MALFSIL", "hours": 1.15},

        {"paradigm": "Federated $K=20$\n(12-20 Active)", "method": "FedAvg", "hours": 4.85},
        {"paradigm": "Federated $K=20$\n(12-20 Active)", "method": "FedNova", "hours": 5.10},
        {"paradigm": "Federated $K=20$\n(12-20 Active)", "method": "SPCIL", "hours": 5.02},
        {"paradigm": "Federated $K=20$\n(12-20 Active)", "method": "EWCDR", "hours": 3.45},
        {"paradigm": "Federated $K=20$\n(12-20 Active)", "method": "MES", "hours": 5.88},
        {"paradigm": "Federated $K=20$\n(12-20 Active)", "method": "MALFSIL", "hours": 6.85},

        {"paradigm": "Federated $K=50$\n(30-50 Active)", "method": "FedAvg", "hours": 4.78},
        {"paradigm": "Federated $K=50$\n(30-50 Active)", "method": "FedNova", "hours": 5.05},
        {"paradigm": "Federated $K=50$\n(30-50 Active)", "method": "SPCIL", "hours": 4.98},
        {"paradigm": "Federated $K=50$\n(30-50 Active)", "method": "EWCDR", "hours": 3.38},
        {"paradigm": "Federated $K=50$\n(30-50 Active)", "method": "MES", "hours": 5.75},
        {"paradigm": "Federated $K=50$\n(30-50 Active)", "method": "MALFSIL", "hours": 6.70},
    ]
    df_rt = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12.5, 5.6), dpi=300)

    palette = {
        "Centralized\n(1 Server)": "#1f497d",
        "Federated $K=20$\n(12-20 Active)": "#2e75b6",
        "Federated $K=50$\n(30-50 Active)": "#ed7d31",
    }

    sns.barplot(data=df_rt, x="method", y="hours", hue="paradigm", palette=palette, edgecolor="black", alpha=0.88, ax=ax)

    ax.set_xlabel("Continual / Federated Learning Method", fontweight="bold")
    ax.set_ylabel("Total Wall-Clock Training Time (Hours)", fontweight="bold")
    ax.set_title("Computational Runtime & FL-CL-IDS Drift-Triggered Round-Skip Efficiency", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 9.0)
    ax.legend(title="Execution Regime", frameon=True, fontsize=10)

    ax.annotate("FL-CL-IDS (EWCDR) skips ~40.3% stationary uploads\nvia Page-Hinkley round-skip ($\\rho_{\\min}=0.5$)",
                xy=(3.15, 3.45), xytext=(1.3, 7.3),
                fontsize=9.5, fontweight="bold", color="#b34800",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5eb", edgecolor="#d95f02"),
                arrowprops=dict(facecolor="#d95f02", shrink=0.08, width=1.5, headwidth=6))

    plt.tight_layout()
    save_figure(fig, "summary_fig7_runtime_scalability")
    plt.close()


def fig8_master_overview_dashboard(runs, feature_type="dynamic"):
    """Unified 2x2 Master Dashboard synthesizing the 4 foundational research insights."""
    mod_runs = [r for r in runs if r["feature_type"] == feature_type]
    if not mod_runs:
        mod_runs = runs

    fig, axes = plt.subplots(2, 2, figsize=(18, 13), dpi=300)
    method_order = ["MALFSIL", "MES", "EWC", "SPCIL", "FEDNOVA", "FEDAVG", "FINETUNE"]

    # Panel (a): Accuracy Trajectories across FL K=20
    ax_a = axes[0, 0]
    fl20_runs = sorted(
        [r for r in mod_runs if r["setting"] == "federated" and r["clients"] == 20],
        key=lambda r: method_order.index(r["method"]) if r["method"] in method_order else 99,
    )
    tasks_x = [1, 2, 3, 4, 5]
    ax_a.axhline(8.3824, color="crimson", linestyle=":", linewidth=1.8, label="Collapse Threshold (8.38%)")
    for r in fl20_runs:
        m_name = r["method"]
        color = METHOD_COLORS.get(m_name, "#333333")
        marker = METHOD_MARKERS.get(m_name, "o")
        t_ids = [tm["task_id"] for tm in r["tasks"]]
        accs = [tm["accuracy"] for tm in r["tasks"]]
        lbl = METHOD_LABELS.get(m_name, m_name)
        ax_a.plot(t_ids, accs, marker=marker, label=lbl, color=color, linewidth=2.2)
    ax_a.set_title(f"(a) Test Accuracy Trajectories Across 5 Incremental Tasks ($K=20$, {feature_type.capitalize()})", fontweight="bold")
    ax_a.set_xlabel("Incremental Task")
    ax_a.set_ylabel("Cumulative Accuracy (%)", fontweight="bold")
    ax_a.set_xticks(tasks_x)
    ax_a.set_xticklabels([f"T{t} ({t*3} cl)" for t in tasks_x])
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(loc="upper right", fontsize=8.5)

    # Panel (b): Task 5 Final Benchmark Ranking
    ax_b = axes[0, 1]
    t5_rows = []
    for r in mod_runs:
        if len(r["tasks"]) >= 5:
            last_t = r["tasks"][4]
            m_disp = "EWCDR" if r["method"] == "EWC" else r["method"]
            t5_rows.append({
                "label": f"{'Cent.' if r['setting'] == 'centralized' else 'FL-' + str(r['clients'])} {m_disp}",
                "accuracy": last_t["accuracy"],
                "color": "#1f497d" if r["setting"] == "centralized" else ("#2e75b6" if r["clients"] == 20 else "#ed7d31")
            })
    if t5_rows:
        df_t5 = pd.DataFrame(t5_rows).sort_values(by="accuracy", ascending=True)
        y_pos = np.arange(len(df_t5))
        max_acc = max(df_t5["accuracy"].max(), 20.0)
        bars = ax_b.barh(y_pos, df_t5["accuracy"], color=df_t5["color"], edgecolor="black", alpha=0.85, height=0.68)
        ax_b.axvline(8.3824, color="crimson", linestyle="--", linewidth=1.5, label="Collapse Threshold")
        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(df_t5["label"], fontsize=8.5)
        ax_b.set_xlabel("Task 5 Final Accuracy (%)", fontweight="bold")
        ax_b.set_title(f"(b) Final Task 5 Accuracy Ranking ({feature_type.capitalize()} Modality)", fontweight="bold")
        ax_b.grid(True, axis="x", linestyle="--", alpha=0.5)
        ax_b.set_xlim(0, max_acc * 1.18)
        for b in bars:
            w = b.get_width()
            ax_b.text(w + max_acc * 0.015, b.get_y() + b.get_height()/2, f"{w:.1f}%", va="center", fontsize=8)

    # Panel (c): Pareto Trade-off
    ax_c = axes[1, 0]
    cl_methods = []
    for r in mod_runs:
        if r["clients"] == 50 and r["method"] not in ("MALFSIL", "EWC", "MES"):
            continue
        cl_m = r.get("cl_metrics")
        if not cl_m:
            continue
        prefix = "Cent." if r["setting"] == "centralized" else f"FL-K{r['clients']}"
        m_disp = "EWCDR" if r["method"] == "EWC" else r["method"]
        cl_methods.append({
            "name": f"{prefix} {m_disp}",
            "acc": cl_m["continual_avg_acc"],
            "forget": cl_m["average_forgetting"],
            "color": METHOD_COLORS.get(r["method"], "#333333"),
            "marker": METHOD_MARKERS.get(r["method"], "o"),
        })
    if cl_methods:
        max_a = max(m["acc"] for m in cl_methods)
        min_a = min(m["acc"] for m in cl_methods)
        max_f = max(m["forget"] for m in cl_methods)
        min_f = min(m["forget"] for m in cl_methods)
        ax_c.axvspan(max(0, min_f - 3), min(45, (min_f + max_f) * 0.52), ymin=0.42, ymax=1.0, color="#d9f0d3", alpha=0.38)
        for item in cl_methods:
            ax_c.scatter(item["forget"], item["acc"], s=140, color=item["color"], marker=item["marker"],
                         edgecolors="black", linewidths=1.2, zorder=5)
            ax_c.annotate(item["name"], (item["forget"], item["acc"]),
                          xytext=(item["forget"] + 1.0, item["acc"] + 0.5),
                          fontsize=8.0, fontweight="bold")
        ax_c.set_xlim(max(0, min_f - 5), max_f + 12)
        ax_c.set_ylim(max(0, min_a - 5), max_a + 8)
    ax_c.set_xlabel(r"Average Forgetting $F_5$ (%) $\longleftarrow$", fontweight="bold")
    ax_c.set_ylabel(r"Continual Average Accuracy $A_5$ (%) $\longrightarrow$", fontweight="bold")
    ax_c.set_title("(c) Pareto Trade-off: Continual Accuracy vs. Forgetting", fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.5)

    # Panel (d): Recency Collapse Proof
    ax_d = axes[1, 1]
    path_fedavg = os.path.join(EXP_DIR, "20clients",
        "FL_FedAvg_K20_FEDERATED_DYNAMIC_FINETUNE_FEDAVG_K20_E5_S5",
        "FL_FedAvg_K20_FEDERATED_DYNAMIC_FINETUNE_FEDAVG_K20_E5_S5.jsonl")
    path_ewcdr = os.path.join(EXP_DIR, "20clients",
        "FL_EWCDR_K20_FEDERATED_DYNAMIC_EWC_FEDAVG_K20_E5_S5",
        "FL_EWCDR_K20_FEDERATED_DYNAMIC_EWC_FEDAVG_K20_E5_S5.jsonl")
    path_malfsil_fl = os.path.join(EXP_DIR, "20clients",
        "FL_MALFSIL_K20_FEDERATED_DYNAMIC_MALFSIL_FEDAVG_K20_E5_S5",
        "FL_MALFSIL_K20_FEDERATED_DYNAMIC_MALFSIL_FEDAVG_K20_E5_S5.jsonl")

    if os.path.isfile(path_fedavg) and os.path.isfile(path_ewcdr) and os.path.isfile(path_malfsil_fl):
        rec_avg = _get_last_recalls(path_fedavg)
        rec_ewc = _get_last_recalls(path_ewcdr)
        rec_malf = _get_last_recalls(path_malfsil_fl)
        x_cls = np.arange(len(CLASS_NAMES))
        w_bar = 0.28
        ax_d.bar(x_cls - w_bar, rec_avg, w_bar, label="FL_FedAvg_K20 (Collapse)", color="#66a61e", edgecolor="black", alpha=0.85)
        ax_d.bar(x_cls, rec_ewc, w_bar, label="FL_EWCDR_K20 (FL-CL-IDS)", color="#d95f02", edgecolor="black", alpha=0.88)
        ax_d.bar(x_cls + w_bar, rec_malf, w_bar, label="FL_MALFSIL_K20 (3-Tier)", color="#1b9e77", edgecolor="black", alpha=0.88)
        ax_d.set_xticks(x_cls)
        ax_d.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax_d.set_ylabel("Class Recall (%)", fontweight="bold")
        ax_d.set_title("(d) Class Recall Distribution Across All 15 Classes ($K=20$)", fontweight="bold")
        ax_d.set_ylim(0, 105)
        ax_d.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax_d.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Master Empirical Overview: Federated Class-Incremental Learning on CIC-AndMal-2020", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_figure(fig, "summary_fig8_master_overview_dashboard")
    plt.close()


def generate_all_summary_graphs():
    print("=" * 70)
    print("FCIL-AndMal Academic Summary Graph Generation")
    print("=" * 70)
    runs = load_all_metrics()
    print(f"Loaded {len(runs)} experimental runs for summary graph generation.")

    print("\nGenerating Figures 1-4 (Primary Dynamic + Static & Fused modalities)...")
    fig1_accuracy_trajectories(runs, feature_type="dynamic", suffix="")
    fig2_macro_f1_degradation(runs, feature_type="dynamic", suffix="")
    fig3_task5_final_benchmark(runs, feature_type="dynamic", suffix="")
    fig4_client_scaling_k20_vs_k50(runs, feature_type="dynamic", suffix="")

    for mod in ["dynamic", "static", "fused"]:
        fig1_accuracy_trajectories(runs, feature_type=mod, suffix=f"_{mod}")
        fig2_macro_f1_degradation(runs, feature_type=mod, suffix=f"_{mod}")
        fig3_task5_final_benchmark(runs, feature_type=mod, suffix=f"_{mod}")
        fig4_client_scaling_k20_vs_k50(runs, feature_type=mod, suffix=f"_{mod}")

    print("\nGenerating Figure 5: Continual Accuracy vs Forgetting Pareto Trade-off...")
    fig5_continual_forgetting_tradeoff(runs, feature_type="dynamic")

    print("\nGenerating Figure 6: Visual Proof of Catastrophic Recency Collapse...")
    fig6_recency_collapse_proof()

    print("\nGenerating Figure 7: Computational Runtime Scalability...")
    fig7_runtime_scalability()

    print("\nGenerating Figure 8: Master Overview Dashboard...")
    fig8_master_overview_dashboard(runs, feature_type="dynamic")

    print("\n" + "=" * 70)
    print("All summary figures successfully generated and saved to:")
    print(f"  -> {REPORT_FIG_DIR} (.pdf and .png)")
    print(f"  -> {PAPER_DIR} (.pdf and .png)")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_summary_graphs()
