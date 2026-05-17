"""
Generate 3 finance-meme-style PNG images for the conclusions notebook.
Run from the project root: python memes/generate_memes.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
plt.rcParams["font.family"] = "DejaVu Sans"


def _save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─── Meme 1: Drake (approving / disapproving) ─────────────────────────────────

def make_drake_meme():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7),
                              gridspec_kw={"width_ratios": [1, 3]})
    fig.patch.set_facecolor("#1a1a1a")

    captions = [
        ("X", "#c0392b", "Using XGBoost on a\ncyclical semiconductor stock\n(RMSE 0.257)"),
        ("+", "#27ae60", "Just fitting EGARCH(1,1)\nand calling it a day\n(RMSE 0.220)"),
    ]
    bg_colors = ["#2c2c2c", "#1a2a1a"]

    for row, ((symbol, sym_color, text), bg) in enumerate(zip(captions, bg_colors)):
        ax_left = axes[row][0]
        ax_left.set_facecolor(bg)
        ax_left.text(0.5, 0.5, symbol, transform=ax_left.transAxes,
                     fontsize=72, color=sym_color, fontweight="black",
                     ha="center", va="center")
        ax_left.set_xticks([]); ax_left.set_yticks([])
        for spine in ax_left.spines.values():
            spine.set_edgecolor("#444")

        ax_right = axes[row][1]
        ax_right.set_facecolor(bg)
        ax_right.text(0.5, 0.5, text, transform=ax_right.transAxes,
                      fontsize=15, color="white", fontweight="bold",
                      ha="center", va="center", multialignment="center")
        ax_right.set_xticks([]); ax_right.set_yticks([])
        for spine in ax_right.spines.values():
            spine.set_edgecolor("#444")

    fig.suptitle("Drake Hotline Bling  |  Every quant finance student, eventually",
                 color="#aaa", fontsize=11, y=0.01)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _save(fig, "drake_garch_vs_ml.png")


# ─── Meme 2: "This is Fine" dog ───────────────────────────────────────────────

def make_this_is_fine():
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#cc4400")
    ax.set_facecolor("#cc4400")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Flames
    rng = np.random.default_rng(42)
    for _ in range(50):
        x = rng.uniform(0, 10)
        y = rng.uniform(0, 2.5)
        w = rng.uniform(0.3, 1.4)
        h = rng.uniform(0.5, 1.8)
        color = rng.choice(["#ff2200", "#ffaa00", "#ffdd00", "#ff5500"])
        ellipse = mpatches.Ellipse((x, y), w, h, color=color, alpha=0.75)
        ax.add_patch(ellipse)

    # Dog silhouette (simple shapes)
    # Body
    body = mpatches.Ellipse((2.0, 1.5), 1.6, 0.9, color="#d4a055")
    ax.add_patch(body)
    # Head
    head = plt.Circle((2.8, 2.1), 0.5, color="#d4a055")
    ax.add_patch(head)
    # Ears
    ax.add_patch(mpatches.Ellipse((2.55, 2.55), 0.25, 0.4, color="#b8842e", angle=-20))
    ax.add_patch(mpatches.Ellipse((3.05, 2.55), 0.25, 0.4, color="#b8842e", angle=20))
    # Eyes
    ax.plot([2.65, 2.65], [2.15, 2.25], color="black", lw=2)
    ax.plot([2.95, 2.95], [2.15, 2.25], color="black", lw=2)
    # Muzzle
    ax.add_patch(mpatches.Ellipse((2.8, 1.95), 0.35, 0.2, color="#c49040"))
    # Coffee mug
    ax.add_patch(mpatches.FancyBboxPatch((1.55, 0.3), 0.55, 0.65,
                                          boxstyle="round,pad=0.05",
                                          facecolor="#e8e8e8", edgecolor="gray"))
    ax.text(1.82, 0.63, "QLIKE", fontsize=6, ha="center", color="#333")

    # Speech bubble
    bubble = mpatches.FancyBboxPatch((3.5, 2.8), 6.0, 2.6,
                                      boxstyle="round,pad=0.25",
                                      facecolor="white", edgecolor="black", linewidth=2.5)
    ax.add_patch(bubble)
    # Bubble tail
    tri = mpatches.Polygon([[3.5, 3.0], [3.2, 2.5], [3.9, 2.9]], closed=True,
                             facecolor="white", edgecolor="black")
    ax.add_patch(tri)

    ax.text(6.5, 4.3,
            "MU realized vol: 92.7% annualized\nEnsemble signal: EXTREME\n\n\"This is fine.\"",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="#cc0000", multialignment="center")

    ax.text(5, 0.2, "The volatility model, watching MU on earnings day",
            ha="center", va="center", fontsize=10, color="white", fontstyle="italic")

    plt.tight_layout()
    _save(fig, "this_is_fine_vol.png")


# ─── Meme 3: Distracted Boyfriend ─────────────────────────────────────────────

def make_distracted_boyfriend():
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Three "people" as coloured circles with labels
    people = [
        (1.5, 4.5, "#7f8c8d", "XGBoost\n(300 trees,\n28 features,\nRMSE 0.257)",  "Being ignored"),
        (5.5, 4.5, "#e74c3c", "HAR-RV\n(3 params,\nno tuning,\nRMSE 0.225)",     "Being admired"),
        (9.0, 3.5, "#3498db", "Me\n(the researcher\nafter 3 weeks)",               "Distracted"),
    ]
    for x, y, color, label, caption in people:
        circle = plt.Circle((x, y), 0.9, color=color, alpha=0.85, zorder=3)
        ax.add_patch(circle)
        initials = label[0]
        ax.text(x, y, initials, ha="center", va="center", fontsize=28,
                color="white", fontweight="black", zorder=4)
        ax.text(x, y - 1.35, label, ha="center", va="top", fontsize=9,
                color=color, fontweight="bold", multialignment="center")
        ax.text(x, y + 1.1, caption, ha="center", va="bottom", fontsize=8,
                color="#aaaaaa", fontstyle="italic")

    # Arrow from researcher toward HAR-RV
    ax.annotate("", xy=(6.2, 4.5), xytext=(8.3, 4.0),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=3,
                                connectionstyle="arc3,rad=0.3"))
    # Dashed line from researcher toward XGBoost (neglected)
    ax.annotate("", xy=(2.3, 4.5), xytext=(8.1, 3.8),
                arrowprops=dict(arrowstyle="-", color="#555", lw=2,
                                linestyle="dashed",
                                connectionstyle="arc3,rad=-0.2"))

    # Title box
    ax.text(5.5, 6.5,
            "Distracted Boyfriend Meme\n"
            "HAR-RV (3 params) beats XGBoost (300 trees) on MU — Corsi (2009) was right",
            ha="center", va="center", fontsize=13, color="white", fontweight="bold",
            multialignment="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a3a1a", edgecolor="#27ae60", lw=2))

    # Source note
    ax.text(5.5, 0.3,
            "RMSE: HAR-RV = 0.225  |  XGBoost = 0.257  |  EGARCH = 0.220  (MU, 2021-2026 test set)",
            ha="center", va="center", fontsize=9, color="#888")

    plt.tight_layout()
    _save(fig, "distracted_harv_xgb.png")


if __name__ == "__main__":
    make_drake_meme()
    make_this_is_fine()
    make_distracted_boyfriend()
    print("All memes generated.")
