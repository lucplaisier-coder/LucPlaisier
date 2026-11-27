"""
present_figures.py: presentation-only redraw of Figure 3 (27 Nov 2026).

    python -m src.present_figures

Plain-language summary
----------------------
The original Figure 3 (outputs/step12/output5_permutation_importance_2014.png,
copied to outputs/final/) cuts off the end of its x-axis label. Redrawing it with
step 12 is NOT allowed: step 12 refits the models.

This script reads ONLY the saved figure data, outputs/final/figure3_data.csv
(the step-12 permutation-importance table as committed), and draws the same bars:
same values, same order (smallest at the bottom), same error bars (one standard
deviation over the 10 repeats), same colours and title. Only the canvas is wider
and the margins are fitted so no text is cut off. It fits, scores and shuffles
nothing.

The original analytical figure is kept unchanged. The redrawn figure goes to
outputs/presentation/figure3_permutation_importance_2014.png and is the one
the manuscript uses.
"""

import pandas as pd

from . import plotstyle  # noqa: F401  (resets matplotlib to its built-in defaults)
import matplotlib.pyplot as plt

from . import config

SRC = config.REPO_ROOT / "outputs" / "final" / "figure3_data.csv"
OUT = config.REPO_ROOT / "outputs" / "presentation"


def figure3():
    imp = pd.read_csv(SRC, index_col="input")
    order = imp.sort_values("mean_auc_drop")          # same ordering rule as step 12
    base_auc = float(imp["base_auc_2014"].iloc[0])
    colors = ["#8c564b" if i == "PLATFORM PRICING BLOCK" else "#ff7f0e" for i in order.index]
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=150)
    ax.barh(order["label"], order["mean_auc_drop"], xerr=order["sd_over_repeats"], color=colors, capsize=2)
    ax.set_xlabel("Drop in 2014 AUC when shuffled (mean of 10 repeats)")
    ax.set_title(f"HGB permutation importance, 2014 test\n(base AUC {base_auc:.3f})", fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure3_permutation_importance_2014.png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


if __name__ == "__main__":
    figure3()
    print(f"Figure 3 redrawn from {SRC.relative_to(config.REPO_ROOT)} (presentation only).")
