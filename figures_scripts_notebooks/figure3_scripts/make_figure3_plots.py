import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path


INTERPHENOTYPECSV='../../outputs/csvs/figure3_interphenotype_g1.csv'
INTEREXPERIMENTERCSV='../../outputs/csvs/figure3_interexperimenter_g2.csv'
IMAGEDIR='../../images/figure3'

interphenotype = pd.read_csv(INTERPHENOTYPECSV)
interexperimenter = pd.read_csv(INTEREXPERIMENTERCSV)
image_dir = Path(IMAGEDIR)
image_dir.mkdir(parents=True, exist_ok=True)

mapping = {
    "ESM2_650M": "ESM-2 650M",
    "ESM2_3B": "ESM-2 3B",
    "ESMC_600M": "ESMC 600M",
    "ESMC_300M": "ESMC 300M",
    "ESM3_sm_both": "ESM-3 (hybrid)",
    "ESM3_sm_sequence": "ESM-3 (sequence)",
    "ESM3_sm_structure": "ESM-3 (structure)",
    "ISMC_600M": "ISMC 600M",
    "ISMC_300M": "ISMC 300M",
    "ISM_650M": "ISM 650M",
    "ISM2-650M": "ISM 650M",
    "SaProt_650M": "SaProt 650M",
    "AMPLIFY_350M": "AMPLIFY 350M",
    "AMPLIFY_120M": "AMPLIFY 120M",
    "SaProt_650M_AF2": "SaProt 650M (AF2)",
    "ProstT5": "ProstT5",
    "SolubleMPNN": "SolubleMPNN",
    "ProteinMPNN": "ProteinMPNN"
}


def interexperimenter_plot(interexperimenter):
    plt.figure(figsize=(8, 6))
    offset = 0
    gap = 0
    score_cols = [c for c in interexperimenter.columns if c != "UniProt_ID"]
    ytick_positions = []
    ytick_labels = []
    colors = plt.cm.tab10(range(len(score_cols)))
    labeled = {col: False for col in score_cols}

    # Computing the col order
    score_cols = [c for c in interexperimenter.columns if c != "UniProt_ID"]
    group_means = (
        interexperimenter.groupby("UniProt_ID")[score_cols]
        .mean()
        .mean(axis=1)
        .sort_values()
    )
    col_order = group_means.index.tolist()
    for group in col_order:
        if group in interexperimenter["UniProt_ID"].values:
            gdf = interexperimenter[interexperimenter['UniProt_ID'] == group]
        else:
            continue
        gdf = gdf.sort_index()
        n = len(gdf)
        y = np.arange(n) + offset

        for i, col in enumerate(score_cols):
            color = colors[i]
            plt.plot(gdf[col], y, linestyle="-", alpha=0.6, color=color)
            plt.scatter(gdf[col], y, label=col if not labeled[col] else None, color=color)
            labeled[col] = True

        ytick_positions.append(y.mean())
        ytick_labels.append(group)
        boundary = offset + n - 0.5
        plt.axhline(boundary, linestyle="--", alpha=0.4)
        offset += n + gap

    plt.xlabel("Spearman $\\rho$",fontsize=14)
    plt.yticks(ytick_positions, ytick_labels,fontsize=14)
    plt.xticks(fontsize=14)
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [mapping.get(l, l) for l in labels]
    plt.legend(handles, new_labels, bbox_to_anchor=(1, 0.7), loc="upper left", fontsize=12)
    plt.tight_layout()
    plt.savefig(image_dir / 'figure3_interexperimenter.png',dpi=300,bbox_inches="tight")
    plt.show()


def interphenotype_plot(interphenotype):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=interphenotype,
        x="expression_rs",
        y="selection_rs",
        palette='rainbow',
        hue="variable_rs",
        style="model",
        s=100,
        ax=ax
    )
    # --- Extract shape-only handles from the auto legend ---
    handles, labels = ax.get_legend_handles_labels()
    style_start = labels.index("model")+1
    shape_handles = handles[style_start:]
    shape_labels  = labels[style_start:]
    shape_labels = [mapping.get(l, l) for l in shape_labels]
    ax.legend(shape_handles, shape_labels, bbox_to_anchor=(1.25, 0.8), loc="upper left",
            fontsize=12, title_fontsize=14)
    # --- Colorbar for variable_rs ---
    norm = mcolors.Normalize(vmin=interphenotype["variable_rs"].min(), vmax=interphenotype["variable_rs"].max())
    sm = cm.ScalarMappable(cmap="rainbow", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Inter-variable $\\rho$", fontsize=14)
    cbar.ax.tick_params(labelsize=11)

    # --- Rest unchanged ---
    xmin = min(interphenotype["expression_rs"].min(), interphenotype["selection_rs"].min())
    xmax = max(interphenotype["expression_rs"].max(), interphenotype["selection_rs"].max())
    ax.plot([xmin, xmax], [xmin, xmax], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Model, Abundance $\\rho$", fontsize=14)
    ax.set_ylabel(f"Model, Activity $\\rho$", fontsize=14)
    ax.tick_params(labelsize=14)

    plt.savefig(image_dir / 'figure3_interphenotype.png',dpi=300,bbox_inches="tight")
    plt.show()


interexperimenter_plot(interexperimenter)
interphenotype_plot(interphenotype)