"""Generate selection-based replacements for the fitness-subset correlation plots."""

from pathlib import Path
import pickle
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LOGIT_DIR = ROOT / "outputs/logits/proteingym"
ASSAY_DIR = ROOT / "data/proteingym_data/DMS_ProteinGym_substitutions"
METADATA_PATH = ROOT / "data/proteingym_data/DMS_substitutions.csv"
OUTPUT_DIR = ROOT / "outputs/csvs"
IMAGE_DIR = ROOT / "images/figure4"
EXAMPLE = "S22A1_HUMAN_Yee_2023_activity.csv"
AA = list("ARNDCQEGHILKMFPSTWYV")
MODELS = {
    "ESM3-hybrid": ("ESM3_sm_open_v0.both.logits.x.combined.pkl", "ESM3_sm_both"),
    "ESMC-600M": ("esmc_600m.logits.x.combined.pkl", "ESMC-600M"),
    "ProteinMPNN": ("protein_mpnn.proteinmpnn_v_48_020.logits.x.combined.use_sequence0.pkl", "ProteinMPNN"),
}
COLORS = {"worse_than_wt": "#9A9A9A", "beneficial": "#A367C6", "top_ten_percent": "#4C9DAF", "top_one_percent": "#E2BB50"}
LABELS = {"worse_than_wt": "Worse than WT", "beneficial": "Beneficial mutations", "top_ten_percent": "Top 10%", "top_one_percent": "Top 1%"}
MUTATION = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")
RNG = np.random.default_rng(20260909)
BUDGETS = [1, 5, 8, 10, 16, 25, 32, 50, 64, 100, 128, 250, 256, 500]
LOG_PLOT_BUDGETS = [8, 16, 32, 64, 128, 256]
ACTIVITY_ASSAYS = set(pd.read_csv(METADATA_PATH).query("coarse_selection_type == 'Activity'")["DMS_filename"])


def load_assay(logits, path):
    protein = "_".join(path.name.split("_")[:2])
    key = next((candidate for candidate in logits if protein in candidate), None)
    if key is None:
        return None
    positions = [int(value) for value in key.split("indices_")[-1].split("_")]
    raw = np.asarray(logits[key], dtype=np.float64)
    probability = np.exp(raw - raw.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    lookup = (pd.DataFrame(probability, index=positions, columns=AA).stack().rename("probability")
              .rename_axis(["position", "amino_acid"]).reset_index())
    assay = pd.read_csv(path)
    parsed = assay["mutant"].astype(str).str.extract(MUTATION)
    assay = assay[parsed[1].notna()].copy()
    parsed = parsed.loc[assay.index]
    assay["position"] = parsed[1].astype(int)
    assay["wild_type"] = parsed[0]
    assay["mutant_amino_acid"] = parsed[2]
    mutant = assay.merge(lookup, left_on=["position", "mutant_amino_acid"], right_on=["position", "amino_acid"])
    mutant = mutant.rename(columns={"probability": "mutant_probability"})
    wild_type = (assay[["position", "wild_type"]].drop_duplicates()
                 .merge(lookup, left_on=["position", "wild_type"], right_on=["position", "amino_acid"])
                 [["position", "probability"]].rename(columns={"probability": "wild_type_probability"}))
    data = mutant.merge(wild_type, on="position")
    data["log_odds"] = np.log(data["mutant_probability"].clip(1e-12)) - np.log(data["wild_type_probability"].clip(1e-12))
    fit = data[data["DMS_score_bin"].eq(1)]
    if len(fit) < 20 or fit["DMS_score"].nunique() < 2:
        return None
    data["top_one_percent"] = data["DMS_score"].ge(data["DMS_score"].quantile(0.99))
    data["top_ten_percent"] = data["DMS_score"].ge(data["DMS_score"].quantile(0.90))
    data["experimentally_fit"] = data["DMS_score_bin"].eq(1)
    data["fitness_class"] = "beneficial"
    data.loc[data["DMS_score"].lt(0), "fitness_class"] = "worse_than_wt"
    data.loc[data["top_ten_percent"], "fitness_class"] = "top_ten_percent"
    data.loc[data["top_one_percent"], "fitness_class"] = "top_one_percent"
    return data


def measure(data, assay):
    selected = data[data["log_odds"].gt(0)]
    if selected.empty:
        return None
    top_rate = data["top_one_percent"].mean()
    ranked = data.sort_values("log_odds", ascending=False)
    metrics = {"assay": assay, "n_assayed": len(data), "n_positive_log_odds": len(selected),
            "selection_fraction": len(selected) / len(data),
            "top_one_percent_precision": selected["top_one_percent"].mean(),
            "top_one_percent_enrichment": selected["top_one_percent"].mean() / top_rate,
            "top_one_percent_recall": selected["top_one_percent"].sum() / data["top_one_percent"].sum(),
            "top_one_percent_retained": selected["top_one_percent"].any(),
            "top_ten_percent_retained": selected["top_ten_percent"].any()}
    best_score = data["DMS_score"].max()
    for budget in BUDGETS:
        screened = ranked.head(min(budget, len(ranked)))
        metrics[f"fit_recall_at_{budget}"] = screened["experimentally_fit"].sum() / data["experimentally_fit"].sum()
        metrics[f"top_one_percent_recall_at_{budget}"] = screened["top_one_percent"].sum() / data["top_one_percent"].sum()
        metrics[f"top_ten_percent_recall_at_{budget}"] = screened["top_ten_percent"].sum() / data["top_ten_percent"].sum()
        metrics[f"best_recall_at_{budget}"] = screened["DMS_score"].eq(best_score).sum() / data["DMS_score"].eq(best_score).sum()
    for budget in [8]:
        screened_budget = min(budget, len(data))
        for target, column in [("top_one_percent", "top_one_percent"), ("top_ten_percent", "top_ten_percent"), ("best", "DMS_score")]:
            target_count = data[column].eq(best_score).sum() if target == "best" else data[column].sum()
            draws = np.arange(screened_budget)
            random_miss = np.prod((len(data) - target_count - draws) / (len(data) - draws))
            metrics[f"{target}_hit_at_{budget}"] = metrics[f"{target}_recall_at_{budget}"] > 0
            metrics[f"random_{target}_hit_at_{budget}"] = 1 - random_miss
    return metrics


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    means = np.array([RNG.choice(values, len(values), replace=True).mean() for _ in range(10000)])
    return values.mean(), *np.quantile(means, [0.025, 0.975])


def standard_error(values):
    values = np.asarray(values, dtype=float)
    return values.std(ddof=1) / np.sqrt(len(values))


def example_panel(axis, data, model):
    for category in ["top_one_percent", "top_ten_percent", "beneficial", "worse_than_wt"]:
        subset = data[data["fitness_class"].eq(category)]
        axis.scatter(subset["log_odds"], subset["DMS_score"], s=8, alpha=0.65,
                     color=COLORS[category], edgecolors="none", label=LABELS[category])
    best_variant = data.loc[data["DMS_score"].idxmax()]
    axis.scatter(best_variant["log_odds"], best_variant["DMS_score"], s=90, marker="*",
                 color="#E2BB50", edgecolors="black", linewidths=.45, zorder=3)
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Mutation log-odds", ylabel="S22A1 activity")
    axis.tick_params(labelsize=9)
    axis.legend(frameon=False, fontsize=8, loc="upper left")


def enrichment_panel(axis, metrics, model):
    experimentally_fit, top_one_percent, top_ten_percent = [], [], []
    experimentally_fit_se, top_one_percent_se, top_ten_percent_se = [], [], []
    for budget in LOG_PLOT_BUDGETS:
        random_recall = np.minimum(budget / metrics["n_assayed"], 1)
        fit_lift = metrics[f"fit_recall_at_{budget}"] / random_recall
        top_one_percent_lift = metrics[f"top_one_percent_recall_at_{budget}"] / random_recall
        top_ten_percent_lift = metrics[f"top_ten_percent_recall_at_{budget}"] / random_recall
        experimentally_fit.append(fit_lift.mean())
        top_one_percent.append(top_one_percent_lift.mean())
        top_ten_percent.append(top_ten_percent_lift.mean())
        experimentally_fit_se.append(standard_error(fit_lift))
        top_one_percent_se.append(standard_error(top_one_percent_lift))
        top_ten_percent_se.append(standard_error(top_ten_percent_lift))
    axis.axhline(1, color="#666666", linestyle=":", linewidth=1.5, label="Random ranking")
    axis.errorbar(LOG_PLOT_BUDGETS, experimentally_fit, yerr=experimentally_fit_se, marker="o", markersize=4, capsize=2, color="#A367C6", label="Experimentally fit variants")
    axis.errorbar(LOG_PLOT_BUDGETS, top_ten_percent, yerr=top_ten_percent_se, marker="o", markersize=4, capsize=2, color="#4C9DAF", label="Top-10% variants")
    axis.errorbar(LOG_PLOT_BUDGETS, top_one_percent, yerr=top_one_percent_se, marker="o", markersize=4, capsize=2, color="#E2BB50", label="Top-1% variants")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(LOG_PLOT_BUDGETS)
    axis.set_xticklabels([str(budget) for budget in LOG_PLOT_BUDGETS])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_yticks([0.5, 1, 2, 3, 4])
    axis.set_yticklabels(["0.5", "1", "2", "3", "4"])
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_ylim(bottom=0.7)
    axis.set(xlabel="Experimental screening budget", ylabel="Recall enrichment over random (fold)")
    axis.tick_params(labelsize=9)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[::-1], labels[::-1], frameon=False, fontsize=7.5, loc="upper right")


def fixed_budget_panel(axis, metrics, model, budget=8):
    targets = ["best", "top_one_percent", "top_ten_percent"]
    labels = ["Exact\nexperimental\nbest variant", "At least one\ntop-1% variant", "At least one\ntop-10% variant"]
    model_hits = [100 * metrics[f"{target}_hit_at_{budget}"].mean() for target in targets]
    random_hits = [100 * metrics[f"random_{target}_hit_at_{budget}"].mean() for target in targets]
    model_se = [100 * standard_error(metrics[f"{target}_hit_at_{budget}"]) for target in targets]
    random_se = [100 * standard_error(metrics[f"random_{target}_hit_at_{budget}"]) for target in targets]
    y_position = np.arange(len(targets))
    height = .34
    error_style = {"ecolor": "#222222", "capsize": 2, "elinewidth": .8}
    axis.barh(y_position - height / 2, random_hits, height, xerr=random_se, error_kw=error_style, color="#9A9A9A", label="Random ranking")
    axis.barh(y_position + height / 2, model_hits, height, xerr=model_se, error_kw=error_style, color="#E2BB50", label="Model ranking")
    for position, value, error in zip(y_position - height / 2, random_hits, random_se):
        axis.text(value + error + 2.0, position, f"{value:.1f}%", va="center", fontsize=8)
    for position, value, error in zip(y_position + height / 2, model_hits, model_se):
        axis.text(value + error + 2.0, position, f"{value:.1f}%", va="center", fontsize=8)
    axis.set(xlim=(0, 100), xlabel="ProteinGym assays (%)", ylabel="")
    axis.set_yticks(y_position, labels)
    axis.invert_yaxis()
    axis.tick_params(labelsize=9)
    axis.legend(frameon=False, fontsize=7.5, loc="upper right")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for model, (filename, slug) in MODELS.items():
        with open(LOGIT_DIR / filename, "rb") as handle:
            logits = pickle.load(handle)
        metrics, example = [], None
        for path in sorted(ASSAY_DIR.glob("*.csv")):
            if path.name not in ACTIVITY_ASSAYS:
                continue
            try:
                data = load_assay(logits, path)
                if data is None:
                    continue
                metric = measure(data, path.stem)
                if metric is not None:
                    metrics.append(metric)
                if path.name == EXAMPLE:
                    example = data
            except (KeyError, TypeError, ValueError):
                continue
        metrics = pd.DataFrame(metrics)
        if example is None or metrics.empty:
            raise RuntimeError(f"Missing data for {model}")
        metrics.to_csv(OUTPUT_DIR / f"figure4_selection_{slug}.csv", index=False)
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))
        example_panel(axes[0], example, model)
        enrichment_panel(axes[1], metrics, model)
        fixed_budget_panel(axes[2], metrics, model)
        figure.tight_layout()
        figure.savefig(IMAGE_DIR / f"fitness_selection_{slug}.png", dpi=300, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()