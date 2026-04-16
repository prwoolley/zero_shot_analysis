import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


FPDBSCORESCSV='../../outputs/csvs/fig1_fpdb_comparison_plot.csv'
PROTEINGYMSCORESCSV='../../outputs/csvs/fig1_pg_comparison_plot.csv'
IMAGEDIR='../../images/figure1'

func_metrics = pd.read_csv(PROTEINGYMSCORESCSV)
fpdb_metrics = pd.read_csv(FPDBSCORESCSV)
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
color_dict = {
    "ESM-2 650M": "black",
    "ESM-2 3B": "black",
    "ESMC 600M": "black",
    "ESMC 300M": "black",
    "ESM-3 (hybrid)": "tab:blue",
    "ESM-3 (sequence)": "black",
    "ESM-3 (structure)": "tab:red",
    "ISMC 600M": "tab:blue",
    "ISMC 200M": "tab:blue",
    "ISM 650M": "tab:blue",
    "ISM2 650M": "tab:blue",
    "SaProt 650M": "tab:blue",
    "AMPLIFY 350M": "black",
    "AMPLIFY 120M": "black",
    "SaProt 650M (AF2)": "tab:blue",
    "ProstT5": "tab:red",
    "SolubleMPNN": "tab:red",
    "ProteinMPNN": "tab:red"
}

func_metrics['model'] = func_metrics['model'].replace(mapping)
fpdb_metrics['model'] = fpdb_metrics['model'].replace(mapping)


def tidy_df(df):
    pivot_df = (df.groupby(['protein', 'model'])['spearman_rho'].max().unstack())
    rank_df = pivot_df.rank(axis=1, ascending=False) # Rank models per protein (higher AUROC = better rank = 1)
    avg_rank = rank_df.mean().sort_values() # Compute average rank per model
    model_order = avg_rank.index.tolist() # THIS ORDERS BY MEAN RANK
    metric = 'spearman_rho' # Possible metrics: 'mean_rank', 'ROC-AUC', 'MCC', 'F1_max', 'F1_mean', 'Accuracy', 'Precision', 'Recall', 'Average_Precision'
    model_order = df.groupby('model')[metric].mean().sort_values(ascending=False).index.tolist() # THIS ORDERS BY MEAN AUROC
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True) # Set categorical order for consistent sorting/plotting
    df = df.sort_values('model') # Sort the DataFrame by the new categorical order
    return df, metric, model_order


def make_comparison_plot(df1, df2, color_dict, metric1_name='Metric 1', metric2_name='Metric 2', 
                         title1='Plot 1', title2='Plot 2', figsize=(9, 6),
                         line_a=1,wspace=0.4):
    """
    Create side-by-side ranked box plots with connecting lines showing rank changes.
    
    Parameters:
    -----------
    df1, df2 : pandas.DataFrame
        DataFrames with same models but potentially different rankings
    metric1_name, metric2_name : str
        Names of the metrics being compared
    title1, title2 : str
        Titles for each subplot
    figsize : tuple
        Figure size (width, height)
    wspace : float
        Horizontal spacing between subplots (default 0.4, increase for more space)
    """
    
    # Process both dataframes
    df1_tidy, metric1, order1 = tidy_df(df1.copy())
    df2_tidy, metric2, order2 = tidy_df(df2.copy())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=False)
    fig.subplots_adjust(wspace=wspace)  # Space between plots for connecting lines
    
    # Plot 1
    ax1.axvline(x=0, color='darkgrey', linestyle='--', alpha=0.5)
    sns.boxplot(data=df1_tidy, x=metric1, y='model', order=order1, 
                color='white', linecolor='black', fliersize=0, 
                linewidth=1, ax=ax1)
    
    means1 = df1_tidy.groupby('model')[metric1].mean()
    for i, model in enumerate(order1):
        ax1.scatter(means1[model], i, color='lightblue', marker='o', s=50, zorder=10)
    
    sns.stripplot(data=df1_tidy, x=metric1, y='model', order=order1,
                  color='black', size=3, jitter=True, alpha=0.3, ax=ax1)
    for label in ax1.get_yticklabels():
        model_name = label.get_text()
        if model_name in color_dict:
            label.set_color(color_dict[model_name])

    ax1.set_ylabel('')
    ax1.set_xlabel(metric1_name,fontsize=14)
    ax1.set_xlim(-1, 1)
    ax1.set_title(title1,fontsize=16)
    ax1.tick_params(right=True, labelright=False, labelsize=12)  # Add ticks on right, no labels
    
    # Plot 2
    ax2.axvline(x=0, color='darkgrey', linestyle='--', alpha=0.5)
    sns.boxplot(data=df2_tidy, x=metric2, y='model', order=order2,
                color='white', linecolor='black', fliersize=0,
                linewidth=1, ax=ax2)
    
    means2 = df2_tidy.groupby('model')[metric2].mean()
    for i, model in enumerate(order2):
        ax2.scatter(means2[model], i, color='lightblue', marker='o', s=50, zorder=10)
    
    sns.stripplot(data=df2_tidy, x=metric2, y='model', order=order2,
                  color='black', size=3, jitter=True, alpha=0.3, ax=ax2)
    
    ax2.set_ylabel('')
    ax2.set_xlabel(metric2_name, fontsize=14)
    ax2.set_xlim(-1, 1)
    ax2.set_title(title2,fontsize=16)
    ax2.yaxis.tick_right()  # Move y-axis ticks to right side
    ax2.yaxis.set_label_position("right")  # Move y-axis label to right side
    for label in ax2.get_yticklabels():
        model_name = label.get_text()
        if model_name in color_dict:
            label.set_color(color_dict[model_name])
    ax2.tick_params(left=True, labelleft=False, labelsize=12)  # Add ticks on left, no labels
    
    # Draw connecting lines between the two plots
    for model in order1:
        if model in order2:
            # Get y-position in each plot (indices)
            y1 = order1.index(model)
            y2 = order2.index(model)
            # Convert to figure coordinates
            # Right edge of left plot - use parameterized x position
            point1 = ax1.transData.transform((1.0, y1))
            # Left edge of right plot - use parameterized x position
            point2 = ax2.transData.transform((-1, y2))
            # Convert to figure coordinates
            point1_fig = fig.transFigure.inverted().transform(point1)
            point2_fig = fig.transFigure.inverted().transform(point2)
            # Calculate rank change for color coding
            rank_change = y2 - y1
            # Color: green if improved (moved up = lower index), red if worsened
            if rank_change < 0:
                color = 'black'
            elif rank_change > 0:
                color = 'black'
            else:
                color = 'black'
            # Draw line
            line = plt.Line2D([point1_fig[0], point2_fig[0]], 
                            [point1_fig[1], point2_fig[1]],
                            transform=fig.transFigure,
                            color=color, alpha=line_a, linewidth=1.5, zorder=1)
            fig.add_artist(line)
    plt.savefig(image_dir / 'comparison_plot.svg', format='svg',bbox_inches='tight')
    plt.savefig(image_dir / 'comparison_plot.png', dpi=300,bbox_inches='tight')
    plt.show()


def make_intermodel_plot(model0,model1,database='fireprotdb'):
    if database=='fireprotdb':
        group0 = fpdb_metrics[fpdb_metrics['model'] == model0].groupby('protein')['spearman_rho'].mean().dropna()
        group1 = fpdb_metrics[fpdb_metrics['model'] == model1].groupby('protein')['spearman_rho'].mean().dropna()
        plot_title = 'Inter-model FireprotDB Stability Predictions'
        plot_path = image_dir / 'inter-model_comparison_plot_fireprotdb'
    else:
        group0 = func_metrics[func_metrics['model'] == model0].groupby('protein')['spearman_rho'].mean().dropna()
        group1 = func_metrics[func_metrics['model'] == model1].groupby('protein')['spearman_rho'].mean().dropna()
        plot_title = 'Inter-model ProteinGym Fitness Predictions'
        plot_path = image_dir / 'inter-model_comparison_plot_proteingym'

    common_proteins = group0.index.intersection(group1.index)
    group0 = group0.loc[common_proteins]
    group1 = group1.loc[common_proteins]
    categories = common_proteins
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    # Boxplot positions
    pos0, pos1 = 1, 2
    # --- Draw boxplots ---
    ax.boxplot(group0, positions=[pos0], widths=0.4,
            patch_artist=True, boxprops=dict(facecolor='white', alpha=0.3),
            medianprops=dict(color='black', linewidth=2, alpha=0.3))
    ax.boxplot(group1, positions=[pos1], widths=0.4,
            patch_artist=True, boxprops=dict(facecolor='white', alpha=0.3),
            medianprops=dict(color='black', linewidth=2, alpha=0.3))
    # --- Scatter + connecting lines ---
    # Use a colormap to give each category a unique color
    diffs = np.abs(group1 - group0)
    diff_min, diff_max = diffs.min(), diffs.max()
    alpha_min, alpha_max = 0.01, 1
    for cat in categories:
        y0 = float(group0.loc[cat])
        y1 = float(group1.loc[cat])
        alpha = float(np.clip(alpha_min + (alpha_max - alpha_min) * (diffs.loc[cat] - diff_min) / (diff_max - diff_min), alpha_min, alpha_max))
        ax.scatter([pos0], [y0], color='black', zorder=5, alpha=0.3, s=60)
        ax.scatter([pos1], [y1], color='black', zorder=5, alpha=0.3, s=60)
        ax.plot([pos0, pos1], [y0, y1], color='gray', alpha=alpha, linewidth=1.2, zorder=4)

    ax.set_xticks([pos0, pos1])
    tick_colors = [color_dict[model0],color_dict[model1]]
    ax.set_xticklabels([model0, model1], fontsize=14)
    for ticklabel, color in zip(ax.get_xticklabels(), tick_colors):
        ticklabel.set_color(color)

    ax.set_ylabel('Spearman $\\rho$', fontsize=14)
    ax.set_title(plot_title,fontsize=16)
    ax.set_xlim(0.5, 2.5)
    ax.tick_params(labelsize=14)
    ax.axhline(y=0, color='darkgrey', linestyle='--',alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path.with_suffix('.svg'), format='svg',bbox_inches='tight')
    plt.savefig(plot_path.with_suffix('.png'), dpi=300,bbox_inches='tight')
    plt.show()


make_comparison_plot(fpdb_metrics, func_metrics, color_dict,
                        metric1_name='Spearman $\\rho$',
                        metric2_name='Spearman $\\rho$',
                        title1='FireprotDB Stability Predictions',
                        title2='ProteinGym Fitness Predictions',
                        wspace=0.25)

make_intermodel_plot('ESM-3 (hybrid)', 'SaProt 650M','proteingym')
make_intermodel_plot('ProteinMPNN', 'SolubleMPNN','fireprotdb')