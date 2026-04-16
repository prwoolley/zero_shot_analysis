import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from itertools import combinations
from functools import reduce
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from pathlib import Path


GROUPTSV='../../data/proteingym_data/DMS_substitutions_groups.tsv'
PROTEINGYMCSVDIR='../../data/proteingym_data/DMS_ProteinGym_substitutions'
IMAGEDIR='../../images/figure2'


imagedir = Path(IMAGEDIR)
interphenotype_g1_imagedir = imagedir / 'interphenotype_g1'
interphenotype_g1_imagedir.mkdir(parents=True, exist_ok=True)
interexperimeter_g2_imagedir = imagedir / 'interexperimeter_g2'
interexperimeter_g2_imagedir.mkdir(parents=True, exist_ok=True)
groups = pd.read_csv(GROUPTSV,sep='\t')
interphenotype_g1_df = groups[groups['group1_member']==1].rename(columns={'group1_str':'group_str'})
interexperimenter_g2_df = groups[groups['group2_member']==1].rename(columns={'group2_str':'group_str'})
csv_dir = Path(PROTEINGYMCSVDIR)


def dataset_analysis_plots(g_df,csv_dir,png_dir,group=1):
    def linear_regression(x,y):
        m, b = np.polyfit(x, y, 1) # Pearson (linear fit)
        y_pred = m * x + b
        pearson_r2 = r2_score(y, y_pred)
        spearman_rho, _ = spearmanr(x, y) # Spearman
        spearman_r2 = spearman_rho**2
        return pearson_r2, spearman_rho, spearman_r2, y_pred
    
    global_stats = []  # store per-protein mean correlations
    if group==1:
        color = '#1f77b4'
    else:
        color = '#ff7f0e'
    for _, row in g_df.iterrows():
        exps = row['group_str'].split(',')
        protein_data = []
        selections = []
        for exp in exps:
            if group==1:
                selection = exp.split(':')[1]
            elif group==2:
                s = exp.split(':')
                selection = f"{s[0].split('_')[2]}_{s[1]}"
            else:
                print('Need to specify group 1 or 2!')
                return
            selections.append(selection)
            csv = os.path.join(csv_dir, exp.split(':')[0])
            exp_data = pd.read_csv(csv)[['mutant', 'DMS_score']]
            exp_data = exp_data.rename(columns={'DMS_score': selection})
            protein_data.append(exp_data)
        protein_data = reduce(lambda left, right: pd.merge(left, right, on='mutant', how='inner'),protein_data)
        pair_stats = [] # --- pairwise correlations ---
        make_plots = True
        for a, b in combinations(selections, 2):
            if b == "Abundance":
                y = protein_data[a]
                x = protein_data[b]
            else:
                x = protein_data[a]
                y = protein_data[b]
            try:
                pearson_r2, spearman_rho, spearman_r2, y_pred = linear_regression(x,y)
                pair_stats.append({'a': a, 'b': b,
                               'pearson': pearson_r2,'spearman_rho': spearman_rho, 'spearman_r2': spearman_r2,
                               'y_pred': y_pred})
            except Exception as e:
                print(e)
                print(x)
                make_plots=False
                break
        if make_plots==False:
            continue
        pair_df = pd.DataFrame(pair_stats)
        worst = pair_df.loc[pair_df['pearson'].idxmin()] # weakest Pearson pair
        a, b, y_pred = worst['a'], worst['b'], worst['y_pred']
        if b == "Abundance":
            pltx,plty = b,a
        else:
            pltx,plty = a,b
        plt.figure(figsize=(4.5, 4))
        ax = plt.gca()
        ax.scatter(protein_data[pltx], protein_data[plty], s=2, c=color)
        if group == 2:
            pltx = f"{pltx.split('_')[0]} {pltx.split('_')[1]}"
            plty = f"{plty.split('_')[0]} {plty.split('_')[1]}"
        ax.set_title(f"{row['name']}", fontsize=16)
        ax.set_xlabel(pltx, fontsize=14)
        ax.set_ylabel(plty, fontsize=14)
        # Limit number of ticks with nice spacing
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.tick_params(axis='both', labelsize=14)
        ax.text(0.05, 0.95,f"Spearman $\\rho$ = {spearman_rho:.3f}", #$\nPearson $R^2 = {pearson_r2:.3f}$
                 transform=plt.gca().transAxes,verticalalignment="top", fontsize=12)
        plt.savefig(os.path.join(png_dir,row['UniProt_ID']+'_spearmanrho.png'),dpi=300,bbox_inches="tight")
        global_stats.append({
            'name': row['name'],
            'UniProt_ID': row['UniProt_ID'],
            'mean_pearson': pair_df['pearson'].mean(),
            'mean_spearman': pair_df['spearman_rho'].mean(),
            'all_pearson':pair_df['pearson'],
            'all_spearman':pair_df['spearman_rho'],
        })

    # Rho scatter plot with all proteins
    stats_df = pd.DataFrame(global_stats).sort_values(['mean_spearman'])
    y = np.arange(len(stats_df)) # plot all, not just the mean
    if group==1:
        hollow = ['KCNE1_HUMAN']
        filled = ['CP2C9_HUMAN','KCNJ2_MOUSE','OXDA_RHOTO','Q53Z42_HUMAN','S22A1_HUMAN','SPIKE_SARS2','VKOR1_HUMAN']
        plt.figure(figsize=(5.5, 5.0))
        plt.yticks(y, stats_df['UniProt_ID'], fontsize=14)
        legend_elements = [
            plt.scatter([], [], facecolors='none', edgecolors=color, s=50, linewidths=2.5, label='Binary fitness measurement'),
            plt.scatter([], [], color=color, s=50, label='Continuous fitness measurement')
        ]
        plt.legend(handles=legend_elements, fontsize=12,loc='upper center', bbox_to_anchor=(0.5, -0.15))
    elif group==2:
        plt.figure(figsize=(5.8, 4.2))
        plt.yticks(y, stats_df['UniProt_ID'], fontsize=14)
    for i, values in enumerate(stats_df['all_spearman']):
        if group == 1 and stats_df['UniProt_ID'].iloc[i] in hollow:
            plt.scatter(values, np.full(len(values), y[i]), 
                        facecolors='none', edgecolors=color, s=50, linewidths=2.5)
        else:
            plt.scatter(values, np.full(len(values), y[i]), color=color, s=50)
    plt.xticks(fontsize=14)
    plt.xlim(0, 1)
    plt.xlabel(f'Spearman $\\rho$', fontsize=14)
    if group==1:
        plt.title(f'Inter-phenotype correlation', fontsize=16)
        plot_path = png_dir / 'interphenotype_group1_rho-scatterplot.png'
    else:
        plt.title(f'Inter-experimentor correlation', fontsize=16)
        plot_path = png_dir / 'interexperimentor_group2_rho-scatterplot.png'
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path,dpi=300,bbox_inches="tight")


dataset_analysis_plots(interphenotype_g1_df,csv_dir,interphenotype_g1_imagedir,group=1)
dataset_analysis_plots(interexperimenter_g2_df,csv_dir,interexperimeter_g2_imagedir,group=2)