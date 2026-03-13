import numpy as np
import pandas as pd
import pickle
import re
from typing import Tuple, Any
from scipy.stats import spearmanr
from pathlib import Path
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



CSVDIR='../../data/denovo_data/fig5_data'
IMAGEDIR='../../images/figure5'

model = 'SaProt_650M_PDB'
pkl = f'../../outputs/logits/denovo/{model}.logits.x.denovo.pkl'
# model = 'protein_mpnn.proteinmpnn_v_48_020'
# pkl = f'../../outputs/logits/denovo/{model}.logits.x.denovo.use_sequence0.pkl'

csvdir=Path(CSVDIR)
imagedir=Path(IMAGEDIR)
imagedir.mkdir(parents=True, exist_ok=True)


Tawfik_2012_preds_name = ['OPD_BREDI.MODEL.logits.indices_254_306_274_233_172_269_272_80_111_204_130_271',
                          'OPD_BREDI_H254R.MODEL.logits.indices_233_306',
                          'OPD_BREDI_H254R_F306I.MODEL.logits.indices_274',
                          'OPD_BREDI_H254R_F306I_I274S.MODEL.logits.indices_233',
                          'OPD_BREDI_H254R_D233E_F306I_I274S.MODEL.logits.indices_172_269',
                          'OPD_BREDI_H254R_D233E_F306I_I274S_T172I.MODEL.logits.indices_269',
                          'OPD_BREDI_H254R_D233E_F306I_I274S_T172I_S269T_M138I_T199I.MODEL.logits.indices_272',
                          'OPD_BREDI_H254R_D233E_F306I_I274S_T172I_S269T_M138I_T199I_L272M.MODEL.logits.indices_80',
                          'OPD_BREDI_H254R_D233E_F306I_I274S_T172I_S269T_M138I_T199I_L272M_A80V.MODEL.logits.indices_111_204',
                          'OPD_BREDI_H254R_D233E_F306I_I274S_T172I_S269T_M138I_T199I_L272M_A80V_S111R.MODEL.logits.indices_204',
                          'OPD_BREDI_H254R_D233E_F306I_I274S_T172I_S269T_M138I_T199I_L272M_A80V_S111R_A204G.MODEL.logits.indices_130_271',
                          'OPD_BREDI_H254R_D233E_F306I_I274S_T172I_S269T_M138I_T199I_L272M_A80V_S111R_A204G_L130V.MODEL.logits.indices_271']
groups = {'Arnold_2012':{'csv':csvdir / 'CPXB_PRIM2_Arnold_2012.csv','preds_name':'CPXB_PRIM2.MODEL.logits.indices_182_264_329_401_438_439'},
          'Arnold_2014':{'csv':csvdir / 'CPXB_PRIM2_Arnold_2014.csv','preds_name':'CPXB_PRIM2.MODEL.logits.indices_182_264_329_401_438_439'},
          'Kaneko_2012':{'csv':csvdir / 'C1F2K5_ACIC5_Kaneko_2012.csv','preds_name':'C1F2K5_ACIC5.MODEL.logits.indices_45_292_334'},
          'Quax_2002':{'csv':csvdir / 'G7AC_PSEU7_Quax_2002.csv','preds_name':'G7AC_PSEU7.MODEL.logits.indices_178'},
          'full-Ozbek_2007':{'csv':csvdir / 'full-Q8IT70_HYDVU_Ozbek_2007_round-order.csv','preds_name':['full-Q8IT70_HYDVU.MODEL.logits.indices_482','full-Q8IT70_HYDVU_K482P.MODEL.logits.indices_472']},
          'truncated-Ozbek_2007':{'csv':csvdir / 'truncated-Q8IT70_HYDVU_Ozbek_2007_round-order.csv','preds_name':['truncated-Q8IT70_HYDVU.MODEL.logits.indices_21','truncated-Q8IT70_HYDVU_K21P.MODEL.logits.indices_11']},
          'Tawfik_2012':{'csv':csvdir / 'OPD_BREDI_Tawfik_2012_round-order.csv','preds_name':Tawfik_2012_preds_name}}


def pull_probs(mutation, preds, preds_name):
    def softmax(x, axis=-1, temperature=1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        x_scaled = x / temperature
        e_x = np.exp(x_scaled - np.max(x_scaled, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    
    def entropy(prob_dist):
        prob_dist = np.asarray(prob_dist)
        prob_dist = prob_dist[prob_dist > 0]
        return -np.sum(prob_dist * np.log2(prob_dist))
        
    AMINO_ACIDS = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    wt_aa = mutation[0]
    mut_aa = mutation[-1]
    prot_index = mutation[1:-1]
    array_index = preds_name.split('.indices_')[-1].split('_')
    array_index = array_index.index(prot_index)
    pred = preds[array_index]
    probs = softmax(pred)
    ent = entropy(probs)
    wt_index = AMINO_ACIDS.index(wt_aa)
    mut_index = AMINO_ACIDS.index(mut_aa)
    wt_prob = probs[wt_index]
    mut_prob = probs[mut_index]
    return wt_prob,mut_prob,np.log(mut_prob/wt_prob),ent


def ozbek_minicollagen_figure():
    group = 'full-Ozbek_2007'
    csv = groups[group]['csv']
    with open(pkl,'rb') as inf:
        all_preds = pickle.load(inf)
    data = pd.read_csv(csv)
    bar1 = []
    bar2 = []
    muts_p = []
    muts = []
    round = []
    muts_p_wtbackground = []
    wt_preds = all_preds[f'full-Q8IT70_HYDVU.{model}.logits.indices_472_482']
    for i,row in data.iterrows():
        if row['name']=='WT':
            bar1.append(row['pose1'])
            bar2.append(row['pose2'])
            muts_p.append(np.nan)
            muts_p_wtbackground.append(np.nan)
            muts.append('WT')
            round.append(row['round'])
            continue
        bar1.append(row['pose1'])
        bar2.append(row['pose2'])
        position = row['mutant'][1:-1]
        pred_name = row['pred'].replace('MODEL', model)
        preds = all_preds[pred_name]
        _,mut_p,_,_ = pull_probs(row['mutant'],preds,pred_name)
        _,mut_p_wtbackground,_,_ = pull_probs(row['mutant'],wt_preds,pred_name)
        muts_p.append(mut_p)
        muts_p_wtbackground.append(mut_p_wtbackground)
        muts.append(row['mutant'])
        round.append(row['round'])
    round = np.array(round)
    fig, ax1 = plt.subplots(figsize=(5.5, 5))
    ax1.plot(round, muts_p, color="#000000", marker='o',markersize=7, linewidth=3, label='Probability', alpha=0.8)
    ax1.plot(round, muts_p_wtbackground, color="#000000", marker='o',markersize=7, linewidth=3, linestyle='--', label='Probability', alpha=0.8)
    ax1.set_ylabel('Probability', fontsize=14)
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    width = 0.33
    colors = ['#1f77b4', '#ff7f0e']  # Distinct colors for each metric
    for i, (metric, label, color) in enumerate(zip([bar1,bar2], ['Pose 1','Pose 2'], colors)):
        offset = (i - 0.5) * width
        values = metric
        ax2.bar(round + offset, values, width, label=label, color=color, alpha=0.8)
    ax1.set_xlabel('Mutation Round', fontsize=14)
    ax2.set_ylabel('Pose percentage', fontsize=14)
    ax2.set_xticks(round)
    xtick_labels = [f"{r}" for r in round]
    ax2.set_xticklabels(xtick_labels)
    ax2.set_ylim(0,100)
    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax1.patch.set_visible(False)
    ax2.legend(bbox_to_anchor=(0.85, -0.20), loc='upper center',fontsize=12)
    from matplotlib.lines import Line2D
    linestyle_legend_elements = [
        Line2D([0], [0], color='black', linewidth=3, linestyle='-',  label='Directed Evolution Background'),
        Line2D([0], [0], color='black', linewidth=3, linestyle='--', label='WT Background'),
    ]
    linestyle_legend = ax1.legend(handles=linestyle_legend_elements, bbox_to_anchor=(0.3, -0.20), loc='upper center', fontsize=11)
    ax1.add_artist(linestyle_legend)
    plt.title('Hydra minicollagen pose flip',fontsize=16)
    plt.tight_layout()
    plt.savefig(imagedir / f'Ozbek-2007_{model}.png',dpi=300,bbox_inches="tight")
    plt.show()


def tawfik_phosphotriesterase_figure():
    group = 'Tawfik_2012'
    csv = groups[group]['csv']
    png_dir = '/home/pwoolley/work/proteingym/images/denovo'
    with open(pkl,'rb') as inf:
        all_preds = pickle.load(inf)
    data = pd.read_csv(csv)
    bar1 = []
    bar2 = []
    muts_p = []
    muts = []
    round = []
    muts_p_wtbackground = []
    wt_preds = all_preds[f'OPD_BREDI.{model}.logits.indices_80_111_130_172_204_233_254_269_271_272_274_306']
    for i,row in data.iterrows():
        if row['name']=='WT':
            bar1.append(row['kcat-Km_2NH'])
            bar2.append(row['kcat-Km_Paroxon'])
            muts_p.append(np.nan)
            muts_p_wtbackground.append(np.nan)
            muts.append('WT')
            round.append(row['round'])
            continue
        bar1.append(row['kcat-Km_2NH'])
        bar2.append(row['kcat-Km_Paroxon'])
        pred_name = row['pred'].replace('MODEL', model)
        preds = all_preds[pred_name]
        _,mut_p,_,_ = pull_probs(row['mutant'],preds,pred_name)
        _,mut_p_wtbackground,_,_ = pull_probs(row['mutant'],wt_preds,pred_name)
        muts_p.append(mut_p)
        muts_p_wtbackground.append(mut_p_wtbackground)
        muts.append(row['mutant'])
        round.append(row['round'])
    round = np.array(round)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(round, muts_p, color="#000000", marker='o',markersize=7, linewidth=3, label='Probability', alpha=0.8)
    ax1.plot(round, muts_p_wtbackground, color="#000000", marker='o',markersize=7, linewidth=3, linestyle='--', label='Probability', alpha=0.8)
    ax1.set_ylabel('Probability', fontsize=14)
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    width = 0.33
    colors = ['#1f77b4', '#ff7f0e']  # Distinct colors for each metric
    for i, (metric, label, color) in enumerate(zip([bar1,bar2], ['2NH','Paroxon'], colors)):
        offset = (i - 0.5) * width
        values = metric
        ax2.bar(round + offset, values, width, label=label, color=color, alpha=0.8)
    # ax1.set_xlabel('Mutation Round and Mutant', fontsize=14)
    ax1.set_xlabel('Mutation Round', fontsize=14)
    ax2.set_ylabel('$k_{\t{cat}}/K_m$', fontsize=14)
    ax2.set_xticks(round)
    # xtick_labels = [f"{r}\n{m}" for m, r in zip(muts, round)]
    xtick_labels = [f"{r}" for r in round]
    ax2.set_xticklabels(xtick_labels)
    ax2.axvline(x=7, linestyle="--", color="gray", alpha=0.5)
    ax2.axvline(x=8, linestyle="--", color="gray", alpha=0.5)
    ax2.set_yscale('log')
    ax2.set_ylim(100)
    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax1.patch.set_visible(False)
    ax2.legend(bbox_to_anchor=(0.64, -0.20), loc='upper center',fontsize=12)
    from matplotlib.lines import Line2D
    linestyle_legend_elements = [
        Line2D([0], [0], color='black', linewidth=3, linestyle='-',  label='Directed Evolution Background'),
        Line2D([0], [0], color='black', linewidth=3, linestyle='--', label='WT Background'),
    ]
    linestyle_legend = ax1.legend(handles=linestyle_legend_elements, bbox_to_anchor=(0.34, -0.20), loc='upper center', fontsize=11)
    ax1.add_artist(linestyle_legend)
    plt.title('Phosphotriesterase Hydrolysis of Paroxon Substrate',fontsize=16)
    plt.tight_layout()
    plt.savefig(imagedir / f'Tawfik_2012_{model}.png',dpi=300,bbox_inches="tight")
    plt.show()


def arnold_transisomerpercent_figure():
    group = 'Arnold_2012'
    csv = groups[group]['csv']
    preds_name = groups[group]['preds_name'].replace('MODEL',model)
    with open(pkl,'rb') as inf:
        all_preds = pickle.load(inf)
    preds = all_preds[preds_name]
    data = pd.read_csv(csv)

    plt_data = data.sort_values('trans_pct')
    plt_data['position'] = plt_data['mutant'].apply(lambda x: x[1:-1])
    WT_ROW = plt_data.loc[plt_data["mutant"] == "WT"].iloc[0]
    plt_data = plt_data[plt_data['mutant']!='WT']
    positions = sorted(plt_data["position"].dropna().unique())

    x_bar = []
    x_line = []
    trans_vals = []
    probs = []
    colors = []
    labels = []
    separators = []
    group_centers = []

    x_ctr = 0
    GROUP_GAP = 1.0
    for pos in positions:
        start_x = x_ctr
        sub = plt_data[plt_data["position"] == pos].copy()
        wt_res = sub.iloc[0]['mutant'][0]
        # add synthetic WT-at-position
        wt_pos = {
            "mutant": f"{wt_res}{pos}{wt_res}",
            "trans_pct": WT_ROW["trans_pct"],
            "cis_pct": WT_ROW["cis_pct"],
            "position": pos
        }
        sub = pd.concat([pd.DataFrame([wt_pos]), sub], ignore_index=True)
        # sort by trans_pct
        sub = sub.sort_values("trans_pct")
        for _, row in sub.iterrows():
            x_bar.append(x_ctr)
            x_line.append(x_ctr)
            trans_vals.append(row["trans_pct"])
            _,mut_p,_,_ = pull_probs(row['mutant'],preds,preds_name)
            probs.append(mut_p)
            colors.append("tab:green" if (row["mutant"][0]==row["mutant"][-1]) else "tab:blue")
            labels.append(row["mutant"][-1])
            x_ctr += 1
        end_x = x_ctr - 1
        group_centers.append((start_x + end_x) / 2)
        x_line.append(np.nan)
        probs.append(np.nan)
        x_ctr += GROUP_GAP
        separators.append(x_ctr - 0.5)
        x_ctr += GROUP_GAP

    plt.figure(figsize=(4,4))
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
            markerfacecolor="tab:green", markersize=9, label='WT'),
        Line2D([0], [0], marker='o', color='w',
            markerfacecolor="tab:blue", markersize=9, label='Mutant'),
    ]
    plt.legend(handles=legend_elements,fontsize=14,bbox_to_anchor=(1.0, 0.7), loc='upper left')
    plt.title('Cytochrome P450 Trans Isomer',fontsize=15)
    plt.scatter([x for x in probs if not np.isnan(x)], trans_vals,c=colors,s=125)
    plt.xlabel('Probability',fontsize=14)
    plt.ylabel('Trans Isomer Percentage', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(imagedir / f'Arnold_2012_scatter_{model}.png',dpi=300,bbox_inches="tight")
    plt.show()


def arnold_ttn_figure():
    group = 'Arnold_2014'
    csv = groups[group]['csv']
    preds_name = groups[group]['preds_name'].replace('MODEL',model)
    with open(pkl,'rb') as inf:
        all_preds = pickle.load(inf)
    preds = all_preds[preds_name]
    data = pd.read_csv(csv)
    plt_data = data.sort_values('TTN')
    probs = []
    for _, row in plt_data.iterrows():
        if row["mutant"] == 'WT':
            _,y,_,_ = pull_probs("C401C",preds, preds_name)
            probs.append(y)
        else:
            _,y,_,_ = pull_probs(row['mutant'],preds, preds_name)
            probs.append(y)
    colors = ['tab:green' if m == 'WT' else 'tab:blue' for m in plt_data['mutant']]
    plt.figure(figsize=(4,4))
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
            markerfacecolor="tab:green", markersize=9, label='WT'),
        Line2D([0], [0], marker='o', color='w',
            markerfacecolor="tab:blue", markersize=9, label='Mutant'),
    ]
    plt.legend(handles=legend_elements,fontsize=14,bbox_to_anchor=(1.0, 0.7), loc='upper left')
    plt.title('Cytochrome P450 Trans Isomer',fontsize=15)
    plt.scatter([x for x in probs if not np.isnan(x)], plt_data['TTN'],c=colors,s=125)
    plt.xlabel('Probability',fontsize=14)
    plt.ylabel('Trans Turnover Number', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks([1000,3000,5000,7000],fontsize=14)
    plt.savefig(imagedir / f'Arnold_2014_scatter_{model}.png',dpi=300,bbox_inches="tight")
    plt.show()


def kaneko_glucuronidase_figure():
    group = 'Kaneko_2012'
    csv = groups[group]['csv']
    png_dir = '/home/pwoolley/work/proteingym/images/denovo'
    preds_name = groups[group]['preds_name'].replace('MODEL',model)
    with open(pkl,'rb') as inf:
        all_preds = pickle.load(inf)
    preds = all_preds[preds_name]
    data = pd.read_csv(csv)

    plt_data = data.iloc[:2,:]
    wt_p,mut_p,_,_ = pull_probs('Y334F',preds,preds_name)
    width = 0.25
    metrics = ['PNP-β-GlcA_specificity-shift', 'PNP-β-Glc_specificity-shift', 'PNP-β-Xyl_specificity-shift']
    metric_labels = ['PNP-β-GlcA', 'PNP-β-Glc', 'PNP-β-Xyl']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Distinct colors for each metric
    x = np.arange(len(plt_data))
    fig, ax1 = plt.subplots(figsize=(5, 5))  # wider figure
    ax1.plot(x, [wt_p, mut_p], color="#000000", marker='o', markersize=7, linewidth=3, label='Probability', alpha=0.8)
    ax1.set_ylabel('Probability', fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='y', labelsize=14)  # y-tick fontsize
    ax2 = ax1.twinx()
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        offset = (i - 1) * width
        values = plt_data[metric].values
        ax2.bar(x + offset, values, width, label=label, color=color, alpha=0.8)
    ax2.set_xlabel('Mutant', fontsize=14)
    ax2.set_ylabel('Specificity Shift', fontsize=14)
    ax2.set_xticks(x)
    ax1.set_xticklabels(plt_data['mutant'], fontsize=14)  # x-tick fontsize
    ax2.tick_params(axis='y', labelsize=14)  # right y-axis tick fontsize
    ax2.set_yscale('log')
    ax2.set_ylim(0.1)
    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_visible(False)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), bbox_transform=ax1.transAxes, fontsize=12)
    plt.title('β-Glucuronidase Promiscuity', fontsize=16)
    plt.tight_layout()
    plt.savefig(imagedir / f'Kaneko_2012_{model}.png', dpi=300, bbox_inches="tight")
    plt.show()


ozbek_minicollagen_figure()
tawfik_phosphotriesterase_figure()
arnold_transisomerpercent_figure()
arnold_ttn_figure()
kaneko_glucuronidase_figure()
