import os
import numpy as np
import pandas as pd
import json
import pickle
import re
from typing import Tuple, Any
from scipy.stats import spearmanr
from functools import reduce
from pathlib import Path
from sklearn.metrics import r2_score

# # User defined variables
OUTPUTCSVDIR='../../outputs/csvs'
PROTEINGYMCSVDIR='../../data/proteingym_data/DMS_ProteinGym_substitutions'
PROTEINGYMLOGITDIR='../../outputs/logits/proteingym'
GROUPTSV='../../data/proteingym_data/DMS_substitutions_groups.tsv'

outdir = Path(OUTPUTCSVDIR)
outdir.mkdir(parents=True, exist_ok=True)
logitdir = Path(PROTEINGYMLOGITDIR)
groups = pd.read_csv(GROUPTSV,sep='\t')
interphenotype_g1_df = groups[groups['group1_member']==1]
interexperimenter_g2_df = groups[groups['group2_member']==1]
csvdir = Path(PROTEINGYMCSVDIR)


models = {
        #  'esmc_300m.logits.x.combined.pkl':'ESMC_300M',
         'esmc_600m.logits.x.combined.pkl':'ESMC_600M',
        #  'AMPLIFY_120M.logits.x.combined.pkl':'AMPLIFY_120M',
         'AMPLIFY_350M.logits.x.combined.pkl':'AMPLIFY_350M',
        #  'ismc_300m.logits.x.combined.pkl':'ISMC_300M',
         'ismc_600m.logits.x.combined.pkl':'ISMC_600M',
         'SaProt_650M_PDB.logits.x.combined.pkl':'SaProt_650M',
         'esm2_t48_15b_UR50D.logits.x.combined.pkl':'ESM2_15B',
        #  'ESM3_sm_open_v0.structure.logits.x.combined.pkl':'ESM3_sm_structure',
         'ESM3_sm_open_v0.both.logits.x.combined.pkl':'ESM3_sm_both',
        #  'ESM3_sm_open_v0.sequence.logits.x.combined.pkl':'ESM3_sm_sequence',
        #  'ism_t33_650M_uc30pdb.logits.x.combined.pkl':'ISM_650M',
        #  'esm2_t33_650M_UR50D.logits.x.combined.pkl':'ESM2_650M',
         'protein_mpnn.proteinmpnn_v_48_020.logits.x.combined.use_sequence0.pkl':'ProteinMPNN',
        #  'protein_mpnn.proteinmpnn_v_48_020.logits.x.combined.use_sequence1.pkl':'ProteinMPNN_seq1',
         'soluble_mpnn.solublempnn_v_48_020.logits.x.combined.use_sequence0.pkl':'SolubleMPNN',
        #  'soluble_mpnn.solublempnn_v_48_020.logits.x.combined.use_sequence1.pkl':'SolubleMPNN_seq1',
         'ProstT5.logits.x.combined.pkl':'ProstT5'
         }
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


pkls = list(logitdir.glob("*.pkl"))


class ProteinMutationModel:
    AMINO_ACIDS = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    def __init__(self, temperature=1.0, eps=1e-12):
        self.temperature = temperature
        self.eps = eps
        self.probs_long = None
        self.entropy_df = None
        self.data = None

    @staticmethod
    def softmax(x, axis=-1, temperature=1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        x_scaled = x / temperature
        e_x = np.exp(x_scaled - np.max(x_scaled, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    @staticmethod
    def entropy(prob_dist):
        prob_dist = np.asarray(prob_dist)
        prob_dist = prob_dist[prob_dist > 0]
        return -np.sum(prob_dist * np.log2(prob_dist))

    @staticmethod
    def parse_mutations(
        mutation_string: Any
    ) -> Tuple[Any, int, str, str]:
        if pd.isna(mutation_string) or not isinstance(mutation_string, str):
            return [], 0, "", ""
        muts = mutation_string.split(':')
        pattern = re.compile(r'([A-Za-z])(\d+)([A-Za-z])')
        indices, wt, mut = [], "", ""
        for m in muts:
            match = pattern.search(m)
            if match:
                wt += match.group(1)
                mut += match.group(3)
                indices.append(int(match.group(2)))
        if len(indices) == 1:
            return indices[0], 1, wt, mut
        return indices, len(indices), wt, mut
    
    def load_pickle(self, pkl_path: str, protein_id: str):
        with open(pkl_path, "rb") as f:
            pkldata = pickle.load(f)
        for k, v in pkldata.items():
            if protein_id in k:
                indices = [int(x) for x in k.split('indices_')[-1].split('_')]
                probs = self.softmax(v, temperature=self.temperature)
                df = pd.DataFrame(probs,index=indices,columns=self.AMINO_ACIDS)
                self.probs_long = (df.stack().reset_index()
                                   .rename(columns={"level_0": "index","level_1": "amino_acid",0: "prob"}))
                return
        raise ValueError(f"Protein ID '{protein_id}' not found in pickle.")

    def compute_entropy(self):
        self.entropy_df = (
            self.probs_long
            .groupby("index")["prob"]
            .apply(self.entropy)
            .reset_index(name="entropy")
        )

    def load_experiment(self, csv_path: str):
        exp = pd.read_csv(csv_path)
        exp[['index','number_mut','wt','mut']] = (exp['mutant'].apply(lambda x: pd.Series(self.parse_mutations(x))))
        self.exp_df = exp[exp['number_mut'] == 1].reset_index(drop=True)

    def build_dataset(self):
        probs = self.probs_long.copy()
        exp_df = self.exp_df.copy()
        # Rank probabilities
        probs['prob'] = probs['prob'].astype('float64')
        probs = probs.sort_values(['index','prob'], ascending=[True, False])
        probs['rank'] = probs.groupby('index')['prob'].rank(method='first', ascending=False)
        # WT info
        wt = exp_df[['index','wt']].drop_duplicates()
        wt_prob = pd.merge(wt, probs,left_on=['index','wt'],right_on=['index','amino_acid']).rename(columns={'prob':'wt_prob'})[['index','wt_prob']]
        wt_rank = pd.merge(wt, probs,left_on=['index','wt'],right_on=['index','amino_acid']).rename(columns={'rank':'wt_rank'})[['index','wt_rank']]
        # Mutation info
        exp = pd.merge(exp_df, probs,left_on=['index','mut'],right_on=['index','amino_acid']).rename(columns={'prob':'mut_prob','rank':'mut_rank'})
        # Merge everything
        exp = (
            exp
            .merge(wt_prob, on='index')
            .merge(wt_rank, on='index')
            .merge(self.entropy_df, on='index')
        )
        exp = exp.rename(columns={'DMS_score':'exp'})
        exp['log_odds'] = (np.log(exp['mut_prob'].clip(self.eps))-np.log(exp['wt_prob'].clip(self.eps)))
        self.data = exp.reset_index(drop=True)
        
    def spearman(self, column='exp', subsetting = None, score_col='log_odds'):
        """
        Subset the data where `column` is between min_val and max_val, then compute Spearman correlation
        with `score_col`.
        Parameters
        ----------
        column : str
            Column name to filter on (default 'exp').
        subsetting : dict
            Keys are variables in the dataframe, values are a dict with two keys:
                min_val : float
                    Minimum value (inclusive) for filtering. If None, no lower bound.
                max_val : float
                    Maximum value (inclusive) for filtering. If None, no upper bound.
        score_col : str
            Column name to compute Spearman correlation against (default 'log_odds').
        Returns
        -------
        float
            Spearman correlation of the filtered data. Returns np.nan if not enough data points.
        """
        if self.data is None:
            raise ValueError("Data not built yet. Run `build_dataset()` first.")
        df = self.data.copy()
        if subsetting is not None:
            for k,v in subsetting.items():
                if v['min_val'] is not None:
                    df = df[df[k] >= v['min_val']]
                if v['max_val'] is not None:
                    df = df[df[k] <= v['max_val']]
        if df.shape[0] < 2:
            return np.nan  # Not enough points to compute correlation
        corr = spearmanr(df[column], df[score_col]).correlation
        return corr


def tidy_interexperimenter_g2_data(g2_df):
    g2_df['group2_str'] = g2_df['group2_str'].str.split(',')
    g2_df = g2_df.explode('group2_str').reset_index(drop=True)
    pm = ProteinMutationModel(temperature=1.0)

    mutation_sets = {}
    mutation_lookup = {}
    for uniprot_id, group_df in g2_df.groupby('UniProt_ID'):
        mutation_sets[uniprot_id] = {}
        positions = []
        for i, row in group_df.iterrows():
            csv = csvdir / row['group2_str'].split(':')[0]
            dataset_name = row['group2_str'].split(':')[0]
            raw = pd.read_csv(csv)
            csv_positions = raw[~raw['mutant'].str.contains(':')]['mutant'].tolist()
            mutation_sets[uniprot_id][dataset_name] = csv_positions
            if len(positions) == 0:
                positions.extend(csv_positions)
            else:
                positions = [x for x in positions if x in set(csv_positions)]
        mutation_lookup[uniprot_id] = list(set(positions))
        print(uniprot_id, len(mutation_lookup[uniprot_id]))

    with open(outdir / 'figure3_interexperimenter_g2_obs.txt', 'w') as f:
        for uniprot_id in mutation_sets:
            parts = [uniprot_id]
            for dataset_name, muts in mutation_sets[uniprot_id].items():
                parts.append(f"{dataset_name}:{len(muts)}")
            parts.append(f"intersection:{len(mutation_lookup[uniprot_id])}")
            f.write(', '.join(parts) + '\n')

    controlled = {v:[] for v in models.values()}
    controlled['UniProt_ID'] = []
    uncontrolled = {v:[] for v in models.values()}
    uncontrolled['UniProt_ID'] = []

    for _, row in g2_df.iterrows():
        proteinid = row['UniProt_ID']
        csv = csvdir / row['group2_str'].split(':')[0]
        controlled_scores = {}
        uncontrolled_scores = {}
        passed_controlled = True
        passed_uncontrolled = True

        for pkl in pkls:
            if pkl.name not in models:
                continue
            model = models[pkl.name]

            try:
                pm.load_pickle(pkl, protein_id=proteinid)
                pm.compute_entropy()
                pm.load_experiment(csv)
                pm.exp_df = pm.exp_df[pm.exp_df['mutant'].isin(mutation_lookup[proteinid])]
                pm.build_dataset()
                controlled_scores[model] = pm.spearman()
            except Exception as err:
                passed_controlled = False
                print(f"Controlled error: {err}")

            try:
                pm.load_pickle(pkl, protein_id=proteinid)
                pm.compute_entropy()
                pm.load_experiment(csv)
                pm.build_dataset()
                uncontrolled_scores[model] = pm.spearman()
            except Exception as err:
                passed_uncontrolled = False
                print(f"Uncontrolled error: {err}")

        if passed_controlled:
            for model, score in controlled_scores.items():
                controlled[model].append(score)
            controlled['UniProt_ID'].append(proteinid)

        if passed_uncontrolled:
            for model, score in uncontrolled_scores.items():
                uncontrolled[model].append(score)
            uncontrolled['UniProt_ID'].append(proteinid)

    controlled_df = pd.DataFrame({k:v for k,v in controlled.items() if len(v) > 0})
    controlled_df = controlled_df.sort_values('ESM3_sm_both', ascending=False).reset_index(drop=True)
    controlled_df.to_csv(outdir / 'figure3_interexperimenter_g2.csv', index=False)

    uncontrolled_df = pd.DataFrame({k:v for k,v in uncontrolled.items() if len(v) > 0})
    uncontrolled_df = uncontrolled_df.sort_values('ESM3_sm_both', ascending=False).reset_index(drop=True)
    uncontrolled_df.to_csv(outdir / 'figure3_interexperimenter_g2_uncontrolled.csv', index=False)
    return


def tidy_interphenotype_g1_data(g1_df):
    def linear_regression(x, y):
        m, b = np.polyfit(x, y, 1)
        y_pred = m * x + b
        pearson_r2 = r2_score(y, y_pred)
        spearman_rho, _ = spearmanr(x, y)
        spearman_r2 = spearman_rho**2
        return pearson_r2, spearman_rho, spearman_r2, y_pred

    pm = ProteinMutationModel(temperature=1.0)

    g1_df_copy = g1_df.copy()
    g1_df_copy['group1_str'] = g1_df_copy['group1_str'].str.split(',')
    g1_df_copy = g1_df_copy.explode('group1_str').reset_index(drop=True)

    mutation_lookup = {}
    mutation_sets = {}
    for uniprot_id, group_df in g1_df_copy.groupby('UniProt_ID'):
        positions = []
        mutation_sets[uniprot_id] = {}
        for i, row in group_df.iterrows():
            csv = csvdir / row['group1_str'].split(':')[0]
            dataset_name = row['group1_str'].split(':')[0]
            raw = pd.read_csv(csv)
            csv_positions = raw[~raw['mutant'].str.contains(':')]['mutant'].tolist()
            mutation_sets[uniprot_id][dataset_name] = csv_positions
            if len(positions) == 0:
                positions.extend(csv_positions)
            else:
                positions = [x for x in positions if x in set(csv_positions)]
        mutation_lookup[uniprot_id] = list(set(positions))
        print(uniprot_id, len(mutation_lookup[uniprot_id]))

    with open(outdir / 'figure3_interphenotype_g1_obs.txt', 'w') as f:
        for uniprot_id in mutation_sets:
            parts = [uniprot_id]
            for dataset_name, muts in mutation_sets[uniprot_id].items():
                parts.append(f"{dataset_name}:{len(muts)}")
            parts.append(f"intersection:{len(mutation_lookup[uniprot_id])}")
            f.write(', '.join(parts) + '\n')

    controlled = {'proteinid': [], 'model': [], 'expression_rs': [], 'selection_rs': [], 'selection_name': []}
    uncontrolled = {'proteinid': [], 'model': [], 'expression_rs': [], 'selection_rs': [], 'selection_name': []}
    all_protein_data = {'proteinid': [], 'variable_rs': []}

    for _, row in g1_df.iterrows():
        proteinid = row['UniProt_ID']
        exps = row['group1_str'].split(',')
        selections = []
        protein_data = []

        for exp in exps:
            selection = exp.split(':')[1]
            selections.append(selection)
            csv = os.path.join(csvdir, exp.split(':')[0])
            exp_data = pd.read_csv(csv)[['mutant', 'DMS_score']]
            exp_data = exp_data.rename(columns={'DMS_score': selection})
            protein_data.append(exp_data)

            for pkl in pkls:
                if pkl.name not in models:
                    continue
                model = models[pkl.name]

                try:
                    pm.load_pickle(pkl, protein_id=proteinid)
                    pm.compute_entropy()
                    pm.load_experiment(csv)
                    pm.exp_df = pm.exp_df[pm.exp_df['mutant'].isin(mutation_lookup[proteinid])]
                    pm.build_dataset()
                    if selection == 'Abundance':
                        controlled['expression_rs'].append(pm.spearman())
                    else:
                        controlled['selection_rs'].append(pm.spearman())
                        controlled['selection_name'].append(selection)
                        controlled['proteinid'].append(proteinid)
                        controlled['model'].append(model)
                except Exception as err:
                    print(f"Controlled error: {err}")

                try:
                    pm.load_pickle(pkl, protein_id=proteinid)
                    pm.compute_entropy()
                    pm.load_experiment(csv)
                    pm.build_dataset()
                    if selection == 'Abundance':
                        uncontrolled['expression_rs'].append(pm.spearman())
                    else:
                        uncontrolled['selection_rs'].append(pm.spearman())
                        uncontrolled['selection_name'].append(selection)
                        uncontrolled['proteinid'].append(proteinid)
                        uncontrolled['model'].append(model)
                except Exception as err:
                    print(f"Uncontrolled error: {err}")

        protein_data = reduce(lambda left, right: pd.merge(left, right, on='mutant', how='inner'), protein_data)
        a, b = selections
        if b == "Abundance":
            y = protein_data[a]
            x = protein_data[b]
        else:
            x = protein_data[a]
            y = protein_data[b]
        try:
            _, spearman_rho, _, _ = linear_regression(x, y)
            all_protein_data['proteinid'].append(proteinid)
            all_protein_data['variable_rs'].append(spearman_rho)
        except:
            print(selections)

    all_protein_data = pd.DataFrame(all_protein_data)

    controlled_df = pd.DataFrame(controlled)
    controlled_df = pd.merge(controlled_df, all_protein_data, how='left')
    controlled_df.to_csv(outdir / 'figure3_interphenotype_g1.csv', index=False)

    uncontrolled_df = pd.DataFrame(uncontrolled)
    uncontrolled_df = pd.merge(uncontrolled_df, all_protein_data, how='left')
    uncontrolled_df.to_csv(outdir / 'figure3_interphenotype_g1_uncontrolled.csv', index=False)
    return


# interexperimenter_g2_df = interexperimenter_g2_df[interexperimenter_g2_df['UniProt_ID']!='SPG1_STRSG'] # manually dropping this row because its slow:
tidy_interexperimenter_g2_data(interexperimenter_g2_df)
tidy_interphenotype_g1_data(interphenotype_g1_df)