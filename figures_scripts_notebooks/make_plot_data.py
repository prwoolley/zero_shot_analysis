import os
import numpy as np
import pandas as pd
import pickle
import re
from itertools import combinations
from functools import reduce
from typing import Tuple, Any
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics import roc_auc_score, ndcg_score, r2_score



class ProteinMutationModel:
    AMINO_ACIDS = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

    def __init__(self, temperature=1.0, eps=1e-12):
        self.temperature = temperature
        self.eps = eps
        self.probs_long = None
        self.entropy_df = None
        self.data = None

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Model output handling
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Experimental data integration
    # ------------------------------------------------------------------

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
        

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------

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
    
    def metrics(self, roc_threshold=0.0, ndcg_k=None):
        df = self.data.copy()
        df['log_likelihood'] = np.log(df['mut_prob'].clip(self.eps))
        out = {
            "spearman": spearmanr(df['exp'], df['log_odds']).correlation,
            "kendall": kendalltau(df['exp'], df['log_odds']).correlation,
            "loglik_spearman": spearmanr(df['exp'], df['log_likelihood']).correlation
        }
        labels = (df['exp'] > roc_threshold).astype(int)
        if labels.nunique() > 1:
            out["roc_auc"] = roc_auc_score(labels, df['log_odds'])
        else:
            out["roc_auc"] = np.nan
        y_true = df['exp'].to_numpy().reshape(1, -1)
        y_score = df['log_odds'].to_numpy().reshape(1, -1)
        minval = y_true.min()
        if minval <= 0:
            y_true = (y_true - minval) + self.eps
        out["ndcg"] = ndcg_score(y_true, y_score, k=ndcg_k)
        return out


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


groups = pd.read_csv('../data/proteingym_data/DMS_substitutions_groups.tsv',sep='\t')
g1_df = groups[groups['group1_member']==1].rename(columns={'group1_str':'group_str'})
g2_df = groups[groups['group2_member']==1].rename(columns={'group2_str':'group_str'})
csv_dir = '../data/proteingym_data/DMS_ProteinGym_substitutions'



def tidy_figure_1_data():
    models_fp = {
         'esmc_300m.logits.x.fireprotdb.pkl':'ESMC_300M',
         'esmc_600m.logits.x.fireprotdb.pkl':'ESMC_600M',
         'AMPLIFY_350M.logits.x.fireprotdb.pkl':'AMPLIFY_350M',
         'ismc_300m.logits.x.fireprotdb.pkl':'ISMC_300M',
         'ismc_600m.logits.x.fireprotdb.pkl':'ISMC_600M',
         'SaProt_650M_PDB.logits.x.fireprotdb.pkl':'SaProt_650M',
         'esm2_t33_650M_UR50D.logits.x.fireprotdb.pkl':'ESM2_650M',
         'esm2_t36_3B_UR50D.logits.x.fireprotdb.pkl':'ESM2_3B',
         'esm2_t48_15b_UR50D.logits.x.fireprotdb.pkl':'ESM2_15B',
         'ESM3_sm_open_v0.structure.logits.x.fireprotdb.pkl':'ESM3_sm_structure',
         'ESM3_sm_open_v0.both.logits.x.fireprotdb.pkl':'ESM3_sm_both',
         'ESM3_sm_open_v0.sequence.logits.x.fireprotdb.pkl':'ESM3_sm_sequence',
         'ism_t33_650M_uc30pdb.logits.x.fireprotdb.pkl':'ISM_650M',
         'protein_mpnn.proteinmpnn_v_48_020.logits.x.combined.use_sequence0.pkl':'ProteinMPNN',
        #  'protein_mpnn.proteinmpnn_v_48_020.logits.x.combined.use_sequence1.pkl':'ProteinMPNN_seq1',
         'soluble_mpnn.solublempnn_v_48_020.logits.x.combined.use_sequence0.pkl':'SolubleMPNN',
        #  'soluble_mpnn.solublempnn_v_48_020.logits.x.combined.use_sequence1.pkl':'SolubleMPNN_seq1',
         'ProstT5.logits.x.fireprotdb.pkl':'ProstT5'
         }

    models_pg = {
         'esmc_300m.logits.x.combined.pkl':'ESMC_300M',
         'esmc_600m.logits.x.combined.pkl':'ESMC_600M',
         'AMPLIFY_350M.logits.x.combined.pkl':'AMPLIFY_350M',
         'ismc_300m.logits.x.combined.pkl':'ISMC_300M',
         'ismc_600m.logits.x.combined.pkl':'ISMC_600M',
         'SaProt_650M_PDB.logits.x.combined.pkl':'SaProt_650M',
         'esm2_t33_650M_UR50D.logits.x.combined.pkl':'ESM2_650M',
         'esm2_t36_3B_UR50D.logits.x.combined.pkl':'ESM2_3B',
         'esm2_t48_15b_UR50D.logits.x.combined.pkl':'ESM2_15B',
         'ESM3_sm_open_v0.structure.logits.x.combined.pkl':'ESM3_sm_structure',
         'ESM3_sm_open_v0.both.logits.x.combined.pkl':'ESM3_sm_both',
         'ESM3_sm_open_v0.sequence.logits.x.combined.pkl':'ESM3_sm_sequence',
         'ism_t33_650M_uc30pdb.logits.x.combined.pkl':'ISM_650M',
         'protein_mpnn.proteinmpnn_v_48_020.logits.x.combined.use_sequence0.pkl':'ProteinMPNN',
        #  'protein_mpnn.proteinmpnn_v_48_020.logits.x.combined.use_sequence1.pkl':'ProteinMPNN_seq1',
         'soluble_mpnn.solublempnn_v_48_020.logits.x.combined.use_sequence0.pkl':'SolubleMPNN',
        #  'soluble_mpnn.solublempnn_v_48_020.logits.x.combined.use_sequence1.pkl':'SolubleMPNN_seq1',
         'ProstT5.logits.x.combined.pkl':'ProstT5'
         }
    
    ### Tidying FireProt CSVs
    def make_csv(group,csv_dir='/home/pwoolley/work/proteingym/fireprot_csvs/individuals',min_mutants=1):
        group = group.copy()
        if group['DDG'].isna().all():
            group = group.rename(columns={
                'SUBSTITUTION': 'mutant',
                'DTM': 'DMS_score'
            })
        else:
            group = group.rename(columns={
                'SUBSTITUTION': 'mutant',
                'DDG': 'DMS_score'
            })
            group['DMS_score'] = -1 * group['DMS_score']
        name = group['UNIPROTKB'].iloc[0].split(',')[0]
        group = group.dropna(subset=['DMS_score'])
        group['DMS_score'] = pd.to_numeric(group['DMS_score'], errors='coerce')
        group = (
            group
            .groupby('mutant', as_index=False)['DMS_score']
            .mean()
        )
        if group['mutant'].nunique() < min_mutants:
            return
        group.to_csv(
            os.path.join(csv_dir, f'{name}.csv'),
            index=False
        )

    df = pd.read_csv('../data/fireprot_data/fireprot_csvs/fireprotdb_20251015-164116.csv')
    df = df.dropna(subset=['SUBSTITUTION','UNIPROTKB'],axis=0).dropna(subset=['DDG', 'DTM'], how='all',axis=0)
    df = df[df['SUBSTITUTION'].str.split(',').str.len() == 1]
    df = df[['UNIPROTKB','SUBSTITUTION','DDG','DTM']]
    for _, group in df.groupby('UNIPROTKB'):
        make_csv(group,min_mutants=5)

    pkl_dir = '/home/pwoolley/work/proteingym/outputs/logits/fireprotdb'
    pkls = [os.path.join(pkl_dir,x) for x in os.listdir(pkl_dir) if x.endswith('.pkl')]


    def evaluate_models(model_pkls, csv_files, pkl_dir, database = 'fireprotdb', temperature=1.0):
        results = []
        for pkl,name in model_pkls.items():
            pkl_path = os.path.join(pkl_dir,pkl)
            model = ProteinMutationModel(temperature=1.0)
            errors = []
            for csv_path in csv_files:
                try:
                    csv_name = os.path.basename(csv_path)
                    if database=='fireprotdb':
                        protein_id = csv_name.split('.csv')[0]
                    else:
                        protein_id = '_'.join(csv_name.split('_')[:1])
                    model.load_pickle(pkl_path, protein_id)
                    model.load_experiment(csv_path)
                    model.build_dataset()
                    rho = model.spearman()
                    results.append({
                        "model": name,
                        "protein": protein_id,
                        "spearman_rho": rho
                    })
                except Exception as e:
                    errors.append(f'{name},{e}')
                    continue
            print(set(errors))
        return pd.DataFrame(results)

    # # Fireprot
    csv_dir = '/home/pwoolley/work/proteingym/fireprot_csvs/individuals'
    csvs = [os.path.join(csv_dir,x) for x in os.listdir(csv_dir)]
    df = evaluate_models(models_fp,csvs,'/home/pwoolley/work/proteingym/outputs/logits/fireprotdb')
    df.to_csv('../outputs/csvs/fpdb_comparison_plot.csv',index=False)

    # ProteinGym
    csv_dir = '/home/pwoolley/work/proteingym/DMS_ProteinGym_substitutions'
    csvs = [os.path.join(csv_dir,x) for x in os.listdir(csv_dir)]
    df = evaluate_models(models_pg,csvs,'/home/pwoolley/work/proteingym/outputs/logits/proteingym',database='proteingym')
    df.to_csv('../outputs/csvs/pg_comparison_plot.csv',index=False)
    return




def tidy_figure_2_data():
    
    return

def tidy_figure_3_data():
    return

def tidy_figure_4_data():
    return

def tidy_figure_5_data():
    return


