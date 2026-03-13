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


# # User defined variables
OUTPUTCSVDIR='../../outputs/csvs'
PROTEINGYMLOGITPKL='../../outputs/logits/proteingym/ESM3_sm_open_v0.both.logits.x.combined.pkl'
PROTEINGYMCSVDIR='../../data/proteingym_data/DMS_ProteinGym_substitutions'
INDIVIDUALCSV='../../data/proteingym_data/DMS_ProteinGym_substitutions/CP2C9_HUMAN_Amorosi_2021_abundance.csv'


outdir = Path(OUTPUTCSVDIR)
outdir.mkdir(parents=True, exist_ok=True)
pkl = Path(PROTEINGYMLOGITPKL)
csvdir = Path(PROTEINGYMCSVDIR)
csvs = csvdir.glob('*.csv')
individualcsv = Path(INDIVIDUALCSV)
model = 'ESM3_sm_both'


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
    

def tidy_fitclass_density_data():
    pm = ProteinMutationModel(temperature=1.0)
    fitclass_rho = {'all_rho':[],'unfit_rho':[],'fit_rho':[],'fittest_rho':[],'model':[]}
    for csv in csvs:
        proteinid = '_'.join(csv.name.split('_')[:2])
        try:
            pm.load_pickle(pkl,protein_id=proteinid)
            pm.compute_entropy()
            pm.load_experiment(csv)
            pm.build_dataset()
            fit_bubble = pm.data[pm.data['DMS_score_bin']==1]
            all_rs = pm.spearman()
            all_min = pm.data['exp'].min()
            fit_min = fit_bubble['exp'].min()
            fittest_min = fit_bubble['exp'].median()
            fit_max = fit_bubble['exp'].max()
            unfit_rs = pm.spearman(subsetting={'exp':{'min_val':all_min,'max_val':fit_min}})
            fit_rs = pm.spearman(subsetting={'exp':{'min_val':fit_min,'max_val':fittest_min}})
            fittest_rs = pm.spearman(subsetting={'exp':{'min_val':fittest_min,'max_val':fit_max}})
            fitclass_rho['all_rho'].append(all_rs)
            fitclass_rho['unfit_rho'].append(unfit_rs)
            fitclass_rho['fit_rho'].append(fit_rs)
            fitclass_rho['fittest_rho'].append(fittest_rs)
            fitclass_rho['model'].append(model)

        except Exception as e:
            print(e)
            continue
    fitclass_rho = pd.DataFrame(fitclass_rho)
    fitclass_rho.to_csv(outdir / 'figure4_fitclass_density.csv',index=False)

tidy_fitclass_density_data()