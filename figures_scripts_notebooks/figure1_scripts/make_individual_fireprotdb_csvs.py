from pathlib import Path
import pandas as pd

FIREPROTCSVPATH='../../data/fireprotdb_data/fireprot_csvs/fireprotdb_20251015-164116.csv'
OUTPUTCSVDIR='../../data/fireprotdb_data/fireprot_csvs/individuals'
MINMUT=20 # minimum number of mutations to save an individual CSV, since many FireprotDB datasets are <10muts

### Tidying FireProt CSVs
def make_csv(group,outdir,min_mutants=1):
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
        outdir / f'{name}.csv',
        index=False
    )


outdir = Path(OUTPUTCSVDIR)
outdir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(FIREPROTCSVPATH)
df = df.dropna(subset=['SUBSTITUTION','UNIPROTKB'],axis=0).dropna(subset=['DDG', 'DTM'], how='all',axis=0)
df = df[df['SUBSTITUTION'].str.split(',').str.len() == 1]
df = df[['UNIPROTKB','SUBSTITUTION','DDG','DTM']]
for _, group in df.groupby('UNIPROTKB'):
    make_csv(group,outdir,min_mutants=MINMUT)