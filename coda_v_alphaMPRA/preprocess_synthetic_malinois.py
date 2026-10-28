# generated opus 5.5, edited grace
"""Add boda's 600bp MPRA flanks to the CODA synthetic sequence table and drop rows that fail
boda's standard-error filter (largest lfcSE across cell types >= 1.0), as in the Malinois
training preprocessing. Nothing else is filtered, reordered, split or augmented; kept rows are
unchanged apart from a new padded_sequence column after sequence."""
from functools import partial

import pandas as pd
from boda.common import utils

IN_PATH = '/home/go274/scratch_pi_skr2/go274/manual_MPRA_models/alphaMPRA/coda_v_alphaMPRA/41586_2024_8070_MOESM14_ESM.txt'
OUT_PATH = '/home/go274/scratch_pi_skr2/go274/manual_MPRA_models/alphaMPRA/coda_v_alphaMPRA/synthetic_seqs_predictions_padded600.txt'
SE_COLUMNS = ['K562_lfcSE', 'HepG2_lfcSE', 'SKNSH_lfcSE']
STDERR_THRESHOLD = 1.0   # same as the Malinois training preprocessing

# read everything as text so every existing value is written back exactly as it was
df = pd.read_csv(IN_PATH, sep='\t', dtype=str, keep_default_na=False)
n_rows, columns_in = len(df), list(df.columns)

# 200bp oligo -> 600bp with boda's MPRA_UPSTREAM / MPRA_DOWNSTREAM flanks, using the same
# function boda's MPRA_DataModule applies before training
pad = partial(utils.row_pad_sequence, in_column_name='sequence', padded_seq_len=600)
df.insert(df.columns.get_loc('sequence') + 1, 'padded_sequence', df.apply(pad, axis=1))

assert len(df) == n_rows, f'row count changed: {n_rows} -> {len(df)}'
assert [c for c in df.columns if c != 'padded_sequence'] == columns_in, 'original columns changed'
assert (df['padded_sequence'].str.len() == 600).all(), 'a padded sequence is not 600bp'
assert all(s in p for s, p in zip(df['sequence'], df['padded_sequence'])), 'oligo not inside its padded sequence'

# stderr stats: boda keeps a row only if its largest lfcSE across cell types is < threshold
# (mpra_datamodule.py: temp_df[stderr_columns].max(axis=1) < stderr_threshold); max skips NaN
se = df[SE_COLUMNS].apply(pd.to_numeric, errors='coerce')
kept = se.max(axis=1) < STDERR_THRESHOLD
print(f'rows: {n_rows}')
print(f'max lfcSE >= {STDERR_THRESHOLD} (boda would drop): {(~kept).sum()} ({(~kept).mean():.1%})')
print(f'rows with a missing lfcSE: {se.isna().any(axis=1).sum()}, all three missing: {se.isna().all(axis=1).sum()}')
for c in SE_COLUMNS:
    print(f'  {c} >= {STDERR_THRESHOLD}: {(se[c] >= STDERR_THRESHOLD).sum()}')
print(pd.DataFrame({'above_cutoff': (~kept).groupby(df['origin']).sum(), 'total': df.groupby('origin').size()}))

# drop the rows boda would drop; rows with every lfcSE missing have no max and are dropped too
df = df[kept].reset_index(drop=True)
assert len(df) == kept.sum(), f'filter kept {len(df)} rows, expected {kept.sum()}'
assert (df[SE_COLUMNS].apply(pd.to_numeric, errors='coerce').max(axis=1) < STDERR_THRESHOLD).all(), \
    'a kept row has lfcSE >= threshold'

df.to_csv(OUT_PATH, sep='\t', index=False)
print(f'wrote {len(df)} rows -> {OUT_PATH}')