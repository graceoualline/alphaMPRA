import torch
import lightning.pytorch as ptl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import boda

data_module = boda.data.MPRA_DataModule
model_module= boda.model.BassetBranched
graph_module= boda.graph.CNNBasicTraining
# this is to preprocess the file so we don't need to do any additional filtering
# when giving the file to other models

# load the data
data = data_module(
    datafile_path='/home/go274/scratch_pi_skr2/go274/coda_data/DATA-Table_S2__MPRA_dataset.txt', 
    sep='\t', sequence_column='sequence',
    stderr_columns=['K562_lfcSE','HepG2_lfcSE','SKNSH_lfcSE'],
    stderr_threshold=1.0, std_multiple_cut=6.0,
    synth_val_pct=0.0, synth_test_pct=99.98,
    val_chrs=['19', '21', 'X'], test_chrs=['7', '13'], 
    activity_columns=['K562_log2FC','HepG2_log2FC', 'SKNSH_log2FC'],
    batch_size=1076, padded_seq_len=600, 
    use_reverse_complements=True, 
    duplication_cutoff=0.5, 
    num_workers=8)
# setup() does every preprocessing step:
#   remove data with std err > 1 (stderr_columns / stderr_threshold)
#   drop outlier rows (std_multiple_cut)
#   add flanking regions (MPRA_UPSTREAM / MPRA_DOWNSTREAM, padded to 600bp)
#   split by chromosome (val_chrs / test_chrs)
#   train only: duplicate the top values (duplication_cutoff) and serve every row
#   with its reverse compliment (use_reverse_complements)
data.setup()

import numpy as np
import pandas as pd
from pathlib import Path
from boda.common import utils, constants

out_dir = Path('/home/go274/scratch_pi_skr2/go274/manual_MPRA_models/baseline_CODA')
out_dir.mkdir(exist_ok=True)
columns = ['K562_log2FC', 'HepG2_log2FC', 'SKNSH_log2FC']
# boda's one-hot channel order; the encoder (utils.row_dna2tensor) and the decoder
# (utils.batch2list) both use it, so decoding is the exact inverse of encoding
assert constants.STANDARD_NT == ['A', 'C', 'G', 'T'], f'unexpected channel order {constants.STANDARD_NT}'

# read every split back out through boda's own dataset, so the rows, duplicates and
# reverse complements are exactly the ones Malinois trains on
# data.dataset_train: asking for item i returns a row or its reverse complement, and the
#   top-activity rows come around twice. Rows are sorted by activity, highest first:
#   item 0 = row 0, item 1 = row 0 reverse complemented, item 2 = row 1, ...
# data.dataset_val / data.dataset_test: every row once, forward only
split_dfs = []
for split, dataset in [('train', data.dataset_train), ('val', data.dataset_val), ('test', data.dataset_test)]:
    seqs, activities = [], []

    # the DataLoader asks the dataset for items 0, 1, 2, ... in order (shuffle=False)
    # and stacks 10000 of them at a time:
    #   dna: (10000, 4, 600) one-hot sequences, channels in A, C, G, T order
    #   act: (10000, 3) log2FC for K562, HepG2, SKNSH
    for dna, act in torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False):

        # one-hot -> string with boda's own decoder, utils.batch2list. For one 4bp sequence:
        #   dna[b] = [[1, 0, 0, 0],    A channel (STANDARD_NT[0])
        #             [0, 0, 1, 0],    C channel (STANDARD_NT[1])
        #             [0, 1, 0, 0],    G channel (STANDARD_NT[2])
        #             [0, 0, 0, 1]]    T channel (STANDARD_NT[3])
        #   argmax over the channels picks the 1 at each position -> [0, 2, 1, 3]
        #   STANDARD_NT[0], [2], [1], [3] -> 'A', 'G', 'C', 'T' -> joined into 'AGCT'
        # N would be an all-zero column, which argmax reads as 'A'; stock boda's encoder
        # crashes on N before this point, and Table S2 has none
        # dna: (batch, 4, 600) -> one 600bp string per row
        seqs.extend(utils.batch2list(dna))

        # keep this batch's activities; row b lines up with seqs for the same batch row
        activities.append(act.numpy())

    # (N, 3) activities, with the sequence and the split name as extra columns:
    #   split  sequence    K562_log2FC  HepG2_log2FC  SKNSH_log2FC
    #   train  AGCT...     5.21         4.87          5.02
    df = pd.DataFrame(np.concatenate(activities), columns=columns)
    df.insert(0, 'sequence', seqs)
    df.insert(0, 'split', split)

    assert len(df) == len(dataset), f'{split}: wrote {len(df)} rows, dataset has {len(dataset)}'
    assert (df['sequence'].str.len() == 600).all(), f'{split}: a sequence is not 600bp'
    assert not df[columns].isna().any().any(), f'{split}: NaN activity survived the filters'
    print(f'{split}: {len(df)} rows')
    split_dfs.append(df)

# one file, all splits stacked: train rows first, then val, then test
all_df = pd.concat(split_dfs, ignore_index=True)
assert len(all_df) == sum(len(df) for df in split_dfs), \
    f'concat changed row count: {sum(len(df) for df in split_dfs)} -> {len(all_df)}'

out_path = out_dir / 'malinois_all_data_filtered_revcomp_preprocessed.tsv'
all_df.to_csv(out_path, sep='\t', index=False)
print(f'all splits: {len(all_df)} rows -> {out_path}')
