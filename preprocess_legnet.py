# preprocessing MPRAlegnet data, or lentiviral data in general

import re

import numpy as np
import pandas as pd
import torch
from pathlib import Path

import boda
from boda.common import utils, constants
from boda.data.mpra_datamodule import DNAActivityDataset

# this is to preprocess the file so we don't need to do any additional filtering
# when giving the file to other models

MPRA_LENTI_UPSTREAM   = 'TGGTTGGGGGAAGGAAAAAAAATGGTGGCAGTGTTTTTTTCAGAATTAGAAGTGAAATGAAAACTTGTTGTGTGTGAGGATTTCTAATGACATGTGGTGGTTGCATACTGAGTGAAGCCGGTGAGCATTCTGCCATGTCACCCCCTCGTGCTCAGTAATGTACTTTACAGAAATCCTAAACTCAAAAGATTGATATAAACCATGCTTCTTGTGTATATCCGGTCTCTTCTCTGGGTAGTCTCACTCAGCCTGCATTTCTGCCAGGGCCCGCTCTAGACCTGCAGGAGGACCGGATCAACT'
MPRA_LENTI_DOWNSTREAM = 'CATTGCGTGAACCGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTGCAAAGTGAACACATCGCTAAGCGAAAGCTAAGNNNNNNNNNNNNNNNACCGGTCGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCC'

DATAFILE = '/nfs/roberts/scratch/pi_skr2/zm278/arden_agarwal/preprocessed/merged_preprocessed.csv'
OUT_DIR  = Path('/home/go274/scratch_pi_skr2/go274/manual_MPRA_models/alphaMPRA')

ACT = ['K562_log2FC', 'HepG2_log2FC', 'WTC11_log2FC']
SE  = ['K562_lfcSE', 'HepG2_lfcSE', 'WTC11_lfcSE']
# boda's MPRA_DataModule defaults, kept so the cutoffs match the Malinois baseline
STDERR_THRESHOLD   = 1.0
STD_MULTIPLE_CUT   = 6.0
UP_CUTOFF_MOVE     = 4.0
VAL_CHRS           = {'19', '21', 'X'}
TEST_CHRS          = {'7', '13'}
PADDED_SEQ_LEN     = 600
DUPLICATION_CUTOFF = 0.5
INSERT_LEN         = 200

# boda's one-hot channel order; the encoder and the decoder below both use it, so
# decoding is the exact inverse of encoding
assert constants.STANDARD_NT == ['A', 'C', 'G', 'T'], f'unexpected channel order {constants.STANDARD_NT}'


def batch2list_withN(batch, vocab_list=constants.STANDARD_NT):
    """batch2list, but an all-zero column decodes to 'N' instead of argmax's 'A'."""
    assert batch.shape[1] == len(vocab_list), f'channel dim {batch.shape[1]} != vocab {len(vocab_list)}'
    idx_tensor = torch.argmax(batch, dim=1)
    is_n = batch.sum(dim=1) == 0
    for seq_idx in range(batch.shape[0]):
        yield ''.join('N' if n else vocab_list[i]
                      for i, n in zip(idx_tensor[seq_idx].numpy(), is_n[seq_idx].numpy()))


def encode(seqs, chunk=50000):
    """Vectorized equivalent of utils.row_dna2tensor_withN over a list of equal-length
    strings. N (or any base outside STANDARD_NT) becomes an all-zero column.
    Returns (N, 4, L) float32."""
    lut = np.full(256, 4, dtype=np.uint8)
    for i, base in enumerate(constants.STANDARD_NT):
        lut[ord(base)] = i
    out = []
    for start in range(0, len(seqs), chunk):
        block = seqs[start:start + chunk]
        # (n, L) raw bytes -> (n, L) vocab indices, 4 marking N
        arr = np.frombuffer(''.join(block).encode(), dtype=np.uint8).reshape(len(block), -1)
        idx = lut[arr]
        # one-hot over 5 classes, then drop class 4 so N is left as all-zeros
        oh = np.zeros((idx.size, 5), dtype=np.float32)
        oh[np.arange(idx.size), idx.ravel()] = 1.0
        oh = oh.reshape(idx.shape[0], idx.shape[1], 5)[..., :4]
        out.append(torch.from_numpy(np.ascontiguousarray(oh.transpose(0, 2, 1))))
    return torch.cat(out)


# ---------------------------------------------------------------------------
# preprocessing, in place of MPRA_DataModule.setup(). Same cutoffs, but a failing
# measurement masks that one assay instead of removing the whole row, so every
# input row survives to the output with NA standing in for what was filtered.
# ---------------------------------------------------------------------------

# every row of the source table, no dropna: each row has 1-3 of the three assays
df = pd.read_csv(DATAFILE, sep=',', low_memory=False)
n_in = len(df)
print(f'read {n_in} rows')
assert df['sequence'].notna().all(), 'a row has no sequence'
assert (df['sequence'].str.len() == INSERT_LEN).all(), 'inserts are not all 200bp'
# activity and stderr are observed together, so masking one implies the other
for a, s in zip(ACT, SE):
    assert (df[a].isna() == df[s].isna()).all(), f'{a}/{s} null patterns differ'
print('observed measurements in:  ' + ', '.join(f'{a}={df[a].notna().sum()}' for a in ACT))

# a noisy measurement masks that one assay; the row keeps its clean assays
for a, s in zip(ACT, SE):
    noisy = (df[s] > STDERR_THRESHOLD).fillna(False)
    df.loc[noisy, [a, s]] = np.nan
    print(f'{a}: masked {int(noisy.sum())} measurements with stderr > {STDERR_THRESHOLD}')

# outliers masked the same way. mean/std skip NaN, so each cutoff comes from that
# assay's observed values only; NaN compares False, so it is never an outlier
means, stds = df[ACT].mean(), df[ACT].std()
up_cut   = means + stds * STD_MULTIPLE_CUT + UP_CUTOFF_MOVE
down_cut = means - stds * STD_MULTIPLE_CUT
extreme = (df[ACT] > up_cut) | (df[ACT] < down_cut)
df[ACT] = df[ACT].mask(extreme)
for a in ACT:
    print(f'{a}: masked {int(extreme[a].sum())} outliers outside '
          f'[{down_cut[a]:.2f}, {up_cut[a]:.2f}]')

assert len(df) == n_in, f'masking changed row count: {n_in} -> {len(df)}'
print('observed measurements out: ' + ', '.join(f'{a}={df[a].notna().sum()}' for a in ACT))

# a row whose every assay is NA carries no training signal, so it is dropped here.
# This is the only place rows leave the table.
n_empty = int(df[ACT].isna().all(axis=1).sum())
df = df[df[ACT].notna().any(axis=1)].reset_index(drop=True)
print(f'dropped {n_empty} rows with no measurement in any assay -> {len(df)} rows')
assert len(df) == n_in - n_empty, f'drop removed the wrong count: {n_in} - {n_empty} != {len(df)}'
assert df[ACT].notna().any(axis=1).all(), 'an all-NA row survived the drop'

# pad to 600bp with the lentiviral flanks. Every insert is the same length, so the
# pad is a constant prefix and suffix; derive it from boda's own padding function
# rather than slicing the flanks here, then check it against that function directly.
probe = utils.row_pad_sequence({'sequence': 'A' * INSERT_LEN},
                               in_column_name='sequence',
                               padded_seq_len=PADDED_SEQ_LEN,
                               upStreamSeq=MPRA_LENTI_UPSTREAM,
                               downStreamSeq=MPRA_LENTI_DOWNSTREAM)
left_pad  = probe[:(PADDED_SEQ_LEN - INSERT_LEN) // 2]
right_pad = probe[(PADDED_SEQ_LEN + INSERT_LEN) // 2:]
df['padded_seq'] = left_pad + df['sequence'] + right_pad
assert (df['padded_seq'].str.len() == PADDED_SEQ_LEN).all(), 'padding produced a wrong length'
# spot-check the constant pad against boda's per-row padding
for i in df.sample(20, random_state=0).index:
    expected = utils.row_pad_sequence(df.loc[i], in_column_name='sequence',
                                      padded_seq_len=PADDED_SEQ_LEN,
                                      upStreamSeq=MPRA_LENTI_UPSTREAM,
                                      downStreamSeq=MPRA_LENTI_DOWNSTREAM)
    assert df.loc[i, 'padded_seq'] == expected, f'constant pad differs from row_pad_sequence at row {i}'

# Recover chr for the handful of rows that lost it but do carry a real human locus
# in their ID. Two ID families must be excluded or they would be assigned a
# chromosome they do not come from:
#   SLEA      a designed oligo: an hg18 scaffold with TF motifs implanted into it,
#             so the chr2/chr9 coordinate in the ID is the backbone, not this sequence
#   mm9       a mouse positive control; its chr19 token is mouse chr19
# Everything else with no chr (shuffled controls, UWpeak, ENSG promoters) has no
# human locus to recover and stays null.
# Builds are mixed across these IDs (hg18, hg19, mm9) and the chr column's own build
# is not recorded anywhere in the file, so only the chromosome is taken, never the
# coordinates: chromosome assignment is stable across human builds, positions are not.
COORD_RE = re.compile(r'chr([0-9XYM]+):\d+-\d+')

def chr_from_id(seq_id):
    if 'SLEA' in seq_id or 'mm9' in seq_id:
        return None
    found = COORD_RE.findall(seq_id)
    return found[-1] if found else None

chrs = df['chr'].astype('string')
missing = chrs.isna()
recovered = df.loc[missing, 'ID'].map(chr_from_id)
# the same rule reproduces the chr column exactly on rows that still have one
check = df.loc[~missing].copy()
check['from_id'] = check['ID'].map(chr_from_id)
check = check[check['from_id'].notna()]
assert (check['from_id'] == check['chr'].astype(str)).all(), \
    'ID-derived chr disagrees with the chr column on rows where both exist'
print(f'chr recovery rule verified on {len(check)} rows that already had a chr')

chrs.loc[missing] = recovered.astype('string')
n_recovered = int(recovered.notna().sum())
print(f'recovered chr for {n_recovered} rows: ' + str(dict(recovered.value_counts())))

# split by chromosome. Rows still without a chr have no human locus, so they cannot
# be held out and cannot leak; they go to train.
is_val  = chrs.isin(VAL_CHRS).fillna(False)
is_test = chrs.isin(TEST_CHRS).fillna(False)
is_train = ~(is_val | is_test)
assert int(is_train.sum() + is_val.sum() + is_test.sum()) == len(df), 'splits do not partition the rows'
print(f'no-chr rows sent to train: {int(chrs.isna().sum())}')

# one-hot encode, checked against boda's own per-row encoder
sample = list(df['padded_seq'].iloc[:8])
assert torch.equal(encode(sample),
                   torch.stack([utils.row_dna2tensor_withN({'s': s}, in_column_name='s') for s in sample])), \
    'vectorized encoder disagrees with utils.row_dna2tensor_withN'

datasets = {}
for split, mask in [('train', is_train), ('val', is_val), ('test', is_test)]:
    part = df[mask]
    seq_tensor = encode(list(part['padded_seq']))
    act_tensor = torch.from_numpy(part[ACT].to_numpy(np.float32))
    assert seq_tensor.shape[0] == act_tensor.shape[0] == len(part), \
        f'{split}: tensor rows {seq_tensor.shape[0]}/{act_tensor.shape[0]} != {len(part)}'
    if split == 'train':
        # sort key for duplication: a row's best observed activity. NaN would sort to
        # the front of a descending sort and get duplicated in place of the real top
        # rows, so unobserved assays are pushed to -inf instead.
        sort_tensor = torch.nan_to_num(act_tensor, nan=-np.inf).max(dim=-1).values
        assert not torch.isnan(sort_tensor).any(), 'NaN survived into the duplication sort key'
        datasets[split] = DNAActivityDataset(seq_tensor, act_tensor,
                                             sort_tensor=sort_tensor,
                                             duplication_cutoff=DUPLICATION_CUTOFF,
                                             use_reverse_complements=True)
    else:
        datasets[split] = torch.utils.data.TensorDataset(seq_tensor, act_tensor)
    print(f'{split}: {len(part)} rows -> dataset of {len(datasets[split])}')

# read every split back out through the dataset, so the rows, duplicates and reverse
# complements are exactly the ones a model would be trained on
# train: asking for item i returns a row or its reverse complement, and the top-activity
#   rows come around twice. Rows are sorted by activity, highest first:
#   item 0 = row 0, item 1 = row 0 reverse complemented, item 2 = row 1, ...
# val / test: every row once, forward only
split_dfs = []
for split in ['train', 'val', 'test']:
    dataset = datasets[split]
    seqs, activities = [], []

    # the DataLoader asks the dataset for items 0, 1, 2, ... in order (shuffle=False)
    # and stacks 10000 of them at a time:
    #   dna: (10000, 4, 600) one-hot sequences, channels in A, C, G, T order
    #   act: (10000, 3) log2FC for K562, HepG2, WTC11
    for dna, act in torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False):

        # one-hot -> string. For one 4bp sequence:
        #   dna[b] = [[1, 0, 0, 0],    A channel (STANDARD_NT[0])
        #             [0, 0, 1, 0],    C channel (STANDARD_NT[1])
        #             [0, 1, 0, 0],    G channel (STANDARD_NT[2])
        #             [0, 0, 0, 1]]    T channel (STANDARD_NT[3])
        #   argmax over the channels picks the 1 at each position -> [0, 2, 1, 3]
        #   STANDARD_NT[0], [2], [1], [3] -> 'A', 'G', 'C', 'T' -> joined into 'AGCT'
        # the 15bp barcode slot in MPRA_LENTI_DOWNSTREAM encodes to an all-zero column,
        # which batch2list_withN decodes back to 'N'
        # dna: (batch, 4, 600) -> one 600bp string per row
        seqs.extend(batch2list_withN(dna))

        # keep this batch's activities; row b lines up with seqs for the same batch row
        activities.append(act.numpy())

    # (N, 3) activities, with the sequence and the split name as extra columns. A
    # measurement that was never taken, or was masked as noisy or extreme, is NA:
    #   split  sequence    K562_log2FC  HepG2_log2FC  WTC11_log2FC
    #   train  AGCT...     5.21         NA            5.02
    out = pd.DataFrame(np.concatenate(activities), columns=ACT)
    out.insert(0, 'sequence', seqs)
    out.insert(0, 'split', split)

    assert len(out) == len(dataset), f'{split}: wrote {len(out)} rows, dataset has {len(dataset)}'
    assert (out['sequence'].str.len() == PADDED_SEQ_LEN).all(), f'{split}: a sequence is not 600bp'
    print(f'{split}: {len(out)} rows, ' + ', '.join(f'{a}={out[a].notna().sum()}' for a in ACT))
    split_dfs.append(out)

# one file, all splits stacked: train rows first, then val, then test
all_df = pd.concat(split_dfs, ignore_index=True)
assert len(all_df) == sum(len(d) for d in split_dfs), \
    f'concat changed row count: {sum(len(d) for d in split_dfs)} -> {len(all_df)}'
# every row carried a sequence in, so every row carries one out
assert all_df['sequence'].notna().all(), 'a row lost its sequence'

out_path = OUT_DIR / 'mpralegnet_all_data_filtered_revcomp_preprocessed.tsv'
all_df.to_csv(out_path, sep='\t', index=False, na_rep='NA')
print(f'all splits: {len(all_df)} rows -> {out_path}')
