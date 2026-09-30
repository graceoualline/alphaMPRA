# alphaMPRA

README is generated and updated continually by Opus 5.5 based on contents of the repo. Please read with caution.

AlphaGenome's sequence encoder fine-tuned to predict episomal MPRA activity (log2FC) in K562, HepG2
and SK-N-SH, trained on the CODA / Gosai et al. 2024 data preprocessed exactly as Malinois sees it.
The point is a like-for-like comparison with Malinois: same sequences, same filtering, same
chromosome splits.

Two model layouts:

- **Single-task**: one model per cell type, one MPRA head each.
- **Multi-head**: one model, a shared encoder with one MPRA head per cell type, trained jointly
  (Malinois-style multi-task).

The model and head come from
[alphagenome_FT_MPRA](https://github.com/Al-Murphy/alphagenome_FT_MPRA)
(`scripts/finetune_episomal_mpra.py`), with the same head architecture and hyperparameters as its
episomal configs.

## Results

Test set: chr7 + chr13, after boda's filters (62,582 sequences). Pearson r between predicted and
observed log2FC over the whole test set, using the best-validation checkpoint.

| Model | K562 | HepG2 | SK-N-SH |
|---|---|---|---|
| Single-task (one model per cell type) | 0.9244 | 0.9267 | 0.9153 |
| Multi-head (one model, three heads) | 0.9241 | 0.9260 | 0.9164 |

The multi-head model matches the single-task models. It trained in 7 hours on one H100, where each
single-task model took about 2 days.

## Model

- **Input:** 600bp one-hot sequence: the 200bp oligo plus the MPRA plasmid flanks, as padded by
  boda.
- **Backbone:** AlphaGenome `all_folds` weights, sequence encoder only (no transformer or decoder).
  It outputs 128bp-resolution features.
- **Head (per cell type):** LayerNorm, then flatten, then Linear(512) + ReLU + dropout 0.1, then
  Linear(512) + ReLU + dropout 0.1, then Linear(1). This is `EncoderMPRAHead` in `alphampra.py`.
- **Loss:** MSE. Multi-head uses the mean of the heads' MSEs.
- **Training:**
  - stage 1 trains the head(s) only, with the encoder frozen (lr 1e-3);
  - stage 2 trains the encoder + head(s) from stage 1's best weights (lr 1e-5);
  - Adam with weight decay 1e-6, batch 64, validation 4 times per epoch;
  - early stopping after 5 epochs without a better validation loss.

## Data

`preprocess_malinois.py` runs boda's `MPRA_DataModule` on the Gosai supplementary table
(`DATA-Table_S2__MPRA_dataset.txt`) and writes
`malinois_all_data_filtered_revcomp_preprocessed.tsv`. Its columns are `split`, `sequence` and
`{K562,HepG2,SKNSH}_log2FC`. The script:

- drops rows with lfcSE > 1 in any cell type, and outliers beyond 6 SD;
- pads each oligo with the MPRA flanks to 600bp;
- splits by chromosome: test chr7 + chr13, val chr19 + chr21 + chrX, train the rest;
- in train only, duplicates high-activity rows (cutoff 0.5) and adds every row's reverse
  complement.

This gives 1,864,208 train, 58,810 val and 62,582 test rows. The TSV (1.2 GB) is not in the repo.

## Setup

Requirements:

- A conda env (`alphampra`) with JAX (GPU), haiku, optax, pandas, seaborn and scipy.
- [alphagenome_ft](https://github.com/genomicsxai/alphagenome_ft) and `alphagenome_ft_mpra` (from
  alphagenome_FT_MPRA) installed in that env. Both are git-ignored here; clone them separately.
- A local copy of the AlphaGenome `all_folds` weights. Set `base_checkpoint` in the YAML to point
  at it.
- boda, only for `preprocess_malinois.py`.

The YAMLs contain absolute paths on the Yale Bouchet cluster. Edit `data`, `base_checkpoint` and
`output_dir` for another machine.

## Training

Each run is one YAML config. `submit_training.sh` checks the config (bad keys, missing files) before
queueing it on Slurm:

```bash
./submit_training.sh K562_alphampra.yaml                                    # single-task
TIME=2-00:00:00 GPU_TYPE=h100 ./submit_training.sh multihead_alphampra.yaml  # multi-head
```

Environment overrides: `PARTITION` (default `priority_gpu`), `ACCOUNT`, `GPUS`, `CPUS` (10), `MEM`
(128G), `TIME` (2 days), `GPU_TYPE` (default: any card), and `DRY_RUN=1` to print the job script
without submitting.

To run without Slurm, use `python alphampra.py --config <yaml>`. Command-line flags override the
YAML, e.g. `--cell-type`, `--subset-frac`, `--num-epochs`, `--heads K562 HepG2 SKNSH` and
`--fast-train`.

Config keys that choose the model and the training loop:

| Key | Meaning |
|---|---|
| `cell_type: K562` | single-task model for one cell type |
| `heads: [K562, HepG2, SKNSH]` | multi-head model, one head per listed cell type (use instead of `cell_type`) |
| `fast_train: true` | train with `fast_train.py` instead of alphagenome_ft_mpra's `train`. Required for `heads` |

### Two training loops

`alphampra.py` can train with either loop. They run the same model, loss, optimizer and two-stage
schedule:

- **ag_train**: `alphagenome_ft_mpra.training.train`, the upstream loop. Single head only. About
  0.56 s per step, so about 5h per epoch on an H100.
- **fast_train**: `fast_train.py`.
  - It compiles forward, backward and the Adam update into one `jax.jit` step.
  - It computes gradients and optimizer state only for the trainable modules: 94M parameters in
    stage 2 instead of all 455M AlphaGenome parameters.
  - Epochs are about 36 minutes for the three-head model.

Differences in what fast_train reports:

- val/test Pearson is computed over the whole split, not averaged over batches of 64;
- test is evaluated once per epoch;
- `train_pearson` is the mean over training batches, not an extra pass over the train set.

`optimize_training.md` has the timing measurements behind this.

Smoke configs (`smoke.yaml`, `smoke_fast.yaml`, `smoke_multi.yaml`) run 2% of the data for 1 + 1
epochs. On an H100 the single-head smoke test takes 19 minutes with ag_train and 4 minutes with
fast_train.

## Outputs

Under `output_dir`:

```
checkpoints/<model_name>/stage1/   best stage 1 checkpoint (heads only trained)
checkpoints/<model_name>/stage2/   best stage 2 checkpoint (the final model)
<model_name>_test_predictions.tsv  one row per test sequence, in file order
<model_name>_test_predictions.png  predicted vs observed, one panel per head
```

The predictions TSV has `observed` / `predicted` columns for single-task models and
`<cell>_observed` / `<cell>_predicted` for multi-head. Checkpoints are git-ignored.

## Loading a trained model

A checkpoint is the whole `stage2/` directory. Register the heads under their training names
first, then load:

```python
import sys
sys.path.insert(0, "/path/to/alphaMPRA")
import jax, jax.numpy as jnp, numpy as np
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_ft import HeadConfig, HeadType, register_custom_head, load_checkpoint
from alphampra import EncoderMPRAHead

STAGE2 = ".../multihead_episomal_alphampra/checkpoints/episomal_alphampra_multihead/stage2"
BASE = ".../alphagenome_weights/all_folds"
CELLS = ["K562", "HepG2", "SKNSH"]
HEADS = [f"mpra_head_{c}" for c in CELLS]   # a single-task model has one head, "mpra_head"

for head in HEADS:
    register_custom_head(head, EncoderMPRAHead, HeadConfig(
        type=HeadType.GENOME_TRACKS, output_type=dna_output.OutputType.RNA_SEQ, num_tracks=1,
        metadata={"center_bp": 256, "pooling_type": "flatten", "nl_size": [512, 512], "do": 0.1, "activation": "relu"}))
model = load_checkpoint(STAGE2, base_checkpoint_path=BASE, init_seq_len=600)

strand_reindex = jax.device_put(model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing,
                                model._device_context._device)

def predict(seqs):
    """list of 600bp flank-padded strings -> (N, n_heads) predicted log2FC, columns in CELLS order"""
    x = jnp.asarray(np.stack([model._one_hot_encoder.encode(s) for s in seqs]))
    with model._device_context:
        out = model._predict(model._params, model._state, x, jnp.zeros(len(seqs), dtype=jnp.int32),
                             negative_strand_mask=jnp.zeros(len(seqs), dtype=bool),
                             strand_reindexing=strand_reindex, requested_outputs=tuple(dna_output.OutputType))
    return np.concatenate([np.asarray(out[h], dtype=np.float32).reshape(-1, 1) for h in HEADS], axis=1)
```

Inputs must be 600bp and padded with the same flanks as the training TSV. Predict in chunks of a
few hundred sequences.

## Files

| File | Purpose |
|---|---|
| `alphampra.py` | model, head, data loading, training entry point, test-set plot |
| `fast_train.py` | jitted two-stage training loop, single- or multi-head |
| `parse_args.py` | YAML + command-line config and its checks |
| `submit_training.sh` | validate a config and submit it to Slurm |
| `preprocess_malinois.py` | build the training TSV with boda |
| `*_alphampra.yaml`, `multihead_alphampra.yaml` | full training configs |
| `smoke*.yaml` | 2%-subset smoke-test configs |
| `optimize_training.md` | where training time went, and the speedups |
| `load_published_alphampra.ipynb` | loads the published alphagenome_FT_MPRA Gosai models (K562, HepG2, SK-N-SH) and scores them on the chr7 + chr13 test set |
| `individual_episomal_alphampra/`, `multihead_episomal_alphampra/` | test-set plots (checkpoints git-ignored) |
