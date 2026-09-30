import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

# Every setting for one AlphaGenome MPRA run: one model, one cell type.
#
# The data file is already preprocessed by baseline_CODA/preprocess_malinois.py:
# filtered, flank-padded to 600bp, split by chromosome, and with the reverse
# complements and duplicated top rows written into train. So there is nothing to
# filter or augment here, only which column to model and how to train.
#
# Training defaults are the alphagenome_FT_MPRA episomal settings
# (configs/episomal_{K562,HepG2,SKNSH}.json, identical across the three cells).
# The head's settings live in create_model in alphampra.py.

DEFAULTS = {
    'data': None,
    'cell_type': None,
    'model_name': None,
    'base_checkpoint': None,
    'output_dir': '.',
    'construct_length': 600,
    # data loading
    'batch_size': 64,
    'subset_frac': 1.0,
    'seed': 42,
    # stage 1: head only, encoder frozen
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-6,
    'early_stopping_patience': 5,
    'val_eval_frequency': 4,
    # stage 2: whole model, from stage 1's best checkpoint
    'second_stage_lr': 1e-5,
    'second_stage_epochs': 50,
    'gradient_accumulation_steps': 1,
    # training loop: False = alphagenome_ft_mpra ag_train, True = fast_train.py
    'fast_train': False,
    # several cell types, one mpra head each on the shared encoder, trained jointly (fast_train only).
    # Set this or cell_type, not both
    'heads': None,
}
REQUIRED = ('data', 'model_name', 'base_checkpoint')
CELL_TYPES = ('K562', 'HepG2', 'SKNSH')


@dataclass
class Config:
    data: Path # boda-preprocessed tsv with split, sequence and <cell>_log2FC columns
    cell_type: str # which <cell>_log2FC column a single-head model predicts; None when heads is set
    model_name: str # names the checkpoint folder and the test plot
    base_checkpoint: Path # local AlphaGenome all_folds weights, used to reload the best checkpoint
    output_dir: Path # checkpoints/<model_name>/ and the test plot land here
    construct_length: int # padded sequence length; fixes the flatten head's input size
    batch_size: int # sequences per gradient step
    subset_frac: float # fraction of each split kept, for smoke runs
    seed: int # seed for the subset_frac sample
    num_epochs: int # stage 1 maximum epochs
    learning_rate: float # stage 1 learning rate
    weight_decay: float # Adam weight decay, both stages
    early_stopping_patience: int # counted in validation checks, not epochs
    val_eval_frequency: int # validation checks per epoch
    second_stage_lr: float # stage 2 learning rate; None trains stage 1 only
    second_stage_epochs: int # stage 2 maximum epochs
    gradient_accumulation_steps: int # raise if stage 2 runs out of GPU memory; keeps the effective batch
    fast_train: bool # use fast_train.py (jitted step, encoder + head only) instead of ag_train
    heads: list # cell types for a multi-head model, e.g. [K562, HepG2, SKNSH]; None for one head

    @property
    def cell_types(self):
        """Cell type of each head, in head order."""
        return list(self.heads) if self.heads else [self.cell_type]

    @property
    def head_names(self):
        """'mpra_head' for a single-head model, so its checkpoints match ag_train's;
        'mpra_head_<cell>' per head otherwise."""
        return [f'mpra_head_{c}' for c in self.heads] if self.heads else ['mpra_head']

    @property
    def checkpoint_dir(self):
        """ag_train writes stage1/ and stage2/ under here."""
        return self.output_dir / 'checkpoints' / self.model_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Finetune AlphaGenome on the boda-preprocessed MPRA table, one cell type per run.\n

    Example commands:\n
    python alphampra.py --config episomal_alphampra.yaml\n
    python alphampra.py --config episomal_alphampra.yaml --cell-type K562 -n episomal_alphampra_K562\n
    # smoke run\n
    python alphampra.py --config episomal_alphampra.yaml --subset-frac 0.05 --num-epochs 1 -n smoke\n
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="YAML config; command-line values override it")
    parser.add_argument("-n", "--model-name", dest="model_name", help="Name for the checkpoint folder and plot")
    parser.add_argument("--cell-type", dest="cell_type", choices=CELL_TYPES, help="Cell type column to model")
    parser.add_argument("--subset-frac", dest="subset_frac", type=float, help="Fraction of each split to keep")
    parser.add_argument("--num-epochs", dest="num_epochs", type=int, help="Stage 1 maximum epochs")
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="Sequences per gradient step")
    parser.add_argument("--second-stage-lr", dest="second_stage_lr", type=float, help="Stage 2 learning rate")
    parser.add_argument("--gradient-accumulation-steps", dest="gradient_accumulation_steps", type=int,
                        help="Split each batch into this many pieces for stage 2 memory")
    parser.add_argument("--heads", nargs='+', choices=CELL_TYPES,
                        help="Cell types for a multi-head model, one head each (needs --fast-train)")
    parser.add_argument("--fast-train", dest="fast_train", action="store_true", default=None,
                        help="Train with fast_train.py instead of ag_train")
    return parser.parse_args()


def build_config(args=None):
    """Defaults, then the config file, then the command line; check invariants; return a Config."""
    if args is None:
        args = parse_args()

    with open(args.config) as handle:
        config_data = yaml.safe_load(handle) or {}
    # A misspelled key would otherwise fall back to its default without a word
    unknown = set(config_data) - set(DEFAULTS)
    assert not unknown, f"unrecognized keys in {args.config}: {sorted(unknown)}"

    overrides = {key: value for key, value in vars(args).items() if key != 'config' and value is not None}
    merged = {**DEFAULTS, **config_data, **overrides}

    missing = [key for key in REQUIRED if merged[key] is None]
    assert not missing, f"missing required config values: {missing}"
    assert (merged['cell_type'] is None) != (merged['heads'] is None), \
        f"set exactly one of cell_type and heads, got {merged['cell_type']!r} and {merged['heads']!r}"
    if merged['heads'] is None:
        assert merged['cell_type'] in CELL_TYPES, f"cell_type must be one of {CELL_TYPES}, got {merged['cell_type']}"
    else:
        heads = merged['heads']
        assert heads and all(h in CELL_TYPES for h in heads) and len(set(heads)) == len(heads), \
            f"heads must be distinct cell types from {CELL_TYPES}, got {heads}"
        assert merged['fast_train'], "heads (multi-head training) needs fast_train: true"

    # Fail here rather than after the model has loaded
    data_path = Path(merged['data'])
    base_checkpoint = Path(merged['base_checkpoint'])
    assert data_path.is_file(), f"data file not found: {data_path}"
    assert base_checkpoint.is_dir(), f"base_checkpoint not found: {base_checkpoint}"

    assert 0.0 < merged['subset_frac'] <= 1.0, f"subset_frac must be in (0, 1]: {merged['subset_frac']}"
    assert merged['batch_size'] > 0 and merged['num_epochs'] > 0, \
        f"batch_size and num_epochs must be positive: {merged['batch_size']}, {merged['num_epochs']}"
    assert merged['learning_rate'] > 0, f"learning_rate must be positive: {merged['learning_rate']}"
    assert merged['second_stage_lr'] is None or merged['second_stage_lr'] > 0, \
        f"second_stage_lr must be positive or null: {merged['second_stage_lr']}"
    assert merged['gradient_accumulation_steps'] >= 1 and merged['batch_size'] % merged['gradient_accumulation_steps'] == 0, \
        f"batch_size {merged['batch_size']} must split evenly into {merged['gradient_accumulation_steps']} accumulation steps"
    assert not (merged['fast_train'] and merged['gradient_accumulation_steps'] != 1), \
        "fast_train does not implement gradient accumulation"

    merged.update(data=data_path, base_checkpoint=base_checkpoint, output_dir=Path(merged['output_dir']))
    return Config(**merged)
