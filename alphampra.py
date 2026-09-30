# MPRA helpers (from alphagenome_FT_MPRA: https://github.com/Al-Murphy/alphagenome_FT_MPRA)

# all code is mostly adapted from alphagenome_FT_MPRA/scripts/finetune_episomal_mpra.py

from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import jax
import jax.numpy as jnp
import haiku as hk
import pandas as pd
from alphagenome_ft import CustomHead
from alphagenome_research.model import dna_model, layers
try:
    import optax
except ImportError:
    optax = None

import os
import urllib.request
import pandas as pd

from pathlib import Path
from alphagenome.models import dna_output
from alphagenome_ft import HeadConfig, HeadType, register_custom_head, create_model_with_custom_heads

from alphagenome_research.model import layers
import numpy as np

from alphagenome_ft import create_optimizer, load_checkpoint
from alphagenome_ft_mpra.training import train as ag_train
import fast_train

import matplotlib
matplotlib.use('Agg')   # runs headless under sbatch, so set the backend before pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from parse_args import build_config


def create_model(config):
    # one head per cell type (config.head_names: 'mpra_head', or 'mpra_head_<cell>' for each of
    # config.heads), all with the same architecture, all on the shared encoder output
    for head_name in config.head_names:
        register_custom_head(
            head_name,
            EncoderMPRAHead,
            HeadConfig(
                type=HeadType.GENOME_TRACKS,
                output_type=dna_output.OutputType.RNA_SEQ,
                num_tracks=1,
                metadata={
                    "center_bp": 256,
                    "pooling_type": "flatten",
                    "nl_size": [512, 512],
                    "do": 0.1,
                    "activation": "relu",
                },
            ),
        )

    # Create the AlphaGenome model (encoder + custom heads only).
    model = create_model_with_custom_heads(
        "all_folds", custom_heads=config.head_names, use_encoder_output=True, init_seq_len=CONSTRUCT_LENGTH, checkpoint_path="/home/go274/scratch_pi_skr2/go274/coda_data/claude_playground/alphagenome_weights/all_folds",
    )
    # ag_train (single head only) reads the heads-only flag this sets for stage 1; fast_train
    # chooses what to train itself
    if len(config.head_names) == 1:
        model.freeze_except_head(config.head_names[0])
    print("Model ready.")

    loss_fn = model.create_loss_fn_for_head(config.head_names[0])

    optimizer = create_optimizer(
        model._params,
        trainable_head_names=tuple(config.head_names),
        learning_rate=1e-3,
        weight_decay=1e-4,
        heads_only=True,
    )
    opt_state = optimizer.init(model._params)

    return model


def train_step(params, state, opt_state, batch_sequences, batch_targets):
    def loss_inner(current_params):
        preds_dict = model._predict(
            current_params,
            state,
            batch_sequences,
            jnp.zeros((batch_sequences.shape[0],), dtype=jnp.int32),  # organism_index
            negative_strand_mask=jnp.zeros((batch_sequences.shape[0],), 
                                           dtype=bool),
            strand_reindexing=model._metadata[
                next(iter(model._metadata))].strand_reindexing,
        )
        preds = preds_dict["mpra_head"]
        loss_dict = loss_fn(
            preds,
            {"targets": batch_targets, "organism_index": None},
        )
        return loss_dict["loss"]

    loss, grads = jax.value_and_grad(loss_inner)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

class MalinoisMPRADataset:
    def __init__(self, model, split, cell_types,
                 path_to_data="/home/go274/scratch_pi_skr2/go274/manual_MPRA_models/baseline_CODA/malinois_all_data_filtered_revcomp_preprocessed.tsv",
                 subset_frac=1.0, seed=42):
        # one <cell>_log2FC label column per cell type, in the order given (= head order)
        assert split in ["train", "val", "test"] and all(c in ["K562", "HepG2", "SKNSH"] for c in cell_types)
        self.model, self.label_columns = model, [f"{c}_log2FC" for c in cell_types]
        data = pd.read_csv(path_to_data, sep="\t", usecols=["split", "sequence", *self.label_columns])
        self.data = data[data["split"] == split].reset_index(drop=True)
        if subset_frac < 1.0:
            self.data = self.data.sample(frac=subset_frac, random_state=seed).reset_index(drop=True)
        assert len(self.data) > 0, f"no rows for split {split}"
        assert (self.data["sequence"].str.len() == 600).all(), "expected 600bp padded sequences"
        print(f"Loaded {len(self.data)} {split} rows for {', '.join(cell_types)}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {"seq": jnp.array(self.model._one_hot_encoder.encode(row["sequence"])),  # (600, 4)
                "y": row[self.label_columns].to_numpy(np.float32),   # (n_cells,)
                "organism_index": jnp.array([0])}  # 0 = human


class MPRADataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, rng_key=None):
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle
        self.rng_key = rng_key if rng_key is not None else (jax.random.PRNGKey(42) if shuffle else None)
    def __iter__(self):
        indices = jnp.arange(len(self.dataset))
        if self.shuffle:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            indices = jax.random.permutation(subkey, indices)
        num_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(self.dataset))
            batch_samples = [self.dataset[int(k)] for k in indices[start:end].tolist()]
            yield self._stack_batch(batch_samples)
    def _stack_batch(self, samples):
        seqs = np.stack([s["seq"] for s in samples])    
        assert seqs.shape[1:] == (600, 4), f"expected (B, 600, 4), got {seqs.shape}"
        return {"seq": jnp.asarray(seqs),
            "y": jnp.asarray(np.stack([s["y"] for s in samples])),   # (B, n_cells)
            "organism_index": jnp.zeros(len(samples), dtype=jnp.int32)}        # all human
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def validate(model, dataloader, loss_fn, head_name="mpra_head"):
    total_loss, total_pearson, num_batches = 0.0, 0.0, 0
    strand_reindex = jax.device_put(model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing, model._device_context._device)
    for batch in dataloader:
        with model._device_context:
            predictions = model._predict(model._params, model._state, batch["seq"], batch["organism_index"], negative_strand_mask=jnp.zeros(batch["seq"].shape[0], dtype=bool), strand_reindexing=strand_reindex, requested_outputs=tuple(dna_output.OutputType))
        loss_dict = loss_fn(predictions[head_name], {"targets": batch["y"]})
        total_loss += float(loss_dict["loss"])
        total_pearson += float(loss_dict.get("pearson_corr", 0.0))
        num_batches += 1
    n = num_batches or 1
    return {"val_loss": total_loss / n, "val_pearson": total_pearson / n}

def test(model, dataloader, loss_fn, head_name="mpra_head"):
    total_loss, total_pearson, num_batches = 0.0, 0.0, 0
    strand_reindex = jax.device_put(model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing, model._device_context._device)
    for batch in dataloader:
        with model._device_context:
            predictions = model._predict(model._params, model._state, batch["seq"], batch["organism_index"], negative_strand_mask=jnp.zeros(batch["seq"].shape[0], dtype=bool), strand_reindexing=strand_reindex, requested_outputs=tuple(dna_output.OutputType))
        loss_dict = loss_fn(predictions[head_name], {"targets": batch["y"]})
        total_loss += float(loss_dict["loss"])
        total_pearson += float(loss_dict.get("pearson_corr", 0.0))
        num_batches += 1
    n = num_batches or 1
    return {"test_loss": total_loss / n, "test_pearson": total_pearson / n}

def train(model, train_loader, val_loader=None, test_loader=None, num_epochs=10, learning_rate=1e-3, head_name="mpra_head", checkpoint_dir=None, save_minimal_model=True, early_stopping_patience=5, use_wandb=False, **kwargs):
    if optax is None:
        raise ImportError("optax is required. pip install optax")
    rng_key = jax.random.PRNGKey(42)
    loss_fn = model.create_loss_fn_for_head(head_name)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(model._params)
    strand_reindex = jax.device_put(model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing, model._device_context._device)
    # Single gradient function (JIT once, reuse every batch) like finetune_mpra.py
    def _seq_loss_fn(params, seq_batch, organism_index, targets, step_key):
        with model._device_context:
            preds = model._predict(params, model._state, seq_batch, organism_index, negative_strand_mask=jnp.zeros(seq_batch.shape[0], dtype=bool), strand_reindexing=strand_reindex, requested_outputs=tuple(dna_output.OutputType), rng=step_key)
        loss_dict = loss_fn(preds[head_name], {"targets": targets})
        return loss_dict["loss"], loss_dict
    _grad_fn = jax.value_and_grad(_seq_loss_fn, has_aux=True)
    def step_fn(params, seq_batch, org_idx, targets, step_key):
        (loss, loss_dict), grads = _grad_fn(params, seq_batch, org_idx, targets, step_key)
        return grads, float(loss), loss_dict
    # Warmup: trigger JAX compilation before epoch loop (same as finetune_mpra.py)
    print("Warming up (JAX compile, may take a few minutes)...", flush=True)
    warmup_batch = next(iter(train_loader))
    rng_key, step_key = jax.random.split(rng_key)
    _ = step_fn(model._params, warmup_batch["seq"], warmup_batch["organism_index"], warmup_batch["y"], step_key)
    print("Warmup done. Starting epochs.", flush=True)
    history = {"train_loss": [], "train_pearson": [], "val_loss": [], "val_pearson": [], "test_loss": [], "test_pearson": []}
    best_val_loss, patience_counter = float("inf"), 0
    print(f"Starting training for {num_epochs} epochs...", flush=True)
    for epoch in range(num_epochs):
        train_losses, train_pearsons = [], []
        for batch in train_loader:
            rng_key, step_key = jax.random.split(rng_key)
            seq_batch, targets, org_idx = batch["seq"], batch["y"], batch["organism_index"]
            grads, loss, loss_dict = step_fn(model._params, seq_batch, org_idx, targets, step_key)
            updates, opt_state = optimizer.update(grads, opt_state, model._params)
            model._params = optax.apply_updates(model._params, updates)
            train_losses.append(loss)
            train_pearsons.append(float(loss_dict.get("pearson_corr", 0.0)))
        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
        avg_train_pearson = sum(train_pearsons) / len(train_pearsons) if train_pearsons else 0.0
        history["train_loss"].append(avg_train_loss)
        history["train_pearson"].append(avg_train_pearson)
        val_metrics = validate(model, val_loader, loss_fn, head_name) if val_loader else {"val_loss": float("inf"), "val_pearson": 0.0}
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_pearson"].append(val_metrics["val_pearson"])
        if test_loader:
            t = test(model, test_loader, loss_fn, head_name)
            history["test_loss"].append(t["test_loss"])
            history["test_pearson"].append(t["test_pearson"])
        print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.6f}, train_pearson={avg_train_pearson:.4f}, val_loss={val_metrics['val_loss']:.6f}, val_pearson={val_metrics['val_pearson']:.4f}", flush=True)
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss, patience_counter = val_metrics["val_loss"], 0
            if checkpoint_dir:
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                model.save_checkpoint(checkpoint_dir, save_full_model=False, save_minimal_model=save_minimal_model)
        else:
            patience_counter += 1
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}", flush=True)
                break
    return history


#custom head for MPRA - need to define a predict & loss function
class EncoderMPRAHead(CustomHead):
    #predefined by AlphaGenome
    ENCODER_RESOLUTION_BP = 128
    
    def __init__(self, *, name, output_type, num_tracks, num_organisms, metadata):
        super().__init__(name=name, num_tracks=num_tracks, output_type=output_type, num_organisms=num_organisms, metadata=metadata)
        center_bp = metadata.get('center_bp', 256) if metadata else 256
        nl_size = metadata.get('nl_size', 1024) if metadata else 1024
        self._hidden_sizes = [nl_size] if isinstance(nl_size, int) else nl_size
        self._do = metadata.get('do', None) if metadata else None
        self._center_window_positions = max(1, int(center_bp / self.ENCODER_RESOLUTION_BP))
        self._pooling_type = metadata.get('pooling_type', 'sum') if metadata else 'sum'
        assert self._pooling_type in ['sum', 'mean', 'max', 'center', 'flatten']
        self._activation = metadata.get('activation', 'relu') if metadata else 'relu'
        assert self._activation in ['relu', 'gelu']
    
    def predict(self, embeddings, organism_index, **kwargs):
        if not hasattr(embeddings, 'encoder_output') or embeddings.encoder_output is None:
            raise AttributeError("EncoderMPRAHead requires encoder_output. Use use_encoder_output=True.")
        x = embeddings.encoder_output
        #layer norm helped speed up convergence
        x = layers.LayerNorm(name='norm')(x)
        
        if self._pooling_type == 'flatten':
            x = x.reshape(x.shape[0], -1)
        
        for i, hidden_size in enumerate(self._hidden_sizes):
            x = hk.Linear(hidden_size, name=f'hidden_{i}')(x)
            if self._do is not None:
                try:
                    x = hk.dropout(hk.next_rng_key(), self._do, x)
                except (RuntimeError, ValueError, AttributeError):
                    pass
            x = jax.nn.gelu(x) if self._activation == 'gelu' else jax.nn.relu(x)
        
        if self._pooling_type == 'flatten':
            per_position_predictions = hk.Linear(self._num_tracks, name='output')(x)[:, None, :]
        else:
            per_position_predictions = hk.Linear(self._num_tracks, name='output')(x)
        return per_position_predictions
    
    def loss(self, predictions, batch):
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        seq_len = predictions.shape[1]
        if self._pooling_type == 'flatten':
            pred_values = predictions.squeeze(1)
        elif self._pooling_type == 'center':
            center_idx = seq_len // 2
            pred_values = jax.lax.dynamic_slice_in_dim(predictions, center_idx, 1, axis=1).squeeze(1)
        else:
            window_size = min(int(self._center_window_positions), seq_len)
            center_start = max((seq_len - window_size) // 2, 0)
            center_predictions = jax.lax.dynamic_slice_in_dim(predictions, center_start, window_size, axis=1)
            if self._pooling_type == 'mean':
                pred_values = jnp.mean(center_predictions, axis=1)
            elif self._pooling_type == 'max':
                pred_values = jnp.max(center_predictions, axis=1)
            else:
                pred_values = jnp.sum(center_predictions, axis=1)
        if targets.ndim == 1:
            targets = targets[:, None]
        mse_loss = jnp.mean((pred_values - targets) ** 2)
        pred_flat, targets_flat = pred_values.flatten(), targets.flatten()
        pearson_corr = jnp.corrcoef(pred_flat, targets_flat)[0, 1]
        
        return {'loss': mse_loss, 'mse': mse_loss, 'pearson_corr': pearson_corr}


def load_best_model(config):
    """Load the best checkpoint ag_train saved: stage 2 if it ran, otherwise stage 1.

    ag_train leaves the model at its last epoch, not its best, so the plot reads the
    saved best-validation checkpoint, as plot_test_results in grace_train_malinois.py
    reads the saved best-epoch .pt.
    """
    stage2_dir = config.checkpoint_dir / 'stage2'
    stage1_dir = config.checkpoint_dir / 'stage1'
    best_dir = stage2_dir if (stage2_dir / 'checkpoint').exists() else stage1_dir
    assert (best_dir / 'checkpoint').exists(), f"no saved checkpoint under {config.checkpoint_dir}"
    print(f'Loading best model from {best_dir}')
    # load_checkpoint needs the head registered first, which create_model has done
    return load_checkpoint(best_dir, base_checkpoint_path=str(config.base_checkpoint),
                           init_seq_len=config.construct_length)


def plot_test_results(config, test_loader):
    """Score the best saved model on every test row and plot predicted vs observed, one panel per head."""
    assert not test_loader.shuffle, "test_loader must be unshuffled so rows stay in file order"
    model = load_best_model(config)
    strand_reindex = jax.device_put(model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing,
                                    model._device_context._device)

    all_preds, all_labels = [], []
    for batch in test_loader:
        # no rng passed, so the head's dropout is off
        with model._device_context:
            predictions = model._predict(model._params, model._state, batch["seq"], batch["organism_index"],
                                         negative_strand_mask=jnp.zeros(batch["seq"].shape[0], dtype=bool),
                                         strand_reindexing=strand_reindex,
                                         requested_outputs=tuple(dna_output.OutputType))
        # flatten heads: each (batch, 1, 1) -> one column of (batch, n_heads); the model computes in
        # bfloat16, which pandas and seaborn cannot handle, so cast to float32 here
        n = batch["seq"].shape[0]
        batch_preds = [np.asarray(predictions[h], dtype=np.float32) for h in config.head_names]
        assert all(p.shape == (n, 1, 1) for p in batch_preds), f"unexpected head output shapes {[p.shape for p in batch_preds]}"
        all_preds.append(np.concatenate([p.reshape(n, 1) for p in batch_preds], axis=1))
        all_labels.append(np.asarray(batch["y"]))   # (batch, n_heads)

    # concatenate all batches -> shape (N_test, n_heads), columns in config.cell_types order
    Y = np.concatenate(all_preds)    # predicted log2FC
    X = np.concatenate(all_labels)   # observed log2FC
    assert X.shape == Y.shape == (len(test_loader.dataset), len(config.head_names)), \
        f"observed {X.shape} vs predicted {Y.shape}, test set has {len(test_loader.dataset)} rows"

    # one row per test sequence, in file order, for later comparisons against Malinois;
    # single-head keeps observed/predicted, multi-head gets <cell>_observed/<cell>_predicted
    preds_path = config.output_dir / f'{config.model_name}_test_predictions.tsv'
    if config.heads:
        columns = {f'{c}_{kind}': M[:, i] for i, c in enumerate(config.cell_types)
                   for kind, M in (('observed', X), ('predicted', Y))}
    else:
        columns = {'observed': X[:, 0], 'predicted': Y[:, 0]}
    pd.DataFrame(columns).to_csv(preds_path, sep='\t', index=False)
    print(f'Saved test predictions to {preds_path}')

    # ── plot: one panel per head ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(config.cell_types), figsize=(4 * len(config.cell_types), 4), squeeze=False)
    rs = [_plot_panel(axes[0, i], X[:, i], Y[:, i], cell_name) for i, cell_name in enumerate(config.cell_types)]

    r_text = f'{rs[0]:.3f}' if len(rs) == 1 else f'{np.mean(rs):.3f} (mean over heads)'
    fig.suptitle(f'{config.model_name} - test set\nPearson r = {r_text}', fontsize=11, y=1.05)
    fig.tight_layout()
    fig_save_name = config.output_dir / f'{config.model_name}_test_predictions.png'
    fig.savefig(fig_save_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved test scatter to {fig_save_name}')
    print('Test Pearson r: ' + ', '.join(f'{c}={r:.4f}' for c, r in zip(config.cell_types, rs)))


def _plot_panel(ax, x, y, cell_name):
    """Observed (x) vs predicted (y) density for one cell type; returns the Pearson r."""

    # shared axis limits: min/max of both arrays + 5% margin
    all_vals = np.concatenate([x, y])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = (vmax - vmin) * 0.05
    lim = (vmin - margin, vmax + margin)

    # 2D density histogram (viridis colormap, same as make_replicate)
    sns.histplot(x=x, y=y, ax=ax,
                 bins=50, binrange=(lim, lim),
                 cmap='viridis', cbar=False, pthresh=0.02)

    # identity line: perfect model would have all points here
    ax.plot(lim, lim, color='crimson', linewidth=0.9, linestyle='--', zorder=3)
    # zero reference lines
    ax.axhline(0, color='gray', linewidth=0.7, linestyle=':', zorder=2)
    ax.axvline(0, color='gray', linewidth=0.7, linestyle=':', zorder=2)

    ax.set_xlim(lim)
    ax.set_ylim(lim)

    # Pearson r + significance stars, over the whole test set rather than per batch
    r, pval = pearsonr(x, y)
    stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
    ax.text(0.05, 0.95, f'r = {r:.3f}{stars}',
            transform=ax.transAxes, fontsize=7, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    # bold label with observation count
    ax.text(0.5, 0.97, f'{cell_name} (n={len(x)})',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=8, fontweight='bold')

    ax.set_xlabel(f'{cell_name} observed log2FC', fontsize=8)
    ax.set_ylabel(f'{cell_name} predicted log2FC', fontsize=8)
    ax.tick_params(labelsize=6)
    return r


if __name__ == '__main__':
    config = build_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print('-' * 50)
    print(f'Model name  : {config.model_name}')
    print(f'Data        : {config.data}')
    print(f'Cell type   : ' + ', '.join(config.cell_types) + (f' (heads {config.head_names})' if config.heads else ''))
    print(f'Stage 1     : {config.num_epochs} epochs max, lr {config.learning_rate}, batch {config.batch_size}, subset {config.subset_frac}')
    print(f'Stage 2     : ' + (f'{config.second_stage_epochs} epochs max, lr {config.second_stage_lr}' if config.second_stage_lr else 'off'))
    print(f'Checkpoints : {config.checkpoint_dir}')
    print(f'Train loop  : ' + ('fast_train' if config.fast_train else 'ag_train'))
    print('-' * 50)

    # create_model reads this module-level name for init_seq_len
    CONSTRUCT_LENGTH = config.construct_length

    model = create_model(config)

    #### LOAD THE DATA #####
    train_dataset = MalinoisMPRADataset(
        model=model, path_to_data=str(config.data), cell_types=config.cell_types, split="train",
        subset_frac=config.subset_frac, seed=config.seed,
    )
    val_dataset = MalinoisMPRADataset(
        model=model, path_to_data=str(config.data), cell_types=config.cell_types, split="val",
        subset_frac=config.subset_frac, seed=config.seed,
    )
    test_dataset = MalinoisMPRADataset(
        model=model, path_to_data=str(config.data), cell_types=config.cell_types, split="test",
        subset_frac=config.subset_frac, seed=config.seed,
    )
    train_loader = MPRADataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = MPRADataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = MPRADataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    print("Train:", len(train_dataset), "Val:", len(val_dataset), "Test:", len(test_dataset))

    ##### TRAIN THE MODEL #####
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if config.fast_train:
        # jitted step over encoder + head(s) only; writes stage1/ and stage2/ like ag_train
        fast_train.train_two_stage(model, config, train_loader, val_loader, test_loader)
    else:
        history = ag_train(
            model, train_loader, val_loader=val_loader, test_loader=test_loader,
            num_epochs=config.num_epochs,                 # stage 1: head only, encoder frozen via optimizer masking
            learning_rate=config.learning_rate,
            second_stage_lr=config.second_stage_lr,       # stage 2: whole model, from stage 1's best checkpoint
            second_stage_epochs=config.second_stage_epochs,
            early_stopping_patience=config.early_stopping_patience,
            val_eval_frequency=config.val_eval_frequency, test_eval_frequency=config.val_eval_frequency,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            checkpoint_dir=str(config.checkpoint_dir),    # writes stage1/ and stage2/ subfolders
            save_minimal_model=True,
            wandb_config={"optimizer": "adam", "weight_decay": config.weight_decay},
            use_wandb=False,
        )

    ##### PLOT THE BEST MODEL ON THE TEST SET #####
    plot_test_results(config, test_loader)
