"""Two-stage finetuning loop for the encoder-only MPRA model, with one jitted step per batch.

Same schedule as alphagenome_ft_mpra.training.train (ag_train):
  stage 1: mpra head(s) only, encoder frozen
  stage 2: sequence_encoder + mpra head(s), starting from stage 1's best weights
  validation val_eval_frequency times per epoch; a new best val loss saves a minimal checkpoint
  to <checkpoint_dir>/stage{1,2}; early stopping after early_stopping_patience epochs' worth of
  validation checks without improvement

Differences from ag_train:
  - gradients, optimizer state and updates cover only the trainable modules (stage 2: 75 arrays,
    94.2M params) instead of all 454.7M AlphaGenome params, and forward, backward and update run
    as one jitted step
  - val/test loss and Pearson are computed over the whole split, not averaged per batch
  - test is evaluated once per epoch, at the end
  - train_pearson is the mean of per-step batch Pearsons, not an extra forward pass over the
    whole train set

Several heads (config.heads, Malinois-style multi-task): one mpra head per cell type on the shared
encoder, each predicting its own <cell>_log2FC column. The training loss is the mean of the heads'
MSEs, and checkpoints are chosen on the mean val MSE. A single-head model is the one-head case.

Model, data and loaders come from alphampra.py. Batches carry y (B, n_heads), columns in head
order. A training loader that drops its short final batch lets the step compile for one shape.
"""
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_ft.optimizer_utils import create_optimizer

HEAD = "mpra_head"
ENCODER = "sequence_encoder"
LOG_EVERY = 500  # steps between progress lines; each one also bounds how far the host runs ahead


def split_params(params, prefixes):
    """Split the flat haiku params dict into (trainable, frozen) by module-name substring."""
    trainable = {k: v for k, v in params.items() if any(p in k for p in prefixes)}
    frozen = {k: v for k, v in params.items() if k not in trainable}
    assert len(trainable) + len(frozen) == len(params), \
        f"split lost modules: {len(trainable)} + {len(frozen)} != {len(params)}"
    return trainable, frozen


def build_fns(model, heads, optimizer):
    """Jitted train step and prediction function for `heads`, whose order matches the label columns."""
    loss_fns = [model.create_loss_fn_for_head(h) for h in heads]
    strand_reindex = jax.device_put(model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing,
                                    model._device_context._device)
    outputs = tuple(dna_output.OutputType)

    def predict(params, state, seq, org, rng):
        # seq (B, 600, 4) -> one (B, 1, 1) output per head; rng=None switches the heads' dropout off
        preds = model._predict(params, state, seq, org, requested_outputs=outputs,
                               negative_strand_mask=jnp.zeros(seq.shape[0], dtype=bool),
                               strand_reindexing=strand_reindex, rng=rng)
        return [preds[h] for h in heads]

    # trainable and opt_state are donated: their old buffers are reused for the new values
    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(trainable, opt_state, frozen, state, seq, org, y, rng):
        def loss_of(tr):
            # head i is scored against label column i, y (B, n_heads); loss = mean of the heads' MSEs
            outs = [fn(p, {"targets": y[:, i]})
                    for i, (fn, p) in enumerate(zip(loss_fns, predict({**frozen, **tr}, state, seq, org, rng)))]
            return jnp.mean(jnp.stack([o["loss"] for o in outs])), jnp.stack([o["pearson_corr"] for o in outs])
        (loss, pearson), grads = jax.value_and_grad(loss_of, has_aux=True)(trainable)
        updates, opt_state = optimizer.update(grads, opt_state, trainable)
        return optax.apply_updates(trainable, updates), opt_state, loss, pearson

    @jax.jit
    def predict_batch(params, state, seq, org):
        # n_heads x (B, 1, 1) -> (B, n_heads) float32
        return jnp.concatenate([p.reshape(-1, 1) for p in predict(params, state, seq, org, None)],
                               axis=1).astype(jnp.float32)

    return train_step, predict_batch


def evaluate(predict_batch, params, state, loader):
    """Mean-over-heads MSE and per-head Pearson (n_heads,) over a whole split. The short last batch
    is zero-padded to the full batch size so eval compiles once, then its padding rows are dropped."""
    preds, ys = [], []
    for batch in loader:
        n = batch["y"].shape[0]
        seq = batch["seq"]
        if n < loader.batch_size:
            seq = jnp.pad(seq, ((0, loader.batch_size - n), (0, 0), (0, 0)))
        preds.append(predict_batch(params, state, seq, jnp.zeros(seq.shape[0], dtype=jnp.int32))[:n])
        ys.append(batch["y"])
    p = np.asarray(jnp.concatenate(preds))   # (N, n_heads)
    y = np.asarray(jnp.concatenate(ys))      # (N, n_heads)
    assert p.shape == y.shape == (len(loader.dataset), y.shape[1]), \
        f"predictions {p.shape} vs labels {y.shape}, split has {len(loader.dataset)} rows"
    assert np.isfinite(p).all(), "non-finite predictions"
    mse = float(np.mean((p - y) ** 2))   # equal-sized columns, so this is the mean of per-head MSEs
    return mse, np.array([np.corrcoef(p[:, i], y[:, i])[0, 1] for i in range(y.shape[1])])


def fmt(cells, values):
    """'K562 0.9012, HepG2 0.9105' for per-head numbers."""
    return ", ".join(f"{c} {v:.4f}" for c, v in zip(cells, values))


def train_stage(model, heads, cells, stage, prefixes, lr, num_epochs, start_epoch, train_loader, val_loader,
                test_loader, weight_decay, val_eval_frequency, early_stopping_patience, checkpoint_dir, rng):
    """Train the modules matching `prefixes`; returns the best val loss and leaves model._params at
    the best weights."""
    trainable, frozen = split_params(model._params, prefixes)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(trainable))
    print(f"\n{'=' * 80}\n{stage}: training {len(trainable)} modules, {n_params:,d} params, lr {lr}\n{'=' * 80}",
          flush=True)

    optimizer = create_optimizer(trainable, trainable_head_names=tuple(heads), learning_rate=lr,
                                 weight_decay=weight_decay, heads_only=False, optimizer_type="adam")
    opt_state = optimizer.init(trainable)
    train_step, predict_batch = build_fns(model, heads, optimizer)
    state = model._state

    # validation after these batch counts in each epoch, as ag_train does
    n_batches = len(train_loader)
    interval = max(1, n_batches // val_eval_frequency)
    eval_points = sorted({min(i * interval, n_batches) for i in range(1, val_eval_frequency + 1)})
    patience_in_evals = early_stopping_patience * val_eval_frequency

    best_val, best_trainable, evals_since_best = float("inf"), None, 0
    stage_dir = Path(checkpoint_dir) / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        t0 = time.perf_counter()
        losses, pearsons = [], []   # device scalars / (n_heads,) vectors; read back only at sync points
        stop = False
        for step, batch in enumerate(train_loader, start=1):
            rng, key = jax.random.split(rng)
            trainable, opt_state, loss, pearson = train_step(
                trainable, opt_state, frozen, state, batch["seq"], batch["organism_index"], batch["y"], key)
            losses.append(loss)
            pearsons.append(pearson)

            if step % LOG_EVERY == 0:
                # float() waits for this step, so the host never queues more than LOG_EVERY steps
                recent = float(jnp.mean(jnp.stack(losses[-LOG_EVERY:])))
                assert np.isfinite(recent), f"non-finite train loss at epoch {epoch} step {step}"
                rate = step / (time.perf_counter() - t0)
                print(f"  epoch {epoch} step {step}/{n_batches}: loss {recent:.4f}, {rate:.1f} steps/s", flush=True)

            if step in eval_points:
                params = {**frozen, **trainable}
                val_loss, val_pearsons = evaluate(predict_batch, params, state, val_loader)
                frac = epoch - 1 + step / n_batches
                print(f"  eval @ epoch {frac:.2f}: val_loss={val_loss:.6f}, val_pearson={val_pearsons.mean():.4f}"
                      + (f" ({fmt(cells, val_pearsons)})" if len(cells) > 1 else ""), flush=True)
                if val_loss < best_val:
                    best_val, evals_since_best = val_loss, 0
                    # copy, because the next train_step donates (overwrites) trainable's buffers
                    best_trainable = jax.tree.map(jnp.copy, trainable)
                    print(f"  -> New best model at epoch {frac:.2f}! Saving checkpoint (val_loss: {val_loss:.6f})",
                          flush=True)
                    model._params = {**frozen, **best_trainable}
                    model.save_checkpoint(str(stage_dir), save_full_model=False, save_minimal_model=True)
                else:
                    evals_since_best += 1
                    if evals_since_best >= patience_in_evals:
                        print(f"Early stopping: {evals_since_best} validation checks without improvement", flush=True)
                        stop = True
                        break

        train_loss = float(jnp.mean(jnp.stack(losses)))
        train_pearson = float(jnp.mean(jnp.stack(pearsons)))   # over steps and heads
        test_loss, test_pearsons = evaluate(predict_batch, {**frozen, **trainable}, state, test_loader)
        # val_pearson / test_pearson are means over heads; per-head values follow on the same line
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, train_pearson={train_pearson:.4f} (batch mean), "
              f"val_loss={val_loss:.6f}, val_pearson={val_pearsons.mean():.4f}, "
              f"test_loss={test_loss:.6f}, test_pearson={test_pearsons.mean():.4f} "
              f"[{(time.perf_counter() - t0) / 60:.1f} min]"
              + (f" | val: {fmt(cells, val_pearsons)} | test: {fmt(cells, test_pearsons)}" if len(cells) > 1 else ""),
              flush=True)
        if stop:
            break

    assert best_trainable is not None, f"{stage}: no validation check ran"
    model._params = {**frozen, **best_trainable}
    return best_val, rng


def train_two_stage(model, config, train_loader, val_loader, test_loader):
    """Stage 1 (heads only), then stage 2 (encoder + heads) from stage 1's best weights."""
    rng = jax.random.PRNGKey(config.seed)
    common = dict(heads=config.head_names, cells=config.cell_types,
                  train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                  weight_decay=config.weight_decay, val_eval_frequency=config.val_eval_frequency,
                  early_stopping_patience=config.early_stopping_patience,
                  checkpoint_dir=config.checkpoint_dir)
    # every head name contains HEAD ('mpra_head' or 'mpra_head_<cell>'), so (HEAD,) selects all heads
    best1, rng = train_stage(model, stage="stage1", prefixes=(HEAD,), lr=config.learning_rate,
                             num_epochs=config.num_epochs, start_epoch=0, rng=rng, **common)
    print(f"Stage 1 best val_loss: {best1:.6f}", flush=True)
    if not config.second_stage_lr:
        return
    best2, _ = train_stage(model, stage="stage2", prefixes=(HEAD, ENCODER), lr=config.second_stage_lr,
                           num_epochs=config.second_stage_epochs, start_epoch=config.num_epochs, rng=rng, **common)
    print(f"Stage 2 best val_loss: {best2:.6f}", flush=True)
