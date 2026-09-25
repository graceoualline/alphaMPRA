# Speeding up alphaMPRA training

Episomal runs (one cell type each, `alphampra.py` -> `alphagenome_ft_mpra.training.train`) take
about 5h per epoch on an H100: 29,129 steps of batch 64, about 0.6 s per step, with Slurm reporting
3-15% GPU utilization. Almost all of that time is spent in the eager optimizer update and the
un-jitted gradient wrapper in `ag_train`, not in the model.

## Measurements

`bench_step0.py`, job 27484371: H100, 10 CPUs, K562 config, batch 64, stage-2 setup (encoder
unfrozen, Adam over all params) exactly as `ag_train` builds it. The job shared node a1127u15n01 with
the running HepG2 training job, so the loader number may be slightly pessimistic.

| What | ms / step | Share of a training step |
|---|---:|---:|
| Loader alone (`MPRADataLoader`, 200 train batches) | 26.8 | 5% |
| Step as `ag_train` does it (grad + eager update + `float(loss)`) | 561.8 | 95% |
| - of which: optimizer update + apply, eager | 435.9 | 74% |
| - of which: `value_and_grad`, not jitted | 108.0 | 18% |
| Same `value_and_grad` under `jax.jit` | 8.1 | - |
| Loader + step | 588.7 | matches the ~600 observed |

Parameters in `model._params` (454.7M total):

| Component | Params | Used for MPRA predictions |
|---|---:|---|
| `sequence_encoder` | 90.0M (20%) | yes |
| `mpra_head` | ~4.2M | yes |
| `transformer_tower` | 191.8M | no |
| `sequence_decoder` | 111.5M | no |
| pretrained AlphaGenome heads + output embedders | ~57M | no |

What this says:
- The model itself is fast: a jitted forward+backward is 8 ms, 13x faster than the same gradient
  un-jitted.
- The optimizer update is three quarters of the step. It runs eagerly over 597 parameter arrays and
  454.7M parameters, 80% of which never affect a prediction.
- The loader is not the bottleneck today. Once the step is jitted it will be: 27 ms of loading
  against roughly 10 ms of compute.

Projected step after items 1-3 below: about 10-15 ms of compute (the jitted gradient, plus a jitted
Adam update over about 94M parameters), with loading overlapped. That would be 30-50x faster than
now, putting an epoch around 10 minutes instead of 5 hours. This is an estimate from the parts; it
has not been measured end to end.

## Recommendations, in order of payoff

| # | Change | Measured / expected effect | Effort |
|---|---|---|---|
| 1 | Jit the whole train step (grad + optimizer update), sync loss rarely | 562 ms -> ~10-20 ms | own training loop (ask for code) |
| 2 | Optimize only encoder + head in stage 2 | shrinks the update 5x (454.7M -> ~94M params) | part of 1 (ask for code) |
| 3 | Vectorized data loader with prefetch | 27 ms -> a few ms, overlapped | small, code below |
| 4 | Fewer stage-2 epochs | ~2x wall time | config |
| 5 | Test eval once per epoch | fewer eval passes | 1 line |
| 6 | Bigger batch | fewer steps | config + LR retune, after 1-3 |
| 7 | Cached embeddings for stage 1 | small (1 of 9 epochs) | `ag_train` flag |
| 8 | One multi-task model instead of three | 3x fewer runs | design decision |

Not recommended: more GPUs or CPUs. The GPU is idle most of each step, and the bottleneck is on
the host in single-threaded Python.

### 1. Jit the whole train step (ask for code)

`ag_train` calls `jax.value_and_grad(_seq_loss_fn)` without `jax.jit`. `model._predict` is jitted
inside, but the loss, `corrcoef` and the backward bookkeeping run from Python every step (108 ms vs
8 ms jitted). `optimizer.update` and `optax.apply_updates` then run eagerly, one GPU launch per op
per parameter array (436 ms). `float(avg_loss)` every step also blocks the host until the GPU
finishes.

Fix: a single `@jax.jit` step that computes forward, backward, optimizer update and new params, with
`donate_argnums` for params and optimizer state; keep the running loss on the device and read it
every few hundred steps. `ag_train` has no hook for this, so it means our own loop, including the
two-stage logic, per-quarter-epoch validation, checkpointing and early stopping that `ag_train`
provides.

### 2. Optimize only encoder + head in stage 2 (ask for code)

Stage 2 builds `create_optimizer(..., heads_only=False)` over all of `model._params`. Every step
computes zero gradients for the transformer, decoder and pretrained heads, keeps Adam moments for
them (about 2.9 GB in fp32 for the unused 360M), and updates them. Mask them to
`optax.set_to_zero()`, or better, drop those subtrees from the params passed to the step so their
gradients are never computed.

### 3. Vectorized data loader with prefetch

For each batch, the current loader does:
- 64 pandas `.iloc` lookups,
- 64 separate `jnp.array` copies to the GPU,
- an `np.stack` that copies them back to the host,
- another `jnp.asarray` to the GPU,
- a `.tolist()` on a JAX index array, which forces a host sync.

It also does no prefetching.

Replacement: tokenize each split once at startup as uint8 (1.86M x 600 = 1.1 GB for train), one-hot
a batch with a single NumPy fancy index, and do one `device_put` per batch. `device_put` is async, so
putting batch i+1 on the device before yielding batch i overlaps the copy with compute. The
interface is the one `ag_train` and `plot_test_results` already use: `len`, iteration,
`batch["seq"|"y"|"organism_index"]`, `.shuffle` and `.dataset`. It is a drop-in replacement for the
two classes in `alphampra.py`.

```python
class MalinoisMPRADataset:
    def __init__(self, model, split, cell_type,
                 path_to_data="/home/go274/scratch_pi_skr2/go274/manual_MPRA_models/baseline_CODA/malinois_all_data_filtered_revcomp_preprocessed.tsv",
                 subset_frac=1.0, seed=42):
        assert split in ["train", "val", "test"] and cell_type in ["K562", "HepG2", "SKNSH"]
        self.label_column = f"{cell_type}_log2FC"
        data = pd.read_csv(path_to_data, sep="\t", usecols=["split", "sequence", self.label_column])
        data = data[data["split"] == split].reset_index(drop=True)
        if subset_frac < 1.0:
            data = data.sample(frac=subset_frac, random_state=seed).reset_index(drop=True)
        assert len(data) > 0, f"no rows for split {split}"
        assert (data["sequence"].str.len() == 600).all(), "expected 600bp padded sequences"
        # all sequences as bytes, (N, 600) uint8; one-hot happens per batch via the encoder's table
        self.tokens = np.frombuffer("".join(data["sequence"]).encode("latin1"), dtype=np.uint8).reshape(len(data), 600)
        self.lookup = model._one_hot_encoder._lookup_table          # (256, 4) float32, A/C/G/T rows, N -> zeros
        self.y = data[self.label_column].to_numpy(np.float32)       # (N,)
        assert np.isfinite(self.y).all(), f"non-finite labels in {split}"
        print(f"Loaded {len(data)} {split} rows for {cell_type}")
    def __len__(self):
        return len(self.y)
    def batch(self, idx):
        # idx: (B,) int array -> seq (B, 600, 4) float32, y (B,)
        return self.lookup[self.tokens[idx]], self.y[idx]


class MPRADataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, seed=42):
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle
        self.rng = np.random.default_rng(seed)
    def __iter__(self):
        n = len(self.dataset)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        pending = None
        for start in range(0, n, self.batch_size):
            seq, y = self.dataset.batch(order[start:start + self.batch_size])
            # async host->device copy; yield the previous batch while this one transfers
            nxt = {"seq": jax.device_put(seq), "y": jax.device_put(y),
                   "organism_index": jnp.zeros(len(y), dtype=jnp.int32)}   # all human
            if pending is not None:
                yield pending
            pending = nxt
        if pending is not None:
            yield pending
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
```

Notes:
- On its own, this saves only about 5% of today's step. It pays off once item 1 is in.
- The shuffle order changes (NumPy instead of `jax.random.permutation`), so runs are not
  step-for-step reproducible against the existing ones. Results should be statistically the same.
- Check after swapping it in: rerun `bench_step0.py` for the loader number, and do a
  `--subset-frac 0.01` smoke run to confirm the loss scale is unchanged.

### 4. Fewer stage-2 epochs

Test Pearson plateaus by stage-2 epoch 3-4 for K562 and SKNSH (see
`individual_episomal_alphampra/test_pearson_*.png`), while train Pearson keeps rising. The train set
already contains reverse complements and duplicated top rows, so one epoch is about two passes over
each unique sequence.

```yaml
second_stage_epochs: 4
```

`alphampra.py` does not pass `lr_scheduler`, so `ag_train` uses a constant LR, and a 4-epoch run
follows the same trajectory as the first 4 epochs of an 8-epoch run. If a schedule is added later
(`lr_scheduler='cosine'`), it is built over the total step count, and the epoch count then changes
the whole decay.

### 5. Test eval once per epoch

`alphampra.py` passes `val_eval_frequency` to both validation and test, so every quarter epoch runs
919 val + 978 test batches. Model selection uses val only, so in the `ag_train(...)` call:

```python
    val_eval_frequency=config.val_eval_frequency, test_eval_frequency=1,
```

The per-epoch test Pearson in the log (read by `plot_test_pearson.py`) is unchanged. Eval is forward
only through the jitted `_predict`, so it is a minor cost today (roughly 30 ms per batch, mostly the
loader). It matters more once training steps are fast.

### 6. Bigger batch

After items 1-3, try batch 256: 4x fewer steps per epoch and better use of the H100. This changes
the optimization, so retune the LR; square-root scaling from batch 64 is a reasonable first guess
(stage 2: 1e-5 -> 2e-5). Doing this before 1-3 would mostly hide per-step overhead.

### 7. Cached embeddings for stage 1

`ag_train(use_cached_embeddings=True)` computes the frozen encoder's outputs once and trains only the
head in stage 1. Stage 1 is 1 of 9 epochs, so the saving is small, and after item 1 stage 1 is cheap
anyway.

### 8. One multi-task model

A single model predicting HepG2, K562 and SKNSH together (as Malinois does) costs a third of the
compute. It is a different comparison from three single-task models, so this is a design decision,
not a speedup.

## Suggested order

1. Config: stage-2 epochs and test eval frequency (minutes).
2. Jitted step with encoder + head optimizer (items 1-2), plus the loader swap (item 3), since the
   loader becomes the bottleneck as soon as the step is fast.
3. Rerun `bench_step0.py` and check GPU utilization, then try a bigger batch.

The HepG2 run is roughly 2x slower than the others and will likely hit the 72h limit around epoch
7. It saves its best checkpoint at each new best, but `plot_test_results` at the end of
`alphampra.py` will not run, so run that separately against the saved checkpoint.
