#!/bin/bash
# Validate a training config, then submit it to a GPU node.
#
#   ./submit_training.sh episomal_alphampra.yaml
#
# The config is checked before anything is queued, so a bad column name or a
# missing data file fails here in seconds instead of after the job waits in the
# queue and then dies on startup.
#
# One job trains every target in the config, one model after another.
#
# Overrides, as environment variables:
#   PARTITION  (default priority_gpu)  ACCOUNT (default prio_skr2)
#   GPUS       (default 1)             CPUS    (default 10)
#   MEM        (default 128G)          TIME    (default 2-00:00:00)
#   GPU_TYPE   (default unset: take whatever frees up first, e.g. h200)
#   CONDA_ENV  (default alphampra)
#   DRY_RUN    (set to 1 to print the job script without submitting)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
    echo "usage: $(basename "$0") <config.yaml>" >&2
    exit 1
fi

CONFIG="$1"
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

PARTITION="${PARTITION:-priority_gpu}"
ACCOUNT="${ACCOUNT:-prio_skr2}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-10}"
MEM="${MEM:-128G}"
TIME="${TIME:-2-00:00:00}"
CONDA_ENV="${CONDA_ENV:-alphampra}"

# conda is not on PATH by default on Bouchet. Named envs may live under the
# module's base or under ~/.conda, so check both rather than assuming.
module load miniconda 2>/dev/null || true
ENV_PREFIX=""
for base in "$HOME/.conda" "$(conda info --base 2>/dev/null || true)"; do
    [ -n "$base" ] && [ -x "$base/envs/$CONDA_ENV/bin/python" ] && { ENV_PREFIX="$base/envs/$CONDA_ENV"; break; }
done
[ -n "$ENV_PREFIX" ] || { echo "conda env '$CONDA_ENV' not found under ~/.conda or the miniconda base" >&2; exit 1; }
PY="$ENV_PREFIX/bin/python"

# ---- preflight: parse the config and report exactly what will run -----------
# Same build_config the training script calls, so every assert fires here first.
echo "Validating $CONFIG"
PLAN="$(cd "$PROJECT_DIR" && LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
    "$PY" - "$CONFIG" <<'PYEOF'
import sys
sys.argv = ['preflight', '--config', sys.argv[1]]
from parse_args import build_config
c = build_config()
stage2 = f'{c.second_stage_epochs} epochs max, lr {c.second_stage_lr}' if c.second_stage_lr else 'off'
print(f'model_name={c.model_name}')
print(f'PLAN  data        : {c.data}')
print(f'PLAN  cell type   : {c.cell_type} ({c.construct_length}bp)')
print(f'PLAN  stage 1     : {c.num_epochs} epochs max, lr {c.learning_rate}, batch {c.batch_size}, subset {c.subset_frac}')
print(f'PLAN  stage 2     : {stage2}, grad accumulation {c.gradient_accumulation_steps}')
print(f'PLAN  checkpoints : {c.checkpoint_dir}')
PYEOF
)" || { echo "config failed validation, nothing submitted" >&2; exit 1; }

MODEL_NAME="$(sed -n 's/^model_name=//p' <<<"$PLAN")"
sed -n 's/^PLAN  /  /p' <<<"$PLAN"
echo

mkdir -p "$PROJECT_DIR/logs"

# priority_gpu refuses a bare --gpus=N: it wants either --gpus=<type>:N or
# --gpus=N with a constraint naming the card. GPU_TYPE pins one model (and waits
# for it); otherwise OR together every gpu:<type> feature the partition currently
# advertises, so the job takes whichever card frees up first.
if [ -n "${GPU_TYPE:-}" ]; then
    GPU_REQUEST="--gpus=${GPU_TYPE}:${GPUS}"
else
    GPU_FEATURES="$(sinfo -h -p "$PARTITION" -o '%f' \
        | tr ',' '\n' | grep '^gpu:' | sort -u | paste -sd'|' -)"
    if [ -z "$GPU_FEATURES" ]; then
        echo "no gpu:<type> features advertised by partition '$PARTITION'; set GPU_TYPE explicitly" >&2
        exit 1
    fi
    GPU_REQUEST="--gpus=${GPUS} --constraint=\"${GPU_FEATURES}\""
fi

JOB_SCRIPT="$(cat <<EOF
#!/bin/bash
#SBATCH -J ${MODEL_NAME}
#SBATCH -p ${PARTITION}
#SBATCH -A ${ACCOUNT}
#SBATCH -t ${TIME}
#SBATCH -c ${CPUS}
#SBATCH --mem=${MEM}
#SBATCH ${GPU_REQUEST}
#SBATCH -o ${PROJECT_DIR}/logs/${MODEL_NAME}_%j.out
#SBATCH -e ${PROJECT_DIR}/logs/${MODEL_NAME}_%j.err

set -euo pipefail

cd ${PROJECT_DIR}

module load miniconda
# sbatch runs a non-interactive shell with no conda shell function defined;
# sourcing the profile is what 'conda init' would otherwise have set up
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

# The loaded GCCcore module puts its libstdc++ ahead of conda's, which breaks
# matplotlib's C extension on import. sbatch inherits the submitting shell's
# LD_LIBRARY_PATH, so put the env's own lib first regardless of what came in.
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"

# ~/.local site-packages belong to an unrelated python; everything needed is in the env
export PYTHONNOUSERSITE=1
# JAX grabs 75% of the GPU up front by default, which breaks jobs sharing a node
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Slurm's .out is not a TTY, so without this the per-epoch prints sit in a buffer
export PYTHONUNBUFFERED=1

echo "host      : \$(hostname)"
echo "gpu       : \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "python    : \$(which python)"
echo "started   : \$(date)"
echo "----------------------------------------"

python alphampra.py --config ${CONFIG}

echo "----------------------------------------"
echo "finished  : \$(date)"
EOF
)"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "$JOB_SCRIPT"
    exit 0
fi

echo "Submitting to ${PARTITION} (account ${ACCOUNT}, ${GPUS} gpu, ${CPUS} cpu, ${MEM}, ${TIME})"
JOB_ID="$(sbatch --parsable <<<"$JOB_SCRIPT")"

echo "job ${JOB_ID}  -> ${PROJECT_DIR}/logs/${MODEL_NAME}_${JOB_ID}.out"
echo
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${PROJECT_DIR}/logs/${MODEL_NAME}_${JOB_ID}.out"
echo "  scancel ${JOB_ID}"
