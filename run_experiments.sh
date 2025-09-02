#!/usr/bin/env bash

# Run Recycling Robot experiments over a grid of parameters.
# Creates one output directory per combination with rewards.txt, rewards.png, and policy_heatmap.png.

set -euo pipefail

# Directory of this script (repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Output base directory
OUT_DIR="${SCRIPT_DIR}/runs"
mkdir -p "${OUT_DIR}"

# Parameter grids
ALPHAS=(0.3 0.6 0.9)        # --alpha_prob
BETAS=(0.3 0.6 0.9)         # --beta_prob
R_SEARCHES=(4 5 6)          # --r_search
R_WAITS=(1 2 3)             # --r_wait

# Optional: override defaults for training via env vars
: "${EPOCH_STEPS:=1000}"
: "${N_EPOCHS:=50}"
: "${LEARNING_RATE:=0.1}"
: "${GAMMA:=0.9}"
: "${MODE:=softmax}"          # softmax | epsilon_greedy | deterministic
: "${TEMPERATURE:=1.0}"
: "${EPSILON:=0.1}"

echo "Running grid..."
echo "  alphas:      ${ALPHAS[*]}"
echo "  betas:       ${BETAS[*]}"
echo "  r_searches:  ${R_SEARCHES[*]}"
echo "  r_waits:     ${R_WAITS[*]}"
echo "  epochs:      ${N_EPOCHS}, steps/epoch: ${EPOCH_STEPS}, lr: ${LEARNING_RATE}, gamma: ${GAMMA}"
echo "  mode:        ${MODE} (temp=${TEMPERATURE}, eps=${EPSILON})"

TOTAL=0
for alpha in "${ALPHAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for rs in "${R_SEARCHES[@]}"; do
      for rw in "${R_WAITS[@]}"; do
        TOTAL=$((TOTAL+1))
        # Create a readable directory name (replace dots with underscores)
        a_tag=${alpha//./_}
        b_tag=${beta//./_}
        run_dir="${OUT_DIR}/alpha-${a_tag}_beta-${b_tag}_rs-${rs}_rw-${rw}"
        mkdir -p "${run_dir}"

        echo "[${TOTAL}] alpha=${alpha} beta=${beta} r_search=${rs} r_wait=${rw} -> ${run_dir}"

        # Save params for traceability
        cat > "${run_dir}/params.txt" <<EOF
alpha_prob=${alpha}
beta_prob=${beta}
r_search=${rs}
r_wait=${rw}
epoch_steps=${EPOCH_STEPS}
n_epochs=${N_EPOCHS}
learning_rate=${LEARNING_RATE}
gamma=${GAMMA}
mode=${MODE}
temperature=${TEMPERATURE}
epsilon=${EPSILON}
EOF

        # Run the experiment in the run directory so outputs land there
        (
          cd "${run_dir}" >/dev/null
          python3 "${SCRIPT_DIR}/recycling_robot.py" \
            --alpha_prob "${alpha}" \
            --beta_prob "${beta}" \
            --r_search "${rs}" \
            --r_wait "${rw}" \
            --epoch_steps "${EPOCH_STEPS}" \
            --n_epochs "${N_EPOCHS}" \
            --learning_rate "${LEARNING_RATE}" \
            --gamma "${GAMMA}" \
            --mode "${MODE}" \
            --temperature "${TEMPERATURE}" \
            --epsilon "${EPSILON}" \
            >/dev/null
        )
      done
    done
  done
done

echo "Done. Outputs in: ${OUT_DIR}"
