# Recycling Robot — Project Summary (RL01)

Overview

- Implements the “Recycling Robot” MDP (Sutton & Barto, Example 3.3) and solves the exercise via policy iteration: per‑epoch TD(0) policy evaluation followed by greedy policy improvement. This document states the problem, method, how to run experiments, and how to interpret outputs.

Problem Setup (MDP)

- States: HIGH (high battery), LOW (low battery).
- Actions: SEARCH, WAIT, RECHARGE.
- Dynamics and rewards:
  - In HIGH:
    - SEARCH: stays in HIGH with probability α, otherwise goes to LOW. Reward r_search.
    - WAIT: stays in HIGH. Reward r_wait.
    - RECHARGE: not needed in HIGH; implemented as zero reward.
  - In LOW:
    - SEARCH: stays in LOW with probability β (reward r_search); with 1−β it fails and is “rescued” → transitions to HIGH with rescue penalty r_rescue < 0.
    - WAIT: stays in LOW. Reward r_wait.
    - RECHARGE: transitions to HIGH with zero reward.
- Key parameters: α (alpha_prob), β (beta_prob), r_search, r_wait, r_rescue (negative rescue penalty; default −3.0 in code).
- Discount factor γ ∈ (0,1).

Methodology

- TD(0) policy evaluation per epoch: for a fixed policy π, run n_steps transitions and update V(s) ← V(s) + α_lr [r + γ V(s′) − V(s)].
- Greedy policy improvement: compute one‑step look‑ahead Q(s,a) using the known model and pick argmax_a Q(s,a) in each state to form the new π.
- Action‑probability visualization: derive a distribution over actions from Q(s,·) with three modes: softmax (temperature τ), epsilon_greedy (ε), and deterministic (0/1 ties split evenly).
- Initial policy: HIGH → SEARCH; LOW → RECHARGE.

Default Hyperparameters

- Epochs: 50; Steps/epoch: 1000; Learning rate α_lr: 0.1; Discount γ: 0.9.
- Fixed random seed for reproducibility.

Repository Layout

- recycling_robot.py: environment, TD(0), greedy improvement, plotting (rewards curve and policy heatmap).
- run_experiments.sh: runs a parameter grid, saving outputs under runs/.
- notebooks/: team exploration notebooks; e.g., notebooks/marcelo_files/marcelo.ipynb.
- Example single‑run outputs at repo root: rewards.txt, rewards.png, policy_heatmap.png.

How to Run a Single Experiment

- Install dependencies (Python 3.10+):

  ```bash
  pip install -r requirements.txt
  ```

- Run:

  ```bash
  python recycling_robot.py \
    --alpha_prob 0.9 \
    --beta_prob 0.3 \
    --r_search 5.0 \
    --r_wait 1.0 \
    --learning_rate 0.1 \
    --epoch_steps 1000 \
    --n_epochs 50 \
    --gamma 0.9 \
    --mode softmax \
    --temperature 1.0 \
    --epsilon 0.1
  ```

- Outputs: rewards.txt (per‑epoch total reward), rewards.png (curve), policy_heatmap.png (action probabilities by state).
- Tip: use `--mode deterministic` for a 0/1 policy map, or reduce `--temperature` in softmax for peakier probabilities.

Experiment Grid (Script)

- Grid values: α ∈ {0.3, 0.6, 0.9}; β ∈ {0.3, 0.6, 0.9}; r_search ∈ {4,5,6}; r_wait ∈ {1,2,3}.
- Run the grid:

  ```bash
  ./run_experiments.sh
  ```

- Each run is saved under `runs/alpha-*_beta-*_rs-*_rw-*/` with params.txt, rewards.txt, rewards.png, policy_heatmap.png.
- Override training hyperparameters via environment variables, e.g.:

  ```bash
  N_EPOCHS=100 EPOCH_STEPS=2000 MODE=deterministic ./run_experiments.sh
  ```

Interpreting Results

- Rewards curve: should trend upward as the policy improves; some noise remains due to stochastic transitions.
- Policy heatmap:
  - In HIGH: larger α or larger r_search favors SEARCH.
  - In LOW: smaller β and more negative r_rescue favor RECHARGE; larger r_wait can favor WAIT.
  - deterministic shows crisp 0/1 choices; softmax shows graded preferences useful for near‑ties.

Reproducibility

- Fixed Python RNG seed in code; grid runs store exact parameters in each run directory (params.txt).

Limitations & Future Work

- Expose r_rescue as a command‑line parameter and study its effect; compare TD(0) with MC and TD(λ); try control algorithms (SARSA, Q‑learning).
- Extend the environment (e.g., recharge costs, partial observability, more battery levels).
- Add unit tests and additional validation.
- Compare against dynamic programming solutions in simplified settings.

References

- Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. 2nd ed., 2018.

Appendix: Notebooks

- See `notebooks/marcelo_files/marcelo.ipynb` for value evolution, reward curve, and action‑probability heatmaps consistent with this pipeline.
