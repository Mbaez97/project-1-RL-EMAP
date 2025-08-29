Recycling Robot – TD Learning Project

## 1. Problem Description

The Recycling Robot is a classical Reinforcement Learning example where the agent (robot) must decide how to act depending on its battery level:
- High battery: can search or wait.
- Low battery: can search, wait, or recharge.
The goal is to maximize long-term accumulated reward, balancing the risk of searching with low battery (which may cause rescue penalty) against the benefit of finding cans.

## 2. Implementation
The solution was implemented in Python using Temporal-Difference learning:
- States: High (0), Low (1).
- Actions: Search (0), Wait (1), Recharge (2).

Environment: step_env simulates transitions with probabilities:

- α = probability of staying High after searching in High.
- β = probability of staying Low after searching in Low.

Rewards:
- 𝑟_search
- 𝑟_wait
- Rescue penalty < 0.

Outputs:
- rewards.txt
- accumulated_reward.png
- optimal_policy_heatmap.png

## 3. Parameters Chosen

## 4. Results
Learning Curves

- The robot consistently converged to stable high accumulated rewards.

Optimal Policy

1) In High: the robot almost always chooses Search.
2) In Low:

- If β is small (risky), the robot chooses Recharge.
- If β is large (safe), the robot sometimes searches in Low.

This matches intuition: search when energy is safe, recharge when risky.
