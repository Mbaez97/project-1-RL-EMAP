# RL for Recycling Robot

Implementation of Reinforcement Learning (RL) for the Recycling Robot problem, as described in Example 3.3 of the textbook. The goal is to use the Temporal Difference (TD) algorithm so that the robot learns to maximize the total reward over several training epochs.

## Set parameters

alpha = [0.3, 0.6, 0.9]
beta = [0.3, 0.6, 0.9]
reward search = [4, 5, 6]
reward wait = [1, 2, 3]

Run the training script:

python recycling_robot.py --alpha_prob 0.9 --beta_prob 0.3 --r_search 5.0 --r_wait 1.0 --learning_rate 0.1 --epoch_steps 1000
