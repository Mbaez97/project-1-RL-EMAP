import argparse
import random
from enum import Enum
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# For reproducibility
random.seed(42)


class State(Enum):
    """States for the recycling robot."""

    HIGH = 0  # High battery
    LOW = 1  # Low battery


class Action(Enum):
    """Actions the robot can take."""

    SEARCH = 0
    WAIT = 1
    RECHARGE = 2


class RecyclingRobot:
    """The Recycling Robot Environment."""

    def __init__(self, alpha_prob=0.9, beta_prob=0.3, r_search=5.0, r_wait=1.0, r_rescue=-10.0):
        self.states = list(State)
        self.actions = list(Action)
        self.alpha_prob = float(alpha_prob)
        self.beta_prob = float(beta_prob)
        self.r_search = float(r_search)
        self.r_wait = float(r_wait)
        self.r_rescue = float(r_rescue)

    def step(self, state: State, action: Action):
        if state == State.HIGH:
            if action == Action.SEARCH:
                if random.random() < self.alpha_prob:
                    return State.HIGH, self.r_search
                return State.LOW, self.r_search
            elif action == Action.WAIT:
                return State.HIGH, self.r_wait
            return state, 0.0

        elif state == State.LOW:
            if action == Action.SEARCH:
                if random.random() < self.beta_prob:
                    return State.LOW, self.r_search
                return State.HIGH, self.r_rescue
            elif action == Action.WAIT:
                return State.LOW, self.r_wait
            elif action == Action.RECHARGE:
                return State.HIGH, 0.0

        return state, 0.0


def td_policy_evaluation_run(env: RecyclingRobot, policy: dict, gamma: float, alpha: float, n_steps: int, V=None):
    if V is None:
        V = {s: 0.0 for s in env.states}

    current_state = random.choice(env.states)
    total_reward = 0.0
    for i in range(n_steps):
        action = policy[current_state]
        next_state, reward = env.step(current_state, action)
        total_reward += reward
        td_target = reward + gamma * V[next_state]
        td_error = td_target - V[current_state]
        V[current_state] += alpha * td_error
        current_state = next_state
    return V, total_reward


def policy_improvement_from_V(env: RecyclingRobot, V: dict, gamma: float):
    policy = {}
    for s in env.states:
        q_vals = {}
        for a in env.actions:
            if s == State.HIGH:
                if a == Action.SEARCH:
                    alpha = env.alpha_prob
                    q = env.r_search + gamma * (alpha * V[State.HIGH] + (1 - alpha) * V[State.LOW])
                elif a == Action.WAIT:
                    q = env.r_wait + gamma * V[State.HIGH]
                else:
                    q = 0.0 + gamma * V[State.HIGH]
            else:
                if a == Action.SEARCH:
                    beta = env.beta_prob
                    q = env.r_search + gamma * (beta * V[State.LOW] + (1 - beta) * V[State.HIGH])
                elif a == Action.WAIT:
                    q = env.r_wait + gamma * V[State.LOW]
                else:
                    q = 0.0 + gamma * V[State.HIGH]
            q_vals[a] = q
        # use explicit keys list to avoid typing issues with mypy/older toolchains
        best_action = max(list(q_vals.keys()), key=lambda k: q_vals[k])
        policy[s] = best_action
    return policy


def policy_action_probabilities_from_V(env: RecyclingRobot, V: dict, gamma: float, mode: str = 'softmax', temperature: float = 1.0, epsilon: float = 0.1) -> dict:
    assert mode in ('softmax', 'epsilon_greedy', 'deterministic')
    action_probs = {}
    nA = len(env.actions)

    for s in env.states:
        q_vals = {}
        for a in env.actions:
            if s == State.HIGH:
                if a == Action.SEARCH:
                    alpha_p = env.alpha_prob
                    q = env.r_search + gamma * (alpha_p * V[State.HIGH] + (1 - alpha_p) * V[State.LOW])
                elif a == Action.WAIT:
                    q = env.r_wait + gamma * V[State.HIGH]
                else:
                    q = 0.0 + gamma * V[State.HIGH]
            else:
                if a == Action.SEARCH:
                    beta_p = env.beta_prob
                    q = env.r_search + gamma * (beta_p * V[State.LOW] + (1 - beta_p) * V[State.HIGH])
                elif a == Action.WAIT:
                    q = env.r_wait + gamma * V[State.LOW]
                else:
                    q = 0.0 + gamma * V[State.HIGH]
            q_vals[a] = q

        if mode == 'softmax':
            max_q = max(q_vals.values())
            exps = {a: math.exp((q - max_q) / max(1e-8, temperature)) for a, q in q_vals.items()}
            Z = sum(exps.values())
            probs = {a: exps[a] / Z for a in env.actions}
        else:
            best_q = max(q_vals.values())
            best_actions = [a for a, q in q_vals.items() if abs(q - best_q) < 1e-12]
            k = len(best_actions)
            if mode == 'epsilon_greedy':
                base = epsilon / nA
                probs = {a: base for a in env.actions}
                share_best = (1.0 - epsilon) / k
                for a in best_actions:
                    probs[a] += share_best
            else:  # deterministic
                probs = {a: (1.0 / k if a in best_actions else 0.0) for a in env.actions}

        action_probs[s] = probs

    return action_probs


def train_policy_iteration(env: RecyclingRobot, init_policy: dict, gamma: float, alpha: float, epoch_steps: int, n_epochs: int, rewards_file='rewards.txt'):
    open(rewards_file, 'w').close()
    V = None
    policy = dict(init_policy)
    rewards = []
    for epoch in range(1, n_epochs + 1):
        V, total_reward = td_policy_evaluation_run(env, policy, gamma, alpha, epoch_steps, V=V)
        rewards.append(total_reward)
        with open(rewards_file, 'a') as f:
            f.write(str(total_reward) + '\n')
        policy = policy_improvement_from_V(env, V, gamma)
        if epoch % max(1, n_epochs // 10) == 0:
            print(f'Completed epoch {epoch}/{n_epochs} | total_reward={total_reward:.2f}')
    return V, policy, rewards


def plot_rewards(rewards, out_path=None, ax=None):
    if ax is None:
        plt.figure(figsize=(8, 4))
        ax = plt.gca()
    ax.plot(range(1, len(rewards) + 1), rewards, marker='o', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total reward (per epoch)')
    ax.set_title('Total reward per epoch')
    ax.grid(True)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    return ax


def plot_policy_heatmap(policy, env: RecyclingRobot, out_path=None, probs=None, title='Action selection probabilities'):
    actions_order = [Action.SEARCH, Action.WAIT, Action.RECHARGE]
    states_order = [State.HIGH, State.LOW]

    if probs is None:
        probs = {s: {a: (1.0 if policy[s] == a else 0.0) for a in actions_order} for s in states_order}

    matrix = [[probs[s][a] for a in actions_order] for s in states_order]
    labels = [[f"{probs[s][a]:.2f}" for a in actions_order] for s in states_order]

    plt.figure(figsize=(6, 3))
    ax = plt.gca()
    sns.heatmap(
        matrix,
        annot=labels,
        fmt='',
        cmap='Blues',
        cbar=True,
        vmin=0.0,
        vmax=1.0,
        xticklabels=[a.name for a in actions_order],
        yticklabels=[s.name for s in states_order],
        ax=ax,
        linewidths=0.5,
        linecolor='white'
    )
    ax.set_title(title)
    ax.set_xlabel('Action')
    ax.set_ylabel('State')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    return ax


def parse_args():
    p = argparse.ArgumentParser(description='Train Recycling Robot using TD(0) per-epoch evaluation and greedy improvement')
    p.add_argument('--alpha_prob', type=float, default=0.9, help='alpha probability (default 0.9)')
    p.add_argument('--beta_prob', type=float, default=0.3, help='beta probability (default 0.3)')
    p.add_argument('--r_search', type=float, default=5.0, help='reward for search action (default 5.0)')
    p.add_argument('--r_wait', type=float, default=1.0, help='reward for wait action (default 1.0)')
    p.add_argument('--learning_rate', type=float, default=0.1, help='TD learning rate alpha (default 0.1)')
    p.add_argument('--epoch_steps', type=int, default=1000, help='steps per epoch (default 1000)')
    p.add_argument('--n_epochs', type=int, default=50, help='number of epochs to train (default 50)')
    p.add_argument('--gamma', type=float, default=0.9, help='discount factor (default 0.9)')
    p.add_argument('--mode', choices=['softmax', 'epsilon_greedy', 'deterministic'], default='softmax', help='policy probability mode for heatmap (default softmax)')
    p.add_argument('--temperature', type=float, default=1.0, help='softmax temperature (default 1.0)')
    p.add_argument('--epsilon', type=float, default=0.1, help='epsilon for epsilon_greedy (default 0.1)')
    p.add_argument('--show', action='store_true', help='show plots (not available in headless mode)')
    return p.parse_args()


def main():
    args = parse_args()

    env = RecyclingRobot(alpha_prob=args.alpha_prob, beta_prob=args.beta_prob, r_search=args.r_search, r_wait=args.r_wait)

    learning_rate = args.learning_rate
    discount_factor = args.gamma
    epoch_steps = args.epoch_steps
    n_epochs = args.n_epochs

    init_policy = {State.HIGH: Action.SEARCH, State.LOW: Action.RECHARGE}

    V_final, learned_policy, rewards = train_policy_iteration(env, init_policy, discount_factor, learning_rate, epoch_steps, n_epochs, rewards_file='rewards.txt')

    print('\nFinal value estimates:')
    for s, v in V_final.items():
        print(f'  V({s.name}) = {v:.2f}')

    print('\nLearned policy:')
    for s, a in learned_policy.items():
        print(f'  pi({s.name}) = {a.name}')

    # save rewards plot
    plot_rewards(rewards, out_path='rewards.png')

    # derive action probabilities and save heatmap
    probs = policy_action_probabilities_from_V(env, V_final, discount_factor, mode=args.mode, temperature=args.temperature, epsilon=args.epsilon)
    plot_policy_heatmap(learned_policy, env, out_path='policy_heatmap.png', probs=probs, title='Action selection probabilities (derived from V)')

    if args.show:
        try:
            plt.show()
        except Exception:
            print('Unable to show plots in this environment')

    print('\nSaved epoch rewards to rewards.txt')
    print('Saved plots: rewards.png, policy_heatmap.png')


if __name__ == '__main__':
    main()
