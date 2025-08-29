import random
from enum import Enum

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
    """
    The Recycling Robot Environment.

    This class simulates the environment based on the problem description for a
    recycling robot, which is a common example in reinforcement learning.
    """

    def __init__(
        self, alpha_prob=0.9, beta_prob=0.2, r_search=5.0, r_wait=1.0, r_rescue=-10.0
    ):
        """
        Initializes the environment parameters.

        Args:
            alpha_prob (float): Probability of NOT draining battery when searching at HIGH.
            beta_prob (float): Probability of finding a can when searching at LOW.
            r_search (float): Reward for actively searching for a can.
            r_wait (float): Reward for waiting.
            r_rescue (float): Reward (penalty) for battery running out (requires rescue).
        """
        self.states = list(State)
        self.actions = list(Action)

        self.alpha_prob = alpha_prob
        self.beta_prob = beta_prob
        self.r_search = r_search
        self.r_wait = r_wait
        self.r_rescue = r_rescue

    def step(self, state: State, action: Action):
        """
        Simulates one step in the environment.

        Args:
            state (State): The current state of the robot.
            action (Action): The action taken by the robot.

        Returns:
            tuple: A tuple containing (next_state, reward).
        """
        if state == State.HIGH:
            if action == Action.SEARCH:
                # With probability alpha_prob, battery stays high
                if random.random() < self.alpha_prob:
                    return State.HIGH, self.r_search
                # Otherwise, it drains to low
                return State.LOW, self.r_search
            elif action == Action.WAIT:
                # Stays at high, gets wait reward
                return State.HIGH, self.r_wait
            # RECHARGE is not a valid action at HIGH, but we handle it gracefully
            return state, 0

        elif state == State.LOW:
            if action == Action.SEARCH:
                # With probability beta_prob, it finds a can and stays low
                if random.random() < self.beta_prob:
                    return State.LOW, self.r_search
                # Otherwise, battery dies, needs rescue (transitions to HIGH after rescue)
                return State.HIGH, self.r_rescue
            elif action == Action.WAIT:
                # Stays at low, gets wait reward
                return State.LOW, self.r_wait
            elif action == Action.RECHARGE:
                # Recharges to high
                return State.HIGH, 0.0

        # Should not be reached
        return state, 0


def td_policy_evaluation(
    env: RecyclingRobot, policy: dict, gamma: float, alpha: float, n_steps: int
):
    """
    Performs TD(0) policy evaluation for a continuous task.

    Args:
        env (RecyclingRobot): The environment.
        policy (dict): A dictionary mapping a state to a fixed action.
        gamma (float): Discount factor.
        alpha (float): Learning rate.
        n_steps (int): Number of steps to run the simulation for.

    Returns:
        dict: The estimated value function V(s) for each state.
    """
    # 1. Initialize V(s) arbitrarily (e.g., to 0)
    V = {s: 0.0 for s in env.states}

    # Start in a random state
    current_state = random.choice(env.states)

    print(f"Running TD(0) for {n_steps} steps...")
    # Loop for a large number of steps
    for i in range(n_steps):
        # 2. Get action from the fixed policy
        action = policy[current_state]

        # 3. Take action, observe reward and next state
        next_state, reward = env.step(current_state, action)

        # 4. Update the value function using the TD update rule
        # V(s) <- V(s) + alpha * [R + gamma * V(s') - V(s)]
        td_target = reward + gamma * V[next_state]
        td_error = td_target - V[current_state]
        V[current_state] += alpha * td_error

        # Move to the next state for the next iteration
        current_state = next_state

        if (i + 1) % (n_steps // 10) == 0:
            print(
                f"  Step {i+1}/{n_steps} | V(HIGH)={V[State.HIGH]:.2f}, V(LOW)={V[State.LOW]:.2f}"
            )

    return V


if __name__ == "__main__":
    # --- Environment and Algorithm Parameters ---
    env = RecyclingRobot()
    learning_rate = 0.1
    discount_factor = 0.9
    num_steps = 50000

    # --- Policy 1: Always Search (Aggressive) ---
    aggressive_policy = {State.HIGH: Action.SEARCH, State.LOW: Action.SEARCH}
    print("Evaluating policy: Always Search")
    V_aggressive = td_policy_evaluation(
        env=env,
        policy=aggressive_policy,
        gamma=discount_factor,
        alpha=learning_rate,
        n_steps=num_steps,
    )
    print("\n--- Results for 'Always Search' Policy ---")
    for state, value in V_aggressive.items():
        print(f"  V({state.name}) = {value:.2f}")

    print("\n" + "=" * 50 + "\n")

    # --- Policy 2: Search High, Recharge Low (Conservative) ---
    conservative_policy = {State.HIGH: Action.SEARCH, State.LOW: Action.RECHARGE}
    print("Evaluating policy: Search when HIGH, Recharge when LOW")
    V_conservative = td_policy_evaluation(
        env=env,
        policy=conservative_policy,
        gamma=discount_factor,
        alpha=learning_rate,
        n_steps=num_steps,
    )
    print("\n--- Results for 'Conservative' Policy ---")
    for state, value in V_conservative.items():
        print(f"  V({state.name}) = {value:.2f}")
