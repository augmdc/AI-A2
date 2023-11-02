import numpy as np
import GridWorld

# Discount factor gamma determines the agent's consideration for future rewards.
gamma = 0.9

# Learning rate alpha for Q-learning
alpha = 0.5

GOAL_REWARD = GridWorld.GOAL_REWARD
WALL_VALUE = GridWorld.WALL_VALUE

def QLearning(rewards, actions, iterations, alpha = 0.5, epsilon = 0.1, living_reward = -1):
    n = rewards[:, 0].size
    m = rewards[0, :].size
    q_values = np.zeros((n, m, 4))
    # Perform Q-learning with epsilon-greedy exploration
    for it in range(iterations):
        for i in range(n):
            for j in range(m):
                for action_index, action in enumerate(actions):
                    ni, nj = i + action[0], j + action[1]

                    # Ensure the agent stays within the grid
                    ni = max(0, min(n - 1, ni))
                    nj = max(0, min(m - 1, nj))

                    # If current state is (0,1) or (0,2), update the Q-values based on their respective rewards
                    if (i, j) == (0, 1):
                        q_values[i, j, action_index] = 10
                        continue
                    elif (i, j) == (0, 2):
                        q_values[i, j, action_index] = -10
                        continue

                    # Explore with probability epsilon or exploit with probability 1 - epsilon
                    if np.random.rand() < epsilon:
                        action_index = np.random.randint(0, len(actions))
                        action = actions[action_index]

                    # Q-learning update rule:
                    # 1. Calculate the sample using the reward and the maximum Q-value of the next state
                    sample = rewards[i, j] + gamma * np.max(q_values[ni, nj]) + living_reward

                    # 2. Update Q-value using the Q-learning update rule (weighted average)
                    q_values[i, j, action_index] = (1 - alpha) * q_values[i, j, action_index] + alpha * sample

                    # Clip the Q-value to a maximum of 10 for (0,1) and a minimum of -10 for (0,2)
                    if rewards[i, j] == 10:
                        q_values[i, j, action_index] = min(q_values[i, j, action_index], 10.0)
                    elif rewards[i, j] == -10:
                        q_values[i, j, action_index] = max(q_values[i, j, action_index], -10.0)
                    elif rewards[i, j] == WALL_VALUE:
                        q_values[i, j, action_index] = WALL_VALUE
                    elif rewards[i, j] == GOAL_REWARD:
                        q_values[i, j, action_index] = GOAL_REWARD

        epsilon /= 2
    return q_values

def Qfinal_policy(n, m, rewards, values, actions):
    # Compute the policy based on the final state values
    final_policy = np.zeros((n, m), dtype=object)  # Change dtype to object to allow strings and numbers

    for i in range(n):
        for j in range(m):
            # If terminal state
            if rewards[i][j] == GOAL_REWARD:  # rewards[i][j] == 10
                final_policy[i][j] = "*"
                continue

            # For each action, compute the resulting state and its value
            action_values = []
            for action_index, action in enumerate(actions):
                ni, nj = i + action[0], j + action[1]
                if 0 <= ni < n and 0 <= nj < m and values[ni, nj, action_index] != np.inf:
                    action_values.append(values[ni, nj, action_index])
                else:  # boundary case, make the action value small
                    action_values.append(-1000)
            best_action = max(action_values)
            best_action_idx = getmaxidx(action_values, best_action)
            final_policy[i][j] = best_action_idx

    # Convert numeric policy to direction symbols, leaving "*" as is
    directions_map = {0: "↑", 1: "↓", 2: "←", 3: "→", "*": "*"}
    symbolic_policy = np.vectorize(directions_map.get)(final_policy)

    # Print the final symbolic policy
    # print(symbolic_policy)
    return symbolic_policy


def getmaxidx(arr, val):
    for i in range(0, len(arr), 1):
        if arr[i] == val:
            return i