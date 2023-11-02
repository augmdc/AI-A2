import numpy as np
import GridWorld

# Discount factor gamma determines the agent's consideration for future rewards.
gamma = 0.9

# Learning rate alpha for Q-learning
alpha = 0.5

GOAL_REWARD = GridWorld.GOAL_REWARD
WALL_VALUE = GridWorld.WALL_VALUE
GOLD_REWARD = GridWorld.GOLD_REWARD
TRAP_PENALTY = GridWorld.TRAP_PENALTY


def Gen_Path(world, values, rewards, actions, epsilon):
    i, j = world.robot_pos
    n = rewards[:, 0].size
    m = rewards[0, :].size
    path = [(i, j)]
    while rewards[i, j] != GOAL_REWARD:
        # For each action, compute the resulting state and its value
        action_values = []
        for action_index, action in enumerate(actions):
            ti, tj = i + action[0], j + action[1]
            if 0 <= ti < n and 0 <= tj < m and values[ti, tj, action_index] != WALL_VALUE:
                action_values.append(values[ti, tj, action_index])
            else:  # boundary case, make the action value small
                action_values.append(-1000)
        best_action = max(action_values)
        best_action_idx = getmaxidx(action_values, best_action)
        action = actions[best_action_idx]

        # Explore with probability epsilon or exploit with probability 1 - epsilon
        if np.random.rand() < epsilon:
            ti = np.inf
            tj = np.inf
            while not (0 <= i + ti < n and 0 <= j + tj < m):
                action_index = np.random.randint(0, len(actions))
                action = actions[action_index]

        ni, nj = i + action[0], j + action[1]
        i = ni
        j = nj
        path.append((i,j))

    return path



def QLearning(rewards, actions, iterations, alpha = 0.5, epsilon = 0.1, living_reward = -1):
    n = rewards[:, 0].size
    m = rewards[0, :].size
    q_values = np.zeros((n, m, 4))
    # Perform Q-learning with epsilon-greedy exploration
    for it in range(iterations):
        for i in range(n):
            for j in range(m):
                for action_index, action in enumerate(actions):
                    randaction = None
                    # Explore with probability epsilon or exploit with probability 1 - epsilon
                    if np.random.rand() < epsilon:
                        action_index = np.random.randint(0, len(actions))
                        randaction = actions[action_index]
                    if randaction is not None:
                        ni, nj = i + randaction[0], j + randaction[1]
                    else:
                        ni, nj = i + action[0], j + action[1]
                    # if the action takes us out of bounds, just assign the q value in that direction to the wall value
                    if not (0 <= ni < n and 0 <= nj < n):
                        q_values[i, j, action_index] = WALL_VALUE
                        continue
                    # Q-learning update rule:
                    # 1. Calculate the sample using the reward and the maximum Q-value of the next state
                    sample = rewards[i, j] + gamma * np.max(q_values[ni, nj]) + living_reward

                    # 2. Update Q-value using the Q-learning update rule (weighted average)
                    q_values[i, j, action_index] = (1 - alpha) * q_values[i, j, action_index] + alpha * sample

                    # Clip the Q-value if it has reached GOLD, TRAP, GOAL or WALL
                    if rewards[i, j] == GOLD_REWARD:
                        q_values[i, j, action_index] = min(q_values[i, j, action_index], 10.0)
                    elif rewards[i, j] == TRAP_PENALTY:
                        q_values[i, j, action_index] = max(q_values[i, j, action_index], -10.0)
                    elif rewards[i, j] == WALL_VALUE:
                        q_values[i, j, action_index] = WALL_VALUE
                    elif rewards[i, j] == GOAL_REWARD:
                        q_values[i, j, action_index] = min(q_values[i, j, action_index], GOAL_REWARD)

        epsilon /= 1.1 #decrease epsilon each time
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
                    action_values.append(values[i, j, action_index])
                else:  # boundary case, make the action value small
                    action_values.append(-1000)
            best_action = max(action_values)
            best_action_idx = getmaxidx(action_values, best_action)
            final_policy[i][j] = best_action_idx

    # Convert numeric policy to direction symbols, leaving "*" as is(0, -1), (-1, 0), (0, 1), (1, 0)
    directions_map = {0: "←", 1: "↑", 2: "→", 3: "↓", "*": "*"}
    symbolic_policy = np.vectorize(directions_map.get)(final_policy)

    # Print the final symbolic policy
    # print(symbolic_policy)
    return symbolic_policy

#gets the idx of the max value of the array
def getmaxidx(arr, val):
    for i in range(0, len(arr), 1):
        if arr[i] == val:
            return i

