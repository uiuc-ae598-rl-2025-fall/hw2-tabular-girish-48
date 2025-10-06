#!/Users/erased/miniforge3/envs/CS/bin/python

import numpy as np
import gymnasium as gym
from show_policy import disp, evaluate_policy

from tqdm import tqdm
import os
import multiprocessing
import matplotlib.pyplot as plt


def simulation(run_id):
    env = gym.make(
        "FrozenLake-v1",
        desc=["SFFF", "FHFH", "FFFH", "HFFG"],
        map_name="4x4",
        is_slippery=True,
        success_rate=1.0 / 3.0,
        reward_schedule=(1, 0, 0),
    )
    env.reset(seed=run_id)
    np.random.seed(run_id)

    eps_decay = 0.999999
    eps_min = 0.0001
    eps = 1
    gamma = 0.95
    N = int(5e5)

    alpha = 0.2  # 0.01
    alpha_decay = 0.99999
    alpha_min = 0.001
    terminal = [5, 7, 11, 12, 15]

    Q = {state: {action: 0 for action in range(4)} for state in range(16)}

    average_returns = []
    timesteps = []

    def A(q, eps):  # Choosing the action from Q table
        max_action = np.argmax(list(q.values()))
        remaining_actions = [i for i in range(4) if i != max_action]
        if np.random.uniform(0, 1) < 1 - 0.75 * eps:
            return max_action
        else:
            return np.random.choice(remaining_actions)

    for j in range(N):
        state, _ = env.reset()
        action = A(Q[state], eps)

        # Episode
        while True:
            # state != terminal state
            if state not in terminal:
                next_state, reward, terminated, _, _ = env.step(action)
                next_action = A(Q[next_state], eps)
                Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
                state = next_state
                action = next_action
            else:
                break

        # Slowly moving policy towards being greedy
        eps = max(eps_min, eps * eps_decay)
        alpha = max(alpha_min, alpha * alpha_decay)

        if j > 50 and j % 100 == 0:
            avg_return = evaluate_policy(env, Q, num_episodes=100)
            average_returns.append(avg_return)
            timesteps.append(j)

    # disp(Q, "SARSA Slippery")
    env.close()
    return (timesteps, average_returns)


if __name__ == "__main__":
    NUM_RUNS = 100

    multiprocessing.set_start_method("spawn", force=True)

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        run_ids = range(NUM_RUNS)

        print(f"Running {NUM_RUNS} experiments in parallel...")
        results = list(tqdm(pool.imap_unordered(simulation, run_ids), total=NUM_RUNS))

    all_timesteps = results[0][0]
    all_returns = np.array([res[1] for res in results])

    mean_returns = np.mean(all_returns, axis=0)
    std_returns = np.std(all_returns, axis=0)
    output_filename = "sarsa_non_slippery.npz"
    np.savez_compressed(
        output_filename, timesteps=all_timesteps, returns=all_returns, mean=mean_returns, std=std_returns
    )
