import numpy as np
import gymnasium as gym
from show_policy import evaluate_policy, disp
from tqdm import tqdm
import multiprocessing
import os

import matplotlib.pyplot as plt


def simulation(run_id):
    env = gym.make("FrozenLake-v1", desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=False)
    env.reset(seed=run_id)
    np.random.seed(run_id)

    eps_decay = 0.999999
    eps_min = 0.0001
    eps = 1
    gamma = 0.95
    N = int(1e4)

    alpha = 0.1
    alpha_decay = 0.99999
    alpha_min = 0.001

    policy = {state: {action: 0.25 for action in range(4)} for state in range(16)}
    Q = {state: {action: 0 for action in range(4)} for state in range(16)}
    n = {state: {action: 0 for action in range(4)} for state in range(16)}

    average_returns = []
    timesteps = []

    for j in range(N):
        curr_state, _ = env.reset()
        rollout = []
        rewards = []

        while True:
            action = np.random.choice([0, 1, 2, 3], p=list(policy[curr_state].values()))
            state = curr_state
            curr_state, reward, terminated, _, _ = env.step(action)
            rollout.append((state, action))
            rewards.append(reward)
            if terminated:
                break

        G = 0
        for i in reversed(range(len(rollout))):
            G = rewards[i] + gamma * G
            if rollout[i] not in rollout[:i]:
                state, action = rollout[i]
                n[state][action] += 1
                Q[state][action] += alpha * (G - Q[state][action]) / n[state][action]

                max_action = np.argmax(list(Q[state].values()))
                for a in range(4):
                    if a == max_action:
                        policy[state][a] = 1 - 0.75 * eps
                    else:
                        policy[state][a] = 0.25 * eps

        if j > 50 and j % 50 == 0:
            avg_return = evaluate_policy(env, policy, num_episodes=100)
            average_returns.append(avg_return)
            timesteps.append(j)

        eps = max(eps * eps_decay, eps_min)
        alpha = max(alpha_min, alpha * alpha_decay)

    env.close()
    # disp(Q, "MC Control Slippery")
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
    output_filename = "mc_control_non_slippery.npz"

    np.savez_compressed(
        output_filename, timesteps=all_timesteps, returns=all_returns, mean=mean_returns, std=std_returns
    )
