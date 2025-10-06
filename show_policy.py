import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def evaluate_policy(env, policy, num_episodes=100):
    # This is the gamma value from your training loop. It must be used here as well.
    gamma = 0.95

    all_returns = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        rewards_in_episode = []
        step = 0

        while not terminated and step < 1500:
            # Your original action selection - this is fine.
            action = np.argmax(list(policy[state].values()))

            state, reward, terminated, _, _ = env.step(action)
            rewards_in_episode.append(reward)
            step += 1

        # --- THIS IS THE CRITICAL FIX ---
        # Instead of summing rewards, we calculate the discounted return.
        G = 0.0
        for reward in reversed(rewards_in_episode):
            G = reward + gamma * G

        all_returns.append(G)

    return np.mean(all_returns)


def disp(policy, title):
    """
    Creates a graphical plot of the learned policy for the 4x4 Frozen Lake environment,
    with improved arrow aesthetics and centering.
    """
    desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
    grid_size = 4

    arrow_length_ratio = 0.4
    head_width_ratio = 0.08
    head_length_ratio = 0.1

    direction_vectors = {
        0: (-1, 0),  # Left
        1: (0, 1),  # Down
        2: (1, 0),  # Right
        3: (0, -1),  # Up
    }

    color_map = {b"S": "#9fd1a8", b"F": "#a7c1e8", b"H": "#6b7482", b"G": "#e3a959"}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)

    for state in range(grid_size * grid_size):
        row, col = divmod(state, grid_size)
        state_char = desc[row][col].encode()

        rect = Rectangle((col, row), 1, 1, facecolor=color_map[state_char], edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(col + 0.05, row + 0.05, state_char.decode(), ha="left", va="top", fontsize=14, color="black")

        if state_char not in [b"H", b"G"]:
            best_action = np.argmax(list(policy[state].values()))

            base_dx, base_dy = direction_vectors[best_action]
            center_x, center_y = col + 0.5, row + 0.5
            total_arrow_span = arrow_length_ratio

            start_x = center_x - base_dx * (total_arrow_span / 2)
            start_y = center_y - base_dy * (total_arrow_span / 2)
            plot_dx = base_dx * total_arrow_span
            plot_dy = base_dy * total_arrow_span

            ax.arrow(
                start_x,
                start_y,
                plot_dx,
                plot_dy,
                head_width=head_width_ratio,
                head_length=head_length_ratio,
                fc="black",
                ec="black",
                length_includes_head=True,
            )

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()
