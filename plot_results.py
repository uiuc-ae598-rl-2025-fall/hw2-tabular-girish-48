import matplotlib.pyplot as plt
import numpy as np

plt.style.use("./design.mplstyle")

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=False)
# fig.suptitle("Evaluation Return Plots", fontsize=12)

conditions = ["non_slippery", "slippery"]

for i, condition in enumerate(conditions):
    ax = axes[i]

    for name, color in [("sarsa", "C0"), ("qlearning", "C1"), ("mc_control", "C2")]:
        data = np.load(f"{name}_{condition}.npz")
        all_timesteps = data["timesteps"]
        mean_returns = data["mean"]
        std_returns = data["std"]

        ax.plot(all_timesteps, mean_returns, label=name, color=color)
        ax.fill_between(
            all_timesteps,
            mean_returns - std_returns,
            mean_returns + std_returns,
            alpha=0.3,
            label=f"Std for {name}",
            color=color,
        )

    ax.set_title(f"{condition}", fontsize=13)
    ax.set_ylabel("Returns averaged over 100 runs")

    if condition == "slippery":
        ax.set_ylim(0, 0.25)
        ax.axhline(0.18, color="black", linestyle="--", label="Optimal Return")
    else:
        ax.set_ylim(0, 1)
        ax.axhline(0.77, color="black", linestyle="--", label="Optimal Return")

    ax.legend(ncol=4)
    ax.grid(False)


axes[1].set_xlabel("Time Steps")
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
# plt.savefig("evaluation_returns.pdf", dpi=300, bbox_inches="tight")
plt.show()
