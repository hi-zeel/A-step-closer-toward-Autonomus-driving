import os
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# Set your log directory (absolute path)
# =====================================
run_dir = r"C:\Carla\Projects\Reinforcement-Learning-for-Self-Driving-Cars-master\log\round4\PA_R4__DQN__lr=0.0001_input=36-3_conv=2_FC=2_nn=100-5_batch=32"

reward_log = os.path.join(run_dir, "reward.log")
q_log = os.path.join(run_dir, "q_values.log")


def load_log(path, expected_cols):
    if not os.path.exists(path):
        print(f"⚠️ Log not found: {path}")
        return None
    data = np.loadtxt(path, delimiter="\t")
    if data.ndim == 1:  # single row
        data = data.reshape(1, -1)
    if data.shape[1] < expected_cols:
        raise ValueError(f"Unexpected log format in {path}")
    return data


if __name__ == "__main__":
    # -------------------------------
    # Load logs
    # -------------------------------
    reward_data = load_log(reward_log, 4)  # episode, states, ep_reward, mean_reward
    q_data = load_log(q_log, 6)            # episode, states, min, mean, max, std

    # -------------------------------
    # Plot rewards
    # -------------------------------
    if reward_data is not None:
        episodes = reward_data[:, 0]
        ep_rewards = reward_data[:, 2]
        mean_rewards = reward_data[:, 3]

        plt.figure(figsize=(10, 5))
        plt.plot(episodes, ep_rewards, label="Episode Reward", alpha=0.6)
        plt.plot(episodes, mean_rewards, label="Mean Reward (moving avg)", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Rewards")
        plt.legend()
        plt.grid(True)
        plt.show()

    # -------------------------------
    # Plot Q-values
    # -------------------------------
    if q_data is not None:
        episodes = q_data[:, 0]
        q_min = q_data[:, 2]
        q_mean = q_data[:, 3]
        q_max = q_data[:, 4]
        q_std = q_data[:, 5]

        plt.figure(figsize=(10, 5))
        plt.plot(episodes, q_min, label="Q Min", alpha=0.6)
        plt.plot(episodes, q_mean, label="Q Mean", linewidth=2)
        plt.plot(episodes, q_max, label="Q Max", alpha=0.6)
        plt.fill_between(episodes, q_mean - q_std, q_mean + q_std, alpha=0.2, label="Q Std")
        plt.xlabel("Episode")
        plt.ylabel("Q-values")
        plt.title("Q-values over Training")
        plt.legend()
        plt.grid(True)
        plt.show()
