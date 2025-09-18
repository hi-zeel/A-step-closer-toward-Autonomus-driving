import os
import csv
import numpy as np

# --- Setup log directory ---
BASE_LOG_DIR = os.path.join(
    "log", "round4", "PA_R4__DQN__lr=0.0001_input=36-3_conv=2_FC=2_nn=100-5_batch=32"
)
os.makedirs(BASE_LOG_DIR, exist_ok=True)

log_reward_path = os.path.join(BASE_LOG_DIR, "reward.log")
log_q_values_path = os.path.join(BASE_LOG_DIR, "q_values.log")


class Log:
    """
    Base-class for logging data to a text-file during training.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.count_episodes = None
        self.count_states = None
        self.data = None

    def _write(self, count_episodes, count_states, msg):
        with open(self.file_path, mode="a", buffering=1) as file:
            msg_annotated = "{0}\t{1}\t{2}\n".format(
                count_episodes, count_states, msg
            )
            file.write(msg_annotated)

    def _read(self):
        with open(self.file_path) as f:
            reader = csv.reader(f, delimiter="\t")
            self.count_episodes, self.count_states, data = zip(*reader)
        self.data = np.array(data, dtype="float")


class LogReward(Log):
    """Log the rewards obtained for episodes during training."""

    def __init__(self):
        self.episode = None
        self.mean = None
        super().__init__(file_path=log_reward_path)

    def write(self, count_episodes, count_states, reward_episode, reward_mean):
        msg = "{0:.1f}\t{1:.1f}".format(reward_episode, reward_mean)
        self._write(count_episodes=count_episodes, count_states=count_states, msg=msg)

    def read(self):
        self._read()
        self.episode = self.data[0]
        self.mean = self.data[1]


class LogQValues(Log):
    """Log the Q-Values during training."""

    def __init__(self):
        self.min = None
        self.mean = None
        self.max = None
        self.std = None
        super().__init__(file_path=log_q_values_path)

    def write(self, count_episodes, count_states, q_values):
        msg = "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(
            np.min(q_values), np.mean(q_values), np.max(q_values), np.std(q_values)
        )
        self._write(count_episodes=count_episodes, count_states=count_states, msg=msg)

    def read(self):
        self._read()
        self.min = self.data[0]
        self.mean = self.data[1]
        self.max = self.data[2]
        self.std = self.data[3]
