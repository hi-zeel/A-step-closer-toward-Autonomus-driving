import tensorflow as tf
import numpy as np
from random import choice, uniform
from collections import deque

from cnn import Cnn
from config import (
    LEARNING_RATE, EPSILON_GREEDY_START_PROB, EPSILON_GREEDY_END_PROB,
    EPSILON_GREEDY_MAX_STATES, MAX_MEM, BATCH_SIZE, VISION_W,
    VISION_B, VISION_F, TARGET_NETWORK_UPDATE_FREQUENCY, LEARN_START
)

# ✅ import logging utilities
from log import LogReward, LogQValues

# Initialize loggers
log_reward = LogReward()
log_qvalues = LogQValues()

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS


class DeepTrafficAgent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.action_names = ['A', 'D', 'M', 'L', 'R']
        self.num_actions = len(self.action_names)
        self.memory = deque()

        self.model = Cnn(self.model_name, self.memory)
        self.target_model = Cnn(self.model_name, [], target=True)

        self.previous_states = np.zeros([1, VISION_F + VISION_B + 1, VISION_W * 2 + 1, 4])
        self.previous_actions = np.zeros([4])
        self.previous_actions.fill(2)
        self.q_values = np.zeros(5)
        self.action = 2

        self.count_states = self.model.get_count_states()
        self.delay_count = 0

        self.epsilon_linear = LinearControlSignal(
            start_value=EPSILON_GREEDY_START_PROB,
            end_value=EPSILON_GREEDY_END_PROB,
            repeat=False,
        )

        self.advantage = 0
        self.value = 0
        self.score = 0

    def get_action_name(self, action):
        return self.action_names[action]

    def get_action_index(self, action):
        return self.action_names.index(action)

    def act(self, state, is_training=True):
        state = state.reshape(VISION_F + VISION_B + 1, VISION_W * 2 + 1).tolist()
        previous_states = self.previous_states.tolist()
        for n in range(len(previous_states)):
            for y in range(len(previous_states[n])):
                for x in range(len(previous_states[n][y])):
                    previous_states[n][y][x].pop(0)
                    previous_states[n][y][x].append(state[y][x])
        self.previous_states = np.array(previous_states, dtype=int)
        self.previous_states = self.previous_states.reshape(1, VISION_F + VISION_B + 1, VISION_W * 2 + 1, 4)
        self.previous_actions = np.roll(self.previous_actions, -1)
        self.previous_actions[3] = self.action
        self.q_values = self.model.get_q_values(self.previous_states, self.previous_actions)
        self.q_values = self.q_values[0][0]

        # ✅ log Q-values each step
        log_qvalues.write(
            count_episodes=self.model.get_count_episodes(),
            count_states=self.model.get_count_states(),
            q_values=self.q_values,
        )

        if is_training and self.epsilon_linear.get_value(iteration=self.model.get_count_states()) > uniform(0, 1):
            self.action = choice([0, 1, 2, 3, 4])
        else:
            self.action = np.argmax(self.q_values)

        return self.q_values, self.get_action_name(self.action)

    def remember(self, reward, next_state, end_episode=False, is_training=True):
        next_state = next_state.reshape(VISION_F + VISION_B + 1, VISION_W * 2 + 1).tolist()

        previous_states = self.previous_states.tolist()
        for n in range(len(previous_states)):
            for y in range(len(previous_states[n])):
                for x in range(len(previous_states[n][y])):
                    previous_states[n][y][x].pop(0)
                    previous_states[n][y][x].append(next_state[y][x])
        next_state = np.array(previous_states).reshape(-1, VISION_F + VISION_B + 1, VISION_W * 2 + 1, 4)

        next_actions = self.previous_actions.copy()
        next_actions = np.roll(next_actions, -1)
        next_actions[3] = self.action

        self.count_states = self.model.get_count_states()

        if is_training and self.model.get_count_states() > LEARN_START and len(self.memory) > LEARN_START:
            self.model.optimize(
                self.memory,
                learning_rate=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                target_network=self.target_model,
            )

            if self.model.get_count_states() % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                self.model.save_checkpoint(self.model.get_count_states())
                self.target_model.load_checkpoint()
                self.model.log_target_network_update()
                print("Target network updated")
            elif self.model.get_count_states() % 1000 == 0:
                self.model.save_checkpoint(self.model.get_count_states())

        if len(self.memory) > MAX_MEM:
            self.memory.popleft()

        self.memory.append((
            self.previous_states,
            next_state,
            self.action,
            reward - self.score,
            end_episode,
            self.previous_actions,
            next_actions
        ))
        self.score = reward

        if end_episode:
            # ✅ log reward at end of episode
            log_reward.write(
                count_episodes=self.model.get_count_episodes(),
                count_states=self.count_states,
                reward_episode=reward,
                reward_mean=reward,  # GUI will pass mean later
            )

            self.previous_states = np.zeros([1, VISION_F + VISION_B + 1, VISION_W * 2 + 1, 4])
            self.previous_actions = np.zeros([4])
            self.previous_actions.fill(2)
            self.q_values = np.zeros(5)
            self.action = 2
            self.score = 0

        self.count_states = self.model.increase_count_states()


class LinearControlSignal:
    def __init__(self, start_value, end_value, repeat=False):
        self.start_value = start_value
        self.end_value = end_value
        self.num_iterations = EPSILON_GREEDY_MAX_STATES
        self.repeat = repeat
        self._coefficient = (end_value - start_value) / self.num_iterations

    def get_value(self, iteration):
        if self.repeat:
            iteration %= self.num_iterations
        if iteration < self.num_iterations:
            value = iteration * self._coefficient + self.start_value
        else:
            value = self.end_value
        return value
