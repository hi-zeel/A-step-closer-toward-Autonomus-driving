from players.player import Player
import numpy as np


class DeepTrafficPlayer(Player):
    def decide_with_vision(self, vision, score, end_episode, cache=False, is_training=True):
        if cache:
            self.car.move(self.action_cache)
            return None, None

        action = 'M'  # Default action
        if is_training and not cache:
            self.agent.remember(score, vision, end_episode=end_episode, is_training=is_training)

        if self.car.switching_lane < 0:
            q_values, action = self.agent.act(vision, is_training=is_training)
            self.agent_action = True
        else:
            self.agent_action = False

        mismatch_direction = False
        resulted_direction = 'M'

        if self.agent_action:
            if action in ['A', 'D', 'M']:
                self.car.switch_lane('M')
                mismatch_direction = True
            else:
                resulted_direction = self.car.switch_lane(action)
                if resulted_direction[0] != action[0]:
                    self.agent.action = self.agent.action_names.index(resulted_direction)
                    mismatch_direction = True
                action = 'M'

        resulted_action = self.car.move(action)
        if resulted_action != action:
            self.agent.action = self.agent.action_names.index(resulted_action)
        self.action_cache = resulted_action
        result = resulted_direction if not mismatch_direction else resulted_action

        return q_values if self.agent_action else None, \
            result if self.agent_action else None  



"""import tensorflow as tf
import numpy as np
from random import choice, uniform
from collections import deque

from cnn import Cnn
from config import LEARNING_RATE, EPSILON_GREEDY_START_PROB, EPSILON_GREEDY_END_PROB, EPSILON_GREEDY_MAX_STATES, \
    MAX_MEM, BATCH_SIZE, VISION_W, VISION_B, VISION_F, TARGET_NETWORK_UPDATE_FREQUENCY, LEARN_START

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

FLAGS = tf.app.flags.FLAGS


class DeepTrafficAgent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.action_names = ['A', 'D', 'M', 'L', 'R']
        self.num_actions = len(self.action_names)
        self.memory = deque()

        # models
        self.model = Cnn(self.model_name, self.memory)
        self.target_model = Cnn(self.model_name, [], target=True)

        # state buffers
        self.previous_states = np.zeros([1, VISION_F + VISION_B + 1, VISION_W * 2 + 1, 4])
        self.previous_actions = np.zeros([4])
        self.previous_actions.fill(2)
        self.q_values = np.zeros(5)
        self.action = 2

        self.count_states = self.model.get_count_states()
        self.delay_count = 0

        # epsilon schedule
        self.epsilon_linear = LinearControlSignal(start_value=EPSILON_GREEDY_START_PROB,
                                                  end_value=EPSILON_GREEDY_END_PROB,
                                                  repeat=False)

        self.advantage = 0
        self.value = 0
        self.score = 0

        # ✅ summary writer for TensorBoard
        self.summary_writer = tf.compat.v1.summary.FileWriter("log/round4/" + self.model_name)

    def get_action_name(self, action):
        return self.action_names[action]

    def get_action_index(self, action):
        return self.action_names.index(action)

    def act(self, state, is_training=True):
        """ # Choose an action based on epsilon-greedy Q-values
"""
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

        epsilon = self.epsilon_linear.get_value(iteration=self.model.get_count_states())

        if is_training and epsilon > uniform(0, 1):
            self.action = choice([0, 1, 2, 3, 4])
        else:
            self.action = np.argmax(self.q_values)

        # ✅ log epsilon occasionally
        if self.model.get_count_states() % 100 == 0:
            summary = tf.compat.v1.Summary(
                value=[tf.compat.v1.Summary.Value(tag="epsilon", simple_value=epsilon)]
            )
            self.summary_writer.add_summary(summary, self.model.get_count_states())
            self.summary_writer.flush()

        return self.q_values, self.get_action_name(self.action)

    def remember(self, reward, next_state, end_episode=False, is_training=True):
        """ #Store experience and train model if conditions are met
"""
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

        # optimize the model if enough data collected
        if is_training and self.model.get_count_states() > LEARN_START and len(self.memory) > LEARN_START:
            loss = self.model.optimize(self.memory,
                                       learning_rate=LEARNING_RATE,
                                       batch_size=BATCH_SIZE,
                                       target_network=self.target_model)

            # ✅ log loss
            summary = tf.compat.v1.Summary(
                value=[tf.compat.v1.Summary.Value(tag="loss", simple_value=float(loss))]
            )
            self.summary_writer.add_summary(summary, self.model.get_count_states())
            self.summary_writer.flush()

            if self.model.get_count_states() % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                self.model.save_checkpoint(self.model.get_count_states())
                self.target_model.load_checkpoint()
                self.model.log_target_network_update()
                print("Target network updated")
            elif self.model.get_count_states() % 1000 == 0:
                self.model.save_checkpoint(self.model.get_count_states())

        # manage memory
        if len(self.memory) > MAX_MEM:
            self.memory.popleft()
        self.memory.append((self.previous_states,
                            next_state,
                            self.action,
                            reward - self.score,
                            end_episode,
                            self.previous_actions,
                            next_actions))

        # ✅ log reward
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag="reward", simple_value=float(reward))]
        )
        self.summary_writer.add_summary(summary, self.model.get_count_states())
        self.summary_writer.flush()

        self.score = reward

        if end_episode:
            self.previous_states = np.zeros([1, VISION_F + VISION_B + 1, VISION_W * 2 + 1, 4])
            self.previous_actions = np.zeros([4])
            self.previous_actions.fill(2)
            self.q_values = np.zeros(5)
            self.action = 2
            self.score = 0

        self.count_states = self.model.increase_count_states()


class LinearControlSignal:
    """#Linearly decaying epsilon-greedy control signal
"""
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
        return value """
