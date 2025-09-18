"""import math
import pygame
import os
import numpy as np

from players.player import Player
from players.aggresive_player import AggresivePlayer
from players.sticky_player import StickyPlayer
from players.deep_traffic_player import DeepTrafficPlayer

from config import VISION_B, VISION_F, VISION_W, \
    VISUALENABLED, EMERGENCY_BRAKE_MAX_SPEED_DIFF, ROAD_VIEW_OFFSET, \
    VISUAL_VISION_B, VISUAL_VISION_F, VISUAL_VISION_W


MAX_SPEED = 110  # km/h

DEFAULT_CAR_POS = 700

IMAGE_PATH = './images'

if VISUALENABLED:
    red_car = pygame.image.load(os.path.join(IMAGE_PATH, 'red_car.png'))
    red_car = pygame.transform.scale(red_car, (34, 70))
    white_car = pygame.image.load(os.path.join(IMAGE_PATH, 'white_car.png'))
    white_car = pygame.transform.scale(white_car, (34, 70))

direction_weight = {
    'L': 0.01,
    'M': 0.98,
    'R': 0.01,
}

move_weight = {
    'A': 0.30,
    'M': 0.50,
    'D': 0.20
}


class Car():
    def __init__(self, surface, lane_map, speed=0, y=0, lane=4, is_subject=False, subject=None, score=None, agent=None):
        self.surface = surface
        self.lane_map = lane_map
        self.sprite = None if not VISUALENABLED else red_car if is_subject else white_car
        self.speed = min(max(speed, 0), MAX_SPEED)
        self.y = y
        self.lane = lane
        self.x = (self.lane - 1) * 50 + 15 + 8 + ROAD_VIEW_OFFSET
        self.is_subject = is_subject
        self.subject = subject
        self.max_speed = -1
        self.removed = False
        self.emergency_brake = None

        self.switching_lane = -1
        self.available_directions = ['M']
        self.available_moves = ['D']

        self.score = score

        self.player = np.random.choice([
                Player(self),
                AggresivePlayer(self),
                StickyPlayer(self)
            ]) if not self.is_subject else DeepTrafficPlayer(self, agent=agent)

        self.hard_brake_count = 0
        self.alternate_line_switching = 0

    def identify(self):
        min_box = int(math.floor(self.y / 10.0)) - 1
        max_box = int(math.ceil(self.y / 10.0))

        # Out of bound
        if self.y < -200 or self.y > 1200:
            self.removed = True
            return False

        if 0 <= min_box < 100:
            self.lane_map[min_box][self.lane - 1] = self
            if 1 <= self.switching_lane <= 7:
                self.lane_map[min_box][self.switching_lane - 1] = self
        for i in range(-1, 9):
            if 0 <= max_box + i < 100:
                self.lane_map[max_box + i][self.lane - 1] = self
                if 1 <= self.switching_lane <= 7:
                    self.lane_map[max_box + i][self.switching_lane - 1] = self
        return True

    def accelerate(self):
        # If in front has car then cannot accelerate but follow
        self.speed += 1.0 if self.speed < MAX_SPEED else 0.0

    def decelerate(self):
        if self.max_speed > -1:
            self.speed = self.max_speed
        else:
            self.speed -= 1.0 if self.speed > 0 else 0.0

    def check_switch_lane(self):
        if self.switching_lane == -1:
            return
        self.x += (self.switching_lane - self.lane) * 50
        if self.x == ROAD_VIEW_OFFSET + (self.switching_lane - 1) * 50 + 15 + 8:
            self.lane = self.switching_lane
            self.switching_lane = -1

    def move(self, action):
        moves = self.available_moves

        if action not in moves:
            action = moves[0]
            if self.subject is None:
                self.score.action_mismatch_penalty()

        if action == 'A':
            self.accelerate()
        elif action == 'D':
            self.decelerate()

        return action

    def switch_lane(self, direction):
        directions = self.available_directions
        if direction == 'R':
            if 'R' in directions:
                if self.lane < 7:
                    self.switching_lane = self.lane + 1
                    self.identify()
                else:
                    if self.subject is None:
                        self.score.action_mismatch_penalty()
                    return 'M'
        if direction == 'L':
            if 'L' in directions:
                if self.lane > 1:
                    self.switching_lane = self.lane - 1
                    self.identify()
                else:
                    if self.subject is None:
                        self.score.action_mismatch_penalty()
                    return 'M'
        return direction

    def identify_available_moves(self):
        self.max_speed = -1
        moves = ['M', 'A', 'D']
        directions = ['M', 'L', 'R']
        if self.switching_lane >= 0:
            directions = ['M']
        if self.lane == 1 and 'L' in directions:
            directions.remove('L')
        if self.lane == 7 and 'R' in directions:
            directions.remove('R')

        max_box = int(math.ceil(self.y / 10.0)) - 1
        # Front checking
        for i in range(-1, 7):
            if 0 <= max_box + i < 100:
                if self.lane_map[max_box + i][self.lane - 1] != 0 and self.lane_map[max_box + i][self.lane - 1] != self:
                    car_in_front = self.lane_map[max_box + i][self.lane - 1]
                    if 'A' in moves:
                        moves.remove('A')
                    if car_in_front.speed < self.speed:
                        if 'M' in moves:
                            moves.remove('M')
                        self.emergency_brake = self.speed - car_in_front.speed
                        self.max_speed = car_in_front.speed
                    break
        # Consider car in target switching lane
        for i in range(-1, 7):
            if 0 <= max_box + i < 100:
                if self.switching_lane > 0:
                    if self.lane_map[max_box + i][self.switching_lane - 1] != 0 and self.lane_map[max_box + i][
                                self.switching_lane - 1] != self:
                        if 'A' in moves:
                            moves.remove('A')
                        car_in_front = self.lane_map[max_box + i][self.switching_lane - 1]
                        if car_in_front.speed < self.speed:
                            if 'M' in moves:
                                moves.remove('M')
                            # emergency_brake = self.speed - car_in_front.speed
                            self.max_speed = car_in_front.speed \
                                if self.max_speed == -1 or self.max_speed > car_in_front.speed else self.max_speed

        # Left lane checking
        if 'L' in directions:
            for i in range(0, 9):
                if 0 <= max_box + i < 100:
                    if self.lane_map[max_box + i][self.lane - 2] != 0:
                        directions.remove('L')
                        break

        # Right lane checking
        if 'R' in directions:
            for i in range(0, 9):
                if 0 <= max_box + i < 100:
                    if self.lane_map[max_box + i][self.lane] != 0:
                        directions.remove('R')
                        break
        self.available_moves = moves
        self.available_directions = directions

        return moves, directions

    def random(self):
        moves, directions = self.identify_available_moves()

        ds = np.random.choice(direction_weight.keys(), 3, p=direction_weight.values())
        ms = np.random.choice(move_weight.keys(), 3, p=move_weight.values())
        for d in ds:
            if d in directions:
                self.switch_lane(d)
                break

        for m in ms:
            if m in moves:
                self.move(m)
                break

    def relative_pos_subject(self):
        if self.is_subject:
            if self.emergency_brake is not None and self.emergency_brake > EMERGENCY_BRAKE_MAX_SPEED_DIFF:
                self.score.brake_penalty()
                self.hard_brake_count += 1
            self.emergency_brake = None
            return
        dvdt = self.speed - self.subject.speed
        dmds = dvdt / 3.6
        dbdm = 1.0 / 0.25
        dsdf = 1.0 / 50.0
        dmdf = dmds * dsdf
        dbdf = dbdm * dmdf * 10.0
        self.y = self.y - dbdf

        if DEFAULT_CAR_POS - dbdf <= self.y < DEFAULT_CAR_POS:
            self.score.subtract()
        elif DEFAULT_CAR_POS - dbdf > self.y >= DEFAULT_CAR_POS:
            self.score.add()
        self.score.penalty()

    def decide(self, end_episode, cache=False, is_training=True):
        if self.subject is None:
            q_values, result = self.player.decide_with_vision(self.get_vision(),
                                                  self.score.score,
                                                  end_episode,
                                                  cache=cache,
                                                  is_training=is_training)
            # Check for recent lane switching
            if result == 'L' or result == 'R':
                if (result == 'L' and 4 in self.player.agent.previous_actions) or \
                        (result == 'R' and 3 in self.player.agent.previous_actions):
                    self.score.switching_lane_penalty()
                    self.alternate_line_switching += 1
            return q_values, result
        else:
            return self.player.decide(end_episode, cache=cache)

    def draw(self):
        self.relative_pos_subject()
        self.check_switch_lane()
        if VISUALENABLED:
            self.surface.blit(self.sprite, (self.x, self.y, 34, 70))

    def get_vision(self):
        min_x = min(max(0, self.lane - 1 - VISION_W), 6)
        max_x = min(max(0, self.lane - 1 + VISION_W), 6)
        input_min_xx = self.lane - 1 - VISION_W
        input_max_xx = self.lane - 1 + VISION_W

        input_min_y = int(math.floor(self.y / 10.0)) - VISION_F
        input_max_y = int(math.floor(self.y / 10.0)) + VISION_B
        min_y = min(max(0, input_min_y), 100)
        max_y = min(max(0, input_max_y), 100)

        cars_in_vision = set([
            (self.lane_map[y][x].lane - 1, int(math.floor(self.lane_map[y][x].y / 10.0)))
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if self.lane_map[y][x] != 0])

        vision = np.zeros((100, 7), dtype=np.int)
        for car in cars_in_vision:
            for y in range(7):
                vision[car[1] + y][car[0]] = 1

        # Crop vision from lane_map
        vision = vision[min_y: max_y + 1, min_x: max_x + 1]

        # Add padding if required
        vision = np.pad(vision,
                        ((min_y - input_min_y, input_max_y - max_y), (min_x - input_min_xx, input_max_xx - max_x)),
                        'constant',
                        constant_values=(-1))

        vision = np.reshape(vision, [VISION_F + VISION_B + 1, VISION_W * 2 + 1, 1])
        return vision

    def get_subjective_vision(self):
        min_x = min(max(0, self.lane - 1 - VISUAL_VISION_W), 6)
        max_x = min(max(0, self.lane - 1 + VISUAL_VISION_W), 6)
        input_min_xx = self.lane - 1 - VISUAL_VISION_W
        input_max_xx = self.lane - 1 + VISUAL_VISION_W

        input_min_y = int(math.floor(self.y / 10.0)) - VISUAL_VISION_F
        input_max_y = int(math.floor(self.y / 10.0)) + VISUAL_VISION_B
        min_y = min(max(0, input_min_y), 100)
        max_y = min(max(0, input_max_y), 100)

        cars = [
            (self.lane_map[y][x].lane, int(math.floor(self.lane_map[y][x].y / 10.0)))
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if self.lane_map[y][x] != 0 and self.lane_map[y][x].subject is not None]

        return cars """

# ============================================
# car.py — vehicle model + local “vision” logic
# ============================================
# Key changes from original:
# - No hard-coded 7 lanes anymore.
# - All lane-bound checks / indexing use LANE_COUNT and LANE_INDEX_MAX from config.
# - Vision clamping updated to prevent IndexError when road width is changed.

import math
import os
import numpy as np
import pygame

# Built-in / hand-crafted players
from players.player import Player
from players.aggresive_player import AggresivePlayer
from players.sticky_player import StickyPlayer
from players.deep_traffic_player import DeepTrafficPlayer

# Pull in lane layout + UI toggles from config
from config import (
    VISION_B, VISION_F, VISION_W,
    VISUALENABLED, EMERGENCY_BRAKE_MAX_SPEED_DIFF, ROAD_VIEW_OFFSET,
    VISUAL_VISION_B, VISUAL_VISION_F, VISUAL_VISION_W,
    LANE_COUNT, LANE_INDEX_MAX,
)

# -------------------------
# Constants / assets
# -------------------------
MAX_SPEED = 110          # km/h
DEFAULT_CAR_POS = 700    # initial Y of the subject car (pixels)
IMAGE_PATH = './images'

# Only load sprites if we actually render
if VISUALENABLED:
    red_car = pygame.image.load(os.path.join(IMAGE_PATH, 'red_car.png'))
    red_car = pygame.transform.scale(red_car, (34, 70))
    white_car = pygame.image.load(os.path.join(IMAGE_PATH, 'white_car.png'))
    white_car = pygame.transform.scale(white_car, (34, 70))

# Policy priors for random NPCs (not the DL agent)
direction_weight = {
    'L': 0.01,  # small chance to go left
    'M': 0.98,  # mostly keep lane
    'R': 0.01,  # small chance to go right
}

move_weight = {
    'A': 0.30,  # accelerate
    'M': 0.50,  # maintain
    'D': 0.20,  # decelerate
}


class Car:
    """
    A car knows:
      - where it is (lane, x/y)
      - how fast it is moving
      - whether it’s the subject (DL-controlled) or an NPC
      - how to mark itself in the lane_map occupancy grid
      - how to compute the local "vision" tensor for the agent
    """

    def __init__(self, surface, lane_map, speed=0, y=0, lane=4,
                 is_subject=False, subject=None, score=None, agent=None):
        self.surface = surface
        self.lane_map = lane_map
        self.sprite = None if not VISUALENABLED else (red_car if is_subject else white_car)

        # Clamp speed to [0, MAX_SPEED]
        self.speed = float(min(max(speed, 0), MAX_SPEED))

        # Continuous position on screen
        self.y = float(y)
        self.lane = int(lane)  # lanes are 1-indexed in this codebase (1..LANE_COUNT)

        # Convert lane index to on-screen x pixel
        #  - 50px width per lane, with a small inset of 15+8 for drawing alignment
        self.x = (self.lane - 1) * 50 + 15 + 8 + ROAD_VIEW_OFFSET

        # Identity / role
        self.is_subject = is_subject
        self.subject = subject  # if this is an NPC, subject = the subject car
        self.score = score

        # Stats / flags
        self.max_speed = -1
        self.removed = False
        self.emergency_brake = None
        self.hard_brake_count = 0
        self.alternate_line_switching = 0

        # Lane-switch state (-1 means not switching)
        self.switching_lane = -1

        # Movement affordances for the current frame
        self.available_directions = ['M']  # 'L', 'M', 'R'
        self.available_moves = ['D']       # 'A', 'M', 'D'

        # Controller: DL agent for subject; simple scripted player for NPCs
        self.player = (
            DeepTrafficPlayer(self, agent=agent)
            if self.is_subject else
            np.random.choice([
                Player(self),
                AggresivePlayer(self),
                StickyPlayer(self),
            ])
        )

    # -------------------------
    # Occupancy & kinematics
    # -------------------------
    def identify(self):
        """
        Mark this car in the lane_map grid so other cars can "see" it.
        lane_map has shape [100 rows (y-bins)] x [LANE_COUNT lanes].
        """
        min_box = int(math.floor(self.y / 10.0)) - 1
        max_box = int(math.ceil(self.y / 10.0))

        # Cull if far off-screen (safety)
        if self.y < -200 or self.y > 1200:
            self.removed = True
            return False

        # Helper to set a cell if indices are valid
        def set_cell(row, lane_index, value):
            if 0 <= row < 100 and 0 <= lane_index <= LANE_INDEX_MAX:
                self.lane_map[row][lane_index] = value

        # Mark current lane cells
        if 0 <= min_box < 100:
            set_cell(min_box, self.lane - 1, self)
            if 1 <= self.switching_lane <= LANE_COUNT:
                set_cell(min_box, self.switching_lane - 1, self)

        # Spread occupancy downward for a few bins to make detection easier
        for i in range(-1, 9):
            row = max_box + i
            if 0 <= row < 100:
                set_cell(row, self.lane - 1, self)
                if 1 <= self.switching_lane <= LANE_COUNT:
                    set_cell(row, self.switching_lane - 1, self)

        return True

    def accelerate(self):
        # Simple capped acceleration
        if self.speed < MAX_SPEED:
            self.speed += 1.0

    def decelerate(self):
        # If we computed a safe max_speed this frame, snap to it
        if self.max_speed > -1:
            self.speed = float(self.max_speed)
        else:
            if self.speed > 0:
                self.speed -= 1.0

    def check_switch_lane(self):
        """
        If we are mid-switch, advance the sprite’s x toward the target lane center.
        """
        if self.switching_lane == -1:
            return
        # Move one full-lane step instantly (original behavior)
        self.x += (self.switching_lane - self.lane) * 50
        # Snap completed?
        target_x = ROAD_VIEW_OFFSET + (self.switching_lane - 1) * 50 + 15 + 8
        if self.x == target_x:
            self.lane = self.switching_lane
            self.switching_lane = -1

    def move(self, action):
        """
        Apply 'A'ccelerate / 'D'ecelerate / 'M'aintain, respecting available moves.
        """
        moves = self.available_moves
        if action not in moves:
            # If controller asked for an impossible move, penalize NPCs
            action = moves[0]
            if self.subject is None and self.score is not None:
                self.score.action_mismatch_penalty()

        if action == 'A':
            self.accelerate()
        elif action == 'D':
            self.decelerate()
        # 'M' => keep current speed

        return action

    def switch_lane(self, direction):
        """
        Try to switch lanes 'L' or 'R' if allowed; otherwise return 'M'.
        """
        directions = self.available_directions

        if direction == 'R':
            if 'R' in directions:
                if self.lane < LANE_COUNT:
                    self.switching_lane = self.lane + 1
                    self.identify()
                else:
                    if self.subject is None and self.score is not None:
                        self.score.action_mismatch_penalty()
                    return 'M'

        if direction == 'L':
            if 'L' in directions:
                if self.lane > 1:
                    self.switching_lane = self.lane - 1
                    self.identify()
                else:
                    if self.subject is None and self.score is not None:
                        self.score.action_mismatch_penalty()
                    return 'M'

        return direction

    # -------------------------
    # Perception & decisions
    # -------------------------
    def identify_available_moves(self):
        """
        Compute which speed actions and lane directions are legal for this frame.
        Also computes a safe max_speed if something is blocking in front.
        """
        self.max_speed = -1

        # Start permissive; prune as we detect obstacles
        moves = ['M', 'A', 'D']
        directions = ['M', 'L', 'R']

        # If currently switching lanes, don’t allow a new lateral decision
        if self.switching_lane >= 0:
            directions = ['M']

        # Edge lanes cannot go further outward
        if self.lane == 1 and 'L' in directions:
            directions.remove('L')
        if self.lane == LANE_COUNT and 'R' in directions:
            directions.remove('R')

        max_box = int(math.ceil(self.y / 10.0)) - 1

        # --- (1) Front checking on current lane: block A/M if a slower car ahead
        for i in range(-1, 7):
            row = max_box + i
            if 0 <= row < 100:
                here = self.lane_map[row][self.lane - 1]
                if here != 0 and here is not self:
                    car_in_front = here
                    if 'A' in moves:
                        moves.remove('A')
                    if car_in_front.speed < self.speed:
                        if 'M' in moves:
                            moves.remove('M')
                        self.emergency_brake = self.speed - car_in_front.speed
                        self.max_speed = car_in_front.speed
                    break

        # --- (2) If in the middle of switching, also consider the target lane’s front
        for i in range(-1, 7):
            row = max_box + i
            if 0 <= row < 100 and self.switching_lane > 0:
                target = self.lane_map[row][self.switching_lane - 1]
                if target != 0 and target is not self:
                    if 'A' in moves:
                        moves.remove('A')
                    car_in_front = target
                    if car_in_front.speed < self.speed:
                        if 'M' in moves:
                            moves.remove('M')
                        # Keep the most restrictive safe max speed
                        self.max_speed = car_in_front.speed if (self.max_speed == -1 or
                                                                self.max_speed > car_in_front.speed) else self.max_speed

        # --- (3) Left lane occupancy in a short lookahead window: block 'L' if busy
        if 'L' in directions:
            for i in range(0, 9):
                row = max_box + i
                if 0 <= row < 100:
                    # self.lane > 1 guaranteed if 'L' present
                    if self.lane_map[row][self.lane - 2] != 0:
                        directions.remove('L')
                        break

        # --- (4) Right lane occupancy: block 'R' if busy
        if 'R' in directions:
            for i in range(0, 9):
                row = max_box + i
                if 0 <= row < 100:
                    # self.lane < LANE_COUNT guaranteed if 'R' present
                    if self.lane_map[row][self.lane] != 0:
                        directions.remove('R')
                        break

        self.available_moves = moves
        self.available_directions = directions
        return moves, directions

    def random(self):
        """
        Simple random policy used by NPCs (not the DL agent).
        """
        moves, directions = self.identify_available_moves()

        ds = np.random.choice(list(direction_weight.keys()), 3, p=list(direction_weight.values()))
        ms = np.random.choice(list(move_weight.keys()), 3, p=list(move_weight.values()))

        for d in ds:
            if d in directions:
                self.switch_lane(d)
                break

        for m in ms:
            if m in moves:
                self.move(m)
                break

    def relative_pos_subject(self):
        """
        Update this car’s Y relative to the subject car’s speed; apply penalties.
        For the subject itself, count hard brakes.
        """
        if self.is_subject:
            if self.emergency_brake is not None and self.emergency_brake > EMERGENCY_BRAKE_MAX_SPEED_DIFF:
                if self.score is not None:
                    self.score.brake_penalty()
                self.hard_brake_count += 1
            self.emergency_brake = None
            return

        # NPCs move relative to the subject’s speed (discrete-time physics proxy)
        dvdt = self.speed - self.subject.speed
        dmds = dvdt / 3.6
        dbdm = 1.0 / 0.25
        dsdf = 1.0 / 50.0
        dmdf = dmds * dsdf
        dbdf = dbdm * dmdf * 10.0
        self.y = self.y - dbdf

        # Reward / penalty shaping around DEFAULT_CAR_POS band
        if DEFAULT_CAR_POS - dbdf <= self.y < DEFAULT_CAR_POS:
            self.score.subtract()
        elif DEFAULT_CAR_POS - dbdf > self.y >= DEFAULT_CAR_POS:
            self.score.add()
        self.score.penalty()

    def decide(self, end_episode, cache=False, is_training=True):
        """
        Ask the controller for an action. For NPCs this uses shallow heuristics;
        for the subject it uses the DeepTrafficPlayer (DQN).
        """
        if self.subject is None:
            # NPC controlled by DL agent wrapper (decide_with_vision expects CNN input)
            q_values, result = self.player.decide_with_vision(
                self.get_vision(),
                self.score.score,
                end_episode,
                cache=cache,
                is_training=is_training
            )

            # Penalize rapid L<->R alternations
            if result in ('L', 'R'):
                if (result == 'L' and 4 in self.player.agent.previous_actions) or \
                   (result == 'R' and 3 in self.player.agent.previous_actions):
                    self.score.switching_lane_penalty()
                    self.alternate_line_switching += 1
            return q_values, result
        else:
            # Subject car
            return self.player.decide(end_episode, cache=cache)

    def draw(self):
        """
        Advance position, then draw sprite if visuals are enabled.
        """
        self.relative_pos_subject()
        self.check_switch_lane()
        if VISUALENABLED and self.sprite is not None:
            self.surface.blit(self.sprite, (self.x, self.y, 34, 70))

    # -------------------------
    # Vision tensors
    # -------------------------
    def get_vision(self):
        """
        Build the compact occupancy tensor for the CNN:
          shape = [VISION_F + VISION_B + 1, VISION_W*2 + 1, 1]
        This is extracted from a coarse [100 x LANE_COUNT] occupancy grid.
        """
        # Compute x-range (clamped to available lanes)
        min_x = min(max(0, self.lane - 1 - VISION_W), LANE_INDEX_MAX)
        max_x = min(max(0, self.lane - 1 + VISION_W), LANE_INDEX_MAX)
        input_min_xx = self.lane - 1 - VISION_W
        input_max_xx = self.lane - 1 + VISION_W

        # Compute y-range (in 10px “boxes”)
        input_min_y = int(math.floor(self.y / 10.0)) - VISION_F
        input_max_y = int(math.floor(self.y / 10.0)) + VISION_B
        min_y = min(max(0, input_min_y), 100)
        max_y = min(max(0, input_max_y), 100)

        # Collect all cars in the clamped window
        cars_in_vision = set([
            (self.lane_map[y][x].lane - 1, int(math.floor(self.lane_map[y][x].y / 10.0)))
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if self.lane_map[y][x] != 0
        ])

        # Global occupancy canvas (100 x LANE_COUNT), then mark a short vertical stack for each seen car
        vision = np.zeros((100, LANE_COUNT), dtype=np.int)
        for lane_x, box_y in cars_in_vision:
            for dy in range(7):  # keeps original vertical “thickness” heuristic
                yy = box_y + dy
                if 0 <= yy < 100:
                    vision[yy][lane_x] = 1

        # Crop to the [min_y:max_y] x [min_x:max_x] window
        vision = vision[min_y: max_y + 1, min_x: max_x + 1]

        # Pad if the requested window would extend outside the world
        vision = np.pad(
            vision,
            ((min_y - input_min_y, input_max_y - max_y), (min_x - input_min_xx, input_max_xx - max_x)),
            'constant',
            constant_values=(-1)
        )

        # Final shape expected by the CNN
        vision = np.reshape(vision, [VISION_F + VISION_B + 1, VISION_W * 2 + 1, 1])
        return vision

    def get_subjective_vision(self):
        """
        Return a list of (lane_index, y_box) for objects in the larger *visual* overlay window.
        Used by the advanced view UI.
        """
        # x-range clamped to lanes
        min_x = min(max(0, self.lane - 1 - VISUAL_VISION_W), LANE_INDEX_MAX)
        max_x = min(max(0, self.lane - 1 + VISUAL_VISION_W), LANE_INDEX_MAX)
        input_min_xx = self.lane - 1 - VISUAL_VISION_W
        input_max_xx = self.lane - 1 + VISUAL_VISION_W

        # y-range in boxes
        input_min_y = int(math.floor(self.y / 10.0)) - VISUAL_VISION_F
        input_max_y = int(math.floor(self.y / 10.0)) + VISUAL_VISION_B
        min_y = min(max(0, input_min_y), 100)
        max_y = min(max(0, input_max_y), 100)

        # Collect cars with a subject attribute (i.e., the subject itself)
        cars = [
            (self.lane_map[y][x].lane, int(math.floor(self.lane_map[y][x].y / 10.0)))
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if self.lane_map[y][x] != 0 and self.lane_map[y][x].subject is not None
        ]

        return cars

