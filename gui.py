
"""
# Deep Traffic
import os
# Import required packages
import pygame
import sys
from pygame.locals import *
import numpy as np

# Import model and GUI related modules
from car import Car, DEFAULT_CAR_POS
from gui_util import draw_basic_road, \
    draw_road_overlay_safety, \
    draw_road_overlay_vision, \
    control_car, \
    identify_free_lane, \
    Score, \
    draw_inputs, \
    draw_actions, \
    draw_gauge, \
    draw_score
from deep_traffic_agent import DeepTrafficAgent

# Advanced view
from advanced_view.road import AdvancedRoad

import config

# Model name
model_name = config.MODEL_NAME

deep_traffic_agent = DeepTrafficAgent(model_name)

# Define game constant
OPTIMAL_CARS_IN_SCENE = 8
ACTION_MAP = ['A', 'M', 'D', 'L', 'R']
monitor_keys = [pygame.K_UP, pygame.K_RIGHT, pygame.K_LEFT, pygame.K_DOWN]

if config.VISUALENABLED:
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption('DeepTraffic')
    fpsClock = pygame.time.Clock()

    main_surface = pygame.display.set_mode((1600, 800), pygame.DOUBLEBUF | pygame.HWSURFACE)
    advanced_road = AdvancedRoad(main_surface, 0, 550, 1010, 800, lane=4)
else:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    main_surface = None

lane_map = [[0 for x in range(4)] for y in range(100)]

episode_count = deep_traffic_agent.model.get_count_episodes()

speed_counter_avg = []
hard_brake_avg = []
alternate_line_switching = []

action_stats = np.zeros(5, np.int32)

PREDEFINED_MAX_CAR = config.MAX_SIMULATION_CAR

# New episode/game round
while episode_count < config.MAX_EPISODE + config.TESTING_EPISODE * 3:
    is_training = config.DL_IS_TRAINING and episode_count < config.MAX_EPISODE and not config.VISUALENABLED

    # Score object
    score = Score(score=0)

    subject_car = Car(main_surface,
                      lane_map,
                      speed=60,
                      y=DEFAULT_CAR_POS,
                      lane=1,
                      is_subject=True,
                      score=score,
                      agent=deep_traffic_agent)
    object_cars = [Car(main_surface,
                       lane_map,
                       speed=40,
                       y=800,
                       lane=3,
                       is_subject=False,
                       score=score,
                       subject=subject_car)
                   for i in range(3, 4)]

    frame = 0

    game_ended = False

    delay_count = 0
    speed_counter = []
    subject_car_action = 'M'

    while True: # frame < config.MAX_FRAME_COUNT:
        # brick draw
        # bat and ball draw
        # events
        if config.VISUALENABLED and not config.DLAGENTENABLED:
            pressed_key = pygame.key.get_pressed()
            keydown_key = []

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    keydown_key.append(event.key)

        if config.VISUALENABLED:
            ressed_key = pygame.key.get_pressed()
            keydown_key = []

            for event in pygame.event.get():
                if event.type == QUIT or event.type == pygame.K_q:
                    pygame.quit()
                    sys.exit()

        # Setup game background
        draw_basic_road(main_surface, subject_car.speed)

        # Car to identify available moves in the order from top to bottom
        cars = [subject_car]
        cars.extend([o_car for o_car in object_cars if o_car.removed is False])
        cars.sort(key=lambda t_car: t_car.y, reverse=True)

        available_lanes_for_new_car = identify_free_lane(cars)

        # Add more cars to the scene
        if len(cars) < PREDEFINED_MAX_CAR and np.random.standard_normal(1)[0] >= 0:
            # Decide position(Front or back)
            map_position = np.random.choice([0, 1], 1)[0]
            position = available_lanes_for_new_car[map_position]
            if len(position) > 0:
                # Back
                if map_position:
                    #new_car_speed = np.random.random_integers(30, 90)
                    new_car_speed = np.random.randint(30, 91)
                    new_car_y = 1010
                    new_car_lane = int(np.random.choice(position, 1))
                else:
                    #new_car_speed = np.random.random_integers(30, 60)
                    new_car_speed = np.random.randint(30, 61)
                    new_car_y = -100
                    new_car_lane = int(np.random.choice(position, 1))
                # Decide lanes
                new_car = Car(main_surface,
                              lane_map,
                              speed=new_car_speed,
                              y=new_car_y,
                              lane=new_car_lane,
                              is_subject=False,
                              subject=subject_car,
                              score=score)
                object_cars.append(new_car)
                if position:
                    cars.append(new_car)
                else:
                    cars.insert(0, new_car)

        # main game logic
        # Reinitialize lane map
        for y in range(100):
            for x in range(4):
                lane_map[y][x] = 0

        # Identify car position
        for car in cars:
            car.identify()

        for car in cars:
            car.identify_available_moves()

        cache = False
        if delay_count < config.DELAY and not game_ended and is_training:
            delay_count += 1
            cache = True
        else:
            delay_count = 0

        q_values = None
        # Car react to road according to order
        for car in cars[::-1]:
            # For object car
            if car.subject is not None:
                car.decide(game_ended, cache=cache, is_training=is_training)
                continue

            if config.DLAGENTENABLED:
                # Get prediction from DeepTrafficAgent
                q_values, temp_action = car.decide(game_ended, cache=cache, is_training=is_training)
                if not cache:
                    subject_car_action = temp_action
                    q_values = np.sum(q_values)
                    if not is_training:
                        action_stats[deep_traffic_agent.get_action_index(temp_action)] += 1
            elif config.VISUALENABLED:
                # Manual control
                is_controlled = False
                for key in monitor_keys:
                    if pressed_key[key] or key in keydown_key:
                        is_controlled = True
                        control_car(subject_car, key)
                if not is_controlled:
                    car.move('M')

        # Show road overlay (Safety)
        # draw_road_overlay_safety(main_surface, lane_map)
        draw_road_overlay_vision(main_surface, subject_car)

        for car in cars:
            car.draw()

        # Decide end of game
        if game_ended:
            deep_traffic_agent.remember(score.score,
                                        subject_car.get_vision(),
                                        end_episode=True,
                                        is_training=is_training)
            break
        elif frame >= config.MAX_FRAME_COUNT: # abs(score.score) >= config.GOAL:
            game_ended = True

        # Show statistics
        if config.VISUALENABLED:
            draw_score(main_surface, score.score)

            draw_inputs(main_surface, subject_car.get_vision())
            draw_actions(main_surface, subject_car_action)
            draw_gauge(main_surface, subject_car.speed)

            # Setup advanced view
            advanced_road.draw(frame, subject_car)

            # collision detection
            fpsClock.tick(20000)
            pygame.event.poll()
            pygame.display.flip()

        frame += 1
        speed_counter.append(subject_car.speed)

        if q_values is not None:
            deep_traffic_agent.model.log_q_values(q_values)

    episode_count = deep_traffic_agent.model.increase_count_episodes()
    avg_speed = np.average(speed_counter)
    if not is_training:
        speed_counter_avg.append(avg_speed)
        deep_traffic_agent.model.log_testing_speed(avg_speed)
    else:
        print("Average speed for episode{}: {}".format(episode_count, avg_speed))
        deep_traffic_agent.model.log_average_speed(avg_speed)
    deep_traffic_agent.model.log_total_frame(frame)
    deep_traffic_agent.model.log_terminated(frame < config.MAX_FRAME_COUNT - 1)
    deep_traffic_agent.model.log_reward(score.score)

    deep_traffic_agent.model.log_hard_brake_count(subject_car.hard_brake_count)

    if episode_count > config.MAX_EPISODE:
        alternate_line_switching.append(subject_car.alternate_line_switching)
        hard_brake_avg.append(subject_car.hard_brake_count)
        if (episode_count - config.MAX_EPISODE) % config.TESTING_EPISODE == 0:
            avg_speed = np.average(speed_counter_avg)
            median_speed = np.median(speed_counter_avg)
            avg_hard_brake = np.average(hard_brake_avg)
            median_hard_brake = np.median(hard_brake_avg)
            avg_alternate_line_switching = np.average(alternate_line_switching)
            median_alternate_line_switching = np.median(alternate_line_switching)
            print("Car:{},Speed:(Mean: {}, Median: {}),Hard_Brake:(Mean: {}, Median: {}), Line::(Mean: {}, Median: {})"
                  .format(PREDEFINED_MAX_CAR, avg_speed, median_speed, avg_hard_brake, median_hard_brake,
                          avg_alternate_line_switching, median_alternate_line_switching))
            if abs(PREDEFINED_MAX_CAR - 40) < 1:
                deep_traffic_agent.model.log_average_test_speed_40(avg_speed)
                PREDEFINED_MAX_CAR = 20
            elif abs(PREDEFINED_MAX_CAR - 20) < 1:
                deep_traffic_agent.model.log_average_test_speed_20(avg_speed)
                PREDEFINED_MAX_CAR = 60
            else:
                deep_traffic_agent.model.log_average_test_speed_60(avg_speed)
            speed_counter_avg = []
            hard_brake_avg = []
            alternate_line_switching = []
            
    for event in pygame.event.get():
        if event.type == pygame.QUIT:   # when you click the X button
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # press ESC to exit
                pygame.quit()
                quit()


deep_traffic_agent.model.log_action_frequency(action_stats)"""



# ============================================
# gui.py — Main game loop for DeepTraffic
# ============================================

import os
import sys
import pygame
import numpy as np
from pygame.locals import *

# Import model and GUI utilities
from car import Car, DEFAULT_CAR_POS
from gui_util import (
    draw_basic_road,
    draw_road_overlay_vision,
    control_car,
    identify_free_lane,
    Score,
    draw_inputs,
    draw_actions,
    draw_gauge,
    draw_score,
)
from deep_traffic_agent import DeepTrafficAgent
from advanced_view.road import AdvancedRoad
import config
from config import LANE_COUNT

# -------------------------------
# Model initialization
# -------------------------------
model_name = config.MODEL_NAME
deep_traffic_agent = DeepTrafficAgent(model_name)

# -------------------------------
# Game constants
# -------------------------------
ACTION_MAP = ["A", "M", "D", "L", "R"]
monitor_keys = [pygame.K_UP, pygame.K_RIGHT, pygame.K_LEFT, pygame.K_DOWN]

# -------------------------------
# Pygame init
# -------------------------------
if config.VISUALENABLED:
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("DeepTraffic")
    fpsClock = pygame.time.Clock()
    main_surface = pygame.display.set_mode(
        (1800, 800), pygame.DOUBLEBUF | pygame.HWSURFACE
    )
    advanced_road = AdvancedRoad(main_surface, 0, 250, 1010, 550, lanes=LANE_COUNT)
else:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    main_surface = None
    advanced_road = None

# lane_map[y][x] where:
# - y = road slices (100)
# - x = lanes (LANE_COUNT)
lane_map = [[0 for _ in range(LANE_COUNT)] for _ in range(100)]

episode_count = deep_traffic_agent.model.get_count_episodes()

# stats
speed_counter_avg = []
hard_brake_avg = []
alternate_line_switching = []
action_stats = np.zeros(5, np.int32)

PREDEFINED_MAX_CAR = config.MAX_SIMULATION_CAR

# =====================================================
# Main simulation loop (episodes)
# =====================================================
while episode_count < config.MAX_EPISODE + config.TESTING_EPISODE * 3:

    is_training = (
        config.DL_IS_TRAINING
        and episode_count < config.MAX_EPISODE
        and not config.VISUALENABLED
    )

    # -------------------------------
    # New Episode Setup
    # -------------------------------
    score = Score(score=0)

    subject_car = Car(
        main_surface,
        lane_map,
        speed=60,
        y=DEFAULT_CAR_POS,
        lane=LANE_COUNT // 2,  # start in middle lane
        is_subject=True,
        score=score,
        agent=deep_traffic_agent,
    )

    # spawn one object car
    object_cars = [
        Car(
            main_surface,
            lane_map,
            speed=40,
            y=800,
            lane=LANE_COUNT - 1,
            is_subject=False,
            score=score,
            subject=subject_car,
        )
    ]

    frame = 0
    game_ended = False
    delay_count = 0
    speed_counter = []
    subject_car_action = "M"

    # =====================================================
    # Frame loop
    # =====================================================
    while True:

        # -------------------------------
        # handle events
        # -------------------------------
        if config.VISUALENABLED:
            pressed_key = pygame.key.get_pressed()
            keydown_key = []
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    keydown_key.append(event.key)

        # -------------------------------
        # background
        # -------------------------------
        draw_basic_road(main_surface, subject_car.speed)

        # -------------------------------
        # build car list
        # -------------------------------
        cars = [subject_car] + [c for c in object_cars if not c.removed]
        cars.sort(key=lambda c: c.y, reverse=True)

        available_lanes_for_new_car = identify_free_lane(cars)

        # -------------------------------
        # Add new cars
        # -------------------------------
        if len(cars) < PREDEFINED_MAX_CAR and np.random.standard_normal(1)[0] >= 0:
            map_position = np.random.choice([0, 1], 1)[0]
            position = available_lanes_for_new_car[map_position]

            if len(position) > 0:
                if map_position:  # spawn behind
                    new_car_speed = np.random.randint(30, 91)
                    new_car_y = 1010
                else:  # spawn ahead
                    new_car_speed = np.random.randint(30, 61)
                    new_car_y = -100
                new_car_lane = int(np.random.choice(position, 1))

                new_car = Car(
                    main_surface,
                    lane_map,
                    speed=new_car_speed,
                    y=new_car_y,
                    lane=new_car_lane,
                    is_subject=False,
                    subject=subject_car,
                    score=score,
                )
                object_cars.append(new_car)
                cars.append(new_car)

        # -------------------------------
        # reset lane_map
        # -------------------------------
        for y in range(100):
            for x in range(LANE_COUNT):
                lane_map[y][x] = 0

        # mark cars in lane_map
        for car in cars:
            car.identify()

        for car in cars:
            car.identify_available_moves()

        # -------------------------------
        # agent decisions
        # -------------------------------
        cache = False
        if delay_count < config.DELAY and not game_ended and is_training:
            delay_count += 1
            cache = True
        else:
            delay_count = 0

        q_values = None
        for car in cars[::-1]:
            if car.subject is not None:  # object cars
                car.decide(game_ended, cache=cache, is_training=is_training)
                continue

            if config.DLAGENTENABLED:  # deep RL agent
                q_values, temp_action = car.decide(
                    game_ended, cache=cache, is_training=is_training
                )
                if not cache:
                    subject_car_action = temp_action
                    q_values = np.sum(q_values)
                    if not is_training:
                        action_stats[
                            deep_traffic_agent.get_action_index(temp_action)
                        ] += 1
            elif config.VISUALENABLED:  # manual control
                is_controlled = False
                for key in monitor_keys:
                    if pressed_key[key] or key in keydown_key:
                        is_controlled = True
                        control_car(subject_car, key)
                if not is_controlled:
                    car.move("M")

        # -------------------------------
        # overlays + draw
        # -------------------------------
        draw_road_overlay_vision(main_surface, subject_car)
        for car in cars:
            car.draw()

        # -------------------------------
        # check game end
        # -------------------------------
        if game_ended:
            deep_traffic_agent.remember(
                score.score,
                subject_car.get_vision(),
                end_episode=True,
                is_training=is_training,
            )
            break
        elif frame >= config.MAX_FRAME_COUNT:
            game_ended = True

        # -------------------------------
        # draw stats
        # -------------------------------
        if config.VISUALENABLED:
            draw_score(main_surface, score.score)
            draw_inputs(main_surface, subject_car.get_vision())
            draw_actions(main_surface, subject_car_action)
            draw_gauge(main_surface, subject_car.speed)

            if advanced_road is not None:
                advanced_road.draw(frame, subject_car)

            fpsClock.tick(60)
            pygame.display.flip()

        frame += 1
        speed_counter.append(subject_car.speed)

    # =====================================================
    # Episode end
    # =====================================================
    episode_count = deep_traffic_agent.model.increase_count_episodes()
    avg_speed = np.average(speed_counter)

    if not is_training:
        speed_counter_avg.append(avg_speed)
        deep_traffic_agent.model.log_testing_speed(avg_speed)
    else:
        print(f"Average speed for episode {episode_count}: {avg_speed}")
        deep_traffic_agent.model.log_average_speed(avg_speed)

    deep_traffic_agent.model.log_total_frame(frame)
    deep_traffic_agent.model.log_terminated(frame < config.MAX_FRAME_COUNT - 1)
    deep_traffic_agent.model.log_reward(score.score)
    deep_traffic_agent.model.log_hard_brake_count(subject_car.hard_brake_count)

    if episode_count > config.MAX_EPISODE:
        alternate_line_switching.append(subject_car.alternate_line_switching)
        hard_brake_avg.append(subject_car.hard_brake_count)

        if (episode_count - config.MAX_EPISODE) % config.TESTING_EPISODE == 0:
            avg_speed = np.average(speed_counter_avg)
            median_speed = np.median(speed_counter_avg)
            avg_hard_brake = np.average(hard_brake_avg)
            median_hard_brake = np.median(hard_brake_avg)
            avg_alt_switch = np.average(alternate_line_switching)
            median_alt_switch = np.median(alternate_line_switching)

            print(
                f"Car:{PREDEFINED_MAX_CAR}, "
                f"Speed:(Mean:{avg_speed}, Median:{median_speed}), "
                f"Hard_Brake:(Mean:{avg_hard_brake}, Median:{median_hard_brake}), "
                f"Line:(Mean:{avg_alt_switch}, Median:{median_alt_switch})"
            )

            if abs(PREDEFINED_MAX_CAR - 40) < 1:
                deep_traffic_agent.model.log_average_test_speed_40(avg_speed)
                PREDEFINED_MAX_CAR = 20
            elif abs(PREDEFINED_MAX_CAR - 20) < 1:
                deep_traffic_agent.model.log_average_test_speed_20(avg_speed)
                PREDEFINED_MAX_CAR = 60
            else:
                deep_traffic_agent.model.log_average_test_speed_60(avg_speed)

            speed_counter_avg.clear()
            hard_brake_avg.clear()
            alternate_line_switching.clear()

# -------------------------------
# log action frequency
# -------------------------------
deep_traffic_agent.model.log_action_frequency(action_stats)
