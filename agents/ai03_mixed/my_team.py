# ai03_mixed/my_team.py
# Strategy: Balanced adaptive team.
# - StrategicOffender: seeks capsules when ghosts are near, hunts scared ghosts, returns when carrying too much.
# - CounterDefender: patrols defensively but opportunistically attacks when no invaders are visible.

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='StrategicOffender', second='CounterDefender', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class SmartCaptureAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights


class StrategicOffender(SmartCaptureAgent):
    """
    Offensive agent with layered priorities:
      1. Return home when carrying >= MAX_CARRY food or only 2 food remain.
      2. Seek capsule if an active ghost is within CAPSULE_SEEK_RANGE.
      3. Hunt scared ghosts for bonus points.
      4. Otherwise minimize distance to food while keeping distance from active ghosts.
    """
    MAX_CARRY = 4
    CAPSULE_SEEK_RANGE = 6

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        food_left = len(self.get_food(game_state).as_list())

        # Head home if carrying a lot or near end
        if food_left <= 2 or (my_state.is_pacman and my_state.num_carrying >= self.MAX_CARRY):
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action if best_action else random.choice(actions)

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()

        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        if food_list:
            min_food = min([self.get_maze_distance(my_pos, f) for f in food_list])
            features['distance_to_food'] = min_food

        # Capsule seeking
        capsules = self.get_capsules(successor)
        if capsules:
            min_cap = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
            features['distance_to_capsule'] = min_cap

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]

        # Avoid active (non-scared) ghosts
        active_ghosts = [
            a for a in enemies
            if not a.is_pacman and a.scared_timer == 0 and a.get_position() is not None
        ]
        if active_ghosts:
            dists = [self.get_maze_distance(my_pos, g.get_position()) for g in active_ghosts]
            features['ghost_distance'] = min(dists)

        # Chase scared ghosts
        scared_ghosts = [
            a for a in enemies
            if not a.is_pacman and a.scared_timer > 0 and a.get_position() is not None
        ]
        if scared_ghosts:
            dists = [self.get_maze_distance(my_pos, g.get_position()) for g in scared_ghosts]
            features['scared_ghost_distance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,
            'distance_to_food': -1,
            'distance_to_capsule': -3,
            'ghost_distance': 20,
            'scared_ghost_distance': -15,
            'stop': -100,
        }


class CounterDefender(SmartCaptureAgent):
    """
    Defensive agent that:
      - Aggressively chases any visible invaders.
      - When no invaders are visible, crosses into enemy territory for light offense.
      - Penalizes stopping and reversing to keep moving.
    """

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # Direct chase when invaders are spotted
        if invaders:
            best_action = None
            best_dist = 9999
            for action in actions:
                if action == Directions.STOP:
                    continue
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dists = [self.get_maze_distance(pos2, inv.get_position()) for inv in invaders]
                min_dist = min(dists)
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_action = action
            return best_action if best_action else random.choice(actions)

        # No invaders: switch to mild offensive evaluation
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if invaders:
            dists = [self.get_maze_distance(my_pos, inv.get_position()) for inv in invaders]
            features['invader_distance'] = min(dists)

        # Opportunistic food grabbing when no invaders
        if not invaders:
            food_list = self.get_food(successor).as_list()
            if food_list:
                min_food = min([self.get_maze_distance(my_pos, f) for f in food_list])
                features['food_opportunity'] = min_food

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'food_opportunity': -0.5,
            'stop': -100,
            'reverse': -2,
        }
