# ai04_random/my_team.py
# Strategy: Epsilon-greedy stochastic agents.
# One moderately random attacker + one heavily random defender.
# Injects unpredictability into training to prevent the Q-agent
# from overfitting to deterministic opponent behaviour.
#
# The epsilon values are intentionally different so the two agents present
# different levels of noise, covering a wider part of state-space per game.

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='EpsilonAttacker', second='EpsilonDefender', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class EpsilonBase(CaptureAgent):

    # Subclasses override this
    EPSILON = 0.3

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
        return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def _best_actions(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        return [a for a, v in zip(actions, values) if v == max_value], actions

    def choose_action(self, game_state):
        best, all_actions = self._best_actions(game_state)
        # Epsilon-greedy: with probability EPSILON pick a uniformly random action
        if random.random() < self.EPSILON:
            return random.choice(all_actions)
        return random.choice(best)

    def get_features(self, game_state, action):
        return util.Counter()

    def get_weights(self, game_state, action):
        return {}


class EpsilonAttacker(EpsilonBase):
    """
    Offensive epsilon-greedy agent (epsilon=0.25).
    Most of the time seeks food while avoiding active ghosts,
    but 25 % of turns takes a completely random action.
    """
    EPSILON = 0.25

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
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
            return best_action if best_action else random.choice(actions)
        return super().choose_action(game_state)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()

        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)
        if food_list:
            features['distance_to_food'] = min(self.get_maze_distance(my_pos, f) for f in food_list)

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.scared_timer == 0 and a.get_position() is not None]
        if ghosts:
            features['distance_to_ghost'] = min(self.get_maze_distance(my_pos, g.get_position()) for g in ghosts)

        if action == Directions.STOP:
            features['stop'] = 1
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'distance_to_ghost': 15, 'stop': -80}


class EpsilonDefender(EpsilonBase):
    """
    Defensive epsilon-greedy agent (epsilon=0.45).
    Nearly half of its turns are random, making it very unpredictable.
    When not random, it behaves as a standard reflex defender.
    This forces the Q-agent to learn robust offensive patterns that
    work even against chaotic or erratic defenders.
    """
    EPSILON = 0.45

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
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
