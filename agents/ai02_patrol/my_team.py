# ai02_patrol/my_team.py
# Strategy: One strong boundary-patrolling defender + one very timid/scared offensive agent.
# Trains your Q-agent to handle tight defensive coverage and learn not to fear weak attackers.

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='PatrolAgent', second='TimidAttacker', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class BaseAgent(CaptureAgent):

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

    def get_features(self, game_state, action):
        return util.Counter()

    def get_weights(self, game_state, action):
        return {}


class PatrolAgent(BaseAgent):
    """
    Defensive agent that patrols a fixed set of positions along our side of the boundary.
    When an invader is visible, it abandons patrol and chases directly.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.patrol_positions = []
        self.patrol_index = 0

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self._compute_patrol_positions(game_state)

    def _compute_patrol_positions(self, game_state):
        """Build a list of non-wall cells along the boundary x-column on our side."""
        layout = game_state.data.layout
        width = layout.width
        height = layout.height
        walls = game_state.get_walls()

        # Red patrols at width//2 - 1, Blue patrols at width//2
        x = (width // 2 - 1) if self.red else (width // 2)

        self.patrol_positions = [
            (x, y) for y in range(1, height - 1) if not walls[x][y]
        ]

        if not self.patrol_positions:
            self.patrol_positions = [self.start]

        # Start patrol from the middle of the list for better initial coverage
        self.patrol_index = len(self.patrol_positions) // 2

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)

        # If there are visible invaders, chase the nearest one
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
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

        # No visible invaders: advance along patrol route
        target = self.patrol_positions[self.patrol_index]
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Advance patrol index when we reach the current target
        if my_pos == target:
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_positions)
            target = self.patrol_positions[self.patrol_index]

        best_action = None
        best_dist = 9999
        for action in actions:
            if action == Directions.STOP:
                continue
            successor = self.get_successor(game_state, action)
            pos2 = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(pos2, target)
            if dist < best_dist:
                best_dist = dist
                best_action = action

        return best_action if best_action else random.choice(actions)

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


class TimidAttacker(BaseAgent):
    """
    Offensive agent that flees the moment a ghost comes within FEAR_DISTANCE steps.
    Only advances when the path is clear of non-scared ghosts.
    """
    FEAR_DISTANCE = 5

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Check for nearby active (non-scared) ghosts
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_ghosts = [
            a for a in enemies
            if not a.is_pacman and a.scared_timer == 0 and a.get_position() is not None
        ]

        if active_ghosts:
            ghost_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in active_ghosts]
            if min(ghost_dists) <= self.FEAR_DISTANCE:
                # Flee: head back to start as fast as possible
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

        # Safe: use feature evaluation to find food
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
        my_pos = successor.get_agent_state(self.index).get_position()

        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Still encode ghost distances in features for weight-based fallback
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        active_ghosts = [
            a for a in enemies
            if not a.is_pacman and a.scared_timer == 0 and a.get_position() is not None
        ]
        if active_ghosts:
            dists = [self.get_maze_distance(my_pos, g.get_position()) for g in active_ghosts]
            features['distance_to_ghost'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'distance_to_ghost': 40, 'stop': -100}
