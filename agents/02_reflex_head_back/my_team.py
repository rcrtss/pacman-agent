# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point

import time


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveCustomAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

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

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


##################################################################################
################## MY AGENTS #####################################################
##################################################################################


class OffensiveCustomAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        ######################################################################
        #####                   Parameters                               #####
        ######################################################################
        # If we are carrying more than this amount of food, we should head back to the start
        CARRY_THRESHOLD_STRAT = 5
        # If there are only this many pieces of food left, we should head back to the start  
        FOOD_LEFT_STRAT = 2
        # How many steps of lookahead to perform when on offense
        SEARCH_DEPTH = 2
        # If the sum of opponents' score and food they are carrying is greater than this, we should NOT go on offense
        OPPONENT_SCORE_STRAT = 10
        ######################################################################
        
        # Our agent should offense only if:
        # 1) Both opponents are NOT pacman at the same time, AND
        # 2) The sum of opponents' returned + carrying food is NOT greater than OPPONENT_SCORE_STRAT, AND
        # 3) We are NOT currently carrying a lot of food (i.e. more than CARRY_THRESHOLD_STRAT)
        opponents = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        both_pacman = all(opp.is_pacman for opp in opponents)
        opponent_score = sum(opp.num_carrying + opp.num_returned for opp in opponents)
        my_state = game_state.get_agent_state(self.index)
        if not both_pacman and opponent_score <= OPPONENT_SCORE_STRAT and my_state.num_carrying <= CARRY_THRESHOLD_STRAT:
            self.is_offence = True
        else:
            # print("Op! DEFENSE!")
            self.is_offence = False
        
        ######################################################################
        
        actions = game_state.get_legal_actions(self.index)
        agent_info = game_state.get_agent_state(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        
        if self.is_offence:
            best_value = float('-inf')
            best_actions = []
            for action in actions:
                val = self.lookahead(game_state, action, SEARCH_DEPTH)
                if val > best_value:
                    best_value = val
                    best_actions = [action]
                elif val == best_value:
                    best_actions.append(action)
        else:
            values = [self.evaluate(game_state, a) for a in actions]
            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]
                    
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        # Endgame override: choose the action that brings our agent closest to the starting position
        food_left = len(self.get_food(game_state).as_list())
        
        # Win strategy
        head_back = food_left <= FOOD_LEFT_STRAT or (agent_info.is_pacman and agent_info.num_carrying > CARRY_THRESHOLD_STRAT)
        if head_back:
            # print("Ah!! Head back!!")
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

        return random.choice(best_actions)

    def lookahead(self, game_state, action, depth):
        """
        Recursively evaluates actions up to `depth` levels deep.
        At depth 1, returns the direct evaluation.
        At depth > 1, returns the best evaluation reachable from the successor.
        """
        if depth <= 1:
            return self.evaluate(game_state, action)
        successor = self.get_successor(game_state, action)
        next_actions = successor.get_legal_actions(self.index)
        return max(self.lookahead(successor, a, depth - 1) for a in next_actions)

    def get_features(self, game_state, action):
        """If we're a pacman, use offensive features. If we're a ghost, use defensive features."""
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        if self.is_offence:
            return self.get_features_off(game_state, action)
        else:
            return self.get_features_def(game_state, action)
    
    def get_weights(self, game_state, action):
        """If we're a pacman, use offensive weights. If we're a ghost, use defensive weights."""
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        if self.is_offence:
            return self.get_weights_off(game_state, action)
        else:
            return self.get_weights_def(game_state, action)

    def get_features_off(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        
        # Compute distance to power pill
        power_pills = self.get_capsules(successor)
        if len(power_pills) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, power) for power in power_pills])
            features['distance_to_power'] = min_distance
        
        # Compute distance to ghosts
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.scared_timer == 0 and a.get_position() is not None]
        if len(ghosts) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in ghosts]
            features['distance_to_ghost'] = min(dists)

        return features

    def get_weights_off(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'distance_to_ghost': 100, 'distance_to_power': -10}
    
    def get_features_def(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if self.is_offence: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights_def(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
