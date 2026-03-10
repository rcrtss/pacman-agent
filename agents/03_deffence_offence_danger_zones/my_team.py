# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
from statistics import median
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point

import time


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveCustomAgent', second='DefensiveCustomAgent', num_training=0):
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
        
        # Precompute dead-end map for the FULL map (both sides).
        # dead_end_entrance[cell] -> the entrance cell that seals that dead end.
        # Available to all agents: defensive uses it to trap invaders,
        # offensive uses it to decide when to escape a dead end.
        self.dead_end_depth    = self._compute_dead_end_depth(game_state)
        self.dead_end_entrance = self._compute_dead_end_entrances(self.dead_end_depth, game_state)
        
    # ------------------------------------------------------------------ #
    # Dead-end precomputation (runs once at game start)                   #
    # ------------------------------------------------------------------ #
    def _compute_dead_end_depth(self, game_state):
        """
        Assigns a dead-end depth to every cell in the maze using iterative
        tip-peeling (topological leaf removal).

        A 'tip' is a non-wall cell with exactly one open neighbor.
        We peel tips in waves and assign a peel order to each cell.
        After peeling, we invert so that:
          - The innermost tip (true dead end) has the HIGHEST depth value.
          - Cells closer to the junction entrance have lower values.
          - Junctions and open cells are absent (implicit depth 0).

        This makes dead_end_depth directly usable as a risk score:
        higher value = deeper inside a dead end = more dangerous.

        Returns: dict {cell: depth} for all dead-end cells (depth >= 1).
        """
        walls  = game_state.get_walls()
        width  = game_state.data.layout.width
        height = game_state.data.layout.height

        all_cells = {
            (x, y)
            for x in range(width)
            for y in range(height)
            if not walls[x][y]
        }

        # Mutable neighbor sets — we remove cells as we peel them
        open_neighbors = {
            cell: {
                (cell[0] + dx, cell[1] + dy)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if (cell[0] + dx, cell[1] + dy) in all_cells
            }
            for cell in all_cells
        }

        peel_order = {}   # cell -> wave number it was peeled (1 = first peeled = entrance-side)
        current_wave = 1
        tips = [cell for cell in all_cells if len(open_neighbors[cell]) == 1]

        while tips:
            next_tips = []
            for tip in tips:
                if len(open_neighbors[tip]) != 1:
                    continue                          # already promoted or processed
                peel_order[tip] = current_wave
                (neighbor,) = open_neighbors[tip]    # exactly one neighbor
                open_neighbors[neighbor].discard(tip)
                open_neighbors[tip] = set()
                if len(open_neighbors[neighbor]) == 1:
                    next_tips.append(neighbor)
            tips = next_tips
            current_wave += 1

        # Invert: tip (last peeled) gets the highest value, entrance-side gets 1
        max_wave = current_wave - 1
        depth = {cell: max_wave - wave + 1 for cell, wave in peel_order.items()}

        return depth

    def _compute_dead_end_entrances(self, dead_end_depth, game_state):
        """
        Derives the entrance map from the depth map.

        For each dead-end cell, its entrance is the first cell outside the
        dead-end region (depth 0) reachable by walking outward.

        With the new convention (tip = highest depth, entrance-side = 1),
        walking *outward* means stepping to the neighbor with the LOWEST
        depth at each step — until we reach a neighbor not in dead_end_depth
        (depth 0), which is the entrance junction.

        Returns: dict {cell: entrance_cell} for all dead-end cells.
        """
        walls  = game_state.get_walls()
        width  = game_state.data.layout.width
        height = game_state.data.layout.height

        def open_neighbors_of(pos):
            x, y = pos
            return [
                (x + dx, y + dy)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= x + dx < width and 0 <= y + dy < height
                and not walls[x + dx][y + dy]
            ]

        entrance = {}
        for cell in dead_end_depth:
            current = cell
            while current in dead_end_depth:
                neighbors = open_neighbors_of(current)
                # Prefer a neighbor already outside the dead end (depth 0)
                exits = [nb for nb in neighbors if nb not in dead_end_depth]
                if exits:
                    current = exits[0]
                    break
                # Otherwise step outward: toward lower depth (toward the entrance)
                current = min(neighbors, key=lambda nb: dead_end_depth.get(nb, float('inf')))
            entrance[cell] = current

        return entrance

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

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
    A reflex agent focused on offensive strategy that also defends if needed.
    """      
    def choose_action(self, game_state):
        """
        Picks best action based on strategy.
        """
        ######################################################################
        #####                   Parameters                               #####
        ######################################################################
        # If we are carrying more than this amount of food, we should head back to the start
        self.MAX_FOOD_CARRY_LIMIT = 5
        # If the sum of opponents' score and food they are carrying is greater than this, we should NOT go on offense
        self.ENEMY_SCORE_WARNING_LEVEL = 10
        # If there are only this many pieces of food left, we should head back to the start  
        self.FOOD_RETRIEVAL_LIMIT = 2
        # How many steps of lookahead to perform when on offense
        self.OFFENSIVE_LOOKAHEAD_DEPTH = 2
        ######################################################################
        
        actions = game_state.get_legal_actions(self.index)
        
        # Choose strategy depending on the current game state
        self.aggressive_play_mode = self._should_offense(game_state)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        best_actions = self.offensive_action(game_state, actions) if self.aggressive_play_mode else self.defensive_action(game_state, actions)            
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        # Head back home if endgame mode, carrying too much food, etc.
        if self._should_head_back(game_state):
            return self.head_back_strategy(game_state, actions)

        return random.choice(best_actions)
    
    ########################################################################
    
    def _should_offense(self, game_state):
        """
        Decide whether or not to set offensive strategy (switch strategy)
        Returns True if we should offense, False if we should defense
        """
        # Our agent should offense only if:
        # 1) Both opponents are NOT pacman at the same time, AND
        # 2) The sum of opponents' returned + carrying food is NOT greater than ENEMY_SCORE_WARNING_LEVEL, AND
        # 3) We are NOT currently carrying a lot of food (i.e. more than MAX_FOOD_CARRY_LIMIT)
        opponents = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        both_opps_pacman = all(opp.is_pacman for opp in opponents)
        opponent_score = sum(opp.num_carrying + opp.num_returned for opp in opponents)
        my_state = game_state.get_agent_state(self.index)
        
        if not both_opps_pacman and not opponent_score > self.ENEMY_SCORE_WARNING_LEVEL and my_state.num_carrying <= self.MAX_FOOD_CARRY_LIMIT:
            return True
        
        return False
    
    def _should_head_back(self, game_state):
        """
        Decide whether or not to head back to the start position to secure points.
        Returns True if we should head back, False otherwise.
        
        TODO: consider a more advanced head back strategy, for example if a ghost is close,
        we should balance going back with evading the ghost instead of just heading back directly.
        """
        agent_info = game_state.get_agent_state(self.index)
        food_left = len(self.get_food(game_state).as_list())
        
        if food_left <= self.FOOD_RETRIEVAL_LIMIT:
            return True
        
        if agent_info.is_pacman and agent_info.num_carrying > self.MAX_FOOD_CARRY_LIMIT:
            return True
        
        return False
            
    def defensive_action(self, game_state, actions):
        """
        Returns the best defensive action according to reflex evaluation.
        """
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return best_actions

    def offensive_action(self, game_state, actions):
        """
        Returns the best offensive action according to lookahead evaluation.
        """
        best_value = float('-inf')
        best_actions = []
        for action in actions:
            val = self._recursive_lookahead(game_state, action, self.OFFENSIVE_LOOKAHEAD_DEPTH)
            if val > best_value:
                best_value = val
                best_actions = [action]
            elif val == best_value:
                best_actions.append(action)
        return best_actions

    def _recursive_lookahead(self, game_state, action, depth=1):
        """
        Recursively evaluates actions up to `depth` levels deep.
        At depth 1, returns the direct evaluation.
        At depth > 1, returns the best evaluation reachable from the successor.
        """
        if depth <= 1:
            return self.evaluate(game_state, action)
        successor = self.get_successor(game_state, action)
        next_actions = successor.get_legal_actions(self.index)
        return max(self._recursive_lookahead(successor, a, depth - 1) for a in next_actions)

    def get_features(self, game_state, action):
        """If we're a pacman, use offensive features. If we're a ghost, use defensive features."""
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        if self.aggressive_play_mode:
            return self.get_features_off(game_state, action)
        else:
            return self.get_features_def(game_state, action)
    
    def get_weights(self, game_state, action):
        """If we're a pacman, use offensive weights. If we're a ghost, use defensive weights."""
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        if self.aggressive_play_mode:
            return self.get_weights_off(game_state, action)
        else:
            return self.get_weights_def(game_state, action)
        
    def head_back_strategy(self, game_state, actions):
        """
        When we're in endgame mode, we want to head back to the start position to secure our points. 
        This function implements that strategy by choosing the action that brings us closest to the starting position."
        """
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
        
    ########################################################################
    ##### OFFENSIVE FUNCTIONS FOR OFFENSIVE AGENT ##########################
    ########################################################################

    def get_features_off(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        successor_self_state = successor.get_agent_state(self.index)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor_self_state.get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        
        # Compute distance to power pill
        power_pills = self.get_capsules(successor)
        if len(power_pills) > 0:
            my_pos = successor_self_state.get_position()
            min_distance = min([self.get_maze_distance(my_pos, power) for power in power_pills])
            features['distance_to_power'] = min_distance
        
        # Compute distance to ghosts
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.scared_timer == 0 and a.get_position() is not None]
        if len(ghosts) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in ghosts]
            features['distance_to_ghost'] = min(dists)
            
        # If in danger zone (dead end), heavily penalize
        if successor_self_state.is_pacman:
            dead_end_risk = self.dead_end_depth.get(successor_self_state.get_position(), 0)
            
            if features['distance_to_ghost'] > 0:
                proximity_value = features['distance_to_ghost']
            else:
                # proximity proxy is the mean of noisy values
                noisy_distances = game_state.get_agent_distances()
                proximity_value = 5 + sum(noisy_distances) / len(noisy_distances) if noisy_distances else 0
            
            features['danger_zone'] = dead_end_risk / proximity_value

        return features

    def get_weights_off(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -5, 'distance_to_ghost': 100, 'distance_to_power': -10,
                'danger_zone': -10}
    
    
    #######################################################################
    ##### DEFENSIVE FALLBACK FUNCTIONS FOR OFFENSIVE AGENT ################
    #######################################################################
    
    def get_features_def(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1

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

##################################################################################
#############################  DEFENSIVE AGENT ###################################
##################################################################################
class DefensiveCustomAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        
        # Compute boundary positions for patrolling when no invaders are present
        mid_x = game_state.data.layout.width // 2
        boundary_x = mid_x - 1 if self.red else mid_x
        walls = game_state.get_walls()
        height = game_state.data.layout.height
        self.patrol_points = [(boundary_x, y) for y in range(height) if not walls[boundary_x][y]]
    
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        
        # ------------------------------------------------------------------ #
        # TRAP OVERRIDE: if I am a ghost and an observable invader (pacman)   #
        # is on my side and is stuck in a dead end, block the only entrance.  #
        # ------------------------------------------------------------------ #
        trap_action = self._trap_override(game_state, actions)
        if trap_action is not None:
            return trap_action
        # ------------------------------------------------------------------ #

        # You can profile your evaluation time by uncommenting these lines
        start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

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
        
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = nearest_point(my_state.get_position())

        # a. Computes whether we're on defense (1) or offense (0). With positive weight, agent is encouraged to defend.
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        
        # b. num_invaders being penalized means the agent will try to eat invader when possible
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0 and my_state.scared_timer == 0:
            # c. invader_distance neg weight means agent encouraged to get close to invaders.
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        
        # d. near_food_cluster to defend vulnerable food clusters
        food_clusters = self._get_food_clusters(game_state)
        if food_clusters:
            dists = [self.get_maze_distance(my_pos, cluster) for cluster in food_clusters]
            features['distance_to_food_cluster'] = min(dists)
        
        # e. distance to recently eaten food to defend against active invaders
        recently_eaten = self._get_recently_eaten_food(game_state)
        if recently_eaten:
            dists = [self.get_maze_distance(my_pos, food) for food in recently_eaten]
            features['distance_to_eaten_food'] = min(dists)
            
        # f. distance to boundary when no invaders to patrol the boundary
        if not invaders:
            dists = [self.get_maze_distance(my_pos, pos) for pos in self.patrol_points]
            features['distance_to_boundary'] = median(dists)
                
        # g. run away from pacman if scared
        if my_state.scared_timer > 0:
            pacmen = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            if pacmen:
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in pacmen]
                features['distance_to_pacman'] = min(dists)
                
        # h. if opponent is pacman and is against corridor, dont eat them, instead stop there to trap them.
        

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -2000, 'on_defense': 200, 'invader_distance': -500, 'stop': -200, 'reverse': -40,
                'distance_to_food_cluster': -4, 'distance_to_eaten_food': -50, 'distance_to_boundary': -3,
                'distance_to_pacman': 50}
    
    def _get_food_clusters(self, game_state):
        """
        Returns the positions of food clusters on our side. A food cluster is defined as a group of 4 or more adjacent food pieces.
        """
        food = self.get_food_you_are_defending(game_state)
        food_list = food.as_list()
        clusters = []
        visited = set()

        for food_pos in food_list:
            if food_pos not in visited:
                cluster = self._bfs_food_cluster(food, food_pos, visited)
                if len(cluster) > 3:
                    clusters.append(cluster)

        return [self._get_cluster_center(cluster) for cluster in clusters]
    
    def _bfs_food_cluster(self, food, start_pos, visited):
        """
        Performs a BFS to find all connected food pieces starting from start_pos.
        """
        queue = [start_pos]
        cluster = []
        visited.add(start_pos)

        while queue:
            pos = queue.pop(0)
            cluster.append(pos)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (pos[0] + dx, pos[1] + dy)
                if food[next_pos[0]][next_pos[1]] and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)

        return cluster
    
    def _get_cluster_center(self, cluster):
        """
        Returns the position in the cluster closest to its geometric center.
        This ensures the returned position is actually on the grid.
        """
        x_coords = [pos[0] for pos in cluster]
        y_coords = [pos[1] for pos in cluster]
        center_x = sum(x_coords) // len(cluster)
        center_y = sum(y_coords) // len(cluster)
        center = (center_x, center_y)
        
        # Return the cluster position closest to the calculated center
        return min(cluster, key=lambda pos: (pos[0] - center[0])**2 + (pos[1] - center[1])**2)
    
    def _get_recently_eaten_food(self, game_state):
        """
        Returns the position of the most recently eaten food, or None if no food has been eaten.
        """
        if self.get_previous_observation() is None:
            return []
        
        current_food = self.get_food_you_are_defending(game_state).as_list()
        previous_food = self.get_food_you_are_defending(self.get_previous_observation()).as_list()
        eatenFood = []
        if len(current_food) < len(previous_food):
            eatenFood = list(set(previous_food) - set(current_food))

        return eatenFood

    # ---------------------------------------------------------------------- #
    # TRAP OVERRIDE HELPERS                                                   #
    # ---------------------------------------------------------------------- #


    def _trap_override(self, game_state, actions):
        """
        Returns the action to block the dead-end entrance if all conditions
        hold, otherwise returns None so normal evaluation proceeds.

        Conditions (all checked cheaply with no BFS at runtime):
          1. I am a ghost (not pacman) and not scared.
          2. There is at least one observable invader (pacman) on my side.
          3. The closest invader's cell is in self.dead_end_entrance (O(1) lookup).

        If all conditions hold:
          - If I am already at the entrance -> STOP (hold the block).
          - Otherwise -> move towards the entrance.
        """
        my_state = game_state.get_agent_state(self.index)

        # Condition 1: I must be a ghost and not scared
        if my_state.is_pacman:
            return None
        if my_state.scared_timer > 0:
            return None

        my_pos = my_state.get_position()

        # Condition 2: At least one observable invader on my side
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        if not invaders:
            return None

        # Pick the closest invader to act on
        invader = min(invaders, key=lambda a: self.get_maze_distance(my_pos, a.get_position()))
        invader_pos = invader.get_position()

        # Condition 3: O(1) dead-end lookup (no BFS at runtime)
        entrance = self.dead_end_entrance.get(invader_pos)
        if entrance is None:
            return None

        # --- All conditions met: execute trap blocking ---

        # If already at the entrance, stay put to hold the block
        if my_pos == entrance:
            return Directions.STOP

        # Otherwise move towards the entrance
        best_action = None
        best_dist   = float('inf')
        for action in actions:
            if action == Directions.STOP:
                continue
            successor = self.get_successor(game_state, action)
            pos2      = successor.get_agent_state(self.index).get_position()
            dist      = self.get_maze_distance(pos2, entrance)
            if dist < best_dist:
                best_dist   = dist
                best_action = action

        return best_action