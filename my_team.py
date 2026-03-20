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
        
        # time_start = time.time()
        self.dead_end_depth    = self._compute_dead_end_depth(game_state)
        self.dead_end_entrance = self._compute_dead_end_entrances(self.dead_end_depth, game_state)
        # print('Dead-end precomputation time for agent %d: %.4f' % (self.index, time.time() - time_start))
        
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
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

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

##################################################################################
################## MY AGENTS #####################################################
##################################################################################


class OffensiveCustomAgent(ReflexCaptureAgent):
    """
    A reflex agent focused on offensive play, with improved ghost avoidance,
    scared-ghost chasing, and smarter return-home decisions.
    """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        # Precompute the set of cells that deposit food when reached
        walls = game_state.get_walls()
        mid = walls.width // 2
        home_x = mid - 1 if self.red else mid
        self.home_boundary = [
            (home_x, y) for y in range(1, walls.height - 1)
            if not walls[home_x][y]
        ]
        
        # Randomly choose which zone (top or bottom) to target when crossing the centerline
        self.target_vertical_zone = random.choice(['bottom'])        
        # Precompute topmost and bottommost walkable positions on home side
        all_walls = game_state.get_walls()
        home_left = 0 if self.red else mid
        home_right = mid if self.red else all_walls.width
        
        walkable_y_values = set()
        for x in range(home_left, home_right):
            for y in range(all_walls.height):
                if not all_walls[x][y]:
                    walkable_y_values.add(y)
        
        self.topmost_y = max(walkable_y_values) if walkable_y_values else mid
        self.bottommost_y = min(walkable_y_values) if walkable_y_values else 1
    def choose_action(self, game_state):
        MAX_FOOD_CARRY    = 5   # hard cap on food carried before heading home
        FOOD_LEFT_LIMIT   = 2   # remaining food count that triggers endgame return
        LOOKAHEAD_DEPTH   = 3   # reflex lookahead depth
        GHOST_FLEE_DIST   = 5   # if a ghost is this close and we carry food, go home
        CARRY_FLEE_MIN    = 3   # minimum food carried to trigger ghost-proximity flee

        actions = game_state.get_legal_actions(self.index)
        me      = game_state.get_agent_state(self.index)
        my_pos  = me.get_position()

        # Dead-end escape: if we're inside a dead end and a ghost is visible,
        # immediately move toward the precomputed entrance (O(1) lookup).
        if me.is_pacman and my_pos in self.dead_end_depth:
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            ghosts  = [a for a in enemies
                       if not a.is_pacman and a.scared_timer == 0 and a.get_position() is not None]
            if ghosts:
                entrance = self.dead_end_entrance.get(my_pos)
                if entrance:
                    best_dist   = 9999
                    best_action = None
                    for action in actions:
                        successor = self.get_successor(game_state, action)
                        pos2      = successor.get_agent_position(self.index)
                        dist      = self.get_maze_distance(pos2, entrance)
                        if dist < best_dist:
                            best_action = action
                            best_dist   = dist
                    if best_action:
                        return best_action

        self.aggressive_play_mode = self._should_offense(game_state)

        best_actions = (
            self.offensive_action(game_state, actions, LOOKAHEAD_DEPTH)
            if self.aggressive_play_mode
            else self.defensive_action(game_state, actions)
        )

        if self._should_head_back(game_state, MAX_FOOD_CARRY, FOOD_LEFT_LIMIT,
                                   GHOST_FLEE_DIST, CARRY_FLEE_MIN):
            return self._head_back_strategy(game_state, actions)

        return random.choice(best_actions)

    def _should_offense(self, game_state):
        # Only switch to defense when BOTH opponents are actively invading our side
        opponents = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return not all(o.is_pacman for o in opponents)

    def _should_head_back(self, game_state, max_carry, food_left_limit,
                           ghost_flee_dist, carry_flee_min):
        me       = game_state.get_agent_state(self.index)
        food_left = len(self.get_food(game_state).as_list())

        if food_left <= food_left_limit:
            return True
        if me.is_pacman and me.num_carrying > max_carry:
            return True

        # Return early if carrying enough food and a non-scared ghost is nearby
        if me.is_pacman and me.num_carrying >= carry_flee_min:
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            ghosts  = [a for a in enemies
                       if not a.is_pacman and a.scared_timer == 0 and a.get_position() is not None]
            if ghosts:
                my_pos   = me.get_position()
                min_dist = min(self.get_maze_distance(my_pos, g.get_position()) for g in ghosts)
                if min_dist <= ghost_flee_dist:
                    return True
        return False

    def _head_back_strategy(self, game_state, actions):
        # Head to the nearest home-boundary cell, avoiding ghost positions
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghost_cells = {
            a.get_position() for a in enemies
            if not a.is_pacman and a.scared_timer == 0 and a.get_position() is not None
        }
        best_dist   = 9999
        best_action = None
        for action in actions:
            successor = self.get_successor(game_state, action)
            pos2      = successor.get_agent_position(self.index)
            if pos2 in ghost_cells:
                continue
            dist = min(self.get_maze_distance(pos2, b) for b in self.home_boundary)
            if dist < best_dist:
                best_action = action
                best_dist   = dist
        # Fallback if all moves lead toward a ghost
        if best_action is None:
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2      = successor.get_agent_position(self.index)
                dist = min(self.get_maze_distance(pos2, b) for b in self.home_boundary)
                if dist < best_dist:
                    best_action = action
                    best_dist   = dist
        return best_action

    def defensive_action(self, game_state, actions):
        values = [self.evaluate(game_state, a) for a in actions]
        m = max(values)
        return [a for a, v in zip(actions, values) if v == m]

    def offensive_action(self, game_state, actions, depth):
        best_val     = float('-inf')
        best_actions = []
        for action in actions:
            val = self._lookahead(game_state, action, depth)
            if val > best_val:
                best_val     = val
                best_actions = [action]
            elif val == best_val:
                best_actions.append(action)
        return best_actions

    def _lookahead(self, game_state, action, depth):
        if depth <= 1:
            return self.evaluate(game_state, action)
        successor    = self.get_successor(game_state, action)
        next_actions = successor.get_legal_actions(self.index)
        return max(self._lookahead(successor, a, depth - 1) for a in next_actions)

    def get_features(self, game_state, action):
        if self.aggressive_play_mode:
            return self._get_features_off(game_state, action)
        return self._get_features_def(game_state, action)

    def get_weights(self, game_state, action):
        if self.aggressive_play_mode:
            return self._get_weights_off(game_state, action)
        return self._get_weights_def(game_state, action)

    # ------------------------------------------------------------------
    # Offensive features
    # ------------------------------------------------------------------

    def _get_target_zone_y(self, game_state):
        """
        Returns the target y-coordinate based on zone preference (top or bottom).
        Top: topmost walkable position on home side.
        Bottom: bottommost walkable position on home side.
        """
        if self.target_vertical_zone == 'top':
            return self.topmost_y
        else:
            return self.bottommost_y

    def _get_features_off(self, game_state, action):
        features  = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state  = successor.get_agent_state(self.index)
        my_pos    = my_state.get_position()

        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        capsules = self.get_capsules(successor)
        features['capsules_remaining'] = -len(capsules)

        if food_list:
            features['distance_to_food'] = min(self.get_maze_distance(my_pos, f) for f in food_list)

        if capsules:
            features['distance_to_capsule'] = min(self.get_maze_distance(my_pos, c) for c in capsules)

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]

        # Dangerous ghosts — only penalise when they're actually threatening (≤8 steps)
        ghosts = [a for a in enemies
                  if not a.is_pacman and a.scared_timer == 0 and a.get_position() is not None]
        if ghosts:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in ghosts]
            min_dist = min(dists)
            if min_dist <= 8:
                features['ghost_distance'] = min_dist
            if min_dist <= 2:
                features['close_danger'] = 1
            if min_dist <= 1:
                features['immediate_danger'] = 1

        # Scared ghosts — free points, chase them
        scared = [a for a in enemies
                  if not a.is_pacman and a.scared_timer > 0 and a.get_position() is not None]
        if scared:
            features['scared_ghost_distance'] = min(
                self.get_maze_distance(my_pos, a.get_position()) for a in scared
            )

        # Encourage movement toward chosen vertical zone while in own territory (ghost on home side)
        if not my_state.is_pacman:
            target_y = self._get_target_zone_y(game_state)
            vertical_distance = abs(my_pos[1] - target_y)
            features['zone_alignment'] = vertical_distance

        # Dead-end risk: penalised more when a ghost is nearby
        if my_state.is_pacman:
            dead_risk = self.dead_end_depth.get(my_pos, 0)
            features['dead_end_depth'] = dead_risk  # always penalise being inside dead ends
            if dead_risk > 0:
                proximity = features['ghost_distance'] if features['ghost_distance'] > 0 else 8
                features['danger_zone'] = dead_risk / max(proximity, 1)

            # Gentle pull toward home boundary when carrying food
            num_carrying = my_state.num_carrying
            if num_carrying > 0:
                home_dist = min(self.get_maze_distance(my_pos, b) for b in self.home_boundary)
                features['return_urgency'] = num_carrying * home_dist

        return features

    def _get_weights_off(self, game_state, action):
        return {
            'successor_score':       100,
            'capsules_remaining':    5000,
            'distance_to_food':       -5,
            'distance_to_capsule':   -60,
            'ghost_distance':         20,
            'close_danger':         -250,
            'immediate_danger':     -600,
            'scared_ghost_distance':  -60,
            'dead_end_depth':         -8,
            'danger_zone':          -200,
            'return_urgency':         -3,
            'zone_alignment':         -100,
        }

    # ------------------------------------------------------------------
    # Defensive fallback features (used when switching to defense mode)
    # ------------------------------------------------------------------

    def _get_features_def(self, game_state, action):
        features  = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state  = successor.get_agent_state(self.index)
        my_pos    = my_state.get_position()

        features['on_defense'] = 1
        enemies   = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders  = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if invaders:
            features['invader_distance'] = min(self.get_maze_distance(my_pos, a.get_position()) for a in invaders)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        return features

    def _get_weights_def(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100,
                'invader_distance': -10, 'stop': -100, 'reverse': -2}

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

        mid_x = game_state.data.layout.width // 2
        boundary_x = mid_x - 1 if self.red else mid_x
        walls = game_state.get_walls()
        height = game_state.data.layout.height
        self.patrol_points = [(boundary_x, y) for y in range(height) if not walls[boundary_x][y]]

        # Belief distributions over opponent positions (for tracking invisible enemies)
        all_cells = [
            (x, y) for x in range(game_state.data.layout.width)
            for y in range(height) if not walls[x][y]
        ]
        self.opponent_indices = self.get_opponents(game_state)
        self.beliefs = {}
        for idx in self.opponent_indices:
            b = util.Counter()
            for cell in all_cells:
                b[cell] = 1.0
            b.normalize()
            self.beliefs[idx] = b

    def _update_beliefs(self, game_state):
        walls = game_state.get_walls()
        my_pos = game_state.get_agent_state(self.index).get_position()
        noisy_dists = game_state.get_agent_distances()

        for idx in self.opponent_indices:
            opp_state = game_state.get_agent_state(idx)
            pos = opp_state.get_position()

            if pos is not None:
                self.beliefs[idx] = util.Counter()
                self.beliefs[idx][nearest_point(pos)] = 1.0
                continue

            # Predict: spread probability to neighboring cells
            new_b = util.Counter()
            for cell, prob in self.beliefs[idx].items():
                cx, cy = int(cell[0]), int(cell[1])
                nbrs = [
                    (cx + dx, cy + dy)
                    for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= cx + dx < walls.width and 0 <= cy + dy < walls.height
                    and not walls[cx + dx][cy + dy]
                ]
                for nb in nbrs:
                    new_b[nb] += prob / len(nbrs)

            # Observe: weight by how well each cell matches the noisy distance reading
            noisy = noisy_dists[idx]
            for cell in list(new_b.keys()):
                true_dist = self.get_maze_distance(my_pos, cell)
                new_b[cell] *= max(0.001, 1.0 - abs(true_dist - noisy) / 6.0)

            new_b.normalize()
            self.beliefs[idx] = new_b

    def _most_likely_invader_cell(self, game_state):
        """Returns the highest-probability cell among opponents believed to be on our side."""
        walls = game_state.get_walls()
        mid_x = game_state.data.layout.width // 2
        our_xs = range(0, mid_x) if self.red else range(mid_x, walls.width)

        best_cell = None
        best_prob = 0.0
        for idx in self.opponent_indices:
            for cell, prob in self.beliefs[idx].items():
                if cell[0] in our_xs and prob > best_prob:
                    best_prob = prob
                    best_cell = cell
        return best_cell

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        self._update_beliefs(game_state)

        # TRAP OVERRIDE
        trap_action = self._trap_override(game_state, actions)
        if trap_action is not None:
            return trap_action

        my_state = game_state.get_agent_state(self.index)
        my_pos = nearest_point(my_state.get_position())

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        visible_invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # If no visible invaders, navigate toward most likely invader position from beliefs
        if not visible_invaders and not my_state.is_pacman and my_state.scared_timer == 0:
            target = self._most_likely_invader_cell(game_state)
            if target:
                action, _ = a_star_goals(my_pos, [target], game_state)
                if action and action != Directions.STOP:
                    return action

        values = [self.evaluate(game_state, a) for a in actions]
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
            try:
                features['distance_to_boundary'] = median(dists)
            except:
                features['distance_to_boundary'] = 0
                
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

##################################################################################
################## UTILITIES #####################################################
##################################################################################

def a_star_goals(start, goals, game_state, avoid=None):
    """
    A* from start to the nearest cell in goals.
    avoid: set of cells treated as impassable (e.g. ghost danger zones).
    Returns (first_action, distance), or (None, inf) if no path exists.
    """
    start = nearest_point(start)
    goals = set(nearest_point(g) for g in goals)
    if not goals:
        return None, float('inf')
    if avoid is None:
        avoid = set()

    walls = game_state.get_walls()
    width = game_state.data.layout.width
    height = game_state.data.layout.height

    def heuristic(pos):
        return min(abs(pos[0] - g[0]) + abs(pos[1] - g[1]) for g in goals)

    pq = util.PriorityQueue()
    pq.push((start, None, 0), heuristic(start))
    visited = {}

    while not pq.is_empty():
        pos, first_action, g = pq.pop()

        if pos in visited and visited[pos] <= g:
            continue
        visited[pos] = g

        if pos in goals:
            return (first_action if first_action is not None else Directions.STOP), g

        x, y = pos
        for dx, dy, action in [(-1, 0, Directions.WEST), (1, 0, Directions.EAST),
                                (0, -1, Directions.SOUTH), (0, 1, Directions.NORTH)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not walls[nx][ny] and (nx, ny) not in avoid:
                new_g = g + 1
                if (nx, ny) not in visited or visited[(nx, ny)] > new_g:
                    taken = action if first_action is None else first_action
                    pq.push(((nx, ny), taken, new_g), new_g + heuristic((nx, ny)))

    return None, float('inf')