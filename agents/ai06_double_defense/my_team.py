# ai06_double_defense/my_team.py
# Strategy: Two coordinated defenders — zero intentional offense.
# Forces the Q-agent to score against an impenetrable defensive wall.
#
# - SentinelAgent:  hugs the boundary entrance column; intercepts any pacman
#                   that tries to cross, and does not stray deep into own territory.
# - InteriorGuard:  roams between the food positions on our side that are most
#                   likely to be targeted, acting as a second ring of defense
#                   if the SentinelAgent is bypassed.
#
# Together they create maximum defensive pressure with complementary coverage zones.

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='SentinelAgent', second='InteriorGuard', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Helpers #
##########

_DIR_VECTORS = {
    Directions.NORTH: (0, 1),
    Directions.SOUTH: (0, -1),
    Directions.EAST:  (1, 0),
    Directions.WEST:  (-1, 0),
}


def bfs_first_action(start, goal, walls):
    if start == goal:
        return Directions.STOP
    queue = util.Queue()
    queue.push((start, None))
    visited = {start}
    while not queue.is_empty():
        pos, first_action = queue.pop()
        for action, (dx, dy) in _DIR_VECTORS.items():
            nx, ny = int(pos[0] + dx), int(pos[1] + dy)
            npos = (nx, ny)
            if npos in visited or walls[nx][ny]:
                continue
            fa = action if first_action is None else first_action
            if npos == goal:
                return fa
            visited.add(npos)
            queue.push((npos, fa))
    return Directions.STOP


##########
# Agents #
##########

class DefenseBase(CaptureAgent):

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


class SentinelAgent(DefenseBase):
    """
    First line of defense: stands guard along the boundary entrance column.
    Patrols that column continuously.
    Immediately chases the nearest visible invader using BFS.
    Never crosses into enemy territory intentionally.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.boundary_patrol = []
        self.patrol_index = 0

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self._build_boundary(game_state)

    def _build_boundary(self, game_state):
        layout = game_state.data.layout
        walls = game_state.get_walls()
        x = (layout.width // 2 - 1) if self.red else (layout.width // 2)
        self.boundary_patrol = [
            (x, y) for y in range(1, layout.height - 1) if not walls[x][y]
        ]
        if not self.boundary_patrol:
            self.boundary_patrol = [self.start]
        # Start at the top of the boundary for vertical sweep
        self.patrol_index = len(self.boundary_patrol) - 1

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        walls = game_state.get_walls()
        my_pos = game_state.get_agent_position(self.index)

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # Chase nearest invader
        if invaders:
            target = min(invaders, key=lambda a: self.get_maze_distance(my_pos, a.get_position()))
            action = bfs_first_action(my_pos, target.get_position(), walls)
            return action if action != Directions.STOP else random.choice(actions)

        # Patrol the boundary column
        target = self.boundary_patrol[self.patrol_index]
        if my_pos == target:
            self.patrol_index = (self.patrol_index - 1) % len(self.boundary_patrol)
            target = self.boundary_patrol[self.patrol_index]

        action = bfs_first_action(my_pos, target, walls)
        return action if action != Directions.STOP else random.choice(actions)


class InteriorGuard(DefenseBase):
    """
    Second line of defense: roams between the food positions on our side
    that are most exposed (closest to the boundary).
    Provides interior coverage for invaders that slip past the SentinelAgent.
    Also chases visible invaders directly.
    """
    # How many of our food positions (sorted by distance to boundary) to cycle between
    PATROL_FOOD_COUNT = 6

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.interior_waypoints = []
        self.waypoint_index = 0

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self._build_waypoints(game_state)

    def _build_waypoints(self, game_state):
        """Select the N most boundary-exposed food positions on our side."""
        layout = game_state.data.layout
        boundary_x = (layout.width // 2 - 1) if self.red else (layout.width // 2)
        # Food we are defending = food on our side (get_food_you_are_defending)
        defending_food = self.get_food_you_are_defending(game_state).as_list()

        if not defending_food:
            self.interior_waypoints = [self.start]
            return

        # Sort by proximity to the boundary column (x distance)
        defending_food.sort(key=lambda pos: abs(pos[0] - boundary_x))
        self.interior_waypoints = defending_food[:self.PATROL_FOOD_COUNT]

        if not self.interior_waypoints:
            self.interior_waypoints = [self.start]

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        walls = game_state.get_walls()
        my_pos = game_state.get_agent_position(self.index)

        # Refresh waypoints periodically in case food has been eaten
        current_defending = self.get_food_you_are_defending(game_state).as_list()
        self.interior_waypoints = [wp for wp in self.interior_waypoints if wp in current_defending]
        if not self.interior_waypoints:
            self._build_waypoints(game_state)

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # Chase the nearest invader
        if invaders:
            target = min(invaders, key=lambda a: self.get_maze_distance(my_pos, a.get_position()))
            action = bfs_first_action(my_pos, target.get_position(), walls)
            return action if action != Directions.STOP else random.choice(actions)

        # Advance through food waypoints
        if self.interior_waypoints:
            self.waypoint_index = self.waypoint_index % len(self.interior_waypoints)
            target = self.interior_waypoints[self.waypoint_index]
            if my_pos == target:
                self.waypoint_index = (self.waypoint_index + 1) % len(self.interior_waypoints)
                target = self.interior_waypoints[self.waypoint_index]
            action = bfs_first_action(my_pos, target, walls)
            return action if action != Directions.STOP else random.choice(actions)

        return random.choice(actions)
