# ai05_bfs/my_team.py
# Strategy: BFS path-following agents.
# Both agents use breadth-first search to compute and follow the actual shortest
# path to their targets rather than relying on greedy one-step feature evaluation.
# This makes them harder opponents than reflex agents.
#
# - BFSAttacker:  follows the globally shortest path to the nearest food,
#                 immediately re-routes to the nearest capsule when an active ghost
#                 is within DANGER_DIST, then returns home once carrying enough food.
# - BFSDefender:  follows the shortest path to the nearest visible invader;
#                 falls back to patrolling the boundary when no invader is visible.

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='BFSAttacker', second='BFSDefender', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Helpers #
##########

# Manual direction vectors — avoids importing Actions separately.
_DIR_VECTORS = {
    Directions.NORTH: (0, 1),
    Directions.SOUTH: (0, -1),
    Directions.EAST:  (1, 0),
    Directions.WEST:  (-1, 0),
}


def bfs_first_action(start, goal, walls):
    """
    Returns the first action on the BFS-shortest path from `start` to `goal`.
    `walls` is the Grid object from game_state.get_walls().
    Returns Directions.STOP if start == goal or no path exists.
    """
    if start == goal:
        return Directions.STOP

    queue = util.Queue()
    queue.push((start, None))   # (position, first_action_taken)
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

    return Directions.STOP  # no path found


##########
# Agents #
##########

class BFSBase(CaptureAgent):

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


class BFSAttacker(BFSBase):
    """
    Offensive agent that follows the BFS-optimal path to the nearest food.
    Priorities (in order):
      1. Return home when food_left <= 2 or carrying >= MAX_CARRY.
      2. Rush the nearest capsule when an active ghost is within DANGER_DIST.
      3. Follow BFS path to the nearest food.
    """
    MAX_CARRY = 5
    DANGER_DIST = 5

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        walls = game_state.get_walls()
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        food_left = len(self.get_food(game_state).as_list())

        # 1. Head home
        if food_left <= 2 or (my_state.is_pacman and my_state.num_carrying >= self.MAX_CARRY):
            action = bfs_first_action(my_pos, self.start, walls)
            return action if action != Directions.STOP else random.choice(actions)

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_ghosts = [
            a for a in enemies
            if not a.is_pacman and a.scared_timer == 0 and a.get_position() is not None
        ]

        # 2. Seek capsule when a ghost is close
        if active_ghosts:
            ghost_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in active_ghosts]
            if min(ghost_dists) <= self.DANGER_DIST:
                capsules = self.get_capsules(game_state)
                if capsules:
                    target = min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
                    action = bfs_first_action(my_pos, target, walls)
                    return action if action != Directions.STOP else random.choice(actions)
                # No capsule available — flee home
                action = bfs_first_action(my_pos, self.start, walls)
                return action if action != Directions.STOP else random.choice(actions)

        # 3. BFS to nearest food
        food_list = self.get_food(game_state).as_list()
        if food_list:
            target = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
            action = bfs_first_action(my_pos, target, walls)
            return action if action != Directions.STOP else random.choice(actions)

        return random.choice(actions)


class BFSDefender(BFSBase):
    """
    Defensive agent that uses BFS to reach the nearest visible invader.
    Falls back to boundary column patrol when no invader is visible.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.patrol_positions = []
        self.patrol_index = 0

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self._build_patrol(game_state)

    def _build_patrol(self, game_state):
        layout = game_state.data.layout
        walls = game_state.get_walls()
        x = (layout.width // 2 - 1) if self.red else (layout.width // 2)
        self.patrol_positions = [
            (x, y) for y in range(1, layout.height - 1) if not walls[x][y]
        ]
        if not self.patrol_positions:
            self.patrol_positions = [self.start]
        self.patrol_index = len(self.patrol_positions) // 2

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        walls = game_state.get_walls()
        my_pos = game_state.get_agent_position(self.index)

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # Chase nearest invader via BFS
        if invaders:
            target = min(invaders, key=lambda a: self.get_maze_distance(my_pos, a.get_position()))
            action = bfs_first_action(my_pos, target.get_position(), walls)
            return action if action != Directions.STOP else random.choice(actions)

        # Patrol: advance to next patrol waypoint
        target = self.patrol_positions[self.patrol_index]
        if my_pos == target:
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_positions)
            target = self.patrol_positions[self.patrol_index]

        action = bfs_first_action(my_pos, target, walls)
        return action if action != Directions.STOP else random.choice(actions)
