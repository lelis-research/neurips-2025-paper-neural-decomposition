import numpy as np
import math
import gc
import copy


SEEDS = {
    "TL-BR": 0,
    "TR-BL": 1, 
    "BR-TL": 2, 
    "BL-TR": 3,

    "ML-BR": 4,
    "ML-TR": 5,
    "MR-TL": 6,
    "MR-BL": 7,
    "TL-MR": 8,
    "BL-MR": 9,
    "TR-ML": 10,
    "BR-ML": 11,

    "BL-MR-ML-BM-TM": 12,
    "BL-MR-BM": 13,

    "MM-MR-ML|hallways": 14,

    "TL-BR|block": 20,
    "TR-BL|block": 21, 
    "BR-TL|block": 22, 
    "BL-TR|block": 23,
}

PROBLEM_NAMES = {value:key for key, value in SEEDS.items()}

# These are for 5*5 grid
OPTIMAL_TRAJECTORY_LENGTHS = {
    seed: 24 for seed in range(12)
}
OPTIMAL_TRAJECTORY_LENGTHS.update({
    12: 42,
    13: 30
})

OPTIMAL_TRAJECTORY_REWARD = {
    seed: 24 for seed in range(12)
}
OPTIMAL_TEST_REWARD = {
        12: 40,
        13: 20
    }

DIRECTIONS = {(0,0,1): "U",
              (0,1,2): "D",
              (2,1,0): "L",
              (1,0,2): "R"}

class Problem:
    def __init__(self, rows, columns, problem_str):
        self.rows = rows
        self.columns = columns
        self.walls = []

        problem_str_decopled = problem_str.split("|")
        locations_str = problem_str_decopled[0]
        if len(problem_str_decopled) > 1:
            self.setup_walls(problem_str_decopled[1])
        self.initial, self.goals = self._parse_problem(locations_str)
        self.init_goals = copy.deepcopy(self.goals)
        
        self.reset()

    def _parse_problem(self, problem_str):
        problem_parts = problem_str.split("-")
        initial = self._parse_position(problem_parts[0])
        goals = [self._parse_position(goal_str) for goal_str in problem_parts[1:]]
        return initial, goals

    def _parse_position(self, pos_str):
        v_loc, h_loc = pos_str[0], pos_str[1]
        if v_loc == 'T':
            row = 0
        elif v_loc == 'B':
            row = self.rows - 1
        elif v_loc == 'M':
            row = math.floor(self.rows / 2)
        else:
            raise ValueError("Invalid row specifier. Use 'T' for top, 'B' for bottom, or 'M' for middle.")
        
        if h_loc == 'L':
            col = 0
        elif h_loc == 'R':
            col = self.columns - 1
        elif h_loc == 'M':
            col = math.floor(self.columns / 2)
        else:
            raise ValueError("Invalid column specifier. Use 'L' for left, 'R' for right, or 'M' for middle.")
        
        return (row, col)
    
    def setup_walls(self, walls_str):
        if walls_str == "hallways":
            for i in range(self.rows):
                for j in range(self.columns):
                    if i != math.floor(self.rows / 2) and j != math.floor(self.columns / 2):
                        self.walls.append((i, j))
        elif walls_str == "block":
            for i in range(2, self.rows - 2):
                for j in range(2, self.columns - 2):
                    self.walls.append((i, j))

    def remove_goal(self, loc) -> bool:
        """
        Updates the goal coordinations.

        Returns whether or not all goals have been reached.
        
        Returns:
            bool: True if all goals have been reached, False otherwise.
        """
        try:
            self.goals.remove(loc)
        except ValueError:
            raise Exception(f"Goal {loc} not found in the list of goals.")

        finished = len(self.goals) == 0
        return finished
    
    def reset(self):
        self.goals = copy.deepcopy(self.init_goals)

    def is_goal(self, loc):
        return any([(loc[0] == goal[0]) and (loc[1]==goal[1]) for goal in self.goals])
        


class Game:
    """
    The (0, 0) in the matrices show top and left and it goes to the bottom and right as 
    the indices increases.
    """
    def __init__(self, rows, columns, problem_str, action_pattern_length=3, init_x=None, init_y=None):
        self._rows = rows
        self._columns = columns

        self.problem = Problem(rows, columns, problem_str)
        self._matrix_structure = np.zeros((rows, columns))
        for wall in self.problem.walls:
            self._matrix_structure[wall[0]][wall[1]] = 1
        
        if init_x and init_y:
            self.reset((init_x, init_y))
        else:
            self.reset()

        # state of current action sequence
        self._pattern_length = action_pattern_length

        self._action_pattern = {}

        if action_pattern_length == 3:
            """
                Mapping used: 
                0, 0, 1 -> up (0)
                0, 1, 2 -> down (1)
                2, 1, 0 -> left (2)
                1, 0, 2 -> right (3)
            """
            self._action_pattern[(0, 0, 1)] = 0
            self._action_pattern[(0, 1, 2)] = 1
            self._action_pattern[(2, 1, 0)] = 2
            self._action_pattern[(1, 0, 2)] = 3
        elif action_pattern_length == 4:
            """
                Mapping used: 
                0, 2, 2, 1 -> up (0)
                0, 0, 1, 1 -> down (1)
                1, 2, 1, 0 -> left (2)
                1, 0, 2, 2 -> right (3)
            """
            self._action_pattern[(0, 2, 2, 1)] = 0
            self._action_pattern[(0, 0, 1, 1)] = 1
            self._action_pattern[(1, 2, 1, 0)] = 2
            self._action_pattern[(1, 0, 2, 2)] = 3


    def reset(self, init_loc=None):
        self._matrix_unit = np.zeros((self._rows, self._columns))
        self.problem.reset()
        initial = self.problem.initial
        self._x, self._y = init_loc if init_loc else initial
        self._matrix_unit[self._x][self._y] = 1
        self._state = []
        self.setup_goals()   
        gc.collect()

    def setup_goals(self):
        self._matrix_goal = np.zeros((self._rows, self._columns))
        for goal in self.problem.goals:
            self._matrix_goal[goal[0]][goal[1]] = 1

    def get_goals(self):
        return self.problem.goals

    def __repr__(self) -> str:
        str_map = ""
        for i in range(self._rows):
            for j in range(self._columns):
                if self._matrix_unit[i][j] == 1:
                    str_map += " A "
                elif self._matrix_structure[i][j] == 1:
                     str_map += " B "
                elif self._matrix_goal[i][j] == 1:
                     str_map += " G "
                elif self.problem.is_goal((i,j)):
                     str_map += " g "
                else: 
                     str_map += " 0 "
            str_map += "\n"
        return str_map
    
    def represent_options(self, options: dict) -> str:
        str_map = ""
        option_letters = "UDLR"
        for i in range(self._rows):
            for j in range(self._columns):
                if self._matrix_structure[i][j] == 1:
                     str_map += " B "
                elif self._matrix_goal[i][j] == 1:
                     str_map += " G "
                elif (i,j) in options and tuple(options[(i,j)]) in self._action_pattern:
                    str_map += f" {option_letters[self._action_pattern[tuple(options[(i,j)])]]} "
                elif self._matrix_unit[i][j] == 1:
                    str_map += " A "
                else: 
                     str_map += " 0 "
            str_map += "\n"
        return str_map
    
    def get_observation(self):
        one_hot_matrix_state = np.zeros((self._pattern_length, self._pattern_length), dtype=int)
        for i, v in enumerate(self._state):
            one_hot_matrix_state[v][i] = 1
        return np.concatenate((self._matrix_unit.ravel(), one_hot_matrix_state.ravel(), self._matrix_goal.ravel()))
    
    def is_over(self):
        is_goal = self._matrix_goal[self._x][self._y] == 1
        finished = False
        if is_goal:
            finished = self.problem.remove_goal((self._x, self._y))
            self._matrix_goal[self._x][self._y] = 0
        return finished, is_goal
    
    def get_actions(self):
        return [0, 1, 2]
   
    def apply_action(self, action):
        """
        Mapping used: 
        0, 0, 1 -> up (0)
        0, 1, 2 -> down (1)
        2, 1, 0 -> left (2)
        1, 0, 2 -> right (3)
        """
        # each column in _state_matrix represents an action
        self._state.append(action)

        if len(self._state) == self._pattern_length:
            action_tuple = tuple(self._state)
            if action_tuple in self._action_pattern:
                # moving up
                if self._action_pattern[action_tuple] == 0:
                    if self._x - 1 >= 0 and self._matrix_structure[self._x - 1][self._y] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._x -= 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving down
                if self._action_pattern[action_tuple] == 1:
                    if self._x + 1 < self._matrix_unit.shape[0] and self._matrix_structure[self._x + 1][self._y] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._x += 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving left
                if self._action_pattern[action_tuple] == 2:
                    if self._y - 1 >= 0 and self._matrix_structure[self._x][self._y - 1] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._y -= 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving right
                if self._action_pattern[action_tuple] == 3:
                    if self._y + 1 < self._matrix_unit.shape[1] and self._matrix_structure[self._x][self._y + 1] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._y += 1
                        self._matrix_unit[self._x][self._y] = 1
            self._state = []

class basic_actions:
    def __init__(self, action):
        self.action = action

    def predict(self, x):
        return self.action

    def predict_hierarchical(self, x, epsilon):
        return self.predict(x)