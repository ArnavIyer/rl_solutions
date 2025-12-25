from matplotlib import pyplot
import math
import random
from typing import Dict, Tuple, Literal
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod

Action4 = Literal['left', 'right', 'up', 'down']
Action8 = Literal['left', 'leftup', 'leftdown', 'right', 'rightup', 'rightdown', 'up', 'down']

@dataclass(frozen=True)
class Point: 
    x: int
    y: int

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)
    
    def __eq__(self, other: 'Point') -> bool:
        return self.x == other.x and self.y == other.y
    
    def __repr__(self) -> str:
        return f"{self.x}|{self.y}"

class BaseEnv(ABC):
    def __init__(self):
        self.state = Point(0, 3)
        self.max_y = 6
        self.max_x = 9
        self.goal = Point(7, 3)
        self.increments: Dict = {}
        self.A = []

    @staticmethod
    def get_wind(x: int) -> int:
        if (x >= 3 and x <= 5) or x == 8:
            return 1
        if x == 6 or x == 7:
            return 2
        return 0
    
    def reset(self) -> Point:
        self.state = Point(0, 3)
        return self.state

    def step(self, action) -> Tuple[Point, float, bool]:
        self.state += Point(0, self.get_wind(self.state.x))
        self.state += self.increments[action]
        self.state = Point(np.clip(self.state.x, 0, self.max_x), np.clip(self.state.y, 0, self.max_y))
        return self.state, -1, True if self.state == self.goal else False
    
    def get_policy(self, Q: Dict[str, float]) -> Dict[str, any]:
        policy = {}
        for x in range(0, self.max_x + 1):
            for y in range(0, self.max_y + 1):
                maxval = -100000000
                argmax_a = self.A[0]
                for a in self.A:
                    if maxval < Q[f"{x}|{y}|{a}"]:
                        maxval = Q[f"{x}|{y}|{a}"]
                        argmax_a = a
                policy[f"{x}|{y}"] = argmax_a
        return policy
    
    def get_action_epsilon_greedy(self, policy: Dict[str, any], state: Point, epsilon: float) -> any:
        if random.random() < epsilon:
            return random.choice(self.A)
        return policy[str(state)]

class Env8(BaseEnv):
    def __init__(self):
        super().__init__()
        self.increments: Dict[Action8, Point] = {
            'left': Point(-1, 0),
            'leftup': Point(-1, 1),
            'leftdown': Point(-1, -1),
            'right': Point(1, 0),
            'rightup': Point(1, 1),
            'rightdown': Point(1, -1),
            'up': Point(0, 1),
            'down': Point(0, -1),
        }
        self.A = [a for a in self.increments]


class Env4(BaseEnv):
    def __init__(self):
        super().__init__()
        self.increments: Dict[Action4, Point] = {
            'left': Point(-1, 0),
            'right': Point(1, 0),
            'up': Point(0, 1),
            'down': Point(0, -1),
        }
        self.A = [a for a in self.increments]

class Env9(BaseEnv):
    def __init__(self):
        super().__init__()
        self.increments: Dict[Action4, Point] = {
            'left': Point(-1, 0),
            'leftup': Point(-1, 1),
            'leftdown': Point(-1, -1),
            'right': Point(1, 0),
            'rightup': Point(1, 1),
            'rightdown': Point(1, -1),
            'up': Point(0, 1),
            'down': Point(0, -1),
            'stay': Point(0, 0),
        }
        self.A = [a for a in self.increments]



def run_simulation(env_class, alpha, epsilon):
    env = env_class()
    Q = {f"{x}|{y}|{a}": 0.0 for x in range(0, env.max_x + 1) for y in range(0, env.max_y + 1) for a in env.A}
    policy = env.get_policy(Q)
    counter = 0
    ep = 0
    timesteps = []
    episodes = []

    while counter < 15000:
        ep += 1
        state = env.reset()
        action = env.get_action_epsilon_greedy(policy, state, epsilon)
        done = False
        while not done:
            counter += 1
            policy = env.get_policy(Q)
            old_state = state
            old_action = action
            state, reward, done = env.step(action)
            action = env.get_action_epsilon_greedy(policy, state, epsilon)
            Q[f"{old_state}|{old_action}"] = Q[f"{old_state}|{old_action}"] + alpha * (reward + Q[f"{state}|{action}"] - Q[f"{old_state}|{old_action}"])
            timesteps.append(counter)
            episodes.append(ep)
    return timesteps, episodes

alpha = 0.5
epsilon = 0.1
line_styles = ['-', '--', ':']

for i, style in enumerate(line_styles):
    timesteps, episodes = run_simulation(Env4, alpha, epsilon)
    label = '4-directional movement' if i == 0 else None
    pyplot.plot(timesteps, episodes, 'r', linestyle=style, markersize=3, label=label)

for i, style in enumerate(line_styles):
    timesteps, episodes = run_simulation(Env8, alpha, epsilon)
    label = '8-directional movement' if i == 0 else None
    pyplot.plot(timesteps, episodes, 'b', linestyle=style, markersize=3, label=label)

for i, style in enumerate(line_styles):
    timesteps, episodes = run_simulation(Env9, alpha, epsilon)
    label = '8-directional + stay movement' if i == 0 else None
    pyplot.plot(timesteps, episodes, 'g', linestyle=style, markersize=3, label=label)

pyplot.legend()
pyplot.show()


# Create a new plot that graphs the gridworld and the optimal path learned by each policy to reach the goal from the start.
# Don't do the epsilon greedy action selection here; just follow the learned policy directly from start to goal for each environment.
def plot_optimal_path(env_class, alpha, epsilon):
    env = env_class()
    Q = {f"{x}|{y}|{a}": 0.0 for x in range(0, env.max_x + 1) for y in range(0, env.max_y + 1) for a in env.A}
    policy = env.get_policy(Q)
    counter = 0
    ep = 0

    while counter < 15000:
        ep += 1
        state = env.reset()
        action =  policy[str(state)]
        done = False
        while not done:
            counter += 1
            policy = env.get_policy(Q)
            old_state = state
            old_action = action
            state, reward, done = env.step(action)
            action = policy[str(state)]
            Q[f"{old_state}|{old_action}"] = Q[f"{old_state}|{old_action}"] + alpha * (reward + Q[f"{state}|{action}"] - Q[f"{old_state}|{old_action}"])

    state = env.reset()
    path_x = [state.x]
    path_y = [state.y]
    done = False
    steps = 0
    while not done and steps < 1000:
        steps += 1
        policy = env.get_policy(Q)
        action = policy[str(state)]
        state, reward, done = env.step(action)
        path_x.append(state.x)
        path_y.append(state.y)

    pyplot.plot(path_x, path_y, marker='o')
    pyplot.xlim(-1, env.max_x + 1)
    pyplot.ylim(-1, env.max_y + 1)
    pyplot.title(f'Optimal Path in {env_class.__name__}')
    pyplot.xlabel('X-axis')
    pyplot.ylabel('Y-axis')
    pyplot.grid()
    pyplot.show()

plot_optimal_path(Env4, alpha, epsilon)
plot_optimal_path(Env8, alpha, epsilon)
plot_optimal_path(Env9, alpha, epsilon)