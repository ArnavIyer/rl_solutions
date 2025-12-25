from matplotlib import pyplot
import math
import random
from typing import Dict, Tuple, Literal
from dataclasses import dataclass
import numpy as np

Action = Literal['left', 'right', 'up', 'down']

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


class Env:
    def __init__(self):
        self.state = Point(0, 3)

        # inclusive max
        self.max_y = 6
        self.max_x = 9

        self.goal = Point(7, 3)
        self.increments: Dict[Action, Point] = {
            'left': Point(-1, 0),
            'right': Point(1, 0),
            'up': Point(0, 1),
            'down': Point(0, -1),
        }
        self.A = [a for a in self.increments]

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

    def step(self, action: Action) -> Tuple[Point, float, bool]:
        self.state += self.increments[action]
        self.state += Point(0, Env.get_wind(self.state.x))

        self.state = Point(np.clip(self.state.x, 0, self.max_x), np.clip(self.state.y, 0, self.max_y))
        return self.state, -1, True if self.state == self.goal else False

env = Env()        

Q = {f"{x}|{y}|{a}": 0.0 for x in range(0, env.max_x + 1) for y in range(0, env.max_y + 1) for a in env.A}

def get_policy(Q: Dict[str, float]) -> Dict[str, Action]:
    policy = {}

    for x in range(0, env.max_x + 1):
        for y in range(0, env.max_y + 1):

            maxval = -100000000
            argmax_a = 'left'
            for a in env.A:
                if maxval < Q[f"{x}|{y}|{a}"]:
                    maxval = Q[f"{x}|{y}|{a}"]
                    argmax_a = a
            
            policy[f"{x}|{y}"] = argmax_a
    
    return policy

def get_action_epsilon_greedy(policy: Dict[str, Action], state: Point, epsilon: float) -> Action:
    if random.random() < epsilon:
        return random.choice(env.A)
    return policy[str(state)]

policy = get_policy(Q)
alpha = 0.5
epsilon = 0.1
episodes = []
timesteps = []
counter = 0
ep = 0

while counter < 8000:
    ep += 1
    state = env.reset()
    action = get_action_epsilon_greedy(policy, state, epsilon)
    done = False
    while not done:
        counter += 1
        policy = get_policy(Q)
        old_state = state
        old_action = action
        state, reward, done = env.step(action)
        action = get_action_epsilon_greedy(policy, state, epsilon)
        Q[f"{old_state}|{old_action}"] = Q[f"{old_state}|{old_action}"] + alpha * (reward + Q[f"{state}|{action}"] - Q[f"{old_state}|{old_action}"])
        timesteps.append(counter)
        episodes.append(ep)

pyplot.plot(timesteps, episodes)
pyplot.show()