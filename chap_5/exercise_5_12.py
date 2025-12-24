'''
Exercise 5.12: Racetrack (programming): Consider driving a race car around a turn
like those shown in Figure 5.5. You want to go as fast as possible, but not so fast as
to run off the track. In our simplified racetrack, 
- the car is at one of a discrete set of grid positions, the cells in the diagram. 
- The velocity is also discrete, a number of grid cells moved horizontally and vertically per time step. 
- The actions are increments to the velocity components. 
- Each may be changed by +1,-1, or 0 in each step, for a total of nine (3 ⇥3) actions. 

Both velocity components are restricted to be nonnegative and less
than 5, and they cannot both be zero except at the starting line. 

Each episode begins
in one of the randomly selected start states with both velocity components zero and
ends when the car crosses the finish line. 

The rewards are -1 for each step until the car crosses the finish line.

If the car hits the track boundary, it is moved back to a random
position on the starting line, both velocity components are reduced to zero, and the
episode continues.

Before updating the car's location at each time step, check to see if
the projected path of the car intersects the track boundary.

If it intersects the finish line, the episode ends; if it intersects anywhere else, the car is considered to have hit the track
boundary and is sent back to the starting line. 

To make the task more challenging, with probability 0.1 at each time step the velocity increments are both zero, independently of
the intended increments. Apply a Monte Carlo control method to this task to compute
the optimal policy from each starting state. Exhibit several trajectories following the
optimal policy (but turn the noise o↵ for these trajectories).
'''

'''
on-policy first-visit monte carlo control
'''

from typing import List, Tuple, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
from collections import defaultdict
import math
import copy


#  Utility classes -----------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Point: 
    x: int
    y: int

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)
    
    def __eq__(self, other: 'Point') -> bool:
        return self.x == other.x and self.y == other.y


@dataclass(frozen=True)
class Segment:
    start: Point
    end: Point

    def contains(self, p: Point) -> bool:
        if (p.x - self.start.x) * (self.end.y - self.start.y) != (p.y - self.start.y) * (self.end.x - self.start.x):
            return False
        return min(self.start.x, self.end.x) <= p.x <= max(self.start.x, self.end.x) and \
               min(self.start.y, self.end.y) <= p.y <= max(self.start.y, self.end.y)

    def intersects(self, other: 'Segment') -> bool:
        p1, p2 = self.start, self.end
        p3, p4 = other.start, other.end
        
        def ccw(a, b, c):
            return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

        d1, d2, d3, d4 = ccw(p3, p4, p1), ccw(p3, p4, p2), ccw(p1, p2, p3), ccw(p1, p2, p4)
        
        if (d1 * d2 < 0) and (d3 * d4 < 0):
            return True
            
        def on_seg(p, a, b):
            return min(a.x, b.x) <= p.x <= max(a.x, b.x) and min(a.y, b.y) <= p.y <= max(a.y, b.y)
            
        if d1 == 0 and on_seg(p1, p3, p4): return True
        if d2 == 0 and on_seg(p2, p3, p4): return True
        if d3 == 0 and on_seg(p3, p1, p2): return True
        if d4 == 0 and on_seg(p4, p1, p2): return True
        
        return False

# Track definitions -----------------------------------------------------------------------------------------------------------------------------

class LTrack:
    def __init__(self):
        self.starting_line = Segment(Point(0, 0), Point(6, 0))
        self.finish_line = Segment(Point(15, 32), Point(15, 38))
        self.min_x, self.max_x = -8, 17
        self.min_y, self.max_y = -1, 40

    def get_random_start(self) -> Point:
        return Point(random.randint(0, 6), 0)

    def is_inside_track(self, point: Point) -> bool:
        x, y = point.x, point.y
        
        if y >= 0 and y <= 10:
            return x >= 0 and x <= 6
        if y > 10 and y <= 15:
            return x >= -2 and x <= 6
        if y > 15 and y <= 20:
            return x >= -4 and x <= 6
        if y > 20 and y <= 28:
            return x >= -6 and x <= 6
        if y > 28 and y <= 31:
            return x >= -6 and x <= 10
        if y > 31 and y <= 38:
            return x >= -6 and x <= 15
        return False


class SmallLTrack:
    def __init__(self):
        self.starting_line = Segment(Point(1, 0), Point(4, 0))
        self.finish_line = Segment(Point(11, 6), Point(11, 9))
        self.min_x, self.max_x = 0, 12
        self.min_y, self.max_y = 0, 10

    def get_random_start(self) -> Point:
        return Point(random.randint(1, 4), 0)

    def is_inside_track(self, point: Point) -> bool:
        x, y = point.x, point.y
        
        if y >= 0 and y < 6:
            return x >= 1 and x <= 4
        if y >= 6 and y <= 9:
            return x >= 1 and x <= 11
        return False
    

def graph_track(track, traj: List[Point] = None, title="Racetrack"):
    width = track.max_x - track.min_x
    height = track.max_y - track.min_y
    
    fig_w = max(6, width / 2)
    fig_h = max(6, height / 2)
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    
    # Draw all track cells
    for y in range(track.min_y, track.max_y):
        for x in range(track.min_x, track.max_x):
            point = Point(x, y)
            if track.is_inside_track(point):
                color = 'lightgray'
                if track.starting_line.contains(point):
                    color = 'red'
                elif track.finish_line.contains(point):
                    color = 'green'
                
                rect = patches.Rectangle((x, y), 1, 1, linewidth=0.5, edgecolor='black', facecolor=color)
                ax.add_patch(rect)
    
    # Draw trajectory if provided
    if traj and len(traj) > 1:
        # Extract x and y coordinates, offset by 0.5 to center in cells
        xs = [p.x + 0.5 for p in traj]
        ys = [p.y + 0.5 for p in traj]
        ax.plot(xs, ys, 'b-', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlim(track.min_x, track.max_x)
    ax.set_ylim(track.min_y, track.max_y)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid(True, linewidth=0.3)
    
    plt.show()

# RL environment -----------------------------------------------------------------------------------------------------------------------------

@dataclass
class State:
    pos: Point
    vel: Point

class Env:
    def __init__(self, track: LTrack):
        self.track = track
        self.state = State(track.get_random_start(), Point(0, 0))

    def reset(self):
        self.state = State(self.track.get_random_start(), Point(0, 0))

    @staticmethod
    def is_valid_action_for_state(s: State, a: Point) -> Tuple[bool, str]:
        if s.vel.x + a.x == 0 and s.vel.y + a.y == 0:
            return (False, "Invalid action: resulting velocity cannot be zero.")

        if s.vel.x + a.x < 0 or s.vel.x + a.x > 5:
            return (False, "Invalid action: resulting velocity x component out of bounds.")
        
        if s.vel.y + a.y < 0 or s.vel.y + a.y > 5:
            return (False, "Invalid action: resulting velocity y component out of bounds.")
        
        return (True, "")

    # Takes an action and returns a tuple containing: state, reward, done
    def step(self, action: Point, noise: bool = True) -> Tuple[State, int, bool]:
        valid, msg = Env.is_valid_action_for_state(self.state, action)
        if not valid:
            raise ValueError(msg)

        # Zero-out the action with probability 10% if current velocity is non-zero
        accel = Point(0, 0) if noise and random.random() < 0.1 and not (self.state.vel == Point(0, 0)) else action
        self.state.vel += accel

        assert(0 <= self.state.vel.x <= 5 and 0 <= self.state.vel.y <= 5)
        
        old_pos = self.state.pos
        self.state.pos += self.state.vel

        if Segment(old_pos, self.state.pos).intersects(self.track.finish_line):
            return self.state, -1, True
        
        # for each x value between the old x pos and the new x pos, check if we intersect the track boundary
        # by stepping along the line from old_pos to self.state.pos and using the is_inside_track method
        steps = max(abs(self.state.pos.x - old_pos.x), abs(self.state.pos.y - old_pos.y))
        for step in range(1, steps + 1):
            interp_x = old_pos.x + (self.state.pos.x - old_pos.x) * step / steps
            interp_y = old_pos.y + (self.state.pos.y - old_pos.y) * step / steps
            if not self.track.is_inside_track(Point(int(interp_x), int(interp_y))):
                self.reset()
                break
        return self.state, -1, False
    
# Monte Carlo on-policy control ----------------------------------------------------------------------------------------------------
        
track = SmallLTrack()
env = Env(track)

# percent of time we do a random action
epsilon = 0.1

Q: Dict[str, float] = {} # state-action is a string posx|posy|velx|vely|accelx|accely
C: Dict[str, int] = {} # stores counts of each state to recompute the average
A: Dict[str, List[Point]] = {}

for posx in range(track.min_x, track.max_x):
    for posy in range(track.min_y, track.max_y):
        for velx in range(0, 6):
            for vely in range(0, 6):
                for accelx in range(-1, 2):
                    for accely in range(-1, 2):

                        if not Env.is_valid_action_for_state(State(Point(posx, posy), Point(velx, vely)), Point(accelx, accely))[0]:
                            continue

                        state = f"{posx}|{posy}|{velx}|{vely}"
                        state_action = f"{posx}|{posy}|{velx}|{vely}|{accelx}|{accely}"

                        if state not in A:
                            A[state] = []

                        Q[state_action] = 0.0
                        A[state].append(Point(accelx, accely))
                        C[state_action] = 0

def get_policy(Q: Dict[str, float]) -> Dict[str, Point]:
    policy = defaultdict(lambda: Point(0, 0)) 
    for posx in range(track.min_x, track.max_x):
        for posy in range(track.min_y, track.max_y):
            for velx in range(0, 6):
                for vely in range(0, 6):

                    maxval = -1e6-1
                    best_action = Point(1, 1)

                    for accelx in [1, 0, -1]:
                        for accely in [1, 0, -1]:

                            if not Env.is_valid_action_for_state(State(Point(posx, posy), Point(velx, vely)), Point(accelx, accely))[0]:
                                continue

                            key = f"{posx}|{posy}|{velx}|{vely}|{accelx}|{accely}"
                            if Q[key] > maxval:
                                maxval = Q[key]
                                best_action = Point(accelx, accely)
                    policy[f"{posx}|{posy}|{velx}|{vely}"] = best_action
    return policy

policy = get_policy(Q)

for i in range(50000):
    if i % 100 == 0:
        print(f"Episode {i}")
    env.reset()
    states: List[State] = []
    actions = []
    rewards = []
    done = False
    state = env.state
    assert(state.vel == Point(0, 0) and env.track.starting_line.contains(state.pos)), "Episode did not start in a valid starting state."
    while not done:
        assert(env.track.is_inside_track(state.pos)), "State is outside track!"
        states.append(copy.deepcopy(state))

        # epsilon-greedy action selection
        argmax_action = policy[f"{env.state.pos.x}|{env.state.pos.y}|{env.state.vel.x}|{env.state.vel.y}"]
        action = argmax_action
        if random.random() < epsilon:
            action = random.choice(A[f"{env.state.pos.x}|{env.state.pos.y}|{env.state.vel.x}|{env.state.vel.y}"])

        state, reward, done = env.step(action)

        actions.append(action)
        rewards.append(reward)

    # Run through the epsiode and update Q
    # for simplicity, i'm assuming the return at index t is -1 * (len(rewards) - t)
    visited = set()
    for t, (state, action, reward) in enumerate(zip(states, actions, rewards)):
        assert(isinstance(action, Point))
        assert(isinstance(state, State))
        
        G = -1 * (len(rewards) - t)
        # update the Q function based on the reward
        state_action = f"{state.pos.x}|{state.pos.y}|{state.vel.x}|{state.vel.y}|{action.x}|{action.y}"
        if state_action in visited:
            continue

        Q[state_action] = (Q[state_action] * C[state_action] + G) / (C[state_action] + 1)
        C[state_action] += 1

        visited.add(state_action)

    # Update the policy based on q
    policy = get_policy(Q)

    # every so often, generate a sample trajectory following the optimal policy and graph it
    if i % 500 == 499:
        print(f"  Episode finished in {len(rewards)} steps.")
        env.reset()
        done = False
        traj = [env.state.pos]
        while not done:
            optimal_action = policy[f"{env.state.pos.x}|{env.state.pos.y}|{env.state.vel.x}|{env.state.vel.y}"]
            state, _, done = env.step(optimal_action, noise=False)
            traj.append(state.pos)

        graph_track(track, traj[-50:], title=f"Episode {i+1}")
