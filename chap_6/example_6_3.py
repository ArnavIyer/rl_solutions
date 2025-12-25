from typing import Literal, Tuple, Dict, Callable, List
from matplotlib import pyplot
import random
import copy
from functools import reduce
import math
from dataclasses import dataclass

State = Literal['A', 'B', 'C', 'D', 'E', 'terminal-left', 'terminal-right']
Action = Literal['left', 'right']
non_terminal_states = ['A', 'B', 'C', 'D', 'E']

class Env:
    def __init__(self):
        self.state: State = 'C'
        self.next_state: Dict[State, Dict[Action, State]] = {
            'A': {'left': 'terminal-left', 'right': 'B'},
            'B': {'left': 'A', 'right': 'C'},
            'C': {'left': 'B', 'right': 'D'},
            'D': {'left': 'C', 'right': 'E'},
            'E': {'left': 'D', 'right': 'terminal-right'},
        }

    def reset(self) -> State:
        self.state = 'C'
        return self.state

    def step(self, action: Action) -> Tuple[State, float, bool]:
        self.state = self.next_state[self.state][action]
        return self.state, 1 if self.state == 'terminal-right' else 0, self.state in ['terminal-left', 'terminal-right']
    

def calc_rms_err(V: Dict[State, float], V_star: Dict[State, float]) -> float:
    def rms_error(acc: float, value_tuple: Tuple[float, float]) -> float:
        return acc + (value_tuple[0] - value_tuple[1])**2

    return math.sqrt(reduce(rms_error, [(V[s], V_star[s]) for s in V_star], 0.0) / len(V_star))

@dataclass
class Episode:
    states: List[State]
    rewards: List[int]

def gen_episode() -> Episode:
    done = False
    state = env.reset()
    states = []
    rewards = []
    while not done:
        states.append(state)
        action: Action = random.choice(['left', 'right'])
        state, reward, done = env.step(action)
        rewards.append(reward)
    
    return Episode(states, rewards)


def mc_batch_update(V: Dict[State, float], alpha: float, V_star: Dict[State, float], episodes: List[Episode]) -> float:
    increments = {s: 0.0 for s in non_terminal_states}
    for episode in episodes:
        G = sum(episode.rewards)
        for t in range(len(episode.states)):
            increments[episode.states[t]] += alpha * (G - V[episode.states[t]])
        
    for state in non_terminal_states:
        V[state] += increments[state]
    
    return calc_rms_err(V, V_star)
        
def td_batch_update(V: Dict[State, float], alpha: float, V_star: Dict[State, float], episodes: List[Episode]) -> float:
    increments = {s: 0.0 for s in non_terminal_states}
    for episode in episodes:
        for t in range(len(episode.rewards)):
            V_t_p_1 = 0 if t == len(episode.rewards) - 1 else V[episode.states[t + 1]]
            increments[episode.states[t]] += alpha * (episode.rewards[t] + V_t_p_1 - V[episode.states[t]])

    for state in non_terminal_states:
        V[state] += increments[state]
    
    return calc_rms_err(V, V_star)
    
    
env = Env()
alpha = 0.001
V_star = {s: (i+1) / 6 for i, s in enumerate(non_terminal_states)}
error_diff_threshold = 0.001

x = [i for i in range(1, 101)]
mc_errors = [0 for _ in range(100)]
td_errors = [0 for _ in range(100)]

for i in range(100):
    episodes = [gen_episode() for _ in range(100)]
    print(f"attempt {i+1}/100")

    for ep in range(100):
        V = {
            **{s: 0.5 for s in non_terminal_states},
            'terminal-left': 0,
            'terminal-right': 0,
        }

        episode_batch = episodes[:ep+1]

        error_diff = math.inf
        error = math.inf
        while error_diff > error_diff_threshold:
            old_error = error
            error = mc_batch_update(V, alpha, V_star, episode_batch)
            error_diff = abs(old_error - error)
        mc_errors[ep] += error / 100

        V = {
            **{s: 0.5 for s in non_terminal_states},
            'terminal-left': 0,
            'terminal-right': 0,
        }
        error_diff = math.inf
        error = math.inf
        while error_diff > error_diff_threshold:
            old_error = error
            error = td_batch_update(V, alpha, V_star, episode_batch)
            error_diff = abs(old_error - error)
        td_errors[ep] += error / 100

pyplot.plot(x, mc_errors, label='MC')
pyplot.plot(x, td_errors, label='TD')
pyplot.show()