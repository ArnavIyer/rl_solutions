from typing import Literal, Tuple, Dict, Callable
from matplotlib import pyplot
import random
import copy
from functools import reduce
import math

State = Literal['A', 'B', 'C', 'D', 'E', 'terminal-left', 'terminal-right']
Action = Literal['left', 'right']

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

EpisodeGen = Callable[[Dict[State, float], float, Dict[State, float]], float]

def mc_epsiode(V: Dict[State, float], alpha: float, V_star: Dict[State, float]) -> float:
    done = False
    state = env.reset()
    states = []
    G = 0
    while not done:
        states.append(state)
        action: Action = random.choice(['left', 'right'])
        state, reward, done = env.step(action)
        G += reward
    
    for state in states:
        V[state] = V[state] + alpha * (G - V[state])
        
    return calc_rms_err(V, V_star)
        
    
def td_episode(V: Dict[State, float], alpha: float, V_star: Dict[State, float]) -> float:
    done = False
    state = env.reset()
    while not done:
        action: Action = random.choice(['left', 'right'])
        old_state = state
        state, reward, done = env.step(action)
        V[old_state] = V[old_state] + alpha*(reward + V[state] - V[old_state])

    return calc_rms_err(V, V_star)
    
    
env = Env()
alpha = 0.1
non_terminal_states = ['A', 'B', 'C', 'D', 'E']
V = {
    **{s: 0.5 for s in non_terminal_states},
    'terminal-left': 0,
    'terminal-right': 0,
}
V_star = {s: (i+1) / 6 for i, s in enumerate(non_terminal_states)}

# create a plot where the x axis is A, B, C, D, E, and the y axis is value
# plot one line (with visible dots) for V_ep0, V_ep1, V_ep10, V, and V_star.
V_ep0 = copy.deepcopy(V)
for episode in range(100):
    td_episode(V, alpha, V_star)
    if episode == 0:
        V_ep1 = copy.deepcopy(V)
    elif episode == 9:
        V_ep10 = copy.deepcopy(V)

pyplot.figure(figsize=(10, 6))
pyplot.plot(non_terminal_states, [V_ep0[s] for s in non_terminal_states], marker='o', label='Episode 0')
pyplot.plot(non_terminal_states, [V_ep1[s] for s in non_terminal_states], marker='o', label='Episode 1')
pyplot.plot(non_terminal_states, [V_ep10[s] for s in non_terminal_states], marker='o', label='Episode 10')
pyplot.plot(non_terminal_states, [V[s] for s in non_terminal_states], marker='o', label='Episode 100')
pyplot.plot(non_terminal_states, [V_star[s] for s in non_terminal_states], marker='o', label='True values', linestyle='--')

pyplot.xlabel('State')
pyplot.ylabel('Estimated Value')
pyplot.title('TD Learning Progress')
pyplot.legend()
pyplot.grid(True, alpha=0.3)
pyplot.show()

# create a plot where the x axis is number of episodes (0 to 100) and the y axis is rms error
# for different values of alpha
mc_alphas = [0.01, 0.02, 0.03, 0.04]
td_alphas = [0.15, 0.1, 0.05]

def collect_errors(alpha: float, episode_function: EpisodeGen):
    errors = [0 for _ in range(101)]
    for _ in range(100):
        V = {
            **{s: 0.5 for s in non_terminal_states},
            'terminal-left': 0,
            'terminal-right': 0,
        }
        errors[0] += calc_rms_err(V, V_star) / 100
        for ep in range(100):
            errors[ep+1] += episode_function(V, alpha, V_star) / 100
    return errors
            
episode_num = [i for i in range(101)]
mc_errors = {alpha: collect_errors(alpha, mc_epsiode) for alpha in mc_alphas}
td_errors = {alpha: collect_errors(alpha, td_episode) for alpha in td_alphas}

mc_linestyles = ['-', '--', '-.', ':']
td_linestyles = ['-', '--', '-.']
for i, alpha in enumerate(mc_alphas):
    pyplot.plot(episode_num, mc_errors[alpha], label=f'MC α={alpha}', color='red', linestyle=mc_linestyles[i])
for i, alpha in enumerate(td_alphas):
    pyplot.plot(episode_num, td_errors[alpha], label=f'TD α={alpha}', color='lightblue', linestyle=td_linestyles[i])
pyplot.xlabel('Number of Episodes')
pyplot.ylabel('RMS Error')
pyplot.title('RMS Error over Episodes for MC and TD')
pyplot.legend(title='Alpha values')
pyplot.grid(True, alpha=0.3)
pyplot.show()