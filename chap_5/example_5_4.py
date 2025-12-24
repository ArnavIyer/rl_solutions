'''
The value of the following blackjack state is -0.27726 under the policy
of only sticking on a player sum of 20 or 21 (target policy): 
- dealer showing    2
- player sum        13
- useable ace       1

We want to graph the mean square error of the estimate (y-axis) provided
by off-policy estimation with weighted importance sampling vs ordinary
importance sampling for different numbers of episodes (x axis). Our behavior
policy will be the policy that randomly hits or sticks.

ordinary_errors = [] # 3d array. first index in is for i, and third for episode number.
weighted_errors = []

for i in range(100):
    Generate episode
    for each episode:
        for method in (ordinary importance sampling, weighted importance sampling):
            update value of state based on return
            calculate and save error

average over dimension 1 for both error arrays, and graph with x-axis log scale
'''

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm

class Blackjack:
    def __init__(self):
        self.deck = [i for i in range(2, 11)] + [10, 10, 10, 'A']
        self.A = ['hit', 'stick']
    
    def get_behavior_policy_action(self):
        return 'hit' if random.randint(0, 1) == 1 else 'stick'

    def pi_probability(self, a, player_sum):
        action_under_pi = 'stick' if player_sum == 20 or player_sum == 21 else 'hit'
        return 1.0 if a == action_under_pi else 0

    def draw_card(self):
        idx = random.randint(0, len(self.deck)-1)
        return self.deck[idx]
    
    def draw_and_update(self, player_sum, useable_ace):
        card = self.draw_card()

        if card == 'A':
            if useable_ace == 1:
                player_sum += 1
            else:
                player_sum += 11
                useable_ace = 1
        else:
            player_sum += card
        
        if player_sum > 21 and useable_ace == 1:
            player_sum -= 10
            useable_ace = 0

        return player_sum, useable_ace

    # should run an episode and return the return and rho (importance sampling ratio)
    def gen_episode(self):
        rho = 1.0

        # draw two cards for player
        card1 = self.draw_card()
        card2 = self.draw_card()

        if card1 == 'A' and card2 == 'A':
            useable_ace = 1
            player_sum = 12
        else:
            useable_ace = 1 if card1 == 'A' or card2 == 'A' else 0
            player_sum = (11 if card1 == 'A' else card1) + (11 if card2 == 'A' else card2)

        # draw one card for dealer
        dealer_card = self.draw_card()
        dealer_sum = 11 if dealer_card == 'A' else dealer_card
        dealer_useable_ace = 1 if dealer_card == 'A' else 0

        assert(player_sum <= 21)

        # player hits according to policy until he sticks or has total greater than 21
        while True:
            action = self.get_behavior_policy_action()

            if player_sum <= 21:
                # probability of target policy selecting action given state / probability of behavior policy selecting action given state
                rho *= self.pi_probability(action, player_sum) / 0.5
        
            if not (player_sum <= 21 and action == 'hit'):
                break

            player_sum, useable_ace = self.draw_and_update(player_sum, useable_ace)

        # player goes bust
        if player_sum > 21:
            return -1, rho
    
        # dealer plays according to his policy (hit below 17)
        while dealer_sum < 17:
            dealer_sum, dealer_useable_ace = self.draw_and_update(dealer_sum, dealer_useable_ace)

        if dealer_sum > 21:
            return 1, rho
        elif dealer_sum == player_sum:
            return 0, rho
        elif dealer_sum > player_sum:
            return -1, rho
        elif dealer_sum < player_sum:
            return 1, rho
        # We should never see this.
        assert(False) 

ordinary_errors = [] # 3d array. first index is for i, and second for episode number.
weighted_errors = []

VALUE = -0.27726
NUM_EPISODES = 10000
NUM_RUNS = 100

game = Blackjack()

for i in range(NUM_RUNS):
    print(f"Running experiment {i+1}/{NUM_RUNS}")
    numerator = 0.0
    weighted_denominator = 0.0
    ordinary_errors.append([None for i in range(NUM_EPISODES)])
    weighted_errors.append([None for i in range(NUM_EPISODES)])
    # generate 10k episodes
    for episode in range(NUM_EPISODES):
        G, rho = game.gen_episode()

        numerator += G * rho
        weighted_denominator += rho
        
        if rho == 0:
            continue
        
        ordinary_value = numerator / (episode + 1)
        weighted_value = numerator / weighted_denominator

        ordinary_errors[-1][episode] = (VALUE - ordinary_value)**2
        weighted_errors[-1][episode] = (VALUE - weighted_value)**2

    print(weighted_value)

ordinary_y = []
weighted_y = []
for i in range(NUM_EPISODES):
    ordinary_y.append(np.mean([ordinary_errors[j][i] for j in range(NUM_RUNS) if ordinary_errors[j][i] is not None]))
    weighted_y.append(np.mean([weighted_errors[j][i] for j in range(NUM_RUNS) if weighted_errors[j][i] is not None]))

assert(len(ordinary_y) == NUM_EPISODES)

x = np.arange(NUM_EPISODES)

plt.plot(x, ordinary_y, label='ordinary')
plt.plot(x, weighted_y, label='weighted')
plt.xscale('log')
plt.show()
