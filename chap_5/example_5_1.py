import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm

class Blackjack:
    def __init__(self):
        self.deck = [i for i in range(2, 11)] + [10, 10, 10, 'A']
        self.A = ['hit', 'stick']
        self.pi = {} # maps "dealershowing|playersum|usableace" to action. ex: self.pi["9|14|0"] = 'hit'
        self.v = {}
        self.N = {}

        # For this example, we want to estimate the value function under the policy where the player sticks only on a sum of 20 or 21
        for player_sum in range(12, 22):
            a = 'hit' if player_sum < 20 else 'stick'
            self.pi[f"A|{player_sum}|0"] = a
            self.pi[f"A|{player_sum}|1"] = a
            self.v[f"A|{player_sum}|0"] = 0
            self.v[f"A|{player_sum}|1"] = 0
            self.N[f"A|{player_sum}|0"] = 0
            self.N[f"A|{player_sum}|1"] = 0
            for dealer_showing in range(2, 11):
                self.pi[f"{dealer_showing}|{player_sum}|0"] = a
                self.pi[f"{dealer_showing}|{player_sum}|1"] = a
                self.v[f"{dealer_showing}|{player_sum}|0"] = 0
                self.v[f"{dealer_showing}|{player_sum}|1"] = 0
                self.N[f"{dealer_showing}|{player_sum}|0"] = 0
                self.N[f"{dealer_showing}|{player_sum}|1"] = 0

    def draw_card(self):
        idx = random.randint(0, len(self.deck)-1)
        return self.deck[idx]
    
    def gen_episode(self):
        states = []

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

        if player_sum >= 12 and player_sum <= 21:
            states.append(f"{dealer_card}|{player_sum}|{useable_ace}")

        # player hits according to policy until he sticks or has total greater than 21
        while (player_sum < 12) or (player_sum <= 21 and self.pi[f"{dealer_card}|{player_sum}|{useable_ace}"] == 'hit'):
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
            
            if player_sum >= 12 and player_sum <= 21:
                states.append(f"{dealer_card}|{player_sum}|{useable_ace}")
        
        # player goes bust
        if player_sum > 21:
            return -1, states
    
        # dealer plays according to his policy (hit below 17)
        while dealer_sum < 17:
            card = self.draw_card()

            if card == 'A':
                if dealer_useable_ace == 1:
                    dealer_sum += 1
                else:
                    dealer_sum += 11
                    dealer_useable_ace = 1
            else:
                dealer_sum += card

            if dealer_sum > 21 and dealer_useable_ace == 1:
                dealer_sum -= 10
                dealer_useable_ace = 0

        if dealer_sum > 21:
            return 1, states
        elif dealer_sum == player_sum:
            return 0, states
        elif dealer_sum > player_sum:
            return -1, states
        elif dealer_sum < player_sum:
            return 1, states
        # We should never see this.
        assert(False) 

    def iterate(self, n):
        ctr = 1
        while ctr < n:
            ctr += 1
            r, states = self.gen_episode()
            # beacuse states in the same episode are guaranteed to be distinct in blackjack,
            # for first visit we can just loop through the whole states array
            for s in states:
                self.v[s] = (self.N[s] * self.v[s] + r) / (self.N[s] + 1)
                self.N[s] += 1
    
    # --- DISCLAIMER: PLOT CODE AI GENERATED ---
    def graph(self):
        player_range = np.arange(12, 22)
        dealer_range = np.arange(1, 11)
        
        X, Y = np.meshgrid(dealer_range, player_range)
        
        Z_no_ace = np.zeros(X.shape)
        Z_ace = np.zeros(X.shape)

        for i, player_sum in enumerate(player_range):
            for j, dealer_val in enumerate(dealer_range):
                if dealer_val == 1:
                    lookup_key = 'A'
                else:
                    lookup_key = str(dealer_val)
                
                Z_no_ace[i, j] = self.v.get(f"{lookup_key}|{player_sum}|0", 0)
                Z_ace[i, j]    = self.v.get(f"{lookup_key}|{player_sum}|1", 0)

        fig = plt.figure(figsize=(16, 8))

        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_ace, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax1.set_title('Usable Ace')
        ax1.set_xlabel('Dealer Showing (1=Ace)')
        ax1.set_ylabel('Player Sum')
        ax1.set_zlabel('Value')

        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_no_ace, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax2.set_title('No Usable Ace')
        ax2.set_xlabel('Dealer Showing (1=Ace)')
        ax2.set_ylabel('Player Sum')
        ax2.set_zlabel('Value')

        plt.show()

game = Blackjack()
game.iterate(10000)
game.graph()
game.iterate(500000)
game.graph()

