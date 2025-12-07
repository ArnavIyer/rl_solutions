import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm

class Blackjack:
    def __init__(self):
        self.deck = [i for i in range(2, 11)] + [10, 10, 10, 'A']
        self.A = ['hit', 'stick']
        self.pi = {} # maps "dealershowing|playersum|usableace" to action. ex: self.pi["9|14|0"] = 'hit'
        self.q = {}  # maps "dealershowing|playersum|usableace|action" to value.
        self.N = {}

        for player_sum in range(12, 22):
            for dealer_showing in set(self.deck):
                for useable_ace in [0, 1]:
                    self.pi[f"{dealer_showing}|{player_sum}|{useable_ace}"] = 'hit'

                    for a in self.A:
                        self.q[f"{dealer_showing}|{player_sum}|{useable_ace}|{a}"] = 0
                        self.N[f"{dealer_showing}|{player_sum}|{useable_ace}|{a}"] = 0

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

    def gen_episode(self):
        state_actions = []

        # draw two cards for player
        card1 = self.draw_card()
        card2 = self.draw_card()
        first_action = 'hit' if random.randint(0, 1) == 0 else 'stick'

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

        # player always hits under 12, doesn't count as a state
        while player_sum < 12:
            player_sum, useable_ace = self.draw_and_update(player_sum, useable_ace)

        assert(player_sum >= 12)
        assert(player_sum <= 21)
        state_actions.append(f"{dealer_card}|{player_sum}|{useable_ace}|{first_action}")

        # player hits according to policy until he sticks or has total greater than 21
        # if first_action is hit, then this will always be true and the break statement will stop the while loop
        # if first_action is stick, then this loop will be skipped.
        while first_action == 'hit':
            player_sum, useable_ace = self.draw_and_update(player_sum, useable_ace)
            
            if player_sum <= 21:
                state_actions.append(f"{dealer_card}|{player_sum}|{useable_ace}|{self.pi[f"{dealer_card}|{player_sum}|{useable_ace}"]}")
        
            if not (player_sum <= 21 and self.pi[f"{dealer_card}|{player_sum}|{useable_ace}"] == 'hit'):
                break

        # player goes bust
        if player_sum > 21:
            return -1, state_actions
    
        # dealer plays according to his policy (hit below 17)
        while dealer_sum < 17:
            dealer_sum, dealer_useable_ace = self.draw_and_update(dealer_sum, dealer_useable_ace)

        if dealer_sum > 21:
            return 1, state_actions
        elif dealer_sum == player_sum:
            return 0, state_actions
        elif dealer_sum > player_sum:
            return -1, state_actions
        elif dealer_sum < player_sum:
            return 1, state_actions
        # We should never see this.
        assert(False) 

    def iterate(self, n):
        ctr = 1
        while ctr < n:
            ctr += 1
            r, state_actions = self.gen_episode()
            # because state_actions in the same episode are guaranteed to be distinct in blackjack,
            # for first visit we can just loop through the whole state_actions array
            for s in state_actions:
                self.q[s] = (self.N[s] * self.q[s] + r) / (self.N[s] + 1)
                self.N[s] += 1

                # update policy for best hit
                dealer_showing, player_sum, useable_ace, old_action = s.split('|')

                hit_value = self.q[f"{dealer_showing}|{player_sum}|{useable_ace}|hit"]
                stick_value = self.q[f"{dealer_showing}|{player_sum}|{useable_ace}|stick"]

                new_action = "hit" if hit_value > stick_value else "stick"

                self.pi[f"{dealer_showing}|{player_sum}|{useable_ace}"] = new_action

    
    # --- DISCLAIMER: PLOT CODE AI GENERATED ---
    def graph(self):
        player_range = np.arange(12, 22)
        # Dealer showing card: 1 (Ace) to 10.
        dealer_range = np.arange(1, 11) 
        
        # Grid for 3D plots (Q-values)
        X, Y = np.meshgrid(dealer_range, player_range)
        
        # Data for 3D Q-value plots
        Q_hit_ace    = np.zeros(X.shape)
        Q_stick_ace  = np.zeros(X.shape)
        Q_hit_no_ace = np.zeros(X.shape)
        Q_stick_no_ace = np.zeros(X.shape)

        # Data for 2D Policy plots (1='hit', 0='stick')
        P_ace    = np.zeros(X.shape)
        P_no_ace = np.zeros(X.shape)

        for i, player_sum in enumerate(player_range):
            for j, dealer_val in enumerate(dealer_range):
                # Map dealer_val to the key in self.q and self.pi
                if dealer_val == 1:
                    lookup_key = 'A'
                else:
                    lookup_key = str(dealer_val)
                
                # --- Fill Q-values ---
                # Usable Ace
                q_hit_a    = self.q.get(f"{lookup_key}|{player_sum}|1|hit", 0)
                q_stick_a  = self.q.get(f"{lookup_key}|{player_sum}|1|stick", 0)
                Q_hit_ace[i, j]   = q_hit_a
                Q_stick_ace[i, j] = q_stick_a
                
                # No Usable Ace
                q_hit_na   = self.q.get(f"{lookup_key}|{player_sum}|0|hit", 0)
                q_stick_na = self.q.get(f"{lookup_key}|{player_sum}|0|stick", 0)
                Q_hit_no_ace[i, j]   = q_hit_na
                Q_stick_no_ace[i, j] = q_stick_na

                # --- Fill Policy values ---
                # Usable Ace: 1 for 'hit', 0 for 'stick'
                policy_a = self.pi.get(f"{lookup_key}|{player_sum}|1")
                P_ace[i, j] = 1 if policy_a == 'hit' else 0
                
                # No Usable Ace: 1 for 'hit', 0 for 'stick'
                policy_na = self.pi.get(f"{lookup_key}|{player_sum}|0")
                P_no_ace[i, j] = 1 if policy_na == 'hit' else 0

        
        # --- 1. Plot Q-Value Surfaces (Usable Ace) ---
        # fig1 = plt.figure(figsize=(10, 8))
        # ax1 = fig1.add_subplot(111, projection='3d')
        # ax1.plot_surface(X, Y, Q_hit_ace, cmap=cm.Blues, linewidth=0, antialiased=False, alpha=0.7, label='Q(s, hit)')
        # ax1.plot_surface(X, Y, Q_stick_ace, cmap=cm.Reds, linewidth=0, antialiased=False, alpha=0.7, label='Q(s, stick)')
        # ax1.set_title('Action-Value Function Q* (Usable Ace)')
        # ax1.set_xlabel('Dealer Showing (1=Ace)')
        # ax1.set_ylabel('Player Sum')
        # ax1.set_zlabel('Q-Value')
        # # Add a custom legend as plot_surface doesn't support the 'label' argument
        # blue_patch = plt.matplotlib.patches.Patch(color=cm.Blues(0.6), label='Q(s, hit)')
        # red_patch = plt.matplotlib.patches.Patch(color=cm.Reds(0.6), label='Q(s, stick)')
        # ax1.legend(handles=[blue_patch, red_patch])
        # fig1.tight_layout()

        # # --- 2. Plot Q-Value Surfaces (No Usable Ace) ---
        # fig2 = plt.figure(figsize=(10, 8))
        # ax2 = fig2.add_subplot(111, projection='3d')
        # ax2.plot_surface(X, Y, Q_hit_no_ace, cmap=cm.Blues, linewidth=0, antialiased=False, alpha=0.7, label='Q(s, hit)')
        # ax2.plot_surface(X, Y, Q_stick_no_ace, cmap=cm.Reds, linewidth=0, antialiased=False, alpha=0.7, label='Q(s, stick)')
        # ax2.set_title('Action-Value Function Q* (No Usable Ace)')
        # ax2.set_xlabel('Dealer Showing (1=Ace)')
        # ax2.set_ylabel('Player Sum')
        # ax2.set_zlabel('Q-Value')
        # blue_patch = plt.matplotlib.patches.Patch(color=cm.Blues(0.6), label='Q(s, hit)')
        # red_patch = plt.matplotlib.patches.Patch(color=cm.Reds(0.6), label='Q(s, stick)')
        # ax2.legend(handles=[blue_patch, red_patch])
        # fig2.tight_layout()

        # --- 3. Plot Optimal Policy (Usable Ace) ---
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        cax3 = ax3.imshow(P_ace, origin='lower', cmap='RdYlGn', 
                         extent=[0.5, 10.5, 11.5, 21.5], aspect='auto') # 0.5 to center ticks
        ax3.set_title('Optimal Policy $\pi*$ (Usable Ace)')
        ax3.set_xlabel('Dealer Showing (1=Ace, 10=Ten/J/Q/K)')
        ax3.set_ylabel('Player Sum')
        ax3.set_xticks(dealer_range)
        ax3.set_yticks(player_range)
        
        # Annotate with policy action
        for i in range(len(player_range)):
            for j in range(len(dealer_range)):
                action = 'H' if P_ace[i, j] == 1 else 'S'
                ax3.text(dealer_range[j], player_range[i], action, 
                         ha='center', va='center', color='black', fontsize=10)
        
        # Colorbar and Ticks
        cbar3 = fig3.colorbar(cax3, ticks=[0, 1])
        cbar3.ax.set_yticklabels(['Stick', 'Hit'])
        fig3.tight_layout()

        # --- 4. Plot Optimal Policy (No Usable Ace) ---
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        cax4 = ax4.imshow(P_no_ace, origin='lower', cmap='RdYlGn', 
                         extent=[0.5, 10.5, 11.5, 21.5], aspect='auto')
        ax4.set_title('Optimal Policy $\pi*$ (No Usable Ace)')
        ax4.set_xlabel('Dealer Showing (1=Ace, 10=Ten/J/Q/K)')
        ax4.set_ylabel('Player Sum')
        ax4.set_xticks(dealer_range)
        ax4.set_yticks(player_range)

        # Annotate with policy action
        for i in range(len(player_range)):
            for j in range(len(dealer_range)):
                action = 'H' if P_no_ace[i, j] == 1 else 'S'
                ax4.text(dealer_range[j], player_range[i], action, 
                         ha='center', va='center', color='black', fontsize=10)

        # Colorbar and Ticks
        cbar4 = fig4.colorbar(cax4, ticks=[0, 1])
        cbar4.ax.set_yticklabels(['Stick', 'Hit'])
        fig4.tight_layout()

        plt.show()

game = Blackjack()
game.iterate(1000000)
game.graph()

