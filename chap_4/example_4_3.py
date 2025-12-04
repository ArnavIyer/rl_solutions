"""
A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips. 
If the coin comes up heads, he wins as many dollars as he has staked on that flip; if it is tails, he loses his stake. 
The game ends when the gambler wins by reaching his goal of $100, or loses by running out of money. 
On each flip, the gambler must decide what portion of his capital to stake, in integer numbers of dollars. 
This problem can be formulated as an undiscounted, episodic, finite MDP.

State: gambler's capital {1, 2, ..., 99)
Actions: a 2 {0, 1, ..., min(s, 100 - s)}
Reward: +1 when state 100 reached, 0 otherwise

p_h is the probability of the coin coming up heads
"""

import matplotlib.pyplot as plt

class GamblerProblem:
    def __init__(self, p_h):
        self.p_h = p_h
        self.v = [0 for _ in range(101)]
        self.theta = 0.000005
    
    def iterate(self):
        delta = 100
        iter = 0
        while delta > self.theta or iter < 100:
            delta = 0
            iter += 1
            for s in range(1, 100):
                old_v = self.v[s]

                # loop over each action
                self.v[s] = 0
                for a in range(0, min(s, 100-s) + 1):
                    # for a given s and a, there are only 2 possible s_p, each associated with a reward
                    assert(s + a <= 100)
                    assert(s - a >= 0)
                    v = self.p_h * (1 if s + a == 100 else 0 + self.v[s + a]) + (1 - self.p_h) * (self.v[s - a])

                    self.v[s] = max(self.v[s], v)
                
                delta = max(delta, abs(old_v - self.v[s]))
            print(f"{iter} max_delta: {delta}")

    def print_policy(self):
        # calculate policy
        pi = [0 for _ in range(0, 100)]

        for s in range(1, 100):
            # find the action that maximizes value. we want to skip action=0, since that's a no-op
            max_v = 0
            for a in range(1, min(s, 100-s) + 1):
                assert(s + a <= 100)
                assert(s - a >= 0)
                v = self.p_h * (1 if s + a == 100 else 0 + self.v[s + a]) + (1 - self.p_h) * (self.v[s - a])
                print(f"s: {s} a:{a} v:{v}")
                if max_v < v:
                    max_v = v
                    print(f"updating policy | s: {s} a:{a} v:{v}")
                    pi[s] = a
                    
        # generate line graph of value estimate
        x = [i for i in range(1, 100)] # state/capital
        value = [self.v[i] for i in x] # value

        fig, ax = plt.subplots()
        ax.plot(x, value, label='Value')
        plt.show()

        # generate bar graph of policy estimate
        policy = [pi[i] for i in x] # policy
        print(pi)
        plt.bar(x, policy)
        plt.show()

gp = GamblerProblem(0.05)
gp.iterate()
gp.print_policy()