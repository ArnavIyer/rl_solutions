import numpy as np
import math

# Jack has two locations of a car rental dealership. They store a max of 20 cars.
# At dealership 1, poisson(lambda=3) people come to rent car and poisson(3) people return their car
# At dealership 2, poisson(4) people come to rent car and poisson(2) people return their car
# Jack gets 10 dollars for each car rented. 
# He can pay 2 dollars per car to move a max of 5 cars between locations before the day starts.
# Extra modifications:
# - the first car moved from i -> j gets moved for free
# - it costs 4 dollars to store more than 10 cars in a location
class JackCarRental:
    def __init__(self, max_cars: int):
        self.pi = [[0 for _ in range(max_cars + 1)] for _ in range(max_cars + 1)]
        self.v = [[0.0 for _ in range(max_cars + 1)] for _ in range(max_cars + 1)]
        self.n = max_cars
        self.gamma = 0.9
        self.RETURN_I_LAMBDA = 3
        self.RETURN_J_LAMBDA = 2
        self.REQUEST_I_LAMBDA = 3
        self.REQUEST_J_LAMBDA = 4
        self.OVERFLOW_COST = 4
    
    def poisson(self, lamb: int, x: int):
        return ((lamb**x)/math.factorial(x))*(math.e**(-lamb))
    
    def calculate_expected_value(self, a, i, j):
        value = 0

        new_i = min(i + a, self.n)
        new_j = min(j - a, self.n)
        assert(new_i >= 0)
        assert(new_j >= 0)

        move_cost = abs(a) * -2

        # if a is less than 0, then it means that we are taking cars from i and delivering to j
        # we get one move like this for free, so we can add +2 to the move cost.
        if a < 0:
            move_cost += 2

        # calculate the probabilities of all of the combinations of requested cars
        for request_i in range(0, 10):
            for request_j in range(0, 10):

                # calculate reward
                rented_i = min(new_i, request_i)
                rented_j = min(new_j, request_j)
                r = move_cost + (rented_i + rented_j) * 10

                # for each combination of returned cars
                for returned_i in range(0, 10):
                    for returned_j in range(0, 10):
                        
                        # calculate probability of this 4-tuple happening
                        p = self.poisson(self.REQUEST_I_LAMBDA, request_i) 
                        p *= self.poisson(self.REQUEST_J_LAMBDA, request_j)
                        p *= self.poisson(self.RETURN_I_LAMBDA, returned_i)
                        p *= self.poisson(self.RETURN_J_LAMBDA, returned_j)

                        # calculate resulting state
                        sp_i = min(new_i - rented_i + returned_i, self.n)
                        sp_j = min(new_j - rented_j + returned_j, self.n)
                        assert(sp_i >= 0)
                        assert(sp_j >= 0)

                        overflow_cost = self.OVERFLOW_COST if sp_i > 10 else 0 + self.OVERFLOW_COST if sp_j > 10 else 0
                        value += p * (self.gamma*self.v[sp_i][sp_j] + r - overflow_cost)
        return value

    def iterate(self):
        delta = 2
        while delta > 1:
            delta = 0
            for i in range(self.n + 1):
                for j in range(self.n + 1):
                    old_v = self.v[i][j]
                    self.v[i][j] = self.calculate_expected_value(self.pi[i][j], i, j)
                    delta = max(delta, abs(old_v - self.v[i][j]))
            print(f"max delta: {delta}")

    def improve(self) -> bool:
        policy_stable = True
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                old_a = self.pi[i][j]
                
                # go through all the actions and figure out which one gets you to a state with the highest value
                argmax_a = -5
                max_return = -1000.0
                for a in range(-5, 6):
                    # Make sure that you don't update to an action that is impossible
                    if i + a < 0 or j - a < 0:
                        continue

                    v_s = self.calculate_expected_value(a, i, j)
                    if max_return < v_s:
                        max_return = v_s
                        argmax_a = a
                
                self.pi[i][j] = argmax_a
                
                if old_a != argmax_a:
                    policy_stable = False
        return policy_stable
    
    def main(self):
        done = False
        while not done:
            self.iterate()
            done = self.improve()
        print(np.array(self.pi))

a = JackCarRental(20)
a.main()

"""
Answer:
 [ 0  0  0  0  0  0  0  1  2  2  2  3  3  4  4  4  5  5  5  5  5]
 [-1  0  0  0  0  0  0  1  1  1  2  2  3  3  3  4  4  4  4  4  4]
 [-1 -1  0  0  0  0  0  0  0  1  1  2  2  2  3  3  3  3  3  3  3]
 [-1 -1 -1 -1  0  0  0  0  0  0  1  1  1  2  2  2  2  2  2  2  2]
 [-1 -1 -1 -1 -1  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  2]
 [-1 -1 -1 -1 -1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1]
 [-2 -1 -1 -1 -1 -1 -1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [-2 -2 -2 -1 -1 -1 -1 -1 -1  0  0  0  0  0  0  0  0  0  0  0  0]
 [-3 -3 -2 -2 -1 -1 -1 -1 -1 -1  0  0  0  0  0  0  0  0  0  0  0]
 [-4 -3 -3 -2 -1 -1 -1 -1 -1 -1 -1  0  0  0  0  0  0  0  0  0  0]
 [-4 -4 -3 -2 -2 -1 -1 -1 -1 -1 -1 -1 -1  0  0  0  0  0  0  0  0]
 [-5 -4 -3 -3 -2 -2 -1 -1 -1 -1 -1 -1 -1 -1  0  0  0  0  0  0  0]
 [-5 -4 -4 -3 -3 -2 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1  0  0  0  0  0]
 [-5 -5 -4 -4 -3 -3 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0  0  0  0]
 [-5 -5 -5 -4 -4 -3 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0  0  0]
 [-5 -5 -5 -5 -4 -3 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0  0]
 [-5 -5 -5 -5 -4 -3 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0  0]
 [-5 -5 -5 -5 -4 -3 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0]
 [-5 -5 -5 -5 -4 -3 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0]
 [-5 -5 -5 -5 -4 -3 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0]
 [-5 -5 -5 -5 -4 -3 -2 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0]
"""