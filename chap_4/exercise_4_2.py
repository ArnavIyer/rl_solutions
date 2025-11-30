dx = [1,0,-1,0]
dy = [0,1,0,-1]

class GridWorld:
    def __init__(self, n):
        self.v = [[0 for _ in range(n)] for _ in range(n)]
        self.v_15 = 0
        self.n = n

    def is_terminal(self, i: int, j: int) -> bool:
        return (i == 0 and j == 0) or (i == self.n-1 and j == self.n-1)

    def iterate(self):
        for i in range(4):
            for j in range(4):
                #  p(s', r|s, a) = 0 when s is the terminal state, so the value will always be 0
                if self.is_terminal(i, j):
                    continue
                
                # For the special case where going down on state 13 puts you in state 15:
                if i == 3 and j == 1:
                    self.v[3][1] = 0.25*(-1 + self.v[3][0]) + 0.25*(-1 + self.v[2][1]) + 0.25*(-1 + self.v[3][2]) + 0.25*(-1 + self.v_15)
                    continue

                v_s = 0
                for a in range(4):
                    ip = max(0, min(i + dx[a], 3))
                    jp = max(0, min(j + dy[a], 3))
                    v_s += 0.25*(-1 + self.v[ip][jp])

                self.v[i][j] = v_s
        
        # Calculate the value of state 15 or (3,1)
        self.v_15 = 0.25*(-1 + self.v[3][0]) + 0.25*(-1 + self.v[3][1]) + 0.25*(-1 + self.v[3][2]) + 0.25*(-1 + self.v_15)
    
    def main(self):
        for i in range(100):
            self.iterate()
 
    def print(self):
        for i in self.v:
            for x in i:
                print(f"{round(x, 1)}\t", end="")
            print("")
        print(f"\t{round(self.v_15, 1)}")

gw = GridWorld(4)
gw.main()
gw.print()