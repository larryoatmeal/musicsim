from collections import deque


class LookbackQueue:

    def __init__(self, N):
        self.q = deque()
        self.N = N

    def add(self, x):
        if len(self.q) == self.N:
            self.q.popleft()

        self.q.append(x)

    def __getitem__(self, key):
        return self.q[key]


    def test(self):
        self.add(-1)
        for i in range(0, 100):
            if i != self[-1] + 1:
                print "ERROR"
                break
            self.add(i)
        print "OKAY"