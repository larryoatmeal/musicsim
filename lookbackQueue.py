from collections import deque


class LookbackQueue:

    def __init__(self, N):
        self.q = deque()
        self.N = N

    def add(self, x):
        if len(self.q) == self.N - 1:
            self.q.popleft()

        self.q.append(x)

    def __getitem__(self, key):
        return self.q[key]
