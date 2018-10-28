import numpy as np
import simulation
import cProfile

w = 220
h = 110

wall = np.zeros([h, w])
wall[50, 40:150] = 1
wall[55, 40:150] = 1

wall[55, 130] = 0
wall[55, 100] = 0

excitor = np.zeros([h, w])
excitor[51:55, 40] = 1
p_bore_coord = (53, 41)
listen_coord = (45, 155)
sim = simulation.Simulation(w, h, wall, excitor, p_bore_coord, listen_coord, 6)


N = 1000

def profile():
    for i in range(N):
        sim.step()

cProfile.run('profile()')