import numpy as np

w = 200
h = 100

RHO = 1.1760
C = 3.4723e2
DT = 7.81e-6
DS = 3.83e-3

P_UPDATE = -RHO * C * C * DT / DS
V_UPDATE = -DT / RHO / DS

SRC_HZ = 440

for A in [1, -1]:
    for B in [1, 0]:
        for C in [1, -1]:
            for D in [1, 0]:
                for E in [1, -1]:
                    for F in [1, 0]:
                        for G in [1, -1]:
                            for H in [1, 0]:
                                p = np.zeros([w, h])
                                vx = np.zeros([w, h])
                                vy = np.zeros([w, h])
                                # print p.shape

                                for i in range(100):
                                    p = p - P_UPDATE * (vx - np.roll(vx, A, B) + vy - np.roll(vy, C, D))
                                    vx = vx - V_UPDATE * (np.roll(p, E, F) - p)
                                    vy = vy - V_UPDATE * (np.roll(p, G, H) - p)
                                    # print p.shape
                                    p[50, 50] = p[50, 50] + 0.001 * np.sin(i * DT * 2 * 3.14 * SRC_HZ)

                                # print np.max(p)
                                if np.max(p) < 0.5:
                                    print np.max(p)
                                    print [A, B, C, D, E, F, G, H]


