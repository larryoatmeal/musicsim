import kivy
import gui
import simulation
import numpy as np

print "HELLO WORLD"

# wall = np.zeros([220, 110])
# wall[110, 40:150] = 1
# wall[115, 40:150] = 1
# excitor = np.zeros([220, 110])
# excitor[111:115, 40] = 1
#
# sim = simulation.Simulation(None, 220, 110, wall, excitor, (113, 41), (170, 25), 6)
# for i in range(100):
#     sim.step()

app = gui.KivyApp()
# sim.listener = app.setup_texture_and_return_handler(220, 110)

app.run()

