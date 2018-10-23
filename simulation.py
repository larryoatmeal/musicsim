import numpy as np
from lookbackQueue import LookbackQueue
from collections import namedtuple

Velocity = namedtuple('Point', 'x y')

RHO = 1.1760
C = 3.4723e2
GAMMA = 1.4017
MU = 1.8460e-5
PRANDLT = 0.7073
DS = 3.83e-3

DT = DS/(np.sqrt(2) * C) * 0.999999 # make sure we're actually below the condition


DT_LISTEN = 1.0/44000
SAMPLE_EVERY_N = int(round(DT_LISTEN/DT))

REAL_LISTENING_FREQUENCY = 1/(SAMPLE_EVERY_N * DT)

print SAMPLE_EVERY_N

print DT_LISTEN/DT
print REAL_LISTENING_FREQUENCY

# DT = 7.81e-7

# DT = 7.81e-9

AN = 0.01
ADMITTANCE = 1 / (RHO * C * (1 + np.sqrt(1 - AN))/(1 - np.sqrt(1 - AN)))
# W = 220
# H = 110

AXIS_X = 'x'
AXIS_Y = 'y'

DIR_LEFT = 'left'
DIR_RIGHT = 'right'
DIR_UP = 'up'
DIR_DOWN = 'down'

D_DIR_FORWARD = 'forward'
D_DIR_BACKWARD = 'backward'

RHO_C_2_DT = RHO * C * C * DT

H = 0.015  # 15mm, bore diameter of clarinet

W_J = 1.2e-2
H_R = 6e-4
K_R = 8e6
DELTA_P_MAX = K_R * H_R

EXCITATION_IMPULSE = "IMPULSE"
EXCITATION_CLARINET = "CLARINET"
MAX_AUDIO_SIZE = int(3 * REAL_LISTENING_FREQUENCY)

# checked
def divergence(v):
    return gradient(v.x, AXIS_X, D_DIR_BACKWARD) + gradient(v.y, AXIS_Y, D_DIR_BACKWARD)


# checked
def gradient(m, axis, d_dir):
    if d_dir == D_DIR_FORWARD:
        if axis == AXIS_X:
            return (shift(m, DIR_LEFT) - m) / DS
        elif axis == AXIS_Y:
            return (shift(m, DIR_UP) - m) / DS
    else:
        if axis == AXIS_X:
            return (m - shift(m, DIR_RIGHT)) / DS
        elif axis == AXIS_Y:
            return (m - shift(m, DIR_DOWN)) / DS

    print "ERROR, INVALID ARGS"
    return m

# checked
def pressure_step(p, v, sigma_prime_dt):
    num = p - RHO_C_2_DT * divergence(v)
    den = sigma_prime_dt + 1
    return num / den


def vb_step(p, excitation_mode, p_mouth, p_bore, excitor_cells, num_excite_cells, wall_cells, iter_number):
    if excitation_mode == EXCITATION_CLARINET:
        # excitation
        delta_p = p_mouth - p_bore

        u_bore = 0
        if delta_p > 0:
            u_bore = W_J * H_R * (1 - delta_p / DELTA_P_MAX) * np.sqrt(2 * delta_p / RHO)
        mag_v_bore = u_bore / (H * DS * num_excite_cells)

        vb_x_excitor = excitor_cells * mag_v_bore

        # wall velocities
        vb_x_wall = np.zeros(wall_cells.shape)

        facing_up_toward_wall = wall_cells * shift(p, DIR_UP) * -1 * ADMITTANCE
        facing_down_toward_wall = shift(wall_cells, DIR_UP) * p * ADMITTANCE
        # facing_down_toward_wall = -1 * shift(facing_up_toward_wall, DIR_UP)

        vb_y_wall = facing_up_toward_wall + facing_down_toward_wall

        vb_x = vb_x_wall + vb_x_excitor
        vb_y = vb_y_wall

        vb = Velocity(x=vb_x, y=vb_y)
        return vb
    elif excitation_mode == EXCITATION_IMPULSE:
        # checked
        sine = excitor_cells * 0.001 * np.sin(iter_number * DT * 2 * 3.14 * 100)
        return Velocity(x=sine, y=sine)

# checked
def shift(m, direction):
    result = np.empty_like(m)

    if direction == DIR_DOWN:
        result[0, :] = m[0, :]
        result[1:, :] = m[0:-1, :]

    elif direction == DIR_UP:
        result[0:-1, :] = m[1:, :]
        result[-1, :] = m[-1, :]

    elif direction == DIR_RIGHT:
        result[:, 0] = m[:, 0]
        result[:, 1:] = m[:, 0:-1]

    elif direction == DIR_LEFT:
        result[:, 0:-1] = m[:, 1:]
        result[:, -1] = m[:, -1]

    else:
        print "ERROR, NOT VALID DIRECTION"

    return result


def compute_sigma_prime_dt(sigma, beta):
    return (sigma - beta + 1) * DT


def beta_fix(beta, direction):
    return np.minimum(beta, shift(beta, direction))


def velocity_step(p, v, vb, beta_vx, beta_vy, sigma_prime_dt_vx, sigma_prime_dt_vy):
    vNewX = velocity_step_dir(p, v.x, beta_vx, vb.x, sigma_prime_dt_vx, AXIS_X)
    vNewY = velocity_step_dir(p, v.y, beta_vy, vb.y, sigma_prime_dt_vy, AXIS_Y)
    return Velocity(x=vNewX, y=vNewY)


# beta, sigma_prime_dt need to be based on correctly modified beta for boundary. Should only be called by velocity_step
def velocity_step_dir(p, v, beta, vb, sigma_prime_dt, axis):
    num = beta * v - (beta * beta * DT * gradient(p, axis, D_DIR_FORWARD) / RHO) + sigma_prime_dt * vb
    den = beta + sigma_prime_dt
    return num / den


class Simulation:
    def __init__(self, width, height, wall_template, excitor_template, p_bore_coord, listen_coord,
                 pml_layers):
        self.width = width
        self.height = height
        # Caching these for performance
        self.beta = generate_beta(wall_template, excitor_template)
        self.sigma = generate_sigma(width, height, pml_layers)

        self.sigma_prime_dt = compute_sigma_prime_dt(self.sigma, self.beta)
        self.beta_vx = beta_fix(self.beta, DIR_LEFT)
        self.beta_vy = beta_fix(self.beta, DIR_UP)

        self.sigma_prime_dt_vx = compute_sigma_prime_dt(self.sigma, self.beta_vx)
        self.sigma_prime_dt_vy = compute_sigma_prime_dt(self.sigma, self.beta_vy)

        self.p_bore_coord = p_bore_coord
        self.listen_coord = listen_coord
        self.excitor_template = excitor_template
        self.num_excite_cells = np.count_nonzero(excitor_template)
        self.wall_template = wall_template

        # Queue for previous results
        self.pressures = LookbackQueue(2)
        self.velocities = LookbackQueue(2)
        self.vbs = LookbackQueue(2)

        self.p_mouth = 3000  # not sure what this should actually be

        ZERO_PRESSURE_TEMPLATE = self.empty()
        ZERO_VELOCITY_TEMPLATE = Velocity(x=self.empty(), y=self.empty())
        self.pressures.add(ZERO_PRESSURE_TEMPLATE)
        self.pressures.add(ZERO_PRESSURE_TEMPLATE)
        self.velocities.add(ZERO_VELOCITY_TEMPLATE)
        self.velocities.add(ZERO_VELOCITY_TEMPLATE)
        self.vbs.add(ZERO_VELOCITY_TEMPLATE)
        self.vbs.add(ZERO_VELOCITY_TEMPLATE)

        self.iter = 0

        self.excitation_mode = EXCITATION_CLARINET

        self.audio = np.zeros(MAX_AUDIO_SIZE)


    def empty(self):
        return np.zeros([self.height, self.width])

    def empty_color(self):
        return np.zeros([self.height, self.width, 3])

    def start(self):
        pass

    def stop(self):
        pass

    def step(self):
        newP = pressure_step(self.pressures[-1], self.velocities[-1], self.sigma_prime_dt)

        p_bore = newP[self.p_bore_coord[0], self.p_bore_coord[1]] # should be new pressure I think
        newVb = vb_step(newP, self.excitation_mode, self.p_mouth, p_bore, self.excitor_template, self.num_excite_cells,
                        self.wall_template, self.iter)

        newV = velocity_step(newP, self.velocities[-1], newVb, self.beta_vx, self.beta_vy,
                             self.sigma_prime_dt_vx, self.sigma_prime_dt_vy)

        self.pressures.add(newP)
        self.velocities.add(newV)
        self.vbs.add(newVb)
        self.iter = self.iter + 1

        if self.iter % SAMPLE_EVERY_N == 0:
            i = self.iter / SAMPLE_EVERY_N
            if i < len(self.audio):
                self.audio[i] = newP[self.listen_coord]


def generate_beta(wall_template, excitor_template):
    # excitor template and wall template should not overlap
    # beta=0 when there exists a wall/excitor
    return 1 - (wall_template + excitor_template)


def generate_sigma(w, h, pml_layers):
    sigma = np.zeros([h, w])

    pmlValues = np.linspace(0.5 / DT, 0, pml_layers + 1)
    # slow but doesn't matter, we compute this once
    for i in range(0, len(pmlValues)):
        for x in range(i, w - i):
            for y in range(i, h - i):
                sigma[y, x] = pmlValues[i]

    return sigma
