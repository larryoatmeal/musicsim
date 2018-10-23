from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.graphics.vertex_instructions import Rectangle
from kivy.properties import ListProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse

import numpy as np
import simulation
import threading
import time


# class TextureDrawer(Widget):


class TextureWidget(Widget):
    # create a 64x64 texture, defaults to rgba / ubyte
    width = NumericProperty(100)
    height = NumericProperty(100)
    scale = NumericProperty(1)

    pos = ListProperty([0, 0])

    def __init__(self, **kwargs):
        print self.width
        print self.height
        print self.pos
        super(TextureWidget, self).__init__(**kwargs)

        self.texture = Texture.create(size=(self.width, self.height))

        with self.canvas:
            Rectangle(texture=self.texture, pos=self.pos, size=(self.width * self.scale, self.height * self.scale))

        self.buffer = np.ones([self.width, self.height, 3])
        # self.bufferView = self.buffer.view().reshape(-1).astype(np.float32)
        # # print buf.shape
        # # buf[2, :, :] = np.zeros([self.width, self.height])
        #
        # buf[2, :, :] = np.ones([self.width, self.height]) * 1000
        #
        #
        # bufView = buf.view(np.float32).reshape(-1)

        # buf = np.array([int(x * 255 / size) for x in range(size)]).astype(np.uint8)

    # initialize the array with the buffer values
    # arr = array('B', buf)
    # now blit the array

    # # that's all ! you can use it in your graphics now :)
    # # if self is a widget, you can do this

    #     with self.canvas:
    #         # draw a line using the default color
    #
    #         # lets draw a semi-transparent red square
    #         Color(1, 0, 0, .5, mode='rgba')
    #         Rectangle(pos=self.pos, size=self.size)

    def update(self, buf):
        # buf might not be size of buffer (for example, could be NPOT
        # just replace the slice

        if len(buf.shape) == 3:  # multichannel
            (sW, sH, sC) = buf.shape
            self.buffer[0:sW, 0:sH, :sC] = buf
        else:  # single channel, set red
            (sW, sH) = buf.shape
            self.buffer[0:sW, 0:sH, 0] = buf

        self.texture.blit_buffer(self.buffer.reshape(-1).astype(np.float32), colorfmt='rgb', bufferfmt='float')
        self.canvas.ask_update()


class DotWidget(Widget):
    color = ListProperty([1, 0, 0])

    def __init__(self, **kwargs):
        super(DotWidget, self).__init__(**kwargs)
        # self.color = kwargs.color
        # print self.color

    def on_touch_down(self, touch):
        print(touch)
        with self.canvas:
            Color(self.color[0], self.color[1], self.color[2])
            d = 30.
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))


DRAW_BETA = 'DRAW_BETA'
DRAW_SIGMA = 'DRAW_SIGMA'
DRAW_PRESSURE = 'DRAW_PRESSURE'
DRAW_VX = 'DRAW_VX'
DRAW_VY = 'DRAW_VY'
DRAW_VBX = "DRAW_VBX"
DRAW_VBY = "DRAW_VBY"
DRAW_BETA_VX = "DRAW_BETA_VX"
DRAW_BETA_VY = "DRAW_BETA_VY"
DRAW_EXCITOR = "DRAW_EXCITOR"
DRAW_LISTENER = "DRAW_LISTENER"
DRAW_P_BORE_COORD = "DRAW_P_BORE_COORD"
DRAW_STRUCTURE = "DRAW_STRUCTURE"
DRAW_SIGMA_PRIME = "DRAW_SIGMA_PRIME"
DRAW_SIGMA_PRIME_VX = "DRAW_SIGMA_PRIME_VX"
DRAW_SIGMA_PRIME_VY = "DRAW_SIGMA_PRIME_VY"
DRAW_MODES = [
    (DRAW_BETA, "beta"),
    (DRAW_SIGMA, "sigma"),
    (DRAW_PRESSURE, "p"),
    (DRAW_VX, "vx"),
    (DRAW_VY, "vy"),
    (DRAW_VBX, "vbx"),
    (DRAW_VBY, "vby"),
    (DRAW_BETA_VX, "beta_vx"),
    (DRAW_BETA_VY, "beta_vy"),
    (DRAW_EXCITOR, "excitor"),
    (DRAW_LISTENER, "listener"),
    (DRAW_P_BORE_COORD, "p_bore_coord"),
    (DRAW_STRUCTURE, "structure"),
    (DRAW_SIGMA_PRIME, "sigma_prime"),
    (DRAW_SIGMA_PRIME_VX, "sigma_prime_vx"),
    (DRAW_SIGMA_PRIME_VY, "sigma_prime_vy"),

]


class KivyApp(App):
    stopSimulation = threading.Event()

    def __init__(self, **kwargs):
        self.started = False

        self.sim = self.init_sim()
        self.simulationTex = None

        self.draw_mode = DRAW_PRESSURE

        super(KivyApp, self).__init__(**kwargs)
        # self.mainTex = None

    @staticmethod
    def init_sim():
        w = 220
        h = 110

        wall = np.zeros([h, w])
        wall[50, 40:150] = 1
        wall[55, 40:150] = 1
        excitor = np.zeros([h, w])


        # excitor[60, 60] = 1

        excitor[51:55, 40] = 1
        p_bore_coord = (53, 41)
        listen_coord = (45, 155)
        sim = simulation.Simulation(w, h, wall, excitor, p_bore_coord, listen_coord, 20)
        return sim

    def build(self):
        layout = BoxLayout(padding=10)
        self.simulationTex = self.create_texture_widget(self.sim)

        self.simulationTex.size_hint = (None, 1.0)
        self.simulationTex.width = 1020
        layout.add_widget(self.simulationTex)
        layout.add_widget(self.configure_buttons())

        self.start_gui_poll()
        return layout

    def configure_buttons(self):
        buttonLayout = BoxLayout(orientation='vertical')

        btn1 = Button(text='Start')
        btn1.bind(on_press=lambda x: self.start_simulation())

        btn2 = Button(text='Stop')
        btn2.bind(on_press=lambda x: self.stop_simulation())

        buttonLayout.add_widget(btn1)
        buttonLayout.add_widget(btn2)

        dropdown = DropDown()
        for (mode, title) in DRAW_MODES:
            # When adding widgets, we need to specify the height manually
            # (disabling the size_hint_y) so the dropdown can calculate
            # the area it needs.
            btn = Button(text=title, size_hint_y=None, height=44)

            # for each button, attach a callback that will call the select() method
            # on the dropdown. We'll pass the text of the button as the data of the
            # selection

            def btn_release_listener(display_mode):
                def listener(b):
                    print "Set draw mode", display_mode
                    self.draw_mode = display_mode
                    dropdown.select(b.text)

                return listener

            btn.bind(on_release=btn_release_listener(mode))

            # then add the button inside the dropdown
            dropdown.add_widget(btn)

        dropdownButton = Button(text='Display')
        dropdownButton.bind(on_release=dropdown.open)
        dropdown.bind(on_select=lambda instance, x: setattr(dropdownButton, 'text', x))

        buttonLayout.add_widget(dropdownButton)
        return buttonLayout

    def start_gui_poll(self):

        @mainthread
        def my_callback(dt):
            # print "UI callback", dt
            self.update_texture()

        # call my_callback every 0.5 seconds
        Clock.schedule_interval(my_callback, 0.1)

    def create_texture_widget(self, sim):

        def round_up_power_2(x):
            return 1 << (x - 1).bit_length()

        w_pow2 = round_up_power_2(sim.width)
        h_pow2 = round_up_power_2(sim.height)

        dim = max(w_pow2, h_pow2)

        tex = TextureWidget(width=dim, height=dim, scale=5, pos=[0, 0])

        buf = np.zeros([sim.height, sim.width, 3])
        # buf[:, :, 0] = 0
        # buf[:sim.width / 2, :sim.height / 2, 1] = 1
        tex.update(buf)

        return tex

    def update_texture(self):
        def normalize_positive_max(m):
            return m / np.max(m)

        def fill_color(canvas, template, color):
            nonzero = np.nonzero(template)
            canvas[nonzero[0], nonzero[1], :] = color

        def fill_color_single(canvas, coord, color):
            canvas[coord[0], coord[1], :] = color

        if self.draw_mode == DRAW_BETA:
            self.simulationTex.update(self.sim.beta)
        elif self.draw_mode == DRAW_BETA_VY:
            self.simulationTex.update(self.sim.beta_vy)
        elif self.draw_mode == DRAW_BETA_VX:
            self.simulationTex.update(self.sim.beta_vx)
        elif self.draw_mode == DRAW_EXCITOR:
            self.simulationTex.update(self.sim.excitor_template)
        elif self.draw_mode == DRAW_LISTENER:
            empty = self.sim.empty()
            empty[self.sim.listen_coord[0], self.sim.listen_coord[1]] = 1
            self.simulationTex.update(empty)
        elif self.draw_mode == DRAW_P_BORE_COORD:
            empty = self.sim.empty()
            empty[self.sim.p_bore_coord[0], self.sim.p_bore_coord[1]] = 1
            self.simulationTex.update(empty)
        elif self.draw_mode == DRAW_SIGMA:
            self.simulationTex.update(normalize_positive_max(self.sim.sigma))
        elif self.draw_mode == DRAW_STRUCTURE:
            empty = self.sim.empty_color()
            # walls
            fill_color(empty, self.sim.wall_template, color=(0, 1, 0))
            fill_color(empty, self.sim.excitor_template, color=(0, 1, 1))
            fill_color_single(empty, self.sim.p_bore_coord, color=(0, 1, 1))
            fill_color_single(empty, self.sim.listen_coord, color=(0, 0, 1))
            self.simulationTex.update(empty)

        elif self.draw_mode == DRAW_PRESSURE:
            pressureScaled = self.sim.pressures[-1]
            print np.max(pressureScaled)
            self.simulationTex.update(pressureScaled)
        elif self.draw_mode == DRAW_VBX:

            self.simulationTex.update(self.sim.vbs[-1].x)
        elif self.draw_mode == DRAW_VBY:
            self.simulationTex.update(self.sim.vbs[-1].y)
        elif self.draw_mode == DRAW_VX:
            self.simulationTex.update(self.sim.velocities[-1].x)
        elif self.draw_mode == DRAW_VY:
            self.simulationTex.update(self.sim.velocities[-1].y)
        elif self.draw_mode == DRAW_SIGMA_PRIME:
            self.simulationTex.update(self.sim.sigma_prime_dt/simulation.DT)
        elif self.draw_mode == DRAW_SIGMA_PRIME_VX:
            self.simulationTex.update(self.sim.sigma_prime_dt_vx/simulation.DT)
        elif self.draw_mode == DRAW_SIGMA_PRIME_VY:
            self.simulationTex.update(self.sim.sigma_prime_dt_vy/simulation.DT)
        else:
            print "DRAW_MODE", self.draw_mode, " NOT SUPPORTED"

    def simulation_thread(self):
        iteration = 0
        while True:
            if self.stopSimulation.is_set():
                # Stop running this thread so the main Python process can exit.
                return
            iteration += 1

            if iteration % 1000 == 0:
                print('Simulation, iteration {}.'.format(iteration))

            self.sim.step()
            # time.sleep(0.01)

    def start_simulation(self):

        if not self.started:
            self.started = True
            self.stopSimulation.clear()
            threading.Thread(target=self.simulation_thread, args=()).start()

    def stop_simulation(self):

        if self.started:
            self.started = False
            self.stopSimulation.set()

    def on_stop(self):
        # The Kivy event loop is about to stop, set a stop signal;
        # otherwise the app window will close, but the Python process will
        # keep running until all secondary threads exit.
        self.stopSimulation.set()

    # # returns drawing function
    # def _setup_texture_and_return_handler(self, w, h, pos=(0, 0)):
    #
    #     # defer to build time
    #     tex = TextureWidget(width=w, height=h, pos=pos)
    #
    #     def draw_func(data):
    #         tex.update(data)
    #
    #     self.root.add_widget(tex)
    #     return draw_func


if __name__ == '__main__':
    KivyApp().run()
