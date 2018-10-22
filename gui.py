from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics.vertex_instructions import Rectangle
from kivy.properties import ListProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
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


class KivyApp(App):
    stopSimulation = threading.Event()

    def __init__(self, **kwargs):
        self.started = False

        self.sim = self.init_sim()
        self.simulationTex = None
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
        excitor[51:55, 40] = 1
        p_bore_coord = (53, 41)
        listen_coord = (45, 155)
        sim = simulation.Simulation(None, w, h, wall, excitor, p_bore_coord, listen_coord, 6)
        return sim

    def build(self):
        layout = BoxLayout(padding=10)
        self.simulationTex = self.create_texture_widget(self.sim)

        self.simulationTex.size_hint = (None, 1.0)
        self.simulationTex.width = 1020
        layout.add_widget(self.simulationTex)

        btn1 = Button(text='Start')
        btn1.bind(on_press=lambda x: self.start_simulation())

        btn2 = Button(text='Stop')
        btn2.bind(on_press=lambda x: self.stop_simulation())

        layout.add_widget(btn1)
        layout.add_widget(btn2)

        # buf = np.zeros([256, 256, 3])
        # buf[:, :, 0] = 1
        # buf[:50, :50, 1] = 1
        #
        # tex.update(buf)

        def my_callback(dt):
            print "UI callback", dt
            self.update_texture()

        # call my_callback every 0.5 seconds
        Clock.schedule_interval(my_callback, 0.5)

        return layout

    def create_texture_widget(self, sim):

        def round_up_power_2(x):
            return 1 << (x - 1).bit_length()

        w_pow2 = round_up_power_2(sim.width)
        h_pow2 = round_up_power_2(sim.height)

        dim = max(w_pow2, h_pow2)

        tex = TextureWidget(width=dim, height=dim, scale=4, pos=[0, 0])

        buf = np.zeros([sim.height, sim.width, 3])
        # buf[:, :, 0] = 0
        # buf[:sim.width / 2, :sim.height / 2, 1] = 1
        tex.update(buf)

        return tex

    def update_texture(self):
        self.simulationTex.update(self.sim.beta)

    # def _hook_in_simulator(self):
    #     wall = np.zeros([220, 110])
    #     wall[110, 40:150] = 1
    #     wall[115, 40:150] = 1
    #     excitor = np.zeros([220, 110])
    #     excitor[111:115, 40] = 1
    #
    #     sim = simulation.Simulation(None, 220, 110, wall, excitor, (113, 41), (170, 25), 6)
    #
    #     sim.listener = self._setup_texture_and_return_handler(220, 110, (0, 0))
    #

    # def start_loop(self):
    #     def
    #     event =

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
            # time.sleep(1)

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
