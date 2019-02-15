from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.graphics.vertex_instructions import Rectangle
from kivy.properties import ListProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse

import numpy as np
import simulation
import threading
import time
import requests
import sounddevice as sd


# class TextureDrawer(Widget):
from audioWriter import write_audio
def round_up_power_2(x):
    return 1 << (x - 1).bit_length()

class TextureWidget(Widget):
    # create a 64x64 texture, defaults to rgba / ubyte
    width = NumericProperty(100)
    height = NumericProperty(100)
    scale = NumericProperty(1)

    pos = ListProperty([0, 0])

    def get_width(self):
        return round_up_power_2(self.width) * self.scale

    def get_height(self):
        return round_up_power_2(self.height) * self.scale

    def __init__(self, touch_listener, **kwargs):

        super(TextureWidget, self).__init__(**kwargs)

        self.orig_width =  self.width
        self.orig_height =  self.height
        # print se.lf.pos

        w_pow2 = round_up_power_2(self.width)
        h_pow2 = round_up_power_2(self.height)

        dim = max(w_pow2, h_pow2)
        print "dim", dim

        self.texture = Texture.create(size=(dim, dim))

        with self.canvas:
            Rectangle(texture=self.texture, pos=self.pos, size=(dim * self.scale, dim * self.scale))

        self.buffer = np.ones([dim, dim, 3])
        self.touch_listener = touch_listener

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

    def on_touch_down(self, touch):

        # print self.orig_width
        # print self.orig_height
        # print self.pos

        # print(touch.pos)
        # # print([self.width, self.height])
        # print([self.orig_width * self.scale, self.orig_height * self.scale])

        x, y = (int(touch.pos[0]/self.scale), int(touch.pos[1]/self.scale))

        if x < self.orig_width and y < self.orig_height:
            self.touch_listener(x, y)
        # print x, y


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
DRAW_EDIT = "DRAW_EDIT"
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
    (DRAW_EDIT, "edit"),

]

WRITE_LATCH = "WRITE_LATCH"


class KivyApp(App):
    stopSimulation = threading.Event()

    def __init__(self, **kwargs):
        self.started = False

        self.sim = self.init_sim()
        self.simulationTex = None

        self.draw_mode = DRAW_EDIT
        self.cursor = [0, 0]

        self.write_mode = WRITE_LATCH
        super(KivyApp, self).__init__(**kwargs)
        # self.mainTex = None

    @staticmethod
    def init_sim():
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

        return sim

    def build(self):
        layout = BoxLayout(padding=10)
        self.simulationTex = self.create_texture_widget(self.sim)

        self.simulationTex.size_hint = (None, 1.0)
        self.simulationTex.width = self.simulationTex.get_width()
        self.simulationTex.height = 300

        layout.add_widget(self.simulationTex)
        layout.add_widget(self.configure_buttons())
        self.pressure_canvas = self.sim.empty_color()

        self.start_gui_poll()

        Window.bind(on_key_down=self.key_action)
        return layout

    def write_cell(self):
        self.sim.wall_template[self.cursor[0], self.cursor[1]] = 1
        self.sim.update_aux_cells()
        pass

    def delete_cell(self):
        self.sim.wall_template[self.cursor[0], self.cursor[1]] = 0
        self.sim.update_aux_cells()
        pass

    def key_action(self, *args):
        print "got a key event: %s" % list(args)
        k = args[1]
        if k == 13: #enter
            self.write_cell()
        elif k == 8:
            self.delete_cell()
        elif k == 275: #right
            self.cursor[1] += 1
            if self.write_mode == WRITE_LATCH:
                self.write_cell()
        elif k == 276: #left
            self.cursor[1] -= 1
            if self.write_mode == WRITE_LATCH:
                self.write_cell()
        elif k == 273: #up
            self.cursor[0] += 1
            if self.write_mode == WRITE_LATCH:
                self.write_cell()
        elif k == 274: #down
            self.cursor[0] -= 1
            if self.write_mode == WRITE_LATCH:
                self.write_cell()


    def play_audio(self):
        normalized_audio = self.sim.audio / np.max(np.abs(self.sim.audio))
        # print normalized_audio
        write_audio("sound.wav", normalized_audio)


    def start_gpu_sim(self):
        resp = requests.get('http://35.226.198.237:8080/sim/50000')
        audio = np.array(resp.json())
        normalized = audio / np.max(np.abs(audio))

        sd.play(normalized, 128000)
        # print normalized

    def configure_buttons(self):
        buttonLayout = BoxLayout(orientation='vertical')

        btn0 = Button(text="Run GPU")
        btn0.bind(on_press=lambda  x: self.start_gpu_sim())

        btn1 = Button(text='Start')
        btn1.bind(on_press=lambda x: self.start_simulation())

        btn2 = Button(text='Stop')
        btn2.bind(on_press=lambda x: self.stop_simulation())

        btn3 = Button(text='Play')
        btn3.bind(on_press=lambda x: self.play_audio())

        buttonLayout.add_widget(btn0)
        buttonLayout.add_widget(btn1)
        buttonLayout.add_widget(btn2)
        buttonLayout.add_widget(btn3)

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


        editLayout = BoxLayout(orientation='horizontal')

        walls = ToggleButton(text='Walls', group='edit',)
        excitor = ToggleButton(text='Excitor', group='edit',)

        editLayout.add_widget(walls)
        editLayout.add_widget(excitor)

        buttonLayout.add_widget(editLayout)
        # excitor = ToggleButton(text='Female', group='sex', state='down')
        # btn3 = ToggleButton(text='Mixed', group='sex')


        return buttonLayout

    def start_gui_poll(self):

        @mainthread
        def my_callback(dt):
            # print "UI callback", dt
            self.update_texture()

        # call my_callback every 0.5 seconds
        Clock.schedule_interval(my_callback, 0.1)



    def on_touch(self, x, y):
        print x, y
        self.cursor = [y, x]

    def create_texture_widget(self, sim):

        tex = TextureWidget(self.on_touch, width=sim.width, height=sim.height, scale=5, pos=[0, 0])
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
            pressureScaled = self.sim.pressures[-1]/1000
            # print np.min(pressureScaled)
            # self.simulationTex.update((pressureScaled + 2000)/4000)
            self.pressure_canvas[:, :, 0] = pressureScaled
            self.pressure_canvas[:, :, 1] = -pressureScaled
            self.simulationTex.update(self.pressure_canvas)
        elif self.draw_mode == DRAW_VBX:
            self.simulationTex.update(self.sim.vbs[-1].x)
        elif self.draw_mode == DRAW_VBY:
            self.simulationTex.update(self.sim.vbs[-1].y)
            print self.sim.vbs[-1].y.max()
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

        elif self.draw_mode == DRAW_EDIT:
            pressureScaled = self.sim.pressures[-1]/1000
            # print np.min(pressureScaled)
            # self.simulationTex.update((pressureScaled + 2000)/4000)
            self.pressure_canvas[:, :, 0] = pressureScaled
            self.pressure_canvas[:, :, 1] = -pressureScaled

            fill_color(self.pressure_canvas, self.sim.wall_template, color=(1, 1, 1))
            fill_color(self.pressure_canvas, self.sim.excitor_template, color=(0, 1, 1))
            fill_color_single(self.pressure_canvas, self.sim.p_bore_coord, color=(0, 1, 1))
            fill_color_single(self.pressure_canvas, self.sim.listen_coord, color=(0, 0, 1))
            fill_color_single(self.pressure_canvas, self.cursor, color=(1, 1, 0))

            self.simulationTex.update(self.pressure_canvas)

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


if __name__ == '__main__':
    KivyApp().run()
