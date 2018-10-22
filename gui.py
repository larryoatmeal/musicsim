
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.graphics.vertex_instructions import Rectangle
from kivy.properties import ListProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse

import numpy as np
# class TextureDrawer(Widget):


class TextureWidget(Widget):
    # create a 64x64 texture, defaults to rgba / ubyte
    width = NumericProperty(100)
    height = NumericProperty(100)
    pos = ListProperty([0, 0])
    def __init__(self, **kwargs):
        super(TextureWidget, self).__init__(**kwargs)

        print self.width
        print self.height
        print self.pos
        self.texture = Texture.create(size=(self.width, self.height))


        with self.canvas:
            Rectangle(texture=self.texture, pos=self.pos, size=(self.width, self.height))


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
        bufView = buf.reshape(-1).astype(np.float32)
        self.texture.blit_buffer(bufView, colorfmt='rgb', bufferfmt='float')


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

    def build(self):
        layout = BoxLayout(padding=10)
        tex = TextureWidget(width=256, height=256, pos=[100, 256])
        tex2 = TextureWidget(width=256, height=256, pos=[800, 500])

        layout.add_widget(tex)
        layout.add_widget(tex2)

    # layout.add_widget(DotWidget(color=(0.73, 0.3, 0.1)))
        layout.add_widget(Label(text="YO"))
        layout.add_widget(Label(text="FUN"))

        btn1 = Button(text='Hello')
        btn2 = Button(text='World')

        layout.add_widget(btn1)
        layout.add_widget(btn2)

        buf = np.zeros([256, 256, 3])
        buf[:, :, 0] = 1
        buf[:124, :124, 1] = 1

        tex.update(buf)

        tex2.update(buf)
        return layout


if __name__ == '__main__':
    KivyApp().run()
