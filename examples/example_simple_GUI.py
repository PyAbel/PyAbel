#!/usr/bin/env python

# Illustrative GUI driving a small subset of PyAbel methods

# code adapted from http://matplotlib.org/examples/user_interfaces/embedding_in_tk.html

import numpy as np

import matplotlib
matplotlib.use('TkAgg')

from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,\
                                              NavigationToolbar2TkAgg

from matplotlib.figure import Figure

from abel.hansenlaw import iabel_hansenlaw
from abel.tools.vmi import calculate_speeds, find_image_center_by_slice

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
from tkFileDialog import askopenfilename

# ------------

root = tk.Tk()
root.wm_title("Simple GUI PyAbel")

f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(111)

def _display():
    global IM, canvas
    f.clf()
    a = f.add_subplot(111)
    a.imshow(IM, vmin=0)
    canvas.show()

def _getfilename():
    global IM
    fn = askopenfilename()
    IM = np.loadtxt(fn)
    _display()

def _center():
    global IM
    IM, offset = find_image_center_by_slice(IM)
    _display()

def _transform():
    global IM, canvas
    AIM = iabel_hansenlaw(IM)
    f.clf()
    a = f.add_subplot(111)
    a.imshow(AIM, vmin=0, vmax=AIM.max()/2)
    canvas.show()

def _speed():
    global IM, canvas
    AIM = iabel_hansenlaw(IM)
    speed, radial = calculate_speeds(AIM)
    f.clf()
    a = f.add_subplot(111) 
    a.plot(radial, speed/speed[50:].max())
    a.axis(xmax=500,ymin=-0.05)
    canvas.show()


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


# buttons 
tk.Button(master=root, text='Load image file', command=_getfilename).pack(anchor=tk.W)
tk.Button(master=root, text='center image', command=_center).pack(anchor=tk.N)
tk.Button(master=root, text='raw image', command=_display).pack(anchor=tk.N)
tk.Button(master=root, text='inverse Abel tansform', command=_transform).pack(anchor=tk.N)
tk.Button(master=root, text='speed distribution', command=_speed).pack(anchor=tk.N)
tk.Button(master=root, text='Quit', command=_quit).pack(anchor=tk.SW)

# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
a.annotate("load image file", (0.5, 0.5), horizontalalignment="center")
canvas.show()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2TkAgg(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)


tk.mainloop()
