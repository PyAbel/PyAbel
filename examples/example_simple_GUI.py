#!/usr/bin/env python

# Illustrative GUI driving a small subset of PyAbel methods

# tkinter code adapted from
# http://matplotlib.org/examples/user_interfaces/embedding_in_tk.html

import numpy as np

from abel.hansenlaw import iabel_hansenlaw
from abel.tools.vmi import calculate_speeds
from abel.tools.vmi import find_image_center_by_slice
from abel.tools.vmi import calculate_angular_distributions
from abel.tools.vmi import anisotropy_parameter

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
from tkFileDialog import askopenfilename

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,\
                                              NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.pyplot import imread

# GUI window

root = tk.Tk()
root.wm_title("Simple GUI PyAbel")

# matplotlib figure

f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(111)

# button call back functions

def _display():
    global IM, canvas
    f.clf()
    a = f.add_subplot(111)
    a.imshow(IM, vmin=0)
    canvas.show()


def _getfilename():
    global IM
    print("reading image file")
    fn = askopenfilename()
    if ".txt" in fn:
        IM = np.loadtxt(fn)
    else:
        IM = imread(fn)
    _display()


def _center():
    global IM
    IM, offset = find_image_center_by_slice(IM)
    _display()


def _transform():
    global IM, canvas
    print("inverse Abel transform")
    AIM = iabel_hansenlaw(IM)
    f.clf()
    a = f.add_subplot(111)
    a.imshow(AIM, vmin=0, vmax=AIM.max()/5.0)
    canvas.show()


def _speed():
    global IM, canvas
    AIM = iabel_hansenlaw(IM)
    print("calculating speed distribution")
    speed, radial = calculate_speeds(AIM)
    f.clf()
    a = f.add_subplot(111)
    a.plot(radial, speed/speed[50:].max())
    a.axis(xmax=500, ymin=-0.05)
    canvas.show()

def _anisotropy():
    global IM, canvas, rmin, rmax

    def P2(x):   # 2nd order Legendre polynomial
        return (3*x*x-1)/2


    def PAD(theta, beta, amp):
        return amp*(1 + beta*P2(np.cos(theta)))

    rmx = (int(rmin.get()), int(rmax.get()))
    print("calculating anisotropy parameter pixel arange {:} to {:}".format(*rmx))
    AIM = iabel_hansenlaw(IM)
    intensity, theta = calculate_angular_distributions(AIM,\
                                radial_ranges=[rmx,])
    beta, amp = anisotropy_parameter(theta, intensity[0])
    print("beta = {:g}+-{:g}".format(*beta))
    f.clf()
    a = f.add_subplot(111)
    a.plot(theta, intensity[0], 'r-')
    a.plot(theta, PAD(theta, beta[0], amp[0]), 'b-', lw=2)
    a.annotate("$\\beta({:d},{:d})={:.2g}\pm{:.2g}$".format(*rmx+beta), (-np.pi/2,-2))
    canvas.show()

def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


# buttons with callbacks ----------------
# file input
tk.Button(master=root, text='Load image file', command=_getfilename)\
   .pack(anchor=tk.W)

# image centering
tk.Button(master=root, text='center image', command=_center).pack(anchor=tk.N)

# display raw input image
tk.Button(master=root, text='raw image', command=_display).pack(anchor=tk.N)

# Abel transform
tk.Button(master=root, text='inverse Abel transform', command=_transform)\
   .pack(anchor=tk.N)

# speed
tk.Button(master=root, text='speed distribution', command=_speed)\
   .pack(anchor=tk.N)

# anisotropy
tk.Button(master=root, text='anisotropy parameter', command=_anisotropy)\
   .pack(anchor=tk.N)
rmin = tk.Entry(master=root, text='rmin')
rmin.place(anchor=tk.W, relx=0.66, rely=0.24, width=40)
rmin.insert(0, 368)
tk.Label(master=root, text="to").place(relx=0.74, rely=0.22)
rmax = tk.Entry(master=root, text='rmax')
rmax.place(anchor=tk.W, relx=0.78, rely=0.24, width=40)
rmax.insert(0, 389)

tk.Button(master=root, text='Quit', command=_quit).pack(anchor=tk.SW)

# a tk.DrawingArea ---------------
canvas = FigureCanvasTkAgg(f, master=root)
a.annotate("load image file", (0.5, 0.5), horizontalalignment="center")
canvas.show()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2TkAgg(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)


tk.mainloop()
