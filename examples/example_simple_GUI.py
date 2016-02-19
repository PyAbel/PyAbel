#!/usr/bin/env python

# Illustrative GUI driving a small subset of PyAbel methods

# tkinter code adapted from
# http://matplotlib.org/examples/user_interfaces/embedding_in_tk.html

import numpy as np

import abel

from scipy.ndimage.interpolation import shift

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
from tkFileDialog import askopenfilename
import ttk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,\
                                              NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.pyplot import imread

Abel_methods = ['basex', 'direct', 'hansenlaw', #'onion-peeling'
                'three_point']

# GUI window -------------------

root = tk.Tk()
root.wm_title("Simple GUI PyAbel")

# matplotlib figure

f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(111)

# button call back functions

def _display():
    global IM, canvas, text

    # update information text box
    text.insert(tk.END,"raw image\n")

    # display image
    f.clf()
    a = f.add_subplot(111)
    a.imshow(IM, vmin=0)
    canvas.show()


def _getfilename():
    global IM, text
    fn = askopenfilename()

    # update what is occurring text box
    text.delete(1.0, tk.END)
    text.insert(tk.END, "reading image file {:s}\n".format(fn))
    canvas.show()

    # read image file
    if ".txt" in fn:
        IM = np.loadtxt(fn)
    else:
        IM = imread(fn)

    if IM.shape[0] % 2 == 0:
        text.insert(tk.END, "make image odd size")
        IM = shift(IM, (-0.5, -0.5))[:-1,:-1]

    # show the image
    _display()


def _center():
    global IM, text

    # update information text box
    text.delete(1.0, tk.END)
    text.insert(tk.END, "centering image using abel.tools.center.find_image_center_by_slice()\n")
    canvas.show()

    # center image via horizontal (left, right), and vertical (top, bottom)
    # intensity slices
    IM, offset = abel.tools.center.find_image_center_by_slice(IM)
    text.insert(tk.END, "center offset = {:}\n".format(offset))

    _display()


def _transform():
    global IM, AIM, canvas, transform, text

    method = transform.get()

    text.delete(1.0, tk.END)
    text.insert(tk.END,"inverse Abel transform: {:s}\n".format(method))
    if "basex" in method:
        text.insert(tk.END,"  first time calculation of the basis functions may take a while ...\n")
    if "onion" in method:
       text.insert(tk.END,"   onion_peeling method is in early testing and may not produce reliable results\n")
    if "direct" in method:
       text.insert(tk.END,"   calculation is slowed if Cython unavailable ...\n")
    canvas.show()

    # inverse Abel transform of whole image
    AIM = abel.transform(IM, method=method, direction="inverse",
                         vertical_symmetry=False, horizontal_symmetry=False)['transform']

    f.clf()
    a = f.add_subplot(111)
    a.imshow(AIM, vmin=0, vmax=AIM.max()/5.0)
    canvas.show()


def _speed():
    global IM, AIM, canvas, transform, text

    # inverse Abel transform
    _transform()

    # update text box in case something breaks
    text.insert(tk.END, "speed distribution\n")
    canvas.show()

    # speed distribution
    radial, speed  = abel.tools.vmi.angular_integration(AIM)

    f.clf()
    a = f.add_subplot(111)
    a.plot(radial, speed/speed[50:].max())
    a.axis(xmax=500, ymin=-0.05)
    canvas.show()

def _anisotropy():
    global IM, AIM, canvas, rmin, rmax, transform, text

    def P2(x):   # 2nd order Legendre polynomial
        return (3*x*x-1)/2


    def PAD(theta, beta, amp):
        return amp*(1 + beta*P2(np.cos(theta)))

    # radial range over which to follow the intensity variation with angle
    rmx = (int(rmin.get()), int(rmax.get()))

    text.delete(1.0, tk.END)
    text.insert(tk.END,"anisotropy parameter pixel range {:} to {:}\n".format(*rmx))
    canvas.show()

    # inverse Abel transform
    _transform()

    # intensity vs angle
    intensity, theta = abel.tools.vmi.calculate_angular_distributions(AIM,\
                                                  radial_ranges=[rmx,])

    # fit to P2(cos theta)
    beta, amp = abel.tools.vmi.anisotropy_parameter(theta, intensity[0])

    text.insert(tk.END,"beta = {:g}+-{:g}\n".format(*beta))

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

transform = ttk.Combobox(master=root, values=Abel_methods, state="readonly", width=10, height=len(Abel_methods))
transform.current(2)
transform.place(anchor=tk.W, relx=0.67, rely=0.14)

# speed
tk.Button(master=root, text='speed distribution', command=_speed)\
   .pack(anchor=tk.N)

# anisotropy
tk.Button(master=root, text='anisotropy parameter', command=_anisotropy)\
   .pack(anchor=tk.N)
rmin = tk.Entry(master=root, text='rmin')
rmin.place(anchor=tk.W, relx=0.66, rely=0.22, width=40)
rmin.insert(0, 368)
tk.Label(master=root, text="to").place(relx=0.74, rely=0.20)
rmax = tk.Entry(master=root, text='rmax')
rmax.place(anchor=tk.W, relx=0.78, rely=0.22, width=40)
rmax.insert(0, 389)

tk.Button(master=root, text='Quit', command=_quit).pack(anchor=tk.SW)

# a tk.DrawingArea ---------------
canvas = FigureCanvasTkAgg(f, master=root)
a.annotate("load image file", (0.5, 0.6), horizontalalignment="center")
a.annotate("e.g. data/O2-ANU1024.txt.bz2", (0.5, 0.5), horizontalalignment="center")
canvas.show()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2TkAgg(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# text info box
text = tk.Text(master=root, height=4, fg="blue")
text.pack(fill=tk.X)
text.insert(tk.END, "To start load an image data file using the `Load image file' button\n")


tk.mainloop()
