# Illustrative GUI driving a small subset of PyAbel methods

# tkinter code adapted from
# http://matplotlib.org/examples/user_interfaces/embedding_in_tk.html

import numpy as np

import abel

from scipy.ndimage.interpolation import shift

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
    from tkFileDialog import askopenfilename
else:
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
import tkinter.ttk as ttk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,\
                                              NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.pyplot import imread

Abel_methods = ['basex', 'direct', 'hansenlaw', 'linbasex', 'onion_peeling', 
                'onion_bordas', 'two_point', 'three_point']

center_methods = ['center-of-mass', 'convolution', 'gaussian', 'slice']

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
    global cent, IM, text

    center = cent.get()

    # update information text box
    text.delete(1.0, tk.END)
    text.insert(tk.END, "centering image using abel.tools.center_image(center={})\n".format(center))
    canvas.show()

    # center image via horizontal (left, right), and vertical (top, bottom)
    # intensity slices
    IM = abel.tools.center.center_image(IM, center=center, odd_size=True)
    #text.insert(tk.END, "center offset = {:}\n".format(offset))

    _display()


def _transform():
    global IM, AIM, canvas, transform, text

    method = transform.get()

    text.delete(1.0, tk.END)
    text.insert(tk.END,"inverse Abel transform: {:s}\n".format(method))
    if "basex" in method:
        text.insert(tk.END,"  first time calculation of the basis functions may take a while ...\n")
    if "direct" in method:
       text.insert(tk.END,"   calculation is slowed if Cython unavailable ...\n")
    canvas.show()

    # inverse Abel transform of whole image
    if method == 'linbasex':
        AIM = abel.Transform(IM, method=method, direction="inverse",
                             symmetry_axis=None,
                             transform_options=dict(return_Beta=True))
    else:
        AIM = abel.Transform(IM, method=method, direction="inverse",
                             symmetry_axis=None)

    f.clf()
    a = f.add_subplot(111)
    a.imshow(AIM.transform, vmin=0, vmax=AIM.transform.max()/5.0)
    canvas.show()


def _speed():
    global IM, AIM, canvas, transform, text

    # inverse Abel transform
    _transform()

    # update text box in case something breaks
    text.insert(tk.END, "speed distribution\n")
    canvas.show()

    # speed distribution
    if transform.get() not in ['linbasex']:
        radial, speed  = abel.tools.vmi.angular_integration(AIM.transform)
    else:
        radial, speed = AIM.radial, AIM.Beta[0]

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

    # inverse Abel transform
    _transform()

    method = transform.get()
    if method != 'linbasex':
        # radial range over which to follow the intensity variation with angle
        rmx = (int(rmin.get()), int(rmax.get()))

    else:
        rmx = (0, AIM.radial[-1])


    text.delete(1.0, tk.END)
    text.insert(tk.END,"anisotropy parameter pixel range {:} to {:}\n".format(*rmx))
    canvas.show()

    f.clf()
    a = f.add_subplot(111)
    if method not in ['linbasex']:
        # intensity vs angle
        beta, amp, rad, intensity, theta =\
            abel.tools.vmi.radial_integration(AIM.transform, radial_ranges=[rmx,])

        beta = beta[0]
        amp = amp[0]
        text.insert(tk.END,"beta = {:g}+-{:g}\n".format(beta[0],beta[1]))

        a.plot(theta, intensity[0], 'r-')
        a.plot(theta, PAD(theta, beta[0], amp[0]), 'b-', lw=2)
        a.annotate("$\\beta({:d},{:d})={:.2g}\pm{:.2g}$".format(rmx[0], rmx[1],beta[0], beta[1]), (-np.pi/2,-2))
    else:
        beta = AIM.Beta[1]
        radial = AIM.radial
        a.plot(radial, beta)
        a.annotate("anisotropy parameter vs radial coordinate", (-np.pi/2, 2))
        
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

cent = ttk.Combobox(master=root, values=center_methods, state="readonly",
                    width=11, height=len(center_methods))
cent.current(3)
cent.place(anchor=tk.W, relx=0.65, rely=0.05)

# display raw input image
tk.Button(master=root, text='raw image', command=_display).pack(anchor=tk.W, padx=0.1)

# Abel transform
tk.Button(master=root, text='inverse Abel transform', command=_transform)\
   .pack(anchor=tk.N)

transform = ttk.Combobox(master=root, values=Abel_methods, state="readonly", width=10, height=len(Abel_methods))
transform.current(2)
transform.place(anchor=tk.W, relx=0.67, rely=0.11)

# speed
tk.Button(master=root, text='speed distribution', command=_speed)\
   .pack(anchor=tk.N)

# anisotropy
tk.Button(master=root, text='anisotropy parameter', command=_anisotropy)\
   .pack(anchor=tk.N)
rmin = tk.Entry(master=root, text='rmin')
rmin.place(anchor=tk.W, relx=0.66, rely=0.16, width=40)
rmin.insert(0, 368)
tk.Label(master=root, text="to").place(relx=0.74, rely=0.14)
rmax = tk.Entry(master=root, text='rmax')
rmax.place(anchor=tk.W, relx=0.78, rely=0.16, width=40)
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
