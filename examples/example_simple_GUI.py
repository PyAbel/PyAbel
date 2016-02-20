#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

# Illustrative GUI driving a small subset of PyAbel methods

import numpy as np
import abel

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
from matplotlib.pyplot import imread, colorbar

from scipy.ndimage.interpolation import shift

Abel_methods = ['basex', 'direct', 'hansenlaw', #'onion-peeling'
                'three_point']


class PyAbel:  #(tk.Tk):

    def __init__(self, parent):
        self.parent = parent
        self.initialize()

    def initialize(self):
        self.fn = None
        self.old_fn = None
        self.old_method = None
        self.AIM = None
        self.rmx = (368, 393)

        # matplotlib figure
        self.f = Figure(figsize=(5, 4))
        self.a = self.f.add_subplot(111)

        self.main_container = tk.Frame(self.parent)#, background="bisque")
        self.main_container.pack(side="top", fill="both", expand=True)

        self.top_frame = tk.Frame(self.main_container)#, background="green")
        self.middle_frame = tk.Frame(self.main_container)#, background="blue")
        self.bottom_frame = tk.Frame(self.main_container)#, background="yellow")

        self.top_frame.pack(side="top", fill="x", expand=False)
        self.middle_frame.pack(side="left", fill="both", expand=True)
        self.bottom_frame.pack(side="bottom", fill="x", expand=False)

        self._menus()
        self._button_area()
        self._plot_canvas()
        self._text_info_box()

    def _menus(self):
        # menus and buttons with callbacks ----------------
        self.menubar = tk.Menu(self.parent)
        self.transform_method = tk.IntVar()
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Load image file",
                                  command=self._getfilename)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self._quit)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        self.processmenu = tk.Menu(self.menubar, tearoff=0)
        self.processmenu.add_command(label="Center image", command=self._center)
        self.submenu=tk.Menu(self.processmenu)
        for method in Abel_methods:
            self.submenu.add_radiobutton(label=method,
                 var=self.transform_method, val=Abel_methods.index(method),
                 command=self._transform)
        self.processmenu.add_cascade(label="Inverse Abel transform",
                 menu=self.submenu, underline=0)

        self.processmenu.add_command(label="Speed distribution",
                                     command=self._speed)
        self.processmenu.add_command(label="Angular distribution",
                                     command=self._anisotropy)
        self.angmenu=tk.Menu(self.processmenu)
        self.menubar.add_cascade(label="Processing", menu=self.processmenu)
    
        self.viewmenu = tk.Menu(self.menubar, tearoff=0)
        self.viewmenu.add_command(label="Raw image", command=self._display)
        self.viewmenu.add_command(label="Inverse Abel transformed image",
                                  command=self._transform)
        self.menubar.add_cascade(label="View", menu=self.viewmenu)


    def _button_area(self):
        self.rawimg = tk.Button(master=self.top_frame, text="raw image",
                                command=self._display)
        self.rawimg.grid(row=0, column=0)

        self.center = tk.Button(master=self.top_frame, text="center image",
                                state=tk.DISABLED, command=self._center)
        self.center.grid(row=0, column=1)
        self.center_method = ttk.Combobox(master=self.top_frame,
                         values=["com", "slice", "gaussian", "image_center"],
                         state=tk.DISABLED, width=5, height=4)
        self.center_method.current(1)
        self.center_method.grid(row=1, column=1)

        self.recond = tk.Button(master=self.top_frame,
                                text="reconstructed image",
                                state=tk.DISABLED,
                                command=self._transform)
        self.recond.grid(row=0,column=2)

        self.transform = ttk.Combobox(master=self.top_frame,
                         values=Abel_methods,
                         state=tk.DISABLED, width=10, height=len(Abel_methods))
        self.transform.current(2)
        self.transform.grid(row=1, column=2)

        blank = tk.Label(master=self.top_frame, text="   ", width=4)
        blank.grid(row=0, column=3, columnspan=2)


        self.speed = tk.Button(master=self.top_frame, text="speed",
                               state=tk.DISABLED, command=self._speed)
        self.speed.grid(row=0, column=5)

        self.aniso = tk.Button(master=self.top_frame, text="anisotropy",
                               state=tk.DISABLED, command=self._anisotropy)
        self.aniso.grid(row=0, column=6)

        self.subframe = tk.Frame(self.top_frame)
        self.subframe.grid(row=1, column=6)
        self.rmin = tk.Entry(master=self.subframe, text='rmin', width=3,
                             state=tk.DISABLED) 
        self.rmin.grid(row=0, column=0)
        self.rmin.delete(0, tk.END)
        self.rmin.insert(0, self.rmx[0])
        self.lbl = tk.Label(master=self.subframe, text="to", state=tk.DISABLED)
        self.lbl.grid(row=0, column=1)
        self.rmax = tk.Entry(master=self.subframe, text='rmax', width=3,
                             state=tk.DISABLED)
        self.rmax.grid(row=0, column=2)
        self.rmax.delete(0, tk.END)
        self.rmax.insert(0, self.rmx[1])

        self.quit = tk.Button(master=self.top_frame, text="quit",
                              command=self._quit)
        self.quit.grid(row=1, column=0, sticky="w")

        blankrow = tk.Label(master=self.top_frame, text="   ", width=4)
        blankrow.grid(row=3, column=0, columnspan=5)

       
    def _plot_canvas(self):
        # matplotlib canvas --------------------------
        self.canvas = FigureCanvasTkAgg(self.f, master=self.parent)
        self.a.annotate("load image file using 'raw image' button ", (0.5, 0.6), 
                        horizontalalignment="center")
        self.a.annotate("e.g. data/O2-ANU1024.txt.bz2", (0.5, 0.5), 
                        horizontalalignment="center")

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.parent)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(anchor=tk.W, side=tk.TOP, fill=tk.BOTH, expand=1)

    def _text_info_box(self):
        # text info box ---------------------
        self.text = tk.Text(master=self.bottom_frame, height=3, fg="mediumblue")
        self.text.pack(fill=tk.X)
        self.text.insert(tk.END, "To start load an image data file using"
                         " menu File->`load image file`\n")


    # call back functions -----------------------
    def _display(self):
        if self.fn is None:
            self._getfilename()
        # update information text box
        self.text.insert(tk.END,"raw image\n")

        # display image
        self.f.clf()
        self.a = self.f.add_subplot(111)
        self.a.imshow(self.IM, vmin=0)
        self.f.colorbar(self.a.get_children()[2], ax=self.f.gca())
        self.canvas.show()


    def _getfilename(self):
        self.fn = askopenfilename()

        # update what is occurring text box
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, "reading image file {:s}\n".format(self.fn))
        self.canvas.show()

        # read image file
        if ".txt" in self.fn:
            self.IM = np.loadtxt(self.fn)
        else:
            self.IM = imread(self.fn)

        # if even size image, make odd
        if self.IM.shape[0] % 2 == 0:
            self.text.insert(tk.END, "make image odd size")
            self.IM = shift(self.IM, (-0.5, -0.5))[:-1,:-1]
    
        self.old_method = None
        self.AIM = None
        self.action = "file"
        self.center.config(state=tk.ACTIVE)
        self.center_method.config(state=tk.ACTIVE)
        self.recond.config(state=tk.ACTIVE)
        self.transform.config(state=tk.ACTIVE)
        self.speed.config(state=tk.ACTIVE)
        self.aniso.config(state=tk.ACTIVE)
        self.rmin.config(state=tk.NORMAL)
        self.lbl.config(state=tk.NORMAL)
        self.rmax.config(state=tk.NORMAL)

        # show the image
        self._display()
    

    def _center(self):
        self.action = "center"

        center_method = self.center_method.get()
        # update information text box
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, "centering image using {:s}\n".format(center_method))
        self.canvas.show()
    
        # center image via chosen method
        self.IM, self.offset = abel.tools.center.find_center(self.IM,
                               method=center_method)
        self.text.insert(tk.END, "center offset = {:}\n".format(self.offset))
    
        self._display()
    
    
    def _transform(self):
        #self.method = Abel_methods[self.transform_method.get()]
        self.method = self.transform.get()
    
        if self.method != self.old_method:
            # inverse Abel transform of whole image
            self.text.delete(1.0, tk.END)
            self.text.insert(tk.END,"inverse Abel transform: {:s}\n".format(self.method))
            if "basex" in self.method:
                self.text.insert(tk.END,"  first time calculation of the basis functions may take a while ...\n")
            if "onion" in self.method:
               self.text.insert(tk.END,"   onion_peeling method is in early testing and may not produce reliable results\n")
            if "direct" in self.method:
               self.text.insert(tk.END,"   calculation is slowed if Cython unavailable ...\n")
            self.canvas.show()
    
            self.AIM = abel.transform(self.IM, method=self.method, 
                                      direction="inverse",
                                      vertical_symmetry=False,
                                      horizontal_symmetry=False)['transform']
            self.speed.config(state=tk.ACTIVE)
            self.aniso.config(state=tk.ACTIVE)
            self.rmin.config(state=tk.NORMAL)
            self.rmin.delete(0, tk.END)
            self.rmin.insert(0, self.rmx[0])
            self.lbl.config(state=tk.NORMAL)
            self.rmax.config(state=tk.NORMAL)
            self.rmax.delete(0, tk.END)
            self.rmax.insert(0, self.rmx[1])
    
        if self.old_method != self.method or \
           self.action not in ["speed", "anisotropy"]:
            self.f.clf()
            self.a = self.f.add_subplot(111)
            self.a.imshow(self.AIM, vmin=0, vmax=self.AIM.max()/5.0)
            self.f.colorbar(self.a.get_children()[2], ax=self.f.gca())
            self.text.insert(tk.END, "{:s} inverse Abel transformed image".format(self.method))

        self.old_method = self.method
        self.canvas.show()
    
    def _speed(self):
        self.action = "speed"
        # inverse Abel transform
        self._transform()
        # update text box in case something breaks
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, "speed distribution\n")
        self.canvas.show()
    
        # speed distribution
        self.radial, self.speed_dist = abel.tools.vmi.angular_integration(self.AIM)
    
        self.f.clf()
        self.a = self.f.add_subplot(111)
        self.a.plot(self.radial, self.speed_dist/self.speed_dist[50:].max())
        self.a.axis(xmax=500, ymin=-0.05)
        self.a.set_xlabel("pixel radius")
        self.a.set_ylabel("normalized intensity")
        self.a.set_title("radial speed distribution")

        self.action = None
        self.canvas.show()
    
    def _anisotropy(self):

        def P2(x):   # 2nd order Legendre polynomial
            return (3*x*x-1)/2
    
    
        def PAD(theta, beta, amp):
            return amp*(1 + beta*P2(np.cos(theta)))
    
        self.action = "anisotropy"
        self._transform()
        # radial range over which to follow the intensity variation with angle
        self.rmx = (int(self.rmin.get()), int(self.rmax.get()))
    
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END,"anisotropy parameter pixel range {:} to {:}\n".format(*self.rmx))
        self.canvas.show()
    
        # inverse Abel transform
        self._transform()
    
        # intensity vs angle
        self.intensity, self.theta = abel.tools.vmi.calculate_angular_distributions(self.AIM,\
                                       radial_ranges=[self.rmx,])
    
        # fit to P2(cos theta)
        self.beta, self.amp = abel.tools.vmi.anisotropy_parameter(self.theta, self.intensity[0])
    
        self.text.insert(tk.END,"beta = {:g}+-{:g}\n".format(*self.beta))
    
        self.f.clf()
        self.a = self.f.add_subplot(111)
        self.a.plot(self.theta, self.intensity[0], 'r-')
        self.a.plot(self.theta, PAD(self.theta, self.beta[0], self.amp[0]), 'b-', lw=2)
        self.a.annotate("$\\beta({:d},{:d})={:.2g}\pm{:.2g}$".format(*self.rmx+self.beta), (-np.pi/2,-2))
        self.a.set_title("anisotropy")
        self.a.set_xlabel("angle")
        self.a.set_ylabel("intensity")

        self.action = None
        self.canvas.show()
    
    def _quit(self):
        self.parent.quit()     # stops mainloop
        self.parent.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate
    
        


if __name__ == "__main__":
    root = tk.Tk()
    pyabel = PyAbel(root)
    root.title("PyAbel simple GUI")
    root.config(menu=pyabel.menubar)
    root.mainloop()
