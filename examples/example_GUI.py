# -*- coding: iso-8859-1 -*-

# Illustrative GUI driving a small subset of PyAbel methods

import numpy as np
import abel

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
    from tkFileDialog import askopenfilename
else:
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
import tkinter.ttk as ttk
import tkinter.font as tkFont
from tkinter.scrolledtext import *
#from ScrolledText import *

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,\
                                              NavigationToolbar2TkAgg
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from matplotlib.pyplot import imread, colorbar
from matplotlib import gridspec

from scipy.ndimage.interpolation import shift

Abel_methods = ['basex', 'direct', 'hansenlaw', 'linbasex', 'onion_peeling', 
                'onion_bordas', 'two_point', 'three_point']

center_methods = ['center-of-mass', 'convolution', 'gaussian', 'slice']


class PyAbel:  #(tk.Tk):

    def __init__(self, parent):
        self.parent = parent
        self.initialize()

    def initialize(self):
        self.fn = None
        self.old_fn = None
        self.old_method = None
        self.old_fi = None
        self.AIM = None
        self.rmx = (368, 393)

        # matplotlib figure
        self.f = Figure(figsize=(2, 6))
        self.gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2])
        self.gs.update(wspace=0.2, hspace=0.2)

        self.plt = []
        self.plt.append(self.f.add_subplot(self.gs[0]))
        self.plt.append(self.f.add_subplot(self.gs[1]))
        self.plt.append(self.f.add_subplot(self.gs[2], sharex=self.plt[0],
                        sharey=self.plt[0]))
        self.plt.append(self.f.add_subplot(self.gs[3]))
        for i in [0, 2]:
            self.plt[i].set_adjustable('box-forced')

        # hide until have data
        for i in range(4):
            self.plt[i].axis("off")

        # tkinter 
        # set default font size for buttons
        self.font = tkFont.Font(size=11)
        self.fontB = tkFont.Font(size=12, weight='bold')

        #frames top (buttons), text, matplotlib (canvas)
        self.main_container = tk.Frame(self.parent, height=10, width=100)
        self.main_container.pack(side="top", fill="both", expand=True)

        self.button_frame = tk.Frame(self.main_container)
        #self.info_frame = tk.Frame(self.main_container)
        self.matplotlib_frame = tk.Frame(self.main_container)

        self.button_frame.pack(side="top", fill="x", expand=True)
        #self.info_frame.pack(side="top", fill="x", expand=True)
        self.matplotlib_frame.pack(side="top", fill="both", expand=True)

        self._menus()
        self._button_area()
        self._plot_canvas()
        self._text_info_box()

    def _button_frame(self):
        self.button_frame = tk.Frame(self.main_container)
        self.button_frame.pack(side="top", fill="x", expand=True)
        self._menus()

    def _menus(self):
        # menus with callback ----------------
        # duplicates the button interface
        self.menubar = tk.Menu(self.parent)
        self.transform_method = tk.IntVar()
        self.center_method = tk.IntVar()

        # File - menu
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Load image file",
                                  command=self._loadimage)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self._quit)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        # Process - menu
        self.processmenu = tk.Menu(self.menubar, tearoff=0)
        #self.processmenu.add_command(label="Center image", command=self._center)

        self.subcent=tk.Menu(self.processmenu)
        for cent in center_methods:
            self.subcent.add_radiobutton(label=cent,
                 var=self.center_method, val=center_methods.index(cent),
                 command=self._center)
        self.processmenu.add_cascade(label="Center image",
                 menu=self.subcent, underline=0)
        
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
    
        # view - menu
        self.viewmenu = tk.Menu(self.menubar, tearoff=0)
        self.viewmenu.add_command(label="Raw image", command=self._display)
        self.viewmenu.add_command(label="Inverse Abel transformed image",
                                  command=self._transform)
        self.viewmenu.add_command(label="view buttons",
                                  command=self._on_buttons)
        self.menubar.add_cascade(label="View", menu=self.viewmenu)


    def _button_area(self):
        # grid layout  
        # make expandable
        for col in range(5):
            self.button_frame.columnconfigure(col, weight=1)
            self.button_frame.rowconfigure(col, weight=1)

        # column 0 ---------
        # load image file button
        self.load = tk.Button(master=self.button_frame, text="load image",
                              font=self.fontB, fg="dark blue",
                              command=self._loadimage)
        self.load.grid(row=0, column=0, sticky=tk.W, padx=(5, 10), pady=(5, 0))
        self.sample_image = ttk.Combobox(master=self.button_frame,
                         font=self.font,
                         values=["from file", "from transform",
                                 "sample dribinski", "sample Ominus"],
                         width=14, height=4)
        self.sample_image.current(0)
        self.sample_image.grid(row=1, column=0, padx=(5, 10))

        # quit
        self.quit = tk.Button(master=self.button_frame, text="Quit",
                              font=self.fontB, fg="dark red",
                              command=self._quit)
        self.quit.grid(row=3, column=0, sticky=tk.W, padx=(5, 10), pady=(0, 5))
   
        # column 1 -----------
        # center image
        self.center = tk.Button(master=self.button_frame, text="center image",
                                anchor=tk.W, 
                                font=self.fontB, fg="dark blue",
                                command=self._center)
        self.center.grid(row=0, column=1, padx=(0, 20), pady=(5, 0))
        self.center_method = ttk.Combobox(master=self.button_frame,
                         font=self.font,
                         values=center_methods,
                         width=11, height=4)
        self.center_method.current(1)
        self.center_method.grid(row=1, column=1, padx=(0, 20))

        # column 2 -----------
        # Abel transform image
        self.recond = tk.Button(master=self.button_frame,
                                text="Abel transform image",
                                font=self.fontB, fg="dark blue",
                                command=self._transform)
        self.recond.grid(row=0, column=2, padx=(0, 10), pady=(5, 0))

        self.transform = ttk.Combobox(master=self.button_frame,
                         values=Abel_methods,
                         font=self.font,
                         width=10, height=len(Abel_methods))
        self.transform.current(2)
        self.transform.grid(row=1, column=2, padx=(0, 20))

        self.direction = ttk.Combobox(master=self.button_frame,
                         values=["inverse", "forward"],
                         font=self.font,
                         width=8, height=2)
        self.direction.current(0)
        self.direction.grid(row=2, column=2, padx=(0, 20))


        # column 3 -----------
        # speed button
        self.speed = tk.Button(master=self.button_frame, text="speed",
                               font=self.fontB, fg="dark blue",
                               command=self._speed)
        self.speed.grid(row=0, column=5, padx=20, pady=(5, 0))
        
        self.speedclr = tk.Button(master=self.button_frame, text="clear plot",
                                  font=self.font, command=self._speed_clr)
        self.speedclr.grid(row=1, column=5, padx=20)

        # column 4 -----------
        # anisotropy button
        self.aniso = tk.Button(master=self.button_frame, text="anisotropy",
                               font=self.fontB, fg="dark blue",
                               command=self._anisotropy)
        self.aniso.grid(row=0, column=6, pady=(5, 0))

        self.subframe = tk.Frame(self.button_frame)
        self.subframe.grid(row=1, column=6)
        self.rmin = tk.Entry(master=self.subframe, text='rmin', width=3,
                               font=self.font)
        self.rmin.grid(row=0, column=0)
        self.rmin.delete(0, tk.END)
        self.rmin.insert(0, self.rmx[0])
        self.lbl = tk.Label(master=self.subframe, text="to", font=self.font)
        self.lbl.grid(row=0, column=1)
        self.rmax = tk.Entry(master=self.subframe, text='rmax', width=3,
                             font=self.font)
        self.rmax.grid(row=0, column=2)
        self.rmax.delete(0, tk.END)
        self.rmax.insert(0, self.rmx[1])


        # turn off button interface
        self.hide_buttons = tk.Button(master=self.button_frame,
                                      text="hide buttons",
                                      font=self.fontB, fg='grey',
                                      command=self._hide_buttons)
        self.hide_buttons.grid(row=3, column=6, sticky=tk.E, pady=(0, 20))

    def _text_info_box(self):
        # text info box ---------------------
        self.text = ScrolledText(master=self.button_frame, height=6, 
                            fg="mediumblue",
                            bd=1, relief=tk.SUNKEN)
        self.text.insert(tk.END, "Work in progress, some features may"
                         " be incomplete ...\n")
        self.text.insert(tk.END, "To start: load an image data file using"
                         " e.g. data/O2-ANU1024.txt.bz2\n"
                         " (1) load image button (or file menu)\n"
                         " (2) center image\n"
                         " (3) Abel transform\n"
                         " (4) speed\n"
                         " (5) anisotropy\n"
                         " (6) Abel transform <- change\n"
                         " (:) repeat\n")
        self.text.grid(row=3, column=1, columnspan=3, padx=5)


    def _plot_canvas(self):
        # matplotlib canvas --------------------------
        self.canvas = FigureCanvasTkAgg(self.f, master=self.matplotlib_frame) 
        #self.cid = self.canvas.mpl_connect('button_press_event', self._onclick)

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.parent)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(anchor=tk.W, side=tk.TOP, fill=tk.BOTH, expand=1)


    def _onclick(self,event):
        print('button={:d}, x={:f}, y={:f}, xdata={:f}, ydata={:f}'.format(
        event.button, event.x, event.y, event.xdata, event.ydata))


    # call back functions -----------------------
    def _display(self):
        if self.fn is None:
            self._loadimage()

        # display image
        self.plt[0].imshow(self.IM, vmin=0)
        #rows, cols = self.IM.shape
        #r2 = rows/2
        #c2 = cols/2
        #self.a.plot((r2, r2), (0, cols), 'r--', lw=0.1)
        #self.a.plot((0, rows), (c2, c2),'r--', lw=0.1)
        #self.f.colorbar(self.a.get_children()[2], ax=self.f.gca())
        self.plt[0].set_title("raw image", fontsize=10)
        self.canvas.show()


    def _loadimage(self):

        if self.fn is not None:
            # clear old plot
            for i in range(4):
                self._clr_plt(i)
                self.plt[i].axis("off")

        self.fn = self.sample_image.get()
        # update what is occurring text box
        self.text.insert(tk.END, "\nloading image file {:s}".format(self.fn))
        self.text.see(tk.END)
        self.canvas.show()

        if self.fn == "from file":
            self.fn = askopenfilename()
            # read image file
            if ".txt" in self.fn:
                self.IM = np.loadtxt(self.fn)
            else:
                self.IM = imread(self.fn)
        elif self.fn == "from transform":
            self.IM = self.AIM
            self.AIM = None
            for i in range(1,4):
                self._clr_plt(i)
                self.plt[i].axis("off")
            self.direction.current(0)
        else:
            self.fn = self.fn.split(' ')[-1]
            self.IM = abel.tools.analytical.sample_image(n=1001, name=self.fn)
            self.direction.current(1) # raw images require 'forward' transform
            self.text.insert(tk.END,"\nsample image: (1) Abel transform 'forward', ")
            self.text.insert(tk.END,"              (2) load 'from transform', ") 
            self.text.insert(tk.END,"              (3) Abel transform 'inverse', ")
            self.text.insert(tk.END,"              (4) Speed")
            self.text.see(tk.END)


        # if even size image, make odd
        if self.IM.shape[0] % 2 == 0:
            self.IM = shift(self.IM, (-0.5, -0.5))[:-1,:-1]
    
        self.old_method = None
        self.AIM = None
        self.action = "file"
        self.rmin.delete(0, tk.END)
        self.rmin.insert(0, self.rmx[0])
        self.rmax.delete(0, tk.END)
        self.rmax.insert(0, self.rmx[1])

        # show the image
        self._display()
    

    def _center(self):
        self.action = "center"

        center_method = self.center_method.get()
        # update information text box
        self.text.insert(tk.END, "\ncentering image using {:s}".\
                         format(center_method))
        self.canvas.show()
    
        # center image via chosen method
        self.IM = abel.tools.center.center_image(self.IM, center=center_method,
                                  odd_size=True)
        #self.text.insert(tk.END, "\ncenter offset = {:}".format(self.offset))
        self.text.see(tk.END)
    
        self._display()
    
    
    def _transform(self):
        #self.method = Abel_methods[self.transform_method.get()]
        self.method = self.transform.get()
        self.fi = self.direction.get()
    
        if self.method != self.old_method or self.fi != self.old_fi:
            # Abel transform of whole image
            self.text.insert(tk.END,"\n{:s} {:s} Abel transform:".\
                             format(self.method, self.fi))
            if self.method == "basex":
                self.text.insert(tk.END,
                              "\nbasex: first time calculation of the basis"
                              " functions may take a while ...")
            elif self.method == "direct":
               self.text.insert(tk.END,
                 "\ndirect: calculation is slowed if Cython unavailable ...")
            self.canvas.show()
    
            if self.method == 'linbasex':
                self.AIM = abel.Transform(self.IM, method=self.method, 
                                direction=self.fi,
                                transform_options=dict(return_Beta=True))
            else:
                self.AIM = abel.Transform(self.IM, method=self.method, 
                                direction=self.fi,
                                symmetry_axis=None)
            self.rmin.delete(0, tk.END)
            self.rmin.insert(0, self.rmx[0])
            self.rmax.delete(0, tk.END)
            self.rmax.insert(0, self.rmx[1])
    
        if self.old_method != self.method or self.fi != self.old_fi or\
           self.action not in ["speed", "anisotropy"]:
            self.plt[2].set_title(self.method+" {:s} Abel transform".format(self.fi),
                            fontsize=10)
            self.plt[2].imshow(self.AIM.transform, vmin=0, 
                               vmax=self.AIM.transform.max()/5.0)
            #self.f.colorbar(self.c.get_children()[2], ax=self.f.gca())
            #self.text.insert(tk.END, "{:s} inverse Abel transformed image".format(self.method))

        self.text.see(tk.END)
        self.old_method = self.method
        self.old_fi = self.fi
        self.canvas.show()
    
    def _speed(self):
        self.action = "speed"
        # inverse Abel transform
        self._transform()
        # update text box in case something breaks
        self.text.insert(tk.END, "\nspeed distribution")
        self.text.see(tk.END)
        self.canvas.show()
    
        if self.method == 'linbasex':
            self.speed_dist = self.AIM.Beta[0]
            self.radial = self.AIM.radial
        else:
        # speed distribution
            self.radial, self.speed_dist = abel.tools.vmi.angular_integration(
                                           self.AIM.transform)
    
        self.plt[1].axis("on")
        self.plt[1].plot(self.radial, self.speed_dist/self.speed_dist[10:].max(),
                         label=self.method)
        # make O2- look nice
        if self.fn.find('O2-ANU1024') > -1:
            self.plt[1].axis(xmax=500, ymin=-0.05)
        elif self.fn.find('VMI_art1') > -1:
            self.plt[1].axis(xmax=260, ymin=-0.05)

        self.plt[1].set_xlabel("radius (pixels)", fontsize=9)
        self.plt[1].set_ylabel("normalized intensity")
        self.plt[1].set_title("radial speed distribution", fontsize=12)
        self.plt[1].legend(fontsize=9, loc=0, frameon=False)

        self.action = None
        self.canvas.show()

    def _speed_clr(self):
        self._clr_plt(1)

    def _clr_plt(self, i):
        self.f.delaxes(self.plt[i])
        self.plt[i] = self.f.add_subplot(self.gs[i]) 
        self.canvas.show()
    
    def _anisotropy(self):

        def P2(x):   # 2nd order Legendre polynomial
            return (3*x*x-1)/2
    
    
        def PAD(theta, beta, amp):
            return amp*(1 + beta*P2(np.cos(theta)))
    
        self.action = "anisotropy"
        self._transform()

        if self.method == 'linbasex':
            self.text.insert(tk.END,
                        "\nanisotropy parameter pixel range 0 to {}: "\
                        .format(self.rmx[1]))
        else:
            # radial range over which to follow the intensity variation with angle
            self.rmx = (int(self.rmin.get()), int(self.rmax.get()))
            self.text.insert(tk.END,
                        "\nanisotropy parameter pixel range {:} to {:}: "\
                        .format(*self.rmx))
        self.canvas.show()
    
        # inverse Abel transform
        self._transform()
    
        if self.method == 'linbasex':
            self.beta = self.AIM.Beta[1]
            self.radial = self.AIM.radial
            self._clr_plt(3)
            self.plt[3].axis("on")
            self.plt[3].plot(self.radial, self.beta, 'r-')
            self.plt[3].set_title("anisotropy", fontsize=12)
            self.plt[3].set_xlabel("radius", fontsize=9)
            self.plt[3].set_ylabel("anisotropy parameter")
            # make O2- look nice
            if self.fn.find('O2-ANU1024') > -1:
                self.plt[3].axis(xmax=500, ymin=-1.1, ymax=0.1)
            elif self.fn.find('VMI_art1') > -1:
                self.plt[3].axis(xmax=260, ymin=-1.1, ymax=2)
        else:
            # intensity vs angle
            self.beta, self.amp, self.rad, self.intensity, self.theta =\
               abel.tools.vmi.radial_integration(self.AIM.transform,\
                                                 radial_ranges=[self.rmx,])
    
            self.text.insert(tk.END," beta = {:g}+-{:g}".format(*self.beta[0]))
    
            self._clr_plt(3)
            self.plt[3].axis("on")
        
            self.plt[3].plot(self.theta, self.intensity[0], 'r-')
            self.plt[3].plot(self.theta, PAD(self.theta, self.beta[0][0],
                             self.amp[0][0]), 'b-', lw=2)
            self.plt[3].annotate("$\\beta({:d},{:d})={:.2g}\pm{:.2g}$".
             format(*self.rmx+self.beta[0]), (-3, self.intensity[0].min()/0.8))
            self.plt[3].set_title("anisotropy", fontsize=12)
            self.plt[3].set_xlabel("angle", fontsize=9)
            self.plt[3].set_ylabel("intensity")

        self.action = None
        self.canvas.show()

    def _hide_buttons(self):
        self.button_frame.destroy()

    def _on_buttons(self):
        self._button_frame()
    
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
