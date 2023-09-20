import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter.simpledialog import askfloat
from tkinter import messagebox
import numpy as np
import plasma
import collision
import jets
import matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # , NavigationToolbar2Tk
# import matplotlib.animation as animation
# import matplotlib.colors as colors
from utilities import round_decimals_up, round_decimals_down
import timekeeper
import config

matplotlib.use('TkAgg')  # Use proper matplotlib backend
style.use('Solarize_Light2')


# Define fonts
LARGE_FONT = ('Verdana', 10)

#####################
# Generic Functions #
#####################


# Function to slap together a label string to display from calculated moment data.
def moment_label(moment, angleDeflection, k=0, label='Total'):
    string = 'k = ' + str(k) + ' ' + str(label) + ' Jet Drift Moment: ' + str(moment) + ' GeV\n' \
             + 'Angular Deflection: ' + str(angleDeflection) + ' deg'
    return string


###########################
# Main Application Object #
###########################
# Define the main application object, inheriting from tk.Tk object type.
class PlasmaInspector(tk.Tk):
    # Initialization. Needs self. *args allows any number of parameters to be passed in,  **kwargs allows dictionaries.
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)  # Initializes tkinter business passed into the PlasmaInspector.
        # tk.Tk.iconbitmap(self, default='icon.ico')  # Sets the application icon - not working on Mephisto
        tk.Tk.wm_title(self, 'Plasma Inspector')
        # Create menu bar
        self.menubar = tk.Menu(self)
        tk.Tk.config(self, menu=self.menubar)

        # Dictionary of frames
        # Top one is active frame, can hold many frames for cool stuff to do
        self.frames = {}

        # Instantiate each frame and add them into our dictionary of frames
        for frame in [MainPage]:

            current_frame = frame(self)  # Instantiate frame

            self.frames[frame] = current_frame  # Add to dictionary

            # Sets where the frame goes
            # sticky -> North South East West... Defines which directions things stretch in. This is all directions
            # You can stick stuff wherever you want, you don't need to predefine grid size and shape
            current_frame.grid()

        self.show_frame(MainPage)

    # Method to show a particular frame
    def show_frame(self, framekey):

        # Selects frame to be raised to be frame within frames dictionary with key framekey
        frame = self.frames[framekey]

        # Raises the selected frame to the front
        frame.tkraise()


#############
# Main Page #
#############
# Define the main page, inheriting from the tk.Frame class
class MainPage(tk.Frame):

    def __init__(self, parent):
        # initialize frame inheritence
        tk.Frame.__init__(self, parent)

        # Set initialization flags
        self.file_selected = False

        # ##############################
        # Initialize adjustable values #
        ################################
        # Physical options
        self.time = tk.DoubleVar()  # Holds a float; default value 0.0
        self.theta0 = tk.DoubleVar()
        self.x0 = tk.DoubleVar()
        self.y0 = tk.DoubleVar()
        self.jetE = tk.DoubleVar()
        self.jetE.set(1)
        self.nth = tk.IntVar()
        self.nth.set(10)
        self.calculated = tk.BooleanVar()
        self.calculated.set(False)
        self.drift = tk.BooleanVar()
        self.drift.set(True)
        self.grad = tk.BooleanVar()
        self.grad.set(False)
        self.fg = tk.BooleanVar()
        self.fg.set(True)
        self.el = tk.BooleanVar()
        self.el.set(True)
        self.el_model = tk.StringVar()
        self.el_model.set('SGLV')

        # Integration options
        self.tempHRG = tk.DoubleVar()
        self.tempHRG.set(config.transport.hydro.T_HRG)
        self.tempUnhydro = tk.DoubleVar()
        self.tempUnhydro.set(config.transport.hydro.T_UNHYDRO)

        # Plotting options
        self.plot_temp = tk.BooleanVar()
        self.plot_temp.set(True)
        self.plot_vel = tk.BooleanVar()
        self.plot_vel.set(True)
        self.plot_grad = tk.BooleanVar()
        self.plot_grad.set(False)
        self.velocityType = tk.StringVar()
        self.velocityType.set('stream')
        self.gradientType = tk.StringVar()
        self.gradientType.set('stream')
        self.contourNumber = tk.IntVar()
        self.contourNumber.set(15)
        self.tempType = tk.StringVar()
        self.tempType.set('contour')
        self.propPlotRes = tk.DoubleVar()
        self.propPlotRes.set(0.2)
        self.plotColors = tk.BooleanVar()
        self.plotColors.set(True)
        self.zoom = tk.DoubleVar()
        self.zoom.set(1)

        # Moment variables
        self.jet_dataframe = None
        self.jet_xarray = None
        self.moment = 0
        self.angleDeflection = 0
        self.momentPlasma = 0
        self.angleDeflectionPlasma = 0
        self.momentHRG = 0
        self.angleDeflectionHRG = 0
        self.K = 0
        self.momentDisplay = tk.StringVar()
        self.momentDisplay.set(moment_label(moment=None, angleDeflection=None,
                                            k=self.K, label='Total'))
        self.ELDisplay = tk.StringVar()
        self.ELDisplay.set('...')
        self.momentHRGDisplay = tk.StringVar()
        self.momentHRGDisplay.set('...')
        self.momentUnhydroDisplay = tk.StringVar()
        self.momentUnhydroDisplay.set('...')

        ################
        # Plot Objects #
        ################

        # Create the QGP Plot that will dynamically update and set its labels
        self.plasmaFigure = plt.figure(num=0)
        self.plasmaAxis = self.plasmaFigure.add_subplot(1, 1, 1)

        # Define colorbar objects with "1" scalar mappable object so they can be manipulated.
        self.tempcb = 0
        self.velcb = 0
        self.gradcb = 0

        # Define plots
        self.tempPlot = None
        self.velPlot = None

        # Create the medium property plots that will dynamically update
        # Constrained layout does some magic to organize the plots
        self.propertyFigure, self.propertyAxes = plt.subplots(3, 4, constrained_layout=True, num=1)

        # Create canvases and show the empty plots
        self.canvas = FigureCanvasTkAgg(self.plasmaFigure, master=self)
        self.canvas.draw()
        self.canvas1 = FigureCanvasTkAgg(self.propertyFigure, master=self)
        self.canvas1.draw()

        # Make our cute little toolbar for the plasma plot
        # self.toolbar = NavigationToolbar2Tk(self.canvas1, self)
        # self.toolbar.update()

        # Set up the moment display
        self.momentLabel = tk.Label(self, textvariable=self.momentDisplay, font=LARGE_FONT)
        self.momentPlasmaLabel = tk.Label(self, textvariable=self.ELDisplay, font=LARGE_FONT)
        self.momentHRGLabel = tk.Label(self, textvariable=self.momentHRGDisplay, font=LARGE_FONT)
        self.momentUnhydroLabel = tk.Label(self, textvariable=self.momentUnhydroDisplay, font=LARGE_FONT)

        ############
        # Controls #
        ############
        # Create button to change pages
        #buttonPage1 = ttk.Button(self, text='Go to Page 1', command=lambda: parent.show_frame(PageOne))

        # Create button to update plots
        self.update_button = ttk.Button(self, text='Update Plots',
                                   command=self.update_plots)

        # Create button to calculate and display moment
        self.run_jet_button = ttk.Button(self, text='Run Jet',
                                         command=self.run_jet)

        # Create button to sample the event and set jet initial conditions.
        self.sample_button = ttk.Button(self, text='Sample',
                                        command=self.sample_event)



        # Create time slider
        self.timeSlider = tk.Scale(self, orient=tk.HORIZONTAL, variable=self.time,
                                   from_=0, to=10, length=200, resolution=0.1, label='time (fm)')
        # Create x0 slider
        self.x0Slider = tk.Scale(self, orient=tk.HORIZONTAL, variable=self.x0,
                                 from_=0, to=10, length=200, resolution=0.1, label='x0 (fm)')
        # Create y0 slider
        self.y0Slider = tk.Scale(self, orient=tk.HORIZONTAL, variable=self.y0,
                                 from_=0, to=10, length=200, resolution=0.1, label='y0 (fm)')
        # Create theta0 slider
        self.theta0Slider = tk.Scale(self, orient=tk.HORIZONTAL, variable=self.theta0,
                                     from_=0, to=2*np.pi, length=200, resolution=0.1, label='theta0 (rad)')
        # Create jetE slider
        self.jetESlider = tk.Scale(self, orient=tk.HORIZONTAL, variable=self.jetE,
                                   from_=0.1, to=20, length=200, resolution=0.1, label='jetE (GeV)')
        # Create tempHRG slider
        self.tempCutoffSlider = tk.Scale(self, orient=tk.HORIZONTAL,
                                         variable=self.tempHRG, from_=0, to=1, length=200, resolution=0.01,
                                         label='Had. Temp (GeV)')
        # Create tempHRG slider
        self.tempUnhydroSlider = tk.Scale(self, orient=tk.HORIZONTAL,
                                          variable=self.tempUnhydro, from_=0, to=1, length=200, resolution=0.01,
                                          label='Unhydro Temp (GeV)')
        # Create zoom slider
        self.zoomSlider = tk.Scale(self, orient=tk.HORIZONTAL,
                                          variable=self.zoom, from_=0, to=1, length=200, resolution=0.01,
                                          label='Zoom')


        # Register update ON RELEASE - use of command parameter applies action immediately
        #self.update_button.bind("<ButtonRelease-1>", self.update_plots)
        self.timeSlider.bind("<ButtonRelease-1>", self.update_jet)
        self.x0Slider.bind("<ButtonRelease-1>", self.update_jet)
        self.y0Slider.bind("<ButtonRelease-1>", self.update_jet)
        self.theta0Slider.bind("<ButtonRelease-1>", self.update_jet)
        self.jetESlider.bind("<ButtonRelease-1>", self.update_jet)
        self.tempCutoffSlider.bind("<ButtonRelease-1>", self.update_jet)
        self.tempUnhydroSlider.bind("<ButtonRelease-1>", self.update_jet)
        self.zoomSlider.bind("<ButtonRelease-1>", self.update_jet)

        #########
        # Menus #
        #########
        # Create file menu cascade
        # ---
        fileMenu = tk.Menu(parent.menubar, tearoff=0)
        fileMenu.add_command(label='Select File', command=self.select_file)
        fileMenu.add_command(label='Optical Glauber', command=self.optical_glauber)
        fileMenu.add_command(label='Log(Mult) Temp Optical Glauber', command=self.lmt_optical_glauber)
        fileMenu.add_command(label='\"new\" Optical Glauber', command=self.new_optical_glauber)
        fileMenu.add_command(label='Exit', command=self.quit)
        parent.menubar.add_cascade(label='File', menu=fileMenu)
        # ---

        # Create physics menu cascade
        # ---
        physicsMenu = tk.Menu(parent.menubar, tearoff=0)
        # drift submenu
        driftMenu = tk.Menu(physicsMenu, tearoff=0)
        physicsMenu.add_cascade(label='Drift', menu=driftMenu)
        driftMenu.add_radiobutton(label='On', variable=self.drift, value=True,
                                command=self.not_calculated)
        driftMenu.add_radiobutton(label='Off', variable=self.drift, value=False,
                                  command=self.not_calculated)
        # Gradients submenu
        gradMenu = tk.Menu(physicsMenu, tearoff=0)
        physicsMenu.add_cascade(label='Gradients', menu=gradMenu)
        gradMenu.add_radiobutton(label='On', variable=self.grad, value=True,
                                  command=self.not_calculated)
        gradMenu.add_radiobutton(label='Off', variable=self.grad, value=False,
                                  command=self.not_calculated)

        # Flow-Gradients submenu
        flowgradMenu = tk.Menu(physicsMenu, tearoff=0)
        physicsMenu.add_cascade(label='Flow-Gradient', menu=flowgradMenu)
        flowgradMenu.add_radiobutton(label='On', variable=self.fg, value=True,
                                 command=self.not_calculated)
        flowgradMenu.add_radiobutton(label='Off', variable=self.fg, value=False,
                                 command=self.not_calculated)

        # EL submenu
        elMenu = tk.Menu(physicsMenu, tearoff=0)
        el_model_menu = tk.Menu(elMenu, tearoff=0)
        elMenu.add_cascade(label='Model', menu=el_model_menu)
        physicsMenu.add_cascade(label='Energy Loss', menu=elMenu)
        elMenu.add_radiobutton(label='On', variable=self.el, value=True,
                                 command=self.not_calculated)
        elMenu.add_radiobutton(label='Off', variable=self.el, value=False,
                                 command=self.not_calculated)
        el_model_menu.add_radiobutton(label='SGLV', variable=self.el_model, value='SGLV',
                               command=self.not_calculated)
        el_model_menu.add_radiobutton(label='BBMG', variable=self.el_model, value='BBMG',
                               command=self.not_calculated)
        parent.menubar.add_cascade(label='Physics', menu=physicsMenu)

        # Create plasma plot menu cascade
        # ---
        plasmaMenu = tk.Menu(parent.menubar, tearoff=0)
        # Jet submenu
        jetMenu = tk.Menu(plasmaMenu, tearoff=0)
        plasmaMenu.add_cascade(label='Jet', menu=jetMenu)
        jetMenu.add_radiobutton(label='Coarse (Every 20th)', variable=self.nth, value=20,
                                    command=self.update_plots)
        jetMenu.add_radiobutton(label='Medium-Coarse (Every 15th)', variable=self.nth, value=15,
                                    command=self.update_plots)
        jetMenu.add_radiobutton(label='Medium (Every 10th)', variable=self.nth, value=10,
                                    command=self.update_plots)
        jetMenu.add_radiobutton(label='Medium-Fine (Every 5th)', variable=self.nth, value=5,
                                    command=self.update_plots)
        jetMenu.add_radiobutton(label='Fine (Every 2nd)', variable=self.nth, value=2,
                                    command=self.update_plots)
        jetMenu.add_radiobutton(label='Ultra-Fine (1)', variable=self.nth, value=1,
                                    command=self.update_plots)
        # Temperature submenu
        tempMenu = tk.Menu(plasmaMenu, tearoff=0)
        plasmaMenu.add_cascade(label='Temperatures', menu=tempMenu)
        # Contours sub-submenu
        contourMenu = tk.Menu(tempMenu, tearoff=0)
        contourMenu.add_radiobutton(label='Coarse (10)', variable=self.contourNumber, value=10,
                                command=self.update_plots)
        contourMenu.add_radiobutton(label='Medium-Coarse (15)', variable=self.contourNumber, value=15,
                                 command=self.update_plots)
        contourMenu.add_radiobutton(label='Medium (20)', variable=self.contourNumber, value=20,
                                 command=self.update_plots)
        contourMenu.add_radiobutton(label='Medium-Fine (25)', variable=self.contourNumber, value=25,
                                 command=self.update_plots)
        contourMenu.add_radiobutton(label='Fine (30)', variable=self.contourNumber, value=30,
                                 command=self.update_plots)
        contourMenu.add_radiobutton(label='Ultra-Fine (40)', variable=self.contourNumber, value=40,
                                 command=self.update_plots)
        tempMenu.add_cascade(label='No. Contours', menu=contourMenu)
        # Plot type sub-submenu
        tempTypeMenu = tk.Menu(tempMenu, tearoff=0)
        tempMenu.add_cascade(label='Plot Style', menu=tempTypeMenu)
        tempTypeMenu.add_radiobutton(label='Contour', variable=self.tempType, value='contour',
                                    command=self.update_plots)
        tempTypeMenu.add_radiobutton(label='Density', variable=self.tempType, value='density',
                                    command=self.update_plots)
        tempTypeMenu.add_radiobutton(label='Enabled', variable=self.plot_temp, value=True,
                                     command=self.update_plots)
        tempTypeMenu.add_radiobutton(label='Disabled', variable=self.plot_temp, value=False,
                                     command=self.update_plots)



        # Velocity submenu
        velMenu = tk.Menu(plasmaMenu, tearoff=0)
        velTypeMenu = tk.Menu(velMenu)
        velTypeMenu.add_radiobutton(label='Stream Velocities', variable=self.velocityType, value='stream',
                              command=self.update_plots)
        velTypeMenu.add_radiobutton(label='Quiver Velocities', variable=self.velocityType, value='quiver',
                              command=self.update_plots)
        velTypeMenu.add_radiobutton(label='Enabled', variable=self.plot_vel, value=True,
                                    command=self.update_plots)
        velTypeMenu.add_radiobutton(label='Disabled', variable=self.plot_vel, value=False,
                                    command=self.update_plots)
        velMenu.add_cascade(label='Plot Style', menu=velTypeMenu)
        plasmaMenu.add_cascade(label='Velocities', menu=velMenu)
        parent.menubar.add_cascade(label='Plasma Plot', menu=plasmaMenu)

        # Gradient submenu
        gradMenu = tk.Menu(plasmaMenu, tearoff=0)
        gradTypeMenu = tk.Menu(gradMenu)
        gradTypeMenu.add_radiobutton(label='Stream Gradients', variable=self.gradientType, value='stream',
                                    command=self.update_plots)
        gradTypeMenu.add_radiobutton(label='Quiver Gradients', variable=self.gradientType, value='quiver',
                                    command=self.update_plots)
        gradTypeMenu.add_radiobutton(label='Enabled', variable=self.plot_grad, value=True,
                                     command=self.update_plots)
        gradTypeMenu.add_radiobutton(label='Disabled', variable=self.plot_grad, value=False,
                                     command=self.update_plots)
        gradMenu.add_cascade(label='Plot Style', menu=gradTypeMenu)
        plasmaMenu.add_cascade(label='Gradients', menu=gradMenu)

        # Property plot menu
        propMenu = tk.Menu(parent.menubar, tearoff=0)
        propMenu.add_checkbutton(label='Color Plots for Temp Cutoff', variable=self.plotColors, onvalue=1,
                                 offvalue=0, command=self.update_plots)
        propResMenu = tk.Menu(propMenu)
        propResMenu.add_radiobutton(label='0.1 fm', variable=self.propPlotRes, value=0.1,
                                    command=self.update_plots)
        propResMenu.add_radiobutton(label='0.2 fm', variable=self.propPlotRes, value=0.2,
                                    command=self.update_plots)
        propResMenu.add_radiobutton(label='0.3 fm', variable=self.propPlotRes, value=0.3,
                                    command=self.update_plots)
        propResMenu.add_radiobutton(label='0.4 fm', variable=self.propPlotRes, value=0.4,
                                    command=self.update_plots)
        propResMenu.add_radiobutton(label='0.5 fm', variable=self.propPlotRes, value=0.5,
                                    command=self.update_plots)
        propMenu.add_cascade(label='Plotting Time Resolution', menu=propResMenu)
        parent.menubar.add_cascade(label='Medium Properties', menu=propMenu)
        # ---

        ###########################
        # Organization and Layout #
        ###########################
        # Smash everything into the window
        self.update_button.grid(row=0, column=0)
        self.sample_button.grid(row=0, column=1)
        self.run_jet_button.grid(row=0, column=2)
        self.timeSlider.grid(row=1, column=0, columnspan=2)
        self.x0Slider.grid(row=1, column=2, columnspan=2)
        self.y0Slider.grid(row=1, column=4, columnspan=2)
        self.theta0Slider.grid(row=1, column=6, columnspan=2)
        self.jetESlider.grid(row=1, column=8, columnspan=2)
        self.zoomSlider.grid(row=1, column=10, columnspan=1)
        self.tempCutoffSlider.grid(row=5, column=8, columnspan=2)
        self.tempUnhydroSlider.grid(row=6, column=8, columnspan=2)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=6, rowspan=3)
        self.canvas1.get_tk_widget().grid(row=2, column=6, columnspan=6, rowspan=3)
        self.momentLabel.grid(row=5, column=0, columnspan=4)
        self.momentPlasmaLabel.grid(row=6, column=0, columnspan=4)
        self.momentHRGLabel.grid(row=5, column=4, columnspan=4)
        self.momentUnhydroLabel.grid(row=6, column=4, columnspan=4)
        # buttonPage1.grid()  # Unused second page

        # Create the jet object
        self.update_jet(0)

    # Define the select file function
    def select_file(self, value=None):
        hydro_file_path = askopenfilename(
            initialdir='/share/apps/Hybrid-Transport/hic-eventgen/results/')  # show an "Open" dialog box and return the path to the selected file
        self.file_selected = True
        print('Selected file: ' + str(hydro_file_path))

        # Open grid file as object
        self.hydro_file = plasma.osu_hydro_file(file_path=hydro_file_path)

        # Create plasma object from current_event object
        self.current_event = plasma.plasma_event(event=self.hydro_file)

        # Find current_event parameters
        self.tempMax = self.current_event.max_temp()
        self.tempMin = self.current_event.min_temp()

        # Set sliders limits to match bounds of the event
        dec = 1  # number of decimals rounding to... Should match resolution of slider.
        self.timeSlider.configure(from_=round_decimals_up(self.current_event.t0, decimals=dec))
        self.timeSlider.configure(to=round_decimals_down(self.current_event.tf, decimals=dec))
        self.time.set(round_decimals_up(self.current_event.t0, decimals=dec))

        self.x0Slider.configure(from_=round_decimals_up(self.current_event.xmin, decimals=dec))
        self.x0Slider.configure(to=round_decimals_down(self.current_event.xmax, decimals=dec))
        self.x0.set(0)

        self.y0Slider.configure(from_=round_decimals_up(self.current_event.ymin, decimals=dec))
        self.y0Slider.configure(to=round_decimals_down(self.current_event.ymax, decimals=dec))
        self.y0.set(0)

        self.update_plots()

    # Define the optical glauber selection
    def optical_glauber(self, value=None):
        # Ask user for optical glauber input parameters:
        R = askfloat("Input", "Enter ion radius (R) in fm: ", minvalue=0.0, maxvalue=25)
        b = askfloat("Input", "Enter impact parameter (b) in fm (max 2R): ", minvalue=0.0, maxvalue=2*R)
        T0 = askfloat("Input", "Enter temperature norm in GeV: ", minvalue=0.0, maxvalue=500)
        V0 = askfloat("Input", "Enter flow velocity norm in c: ", minvalue=0.0, maxvalue=1)
        phi = askfloat("Input", "Enter reaction plane angle in rad: ", minvalue=0.0, maxvalue=2*np.pi)
        rmax = 15
        event_lifetime = 20
        self.file_selected = True  # Set that you have selected an event
        print('Selected optical glauber:\nR = {}, b = {}, phi = {}, T0 = {}, V0 = {}'.format(R, b, phi, T0, V0))

        # Generate optical glauber
        analytic_t, analytic_ux, analytic_uy, mult, e2 = collision.optical_glauber(R=R, b=b, phi=phi, T0=T0, U0=V0)

        # Create plasma object
        self.current_event = plasma.functional_plasma(temp_func=analytic_t, x_vel_func=analytic_ux,
                                                 y_vel_func=analytic_uy,
                                                 xmax=rmax, time=event_lifetime)

        # Find current_event parameters
        self.tempMax = self.current_event.max_temp()
        self.tempMin = self.current_event.min_temp()

        # Set sliders limits to match bounds of the event
        dec = 1  # number of decimals rounding to... Should match resolution of slider.
        self.timeSlider.configure(from_=round_decimals_up(self.current_event.t0, decimals=dec))
        self.timeSlider.configure(to=round_decimals_down(self.current_event.tf, decimals=dec))
        self.time.set(round_decimals_up(self.current_event.t0, decimals=dec))

        self.x0Slider.configure(from_=round_decimals_up(self.current_event.xmin, decimals=dec))
        self.x0Slider.configure(to=round_decimals_down(self.current_event.xmax, decimals=dec))
        self.x0.set(0)

        self.y0Slider.configure(from_=round_decimals_up(self.current_event.ymin, decimals=dec))
        self.y0Slider.configure(to=round_decimals_down(self.current_event.ymax, decimals=dec))
        self.y0.set(0)

        self.update_plots()

    # Define the optical glauber selection
    def lmt_optical_glauber(self, value=None):
        # Ask user for optical glauber input parameters:
        R = askfloat("Input", "Enter ion radius (R) in fm: ", minvalue=0.0, maxvalue=25)
        b = askfloat("Input", "Enter impact parameter (b) in fm (max 2R): ", minvalue=0.0, maxvalue=2 * R)
        T0 = askfloat("Input", "Enter temperature norm in GeV: ", minvalue=0.0, maxvalue=500)
        V0 = askfloat("Input", "Enter flow velocity norm in c: ", minvalue=0.0, maxvalue=1)
        phi = askfloat("Input", "Enter reaction plane angle in rad: ", minvalue=0.0, maxvalue=2 * np.pi)
        rmax = 15
        event_lifetime = 20
        self.file_selected = True  # Set that you have selected an event
        print('Selected lmt optical glauber:\nR = {}, b = {}, phi = {}, T0 = {}, V0 = {}'.format(R, b, phi, T0, V0))

        # Generate optical glauber
        analytic_t, analytic_ux, analytic_uy, mult, e2 = collision.optical_glauber_logT(R=R, b=b, phi=phi, T0=T0, U0=V0)

        # Create plasma object
        self.current_event = plasma.functional_plasma(temp_func=analytic_t, x_vel_func=analytic_ux,
                                                      y_vel_func=analytic_uy,
                                                      xmax=rmax, time=event_lifetime)

        # Find current_event parameters
        self.tempMax = self.current_event.max_temp()
        self.tempMin = self.current_event.min_temp()

        # Set sliders limits to match bounds of the event
        dec = 1  # number of decimals rounding to... Should match resolution of slider.
        self.timeSlider.configure(from_=round_decimals_up(self.current_event.t0, decimals=dec))
        self.timeSlider.configure(to=round_decimals_down(self.current_event.tf, decimals=dec))
        self.time.set(round_decimals_up(self.current_event.t0, decimals=dec))

        self.x0Slider.configure(from_=round_decimals_up(self.current_event.xmin, decimals=dec))
        self.x0Slider.configure(to=round_decimals_down(self.current_event.xmax, decimals=dec))
        self.x0.set(0)

        self.y0Slider.configure(from_=round_decimals_up(self.current_event.ymin, decimals=dec))
        self.y0Slider.configure(to=round_decimals_down(self.current_event.ymax, decimals=dec))
        self.y0.set(0)

        self.update_plots()

    # Define the optical glauber selection
    def new_optical_glauber(self, value=None):
        # Ask user for optical glauber input parameters:
        R = askfloat("Input", "Enter ion radius (R) in fm: ", minvalue=0.0, maxvalue=25)
        b = askfloat("Input", "Enter impact parameter (b) in fm (max 2R): ", minvalue=0.0, maxvalue=2 * R)
        T0 = askfloat("Input", "Enter temperature norm in GeV: ", minvalue=0.0, maxvalue=500)
        V0 = askfloat("Input", "Enter flow velocity norm in c: ", minvalue=0.0, maxvalue=1)
        phi = askfloat("Input", "Enter reaction plane angle in rad: ", minvalue=0.0, maxvalue=2 * np.pi)
        rmax = 15
        event_lifetime = 20
        self.file_selected = True  # Set that you have selected an event
        print('Selected "new" optical glauber:\nR = {}, b = {}, phi = {}, T0 = {}, V0 = {}'.format(R, b, phi, T0, V0))

        # Generate optical glauber
        analytic_t, analytic_ux, analytic_uy, mult, e2 = collision.optical_glauber_logT(R=R, b=b, phi=phi, T0=T0, U0=V0)

        # Create plasma object
        self.current_event = plasma.functional_plasma(temp_func=analytic_t, x_vel_func=analytic_ux,
                                                      y_vel_func=analytic_uy,
                                                      xmax=rmax, time=event_lifetime)

        # Find current_event parameters
        self.tempMax = self.current_event.max_temp()
        self.tempMin = self.current_event.min_temp()

        # Set sliders limits to match bounds of the event
        dec = 1  # number of decimals rounding to... Should match resolution of slider.
        self.timeSlider.configure(from_=round_decimals_up(self.current_event.t0, decimals=dec))
        self.timeSlider.configure(to=round_decimals_down(self.current_event.tf, decimals=dec))
        self.time.set(round_decimals_up(self.current_event.t0, decimals=dec))

        self.x0Slider.configure(from_=round_decimals_up(self.current_event.xmin, decimals=dec))
        self.x0Slider.configure(to=round_decimals_down(self.current_event.xmax, decimals=dec))
        self.x0.set(0)

        self.y0Slider.configure(from_=round_decimals_up(self.current_event.ymin, decimals=dec))
        self.y0Slider.configure(to=round_decimals_down(self.current_event.ymax, decimals=dec))
        self.y0.set(0)

        self.update_plots()

    def tempColoringFunc(self, t):
        colorArray = np.array([])
        for time in t:
            # Check temperature at the given time
            checkTemp = self.current_event.temp(self.current_jet.coords3(time))
            # Assign relevant color to plot
            if self.tempHRG.get() > checkTemp > self.tempUnhydro.get():
                # Hadron gas phase, hydrodynamic
                color = 'g'
            elif checkTemp < self.tempUnhydro.get():
                # Hadron gas phase, unhydrodynamic
                color = 'r'
            else:
                color = 'b'
            colorArray = np.append(colorArray, color)
        hydroHRGTimeIndexes = np.where(colorArray == 'g')
        hydroHRGTimes = np.array([])
        for index in hydroHRGTimeIndexes:
            time = t[index]
            hydroHRGTimes = np.append(hydroHRGTimes, time)
        unhydroHRGTimeIndexes = np.where(colorArray == 'r')
        unhydroHRGTimes = np.array([])
        for index in unhydroHRGTimeIndexes:
            time = t[index]
            unhydroHRGTimes = np.append(unhydroHRGTimes, time)
        return colorArray, hydroHRGTimes, unhydroHRGTimes

    # Define set not calculated function
    def not_calculated(self, value=None):
        self.calculated.set(False)

    # Define jet update function
    def update_jet(self, value=None):
        # Set current_jet object to current slider parameters
        self.current_jet = jets.jet(x_0=self.x0.get(), y_0=self.y0.get(),
                                    phi_0=self.theta0.get(), p_T0=self.jetE.get(), tag=None)
        self.not_calculated(0)

    # Define the update function
    def update_plots(self, value=None):
        if self.file_selected:
            # Clear all the plots and colorbars
            self.plasmaAxis.clear()
            try:
                self.tempcb.remove()
            except AttributeError:
                pass
            try:
                self.velcb.remove()
            except AttributeError:
                pass
            try:
                self.gradcb.remove()
            except AttributeError:
                pass

            for axisList in self.propertyAxes:  # Medium property plots
                for axis in axisList:
                    axis.clear()

            # Select QGP figure as current figure
            plt.figure(self.plasmaFigure.number)

            # Select QGP plot axis as current axis
            # plt.sca(self.plasmaAxis)

            # Plot new temperatures & velocities
            self.tempPlot, self.velPlot, self.gradPlot, self.tempcb, self.velcb, self.gradcb\
                = self.current_event.plot(self.time.get(),
                                          plot_temp=self.plot_temp.get(),
                                          plot_vel=self.plot_vel.get(),
                                          plot_grad=self.plot_grad.get(),
                                          temp_resolution=100,
                                          vel_resolution=30,
                                          gradtype=self.gradientType.get(),
                                          veltype=self.velocityType.get(),
                                          temptype=self.tempType.get(),
                                          numContours=self.contourNumber.get(),
                                          zoom=self.zoom.get())

            # Set moment display
            # !!!!!!!!!!!!! Currently Empty !!!!!!!!!!!!
            self.momentDisplay.set('...')
            self.ELDisplay.set('...')
            self.momentHRGDisplay.set('...')
            self.momentUnhydroDisplay.set('...')

            # Decide if you want to feed tempHRG to the integrand function to bring it to zero.
            if self.plotColors.get():
                decidedCut = 0
            else:
                decidedCut = self.tempHRG.get()

            # Set plot font hardcoded options
            plotFontSize = 8
            markSize = 2
            gridLineWidth = 1
            connectorLineStyle = '-'
            # connectorLineColor = 'black'  # Not set up

            #print(self.calculated.get())
            if self.calculated.get():

                # Plot jet trajectory
                # Select QGP figure as current figure
                plt.figure(self.plasmaFigure.number)
                # Plot initial trajectory
                d_x = np.cos(self.current_jet.phi_0) * self.current_jet.beta_0 * (0.5 * self.current_event.tf)
                d_y = np.sin(self.current_jet.phi_0) * self.current_jet.beta_0 * (0.5 * self.current_event.tf)
                self.plasmaAxis.arrow(self.current_jet.x_0, self.current_jet.y_0, d_x, d_y, color='white', width=0.15)
                # Get trajectory points
                time_array = self.jet_xarray['time'].to_numpy()
                xpos_array = self.jet_xarray['x'].to_numpy()
                ypos_array = self.jet_xarray['y'].to_numpy()
                q_drift_array = self.jet_xarray['q_drift'].to_numpy()
                q_EL_array = self.jet_xarray['q_EL'].to_numpy()
                q_grad_array = self.jet_xarray['q_grad'].to_numpy()
                q_fg_array = self.jet_xarray['q_fg'].to_numpy()
                int_drift_array = self.jet_xarray['int_drift'].to_numpy()
                int_EL_array = self.jet_xarray['int_EL'].to_numpy()
                int_grad_array = self.jet_xarray['int_grad'].to_numpy()
                pT_array = self.jet_xarray['pT'].to_numpy()
                temp_seen_array = self.jet_xarray['temp'].to_numpy()
                grad_perp_temp_array = self.jet_xarray['grad_perp_temp'].to_numpy()
                u_perp_array = self.jet_xarray['u_perp'].to_numpy()
                u_par_array = self.jet_xarray['u_par'].to_numpy()
                u_array = self.jet_xarray['u'].to_numpy()
                phase_array = self.jet_xarray['phase'].to_numpy()
                rpos_array = np.sqrt(xpos_array**2 + ypos_array**2)
                # Plot trajectory
                self.plasmaAxis.plot(xpos_array[::self.nth.get()], ypos_array[::self.nth.get()], marker=',',
                                     color='black', markersize=20)

                # Set moment display
                self.momentDisplay.set('Total F Drift: {} GeV'.format(np.sum(q_drift_array)))
                self.ELDisplay.set('Total EL: {} GeV'.format(np.sum(q_EL_array)))
                self.momentHRGDisplay.set('Total FG Drift: {} GeV'.format(np.sum(q_fg_array)))
                self.momentUnhydroDisplay.set('...')

                # Select medium properties figure as current figure
                plt.figure(self.propertyFigure.number)

                # Plot connector lines for properties
                self.propertyAxes[0, 0].plot(time_array, u_perp_array, ls=connectorLineStyle)
                self.propertyAxes[0, 1].plot(time_array, u_par_array, ls=connectorLineStyle)
                self.propertyAxes[1, 0].plot(time_array, temp_seen_array, ls=connectorLineStyle)
                self.propertyAxes[1, 1].plot(time_array, grad_perp_temp_array, ls=connectorLineStyle)
                self.propertyAxes[1, 2].plot(time_array, q_EL_array, ls=connectorLineStyle)
                self.propertyAxes[2, 1].plot(time_array, pT_array, ls=connectorLineStyle)
                self.propertyAxes[0, 2].plot(time_array, q_grad_array, ls=connectorLineStyle)
                self.propertyAxes[2, 2].plot(time_array, q_drift_array, ls=connectorLineStyle)
                self.propertyAxes[2, 0].plot(time_array, (u_perp_array / (1 - u_par_array)), ls=connectorLineStyle)
                self.propertyAxes[0, 3].plot(time_array, int_grad_array, ls=connectorLineStyle)
                self.propertyAxes[1, 3].plot(time_array, int_EL_array, ls=connectorLineStyle)
                self.propertyAxes[2, 3].plot(time_array, int_drift_array, ls=connectorLineStyle)

                if self.plotColors.get():
                    # Determine colors from temp seen by jet at each time.
                    color_array = np.array([])
                    for phase in phase_array:
                        if phase == 'qgp':
                            color_array = np.append(color_array, 'b')
                        elif phase == 'hrg':
                            color_array = np.append(color_array, 'g')
                        elif phase == 'unh':
                            color_array = np.append(color_array, 'r')
                        else:
                            color_array = np.append(color_array, 'black')

                    # Plot colored markers.
                    for i in range(0, len(time_array)):
                        self.propertyAxes[0, 0].plot(time_array[i], u_perp_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[0, 1].plot(time_array[i], u_par_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[1, 0].plot(time_array[i], temp_seen_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[1, 1].plot(time_array[i], grad_perp_temp_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[1, 2].plot(time_array[i], q_EL_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[2, 1].plot(time_array[i], pT_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[0, 2].plot(time_array[i], q_grad_array[i], 'o', color=color_array[i] , markersize=markSize)
                        self.propertyAxes[2, 2].plot(time_array[i], q_drift_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[2, 0].plot(time_array[i], (u_perp_array[i] / (1 - u_par_array[i])), 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[0, 3].plot(time_array[i], int_grad_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[1, 3].plot(time_array[i], int_EL_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[2, 3].plot(time_array[i], int_drift_array[i], 'o', color=color_array[i], markersize=markSize)

                        # # PERFORMANCE ISSUES:
                        # # Fill under the drift integrand curve for qgp phase
                        # self.propertyAxes[2, 3].fill_between(
                        #     x=time_array,
                        #     y1=int_drift_array,
                        #     where=(phase_array == 'qgp'),
                        #     color='b',
                        #     alpha=0.2)
                        # # Fill under the drift integrand curve for hydrodynamic hadron gas phase
                        # self.propertyAxes[2, 3].fill_between(
                        #     x=time_array,
                        #     y1=int_drift_array,
                        #     where=(phase_array == 'hrg'),
                        #     color='g',
                        #     alpha=0.2)
                        # # Fill under the drift integrand curve for unhydrodynamic phase
                        # self.propertyAxes[2, 3].fill_between(
                        #     x=time_array,
                        #     y1=int_drift_array,
                        #     where=(phase_array == 'unh'),
                        #     color='g',
                        #     alpha=0.2)
                        # # Fill under the EL integrand curve for qgp phase
                        # self.propertyAxes[1, 3].fill_between(
                        #     x=time_array,
                        #     y1=int_EL_array,
                        #     where=(phase_array == 'qgp'),
                        #     color='b',
                        #     alpha=0.2)
                        # # Fill under the EL integrand curve for hydrodynamic hadron gas phase
                        # self.propertyAxes[1, 3].fill_between(
                        #     x=time_array,
                        #     y1=int_EL_array,
                        #     where=(phase_array == 'hrg'),
                        #     color='g',
                        #     alpha=0.1)
                        # # Fill under the EL integrand curve for unhydrodynamic phase
                        # self.propertyAxes[1, 3].fill_between(
                        #     x=time_array,
                        #     y1=int_EL_array,
                        #     where=(phase_array == 'unh'),
                        #     color='g',
                        #     alpha=0.2)

            # Gridlines and ticks
            # Plot horizontal gridlines at y=0
            for axisList in self.propertyAxes:  # Medium property plots
                for axis in axisList:
                    axis.axhline(y=0, color='black', linestyle=':', lw=gridLineWidth)

            # Plot horizontal gridline at temp minTemp for temp plot
            self.propertyAxes[1, 0].axhline(y=self.tempHRG.get(), color='black', linestyle=':', lw=gridLineWidth)

            # Plot vertical gridline at current time from slider
            for axisList in self.propertyAxes:  # Iterate through medium property plots
                for axis in axisList:
                    axis.axvline(x=self.time.get(), color='black', ls=':', lw=gridLineWidth)

            # Plot tick at temp minTemp for temp plot
            self.propertyAxes[1, 0].set_yticks(list(self.propertyAxes[1, 0].get_yticks()) + [self.tempHRG.get()])

            # Plot property titles
            self.propertyAxes[0, 0].set_title(r"$u_\perp$", fontsize=plotFontSize)
            self.propertyAxes[0, 1].set_title(r"$u_\parallel$", fontsize=plotFontSize)
            self.propertyAxes[1, 0].set_title(r"$T$ (GeV)", fontsize=plotFontSize)
            self.propertyAxes[1, 1].set_title(r"$\nabla_{\perp} T$", fontsize=plotFontSize)
            self.propertyAxes[1, 2].set_title(r"$q_{EL}$", fontsize=plotFontSize)
            self.propertyAxes[2, 1].set_title(r"$p_T$", fontsize=plotFontSize)
            self.propertyAxes[0, 2].set_title(r"$q_{grad}$", fontsize=plotFontSize)
            self.propertyAxes[2, 2].set_title(r"$q_{drift}$", fontsize=plotFontSize)
            self.propertyAxes[2, 0].set_title(r"$u_\perp / (1-u_\parallel)$", fontsize=plotFontSize)
            self.propertyAxes[0, 3].set_title(r"Gradient Integrand", fontsize=plotFontSize)
            self.propertyAxes[1, 3].set_title(r"EL Integrand", fontsize=plotFontSize)
            self.propertyAxes[2, 3].set_title(r"Drift Integrand", fontsize=plotFontSize)



        # If you've got no file loaded, just redraw the canvas.
        else:
            print('Select a file!!!')
        self.canvas.draw()
        self.canvas1.draw()

    # Method to animate plasma evolution
    def animate_plasma(self):
        return None

    def run_jet(self, value=None):
        if self.file_selected:
            # Update jet object
            self.update_jet(0)

            # Calculate the jet trajectory
            print('Calculating jet trajectory...')
            # Run the time loop
            self.jet_dataframe, self.jet_xarray = timekeeper.time_loop(event=self.current_event,
                                                                       jet=self.current_jet,
                                                                       drift=self.drift.get(),
                                                                       el=self.el.get(),
                                                                       grad=self.grad.get(),
                                                                       fg=self.fg.get(),
                                                                       temp_hrg=self.tempHRG.get(),
                                                                       temp_unh=self.tempUnhydro.get())

            print('Jet trajectory complete.')
            self.calculated.set(True)
            # Update plots to set current jet and event business.
            self.update_plots()

        else:
            print('Select a file!!!')

    # Method to sample the event and return new jet production point
    def sample_event(self):
        if self.file_selected:
            # Sample T^6 dist. and get point
            sampledPoint = collision.generate_jet_point(self.current_event, 1)

            # Uniform sample an angle
            sampledAngle = float(np.random.uniform(0, 2*np.pi, 1))

            # Set sliders to point
            self.x0.set(sampledPoint[0])
            self.y0.set(sampledPoint[1])
            self.theta0.set(sampledAngle)

            # Update plots
            self.update_plots()
        else:
            print('Select a file!!!')



"""
# Unused second page with back button.
class PageOne(tk.Frame):

    def __init__(self, parent):
        # initialize frame inheritence
        tk.Frame.__init__(self, parent)

        # Define label and smash it in.
        label = tk.Label(self, text='Page 1', font=LARGE_FONT)
        label.grid()

        # Create button to change pages and smack it in
        backButton = ttk.Button(self, text='Back',
                                command=lambda: parent.show_frame(MainPage))
        backButton.grid()
"""







# Make the main application class
app = PlasmaInspector()

# Handle the window close
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        # Close matplotlib objects
        plt.close('all')
        # Close the application
        app.destroy()
        # Be polite. :)
        print('Have a lovely day!')

app.protocol("WM_DELETE_WINDOW", on_closing)

# Generate the window and let it hang out until closed
app.mainloop()


