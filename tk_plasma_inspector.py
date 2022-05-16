import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import numpy as np
import plasma
import plasma_interaction as pi
import hic as hs
import jets
import matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # , NavigationToolbar2Tk
# import matplotlib.animation as animation
# import matplotlib.colors as colors
from utilities import round_decimals_up, round_decimals_down

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
        self.jetE.set(100)

        # Integration options
        self.tempHRG = tk.DoubleVar()
        self.tempHRG.set(0.200)
        self.tempUnhydro = tk.DoubleVar()
        self.tempUnhydro.set(0.150)

        # Plotting options
        self.velocityType = tk.StringVar()
        self.velocityType.set('stream')
        self.contourNumber = tk.IntVar()
        self.contourNumber.set(15)
        self.tempType = tk.StringVar()
        self.tempType.set('contour')
        self.propPlotRes = tk.DoubleVar()
        self.propPlotRes.set(0.2)
        self.plotColors = tk.BooleanVar()
        self.plotColors.set(True)

        # Moment variables
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
        self.momentPlasmaDisplay = tk.StringVar()
        self.momentPlasmaDisplay.set(moment_label(moment=None, angleDeflection=None,
                                                  k=self.K, label='Plasma'))
        self.momentHRGDisplay = tk.StringVar()
        self.momentHRGDisplay.set(moment_label(moment=None, angleDeflection=None,
                                               k=self.K, label='HRG'))
        self.momentUnhydroDisplay = tk.StringVar()
        self.momentUnhydroDisplay.set(moment_label(moment=None, angleDeflection=None,
                                                   k=self.K, label='Unhydro'))

        ################
        # Plot Objects #
        ################

        # Create the QGP Plot that will dynamically update and set its labels
        self.plasmaFigure = plt.figure(num=0)
        self.plasmaAxis = self.plasmaFigure.add_subplot(1, 1, 1)

        # Define colorbar objects with "1" scalar mappable object so they can be manipulated.
        self.tempcb = self.plasmaFigure.colorbar(plt.cm.ScalarMappable(norm=None, cmap='rainbow'), ax=self.plasmaAxis)
        self.velcb = self.plasmaFigure.colorbar(plt.cm.ScalarMappable(norm=None, cmap='rainbow'), ax=self.plasmaAxis)

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
        self.momentPlasmaLabel = tk.Label(self, textvariable=self.momentPlasmaDisplay, font=LARGE_FONT)
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
        self.moment_button = ttk.Button(self, text='Moment',
                                        command=self.calc_moment)

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
                                   from_=10, to=100, length=200, resolution=0.1, label='jetE (GeV)')
        # Create tempHRG slider
        self.tempCutoffSlider = tk.Scale(self, orient=tk.HORIZONTAL,
                                         variable=self.tempHRG, from_=0, to=1, length=200, resolution=0.01,
                                         label='Had. Temp (GeV)')
        # Create tempHRG slider
        self.tempUnhydroSlider = tk.Scale(self, orient=tk.HORIZONTAL,
                                          variable=self.tempUnhydro, from_=0, to=1, length=200, resolution=0.01,
                                          label='Unhydro Temp (GeV)')


        # Register update ON RELEASE - use of command parameter applies action immediately
        #self.update_button.bind("<ButtonRelease-1>", self.update_plots)
        self.timeSlider.bind("<ButtonRelease-1>", self.update_plots)
        #self.x0Slider.bind("<ButtonRelease-1>", self.update_plots)
        #self.y0Slider.bind("<ButtonRelease-1>", self.update_plots)
        #self.theta0Slider.bind("<ButtonRelease-1>", self.update_plots)
        #self.jetESlider.bind("<ButtonRelease-1>", self.update_plots)
        #self.tempCutoffSlider.bind("<ButtonRelease-1>", self.update_plots)
        #self.tempUnhydroSlider.bind("<ButtonRelease-1>", self.update_plots)

        #########
        # Menus #
        #########
        # Create file menu cascade
        # ---
        fileMenu = tk.Menu(parent.menubar, tearoff=0)
        fileMenu.add_command(label='Select File', command=self.select_file)
        fileMenu.add_command(label='Exit', command=self.quit)
        parent.menubar.add_cascade(label='File', menu=fileMenu)
        # ---

        # Create plasma plot menu cascade
        # ---
        plasmaMenu = tk.Menu(parent.menubar, tearoff=0)
        # Temperature submenu
        tempMenu = tk.Menu(plasmaMenu, tearoff=0)
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
        tempTypeMenu.add_radiobutton(label='Contour', variable=self.tempType, value='contour',
                                    command=self.update_plots)
        tempTypeMenu.add_radiobutton(label='Density', variable=self.tempType, value='density',
                                    command=self.update_plots)
        tempMenu.add_cascade(label='Plot Type', menu=tempTypeMenu)
        plasmaMenu.add_cascade(label='Temperatures', menu=tempMenu)

        # Velocity submenu
        velMenu = tk.Menu(plasmaMenu, tearoff=0)
        velTypeMenu = tk.Menu(velMenu)
        velTypeMenu.add_radiobutton(label='Stream Velocities', variable=self.velocityType, value='stream',
                              command=self.update_plots)
        velTypeMenu.add_radiobutton(label='Quiver Velocities', variable=self.velocityType, value='quiver',
                              command=self.update_plots)
        velMenu.add_cascade(label='Plot Type', menu=velTypeMenu)
        plasmaMenu.add_cascade(label='Velocities', menu=velMenu)
        parent.menubar.add_cascade(label='Plasma Plot', menu=plasmaMenu)

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
        self.moment_button.grid(row=0, column=2)
        self.timeSlider.grid(row=1, column=0, columnspan=2)
        self.x0Slider.grid(row=1, column=2, columnspan=2)
        self.y0Slider.grid(row=1, column=4, columnspan=2)
        self.theta0Slider.grid(row=1, column=6, columnspan=2)
        self.jetESlider.grid(row=1, column=8, columnspan=2)
        self.tempCutoffSlider.grid(row=5, column=8, columnspan=2)
        self.tempUnhydroSlider.grid(row=6, column=8, columnspan=2)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=6, rowspan=3)
        self.canvas1.get_tk_widget().grid(row=2, column=6, columnspan=6, rowspan=3)
        self.momentLabel.grid(row=5, column=0, columnspan=4)
        self.momentPlasmaLabel.grid(row=6, column=0, columnspan=4)
        self.momentHRGLabel.grid(row=5, column=4, columnspan=4)
        self.momentUnhydroLabel.grid(row=6, column=4, columnspan=4)
        # buttonPage1.grid()  # Unused second page

    # Define the update function
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
        self.tempMin = self.current_event.max_temp()

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


    # Define the update function
    def update_plots(self, value=None):
        if self.file_selected:
            # Clear all the plots and colorbars
            self.plasmaAxis.clear()
            self.tempcb.remove()
            self.velcb.remove()

            for axisList in self.propertyAxes:  # Medium property plots
                for axis in axisList:
                    axis.clear()

            # Select QGP figure as current figure
            plt.figure(self.plasmaFigure.number)

            # Select QGP plot axis as current axis
            # plt.sca(self.plasmaAxis)

            # Plot new temperatures & velocities
            self.tempPlot, self.velPlot, self.tempcb, self.velcb = self.current_event.plot(self.time.get(),
                                                                  temp_resolution=100, vel_resolution=30,
                                                                  veltype=self.velocityType.get(),
                                                                  temptype=self.tempType.get(),
                                                                  numContours=self.contourNumber.get())

            # Set moment display to None
            self.momentDisplay.set(moment_label(moment=None, angleDeflection=None,
                                                k=self.K, label='Total'))
            self.momentPlasmaDisplay.set(moment_label(moment=None, angleDeflection=None,
                                                      k=self.K, label='Plasma'))
            self.momentHRGDisplay.set(moment_label(moment=None, angleDeflection=None,
                                                   k=self.K, label='HRG'))
            self.momentUnhydroDisplay.set(moment_label(moment=None, angleDeflection=None,
                                                       k=self.K, label='Unhydro'))

            # Set current_jet object to current slider parameters
            self.current_jet = jets.jet(x0=self.x0.get(), y0=self.y0.get(),
                                   theta0=self.theta0.get(), event=self.current_event, energy=self.jetE.get())

            timeRange = np.arange(self.current_event.t0, self.current_event.tf, self.propPlotRes.get())
            t = np.array([])
            for time in timeRange:
                if pi.time_cut(self.current_event, time) and pi.pos_cut(self.current_event, self.current_jet, time) \
                        and pi.temp_cut(self.current_event, self.current_jet, time):
                    t = np.append(t, time)
                else:
                    break

            # Plot jet trajectory
            # Find final time within position bounds
            jetTrajTime = np.array([])
            for time in t:
                if pi.pos_cut(self.current_event, self.current_jet, time):
                    jetTrajTime = np.append(jetTrajTime, time)


            jetInitialX = self.current_jet.xpos(jetTrajTime[0])
            jetInitialY = self.current_jet.ypos(jetTrajTime[0])
            jetFinalX = self.current_jet.xpos(jetTrajTime[-1])
            jetFinalY = self.current_jet.ypos(jetTrajTime[-1])
            self.plasmaAxis.plot([jetInitialX, jetFinalX], [jetInitialY, jetFinalY], ls=':', color='w')

            # Plot new jet position
            self.plasmaAxis.plot(self.current_jet.xpos(self.time.get()), self.current_jet.ypos(self.time.get()), 'ro')

            # Select medium properties figure as current figure
            plt.figure(self.propertyFigure.number)

            # Initialize empty arrays for the plot data
            uPerpArray = np.array([])
            uParArray = np.array([])
            tempArray = np.array([])
            velArray = np.array([])
            overLambdaArray = np.array([])
            iIntArray = np.array([])
            XArray = np.array([])
            YArray = np.array([])
            integrandArray = np.array([])

            # Decide if you want to feed tempHRG to the integrand function to bring it to zero.
            if self.plotColors.get():
                decidedCut = 0
            else:
                decidedCut = self.tempHRG.get()

            # Calculate plot data
            for time in t:
                uPerp = self.current_event.u_perp(jet=self.current_jet, time=time)
                uPar = self.current_event.u_par(jet=self.current_jet, time=time)
                temp = self.current_event.temp(self.current_jet.coords3(time=time))
                vel = self.current_event.vel(jet=self.current_jet, time=time)
                overLambda = self.current_event.rho(jet=self.current_jet, time=time) \
                             * self.current_event.sigma(jet=self.current_jet, time=time)
                iInt = self.current_event.i_int_factor(jet=self.current_jet, time=time)
                xPOS = self.current_jet.xpos(time)
                yPOS = self.current_jet.ypos(time)
                integrand = pi.integrand(event=self.current_event, jet=self.current_jet, k=self.K,
                                         minTemp=decidedCut)(time)

                uPerpArray = np.append(uPerpArray, uPerp)
                uParArray = np.append(uParArray, uPar)
                tempArray = np.append(tempArray, temp)
                velArray = np.append(velArray, vel)

                overLambdaArray = np.append(overLambdaArray, overLambda)

                integrandArray = np.append(integrandArray, integrand)

                iIntArray = np.append(iIntArray, iInt)

                XArray = np.append(XArray, xPOS)
                YArray = np.append(YArray, yPOS)

            # Set plot font hardcoded options
            plotFontSize = 8
            markSize = 2
            gridLineWidth = 1
            connectorLineStyle = '-'
            # connectorLineColor = 'black'  # Not set up

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

            if self.plotColors.get():
                # Determine colors from temp seen by jet at each time.
                colorArray, hydroHRGTimes, unhydroHRGTimes = self.tempColoringFunc(t)

                # Plot connector line
                self.propertyAxes[0, 0].plot(t, uPerpArray, ls=connectorLineStyle)
                self.propertyAxes[0, 1].plot(t, uParArray, ls=connectorLineStyle)
                self.propertyAxes[1, 0].plot(t, tempArray, ls=connectorLineStyle)
                self.propertyAxes[1, 1].plot(t, velArray, ls=connectorLineStyle)
                self.propertyAxes[2, 0].plot(t, (uPerpArray / (1 - uParArray)), ls=connectorLineStyle)
                self.propertyAxes[2, 1].plot(t, 1 / (5 * overLambdaArray), ls=connectorLineStyle)
                self.propertyAxes[1, 2].plot(t, 1 / (overLambdaArray), ls=connectorLineStyle)
                self.propertyAxes[2, 2].plot(t, 4 * tempArray ** 2, ls=connectorLineStyle)
                self.propertyAxes[0, 2].plot(t, iIntArray, ls=connectorLineStyle)
                self.propertyAxes[0, 3].plot(t, XArray, ls=connectorLineStyle)
                self.propertyAxes[1, 3].plot(t, YArray, ls=connectorLineStyle)
                self.propertyAxes[2, 3].plot(t, integrandArray, ls=connectorLineStyle)

                # Plot colored markers.
                for i in range(0,len(t)):
                    self.propertyAxes[0, 0].plot(t[i], uPerpArray[i], 'o', color=colorArray[i], markersize=markSize)
                    self.propertyAxes[0, 1].plot(t[i], uParArray[i], 'o', color=colorArray[i], markersize=markSize)
                    self.propertyAxes[1, 0].plot(t[i], tempArray[i], 'o', color=colorArray[i], markersize=markSize)
                    self.propertyAxes[1, 1].plot(t[i], velArray[i], 'o', color=colorArray[i], markersize=markSize)
                    self.propertyAxes[2, 0].plot(t[i], (uPerpArray[i] / (1 - uParArray[i])), 'o', color=colorArray[i]
                                                 , markersize=markSize)
                    self.propertyAxes[2, 1].plot(t[i], 1 / (5 * overLambdaArray[i]), 'o', color=colorArray[i]
                                                 , markersize=markSize)
                    self.propertyAxes[1, 2].plot(t[i], 1 / (overLambdaArray[i]), 'o', color=colorArray[i]
                                                 , markersize=markSize)
                    self.propertyAxes[2, 2].plot(t[i], 4 * tempArray[i] ** 2, 'o', color=colorArray[i]
                                                 , markersize=markSize)
                    self.propertyAxes[0, 2].plot(t[i], iIntArray[i], 'o', color=colorArray[i], markersize=markSize)
                    self.propertyAxes[0, 3].plot(t[i], XArray[i], 'o', color=colorArray[i], markersize=markSize)
                    self.propertyAxes[1, 3].plot(t[i], YArray[i], 'o', color=colorArray[i], markersize=markSize)
                    self.propertyAxes[2, 3].plot(t[i], integrandArray[i], 'o', color=colorArray[i], markersize=markSize)

                if len(hydroHRGTimes) > 0 and len(unhydroHRGTimes) > 0:
                    # Fill under the curve for hydrodynamic hadron gas phase
                    self.propertyAxes[2, 3].fill_between(
                        x=t,
                        y1=integrandArray,
                        where=hydroHRGTimes[0] < t,
                        color='g',
                        alpha=0.2)
                    # Fill under the curve for unhydrodynamic hadron gas phase
                    self.propertyAxes[2, 3].fill_between(
                        x=t,
                        y1=integrandArray,
                        where=unhydroHRGTimes[0] < t,
                        color='r',
                        alpha=0.2)
                elif len(hydroHRGTimes) > 0 and len(unhydroHRGTimes) == 0:
                    # Fill under the curve for hydrodynamic hadron gas phase
                    self.propertyAxes[2, 3].fill_between(
                        x=t,
                        y1=integrandArray,
                        where= t > hydroHRGTimes[0],
                        color='g',
                        alpha=0.2)
                elif len(hydroHRGTimes) == 0 and len(unhydroHRGTimes) > 0:
                    # Fill under the curve for unhydrodynamic hadron gas phase
                    self.propertyAxes[2, 3].fill_between(
                        x=t,
                        y1=integrandArray,
                        where=unhydroHRGTimes[0] < t,
                        color='r',
                        alpha=0.2)


            else:
                # Plot usual plots based on style sheet
                self.propertyAxes[0, 0].plot(t, uPerpArray)
                self.propertyAxes[0, 1].plot(t, uParArray)
                self.propertyAxes[1, 0].plot(t, tempArray)
                self.propertyAxes[1, 1].plot(t, velArray)
                self.propertyAxes[2, 0].plot(t, (uPerpArray / (1 - uParArray)))
                self.propertyAxes[2, 1].plot(t, 1 / (5 * overLambdaArray))
                self.propertyAxes[1, 2].plot(t, 1 / (overLambdaArray))
                self.propertyAxes[2, 2].plot(t, 4 * tempArray ** 2)
                self.propertyAxes[0, 2].plot(t, iIntArray)
                self.propertyAxes[0, 3].plot(t, XArray)
                self.propertyAxes[1, 3].plot(t, YArray)
                self.propertyAxes[2, 3].plot(t, integrandArray)

            self.propertyAxes[0, 0].set_title("u_perp", fontsize=plotFontSize)
            self.propertyAxes[0, 1].set_title("u_par", fontsize=plotFontSize)
            self.propertyAxes[1, 0].set_title("T (GeV)", fontsize=plotFontSize)
            self.propertyAxes[1, 1].set_title("|u|", fontsize=plotFontSize)
            self.propertyAxes[2, 0].set_title("prp / (1-par)", fontsize=plotFontSize)
            self.propertyAxes[2, 1].set_title("1/Lmda (fm)", fontsize=plotFontSize)
            self.propertyAxes[1, 2].set_title("1/Lmda (GeV^-1)", fontsize=plotFontSize)
            self.propertyAxes[2, 2].set_title("mu^2 (GeV^2)", fontsize=plotFontSize)
            self.propertyAxes[0, 2].set_title("I(k) Factor", fontsize=plotFontSize)
            self.propertyAxes[0, 3].set_title("X Pos", fontsize=plotFontSize)
            self.propertyAxes[1, 3].set_title("Y Pos", fontsize=plotFontSize)
            self.propertyAxes[2, 3].set_title("Integrand", fontsize=plotFontSize)



        # If you've got no file loaded, just redraw the canvas.
        else:
            print('Select a file!!!')
        self.canvas.draw()
        self.canvas1.draw()

    # Method to animate plasma evolution
    def animate_plasma(self):
        return None

    def calc_moment(self):
        if self.file_selected:
            # Update plots to set current jet and event business.
            self.update_plots()

            # Calculate plasma moment
            momentPlasmaRaw = pi.moment_integral(self.current_event, self.current_jet, minTemp=self.tempHRG.get())

            self.angleDeflectionPlasma = np.arctan((momentPlasmaRaw[0] / self.current_jet.energy)) * (180 / np.pi)
            self.momentPlasma = momentPlasmaRaw[0]
            self.momentPlasmaDisplay.set(moment_label(moment=self.momentPlasma,
                                                      angleDeflection=self.angleDeflectionPlasma,
                                                      k=self.K, label='plasma'))

            # Calculate hadron gas (HRG) moment
            momentHRGRaw = pi.moment_integral(self.current_event, self.current_jet, minTemp=self.tempUnhydro.get(),
                                              maxTemp=self.tempHRG.get())

            self.angleDeflectionHRG = np.arctan((momentHRGRaw[0] / self.current_jet.energy)) * (180 / np.pi)
            self.momentHRG = momentHRGRaw[0]
            self.momentHRGDisplay.set(moment_label(moment=self.momentHRG, angleDeflection=self.angleDeflectionHRG,
                                                   k=self.K, label='HRG'))

            # Calculate unhydro hadron gas (HRG) moment
            momentUnhydroRaw = pi.moment_integral(self.current_event, self.current_jet, maxTemp=self.tempUnhydro.get(),
                                                  minTemp=0)

            self.angleDeflectionUnhydro = np.arctan((momentUnhydroRaw[0] / self.current_jet.energy)) * (180 / np.pi)
            self.momentUnhydro = momentUnhydroRaw[0]
            self.momentUnhydroDisplay.set(moment_label(moment=self.momentUnhydro,
                                                       angleDeflection=self.angleDeflectionUnhydro,
                                                       k=self.K, label='Unhydro'))

            # ???
            # momentRaw = pi.moment_integral(self.current_event, self.current_jet, minTemp=self.tempHRG.get())

            self.angleDeflection = self.angleDeflectionHRG + self.angleDeflectionPlasma + self.angleDeflectionUnhydro
            self.moment = self.momentHRG + self.momentPlasma + self.momentUnhydro
            self.momentDisplay.set(moment_label(moment=self.moment, angleDeflection=self.angleDeflection,
                                                k=self.K, label='Total'))
        else:
            print('Select a file!!!')

    # Method to sample the event and return new jet production point
    def sample_event(self):
        if self.file_selected:
            # Sample T^6 dist. and get point
            sampledPoint = hs.generate_jet_point(self.current_event, 1)

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


