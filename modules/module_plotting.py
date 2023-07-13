import numpy as np
from ash.functions.functions_general import print_line_with_mainheader,print_line_with_subheader1,ashexit

#repeated here so that plotting can be stand-alone
class BC:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    OKMAGENTA= '\033[95m'
    OKRED= '\033[31m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_matplotlib():
    print("Trying to load Matplotlib")
    global matplotlib
    try:
        import matplotlib
    except:
        print("Loading MatplotLib failed. Probably not installed. Please install using conda: conda install matplotlib or pip: pip install matplotlib")
        return None
        #ashexit()
    print("Matplotlib loaded")
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
    return plt

def Gaussian(x, mu, strength, sigma):
    "Produces a Gaussian curve"
    bandshape = (strength)  * np.exp(-1*((x-mu))**2/(2*(sigma**2)))
    return bandshape

def Lorentz(x, x0, strength, gamma):
    "Produces a Lorentz lineshape"
    lineshape = (strength / (1 + ((x - x0) / gamma)**2))
    return lineshape

def Voigt(x, x0, strength,sigma, gamma):
    "Produces a Voigt lineshape with approximately matching height to Gaussian"
    from scipy.special import voigt_profile
    # Calculate the Gaussian height
    gaussian_height = strength * np.sqrt(2 * np.pi * sigma**2)
    # Calculate the Voigt profile using scipy's voigt_profile function
    lineshape = 2*gaussian_height * voigt_profile(x - x0, sigma, gamma)
    return lineshape

class ASH_plot():
    def __init__(self, figuretitle='Plottyplot', num_subplots=1, dpi=200, imageformat='png', figsize=(9,5),
        x_axislabel='X-axis', y_axislabel='Energy (X)', x_axislabels=None, y_axislabels=None, title='Plot-title', 
        subplot_titles=None, xlimit=None, ylimit=None,
        legend_pos=None, horizontal=False, tight_layout=True, padding=None):
        print_line_with_mainheader("ASH_energy_plot")

        plt = load_matplotlib() #Load Matplotlib
        if plt == None:
            print("Matplotlib failed to load. Exiting ASH_plot")
            #Making working attribute False so that we can check if ASH_plot instance is useful or not
            self.working=False
            return None
        #If matplotlib loaded then all is good
        self.working=True
        self.num_subplots=num_subplots
        self.imageformat=imageformat
        self.dpi=dpi
        self.figuretitle=figuretitle
        self.subplot_titles=subplot_titles
        self.x_axislabel=x_axislabel
        self.y_axislabel=y_axislabel
        #For multi-subplots
        self.x_axislabels=x_axislabels
        self.y_axislabels=y_axislabels
        #Legend position
        self.legend_pos=legend_pos

        print("Subplots:", self.num_subplots)
        print("Figure size:", figsize)


        if self.num_subplots > 1:
            print(BC.WARNING, "Note: For multiple subplots use:\n ASH_plot(x_axislabels=['X1','X2','X3], y_axislabels=['Y1','Y2','Y3'], subplot_titles='Title1,'Title2','Title3']", BC.END)
        else:
            print("X-axis label:", x_axislabel)
            print("Y-axis label:", y_axislabel)
            print("Title:", title)



        if self.num_subplots == 1:
            self.fig, ax = matplotlib.pyplot.subplots(figsize=figsize)
            self.axs=[ax]
            self.x_axislabels=x_axislabels
            self.y_axislabels=y_axislabels

            #X-limit and y-limit
            if xlimit != None:
                self.axs[0].set_xlim(xlimit[0], xlimit[1])
            if ylimit != None:
                self.axs[0].set_ylim(ylimit[0], ylimit[1])

        elif self.num_subplots == 2:
            if horizontal is True:
                print("Horizontal plot is true")
                self.fig, self.axs = matplotlib.pyplot.subplots(1, 2, figsize=figsize)
                if tight_layout is True:
                    print("Tight layout True")
                    self.fig.tight_layout()
                #Subplot padding
                if padding is not None:
                    self.fig.subplots_adjust(wspace=padding)

            else:
                self.fig, self.axs = matplotlib.pyplot.subplots(2, 1, figsize=figsize)
                if tight_layout is True:
                    print("Tight layout True")
                    self.fig.tight_layout()
                #Subplot padding
                if padding is not None:
                    self.fig.subplots_adjust(hspace=padding)

            self.axiscount=0
            
            #X-limit and y-limit
            #TODO: Allow different limits for each subplot
            if xlimit != None:
                self.axs[0].set_xlim(xlimit[0], xlimit[1])
                self.axs[1].set_xlim(xlimit[0], xlimit[1])
            if ylimit != None:
                self.axs[0].set_ylim(ylimit[0], ylimit[1])
                self.axs[1].set_ylim(ylimit[0], ylimit[1])



        elif self.num_subplots == 3:
            self.plotlistnames=['upleft','upright','low']
            self.fig, axs_dict = matplotlib.pyplot.subplot_mosaic([['upleft', 'upright'],
                               ['low', 'low']])
            self.axs=[axs_dict['upleft'],axs_dict['upright'],axs_dict['low']]
            self.axiscount=0
        elif self.num_subplots == 4:
            self.fig, axs = matplotlib.pyplot.subplots(2, 2, figsize=figsize)  # a figure with a 2x2 grid of Axes
            self.axs=[axs[0][0],axs[0][1], axs[1][0], axs[1][1]]
            self.axiscount=0

        self.addplotcount=0
    def invert_x_axis(self,subplot):
        self.axs[subplot].invert_xaxis()
    def invert_y_axis(self,subplot):
        self.axs[subplot].invert_yaxis()
    def addseries(self,subplot, surfacedictionary=None, x_list=None, y_list=None, x_labels=None, label='Series', color='blue', pointsize=40, 
                    scatter=True, line=True, scatter_linewidth=2, line_linewidth=1, marker='o', legend=True, x_scaling=1.0,y_scaling=1.0,
                    xticklabelrotation=80, x_scale_log=False, y_scale_log=False):
        print("Adding new series to ASH_plot object")
        self.addplotcount+=1
        curraxes=self.axs[subplot]
        
        #Using x_list and y_list unless not provided
        if surfacedictionary == None:
            #If Python lists
            if (type(x_list) != list and type(x_list) != np.ndarray) or ((type(y_list) != list and type(y_list) != np.ndarray)):
                print(BC.FAIL,"Please provide either a valid x_list and y_list (can be Python lists or Numpy arrays) or a surfacedictionary (Python dict)", BC.END)
                print(f"x_list: {x_list}")
                print(f"y_list: {y_list}")
                ashexit()
            else:
                x=list(x_list);y=list(y_list)

        #Alernative dictionary option
        if surfacedictionary != None:
            print("Using provided surfacedictionary")
            x=[];y=[]
            #Sorting keys dictionary before grabbing so that line-plot is correct
            for key in sorted(surfacedictionary.keys()):
                x.append(key)
                y.append(surfacedictionary[key])

        #Optional scaling of x or y-values
        x = [i*x_scaling for i in x]
        y = [i*y_scaling for i in y]

        #Scatterplot
        if scatter is True:
            print("x:", x)
            print("y:", y)
            curraxes.scatter(x,y, color=color, marker = marker,  s=pointsize, linewidth=scatter_linewidth, label=label)
        #Lineplot
        if line is True:
            #Avoid legend for line if scatter is enabled
            if scatter is True:
                label='_nolegend_'
            else:
                legendlabel=label
            curraxes.plot(x, y, linestyle='-', color=color, linewidth=line_linewidth, label=label)
        
        #Add labels to x-axis if
        if x_labels is not None:
            print("Adding xticks labels using rotation parameter:", xticklabelrotation)
            #curraxes.xticks(x,x_labels)
            curraxes.set_xticks(x, minor=False)
            curraxes.set_xticklabels(x_labels, fontdict=None, minor=False, rotation=xticklabelrotation)
        #Log scale
        if x_scale_log is True:
            curraxes.set_xscale('log')
        if y_scale_log is True:
            curraxes.set_yscale('log')

        #Title/axis options for 1 vs multiple subplots
        if self.num_subplots == 1:
            curraxes.set_xlabel(self.x_axislabel)  # Add an x-label to the axes.
            curraxes.set_ylabel(self.y_axislabel)  # Add a y-label to the axes.
            curraxes.set_title(self.figuretitle)  # Add a title to the axes if provided
        else:
            if self.x_axislabels == None:
                print(BC.FAIL, "For multiple subplots, x_axislabels and y_axislabels must be set.", BC.END)
                ashexit()
            curraxes.set_xlabel(self.x_axislabels[subplot])  # Add an x-label to the axes.
            curraxes.set_ylabel(self.y_axislabels[subplot])  # Add a y-label to the axes.
            if self.subplot_titles != None:
                curraxes.set_title(self.subplot_titles[subplot])  # Add a title to the axes if provided
        if legend is True:
            curraxes.legend(shadow=True, fontsize='small')  # Add a legend.
    #def showplot(self):
    #NOTE: Disabled until we support more backends
    #    matplotlib.pyplot.show()
    def savefig(self, filename, imageformat=None, dpi=None):

        #Change legend position
        #https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
        if self.num_subplots == 1:
            if self.legend_pos != None:
                self.axs[0].legend(loc='center left', bbox_to_anchor=(self.legend_pos[0], self.legend_pos[1]))

        if imageformat == None:
            imageformat = self.imageformat
        if dpi == None:
            dpi = self.dpi
        file=filename+'.'+imageformat
        print("\nSaving plot to file: {} with resolution: {} ".format(file,dpi))
        matplotlib.pyplot.savefig(file, format=imageformat, dpi=self.dpi, bbox_inches = "tight")

#Simple reactionprofile_plot function
#Input: dictionary of (X,Y): energy   entries
#NOTE: Partially deprecated thanks to ASHplot. Relative energy option is useful though. 
#TODO: Keep but call ASHplot here instead of doing separate plotting 
def reactionprofile_plot(surfacedictionary, finalunit='',label='Label', x_axislabel='Coord', y_axislabel='Energy', dpi=200, mode='pyplot',
                         imageformat='png', RelativeEnergy=True, pointsize=40, scatter_linewidth=2, line_linewidth=1, color='blue',
                        filename='Plot'):

    print_line_with_mainheader("reactionprofile_plot")

    plt = load_matplotlib()

    conversionfactor = { 'kcal/mol' : 627.50946900, 'kcal/mol' : 627.50946900, 'kJ/mol' : 2625.499638, 'kJpermol' : 2625.499638, 
                        'eV' : 27.211386245988, 'cm-1' : 219474.6313702 }
    e=[]
    coords=[]

    print("surfacedictionary:", surfacedictionary)

    #Sorting keys dictionary before grabbing so that line-plot is correct
    for key in sorted(surfacedictionary.keys()):
        coords.append(key)
        e.append(surfacedictionary[key])

    if RelativeEnergy is True:
        #List of energies and relenergies here
        refenergy=float(min(e))
        rele=[]
        for numb in e:
            rele.append((numb-refenergy)*conversionfactor[finalunit])
        finalvalues=rele
    else:
        finalvalues=e
    print("Coords:", coords)
    print("Relative energies({}): {}".format(finalunit,finalvalues))
    
    if mode == 'pyplot':
        plt.close() #Clear memory of previous plots
        plt.scatter(coords, finalvalues, color=color, marker = 'o',  s=pointsize, linewidth=scatter_linewidth )
        plt.plot(coords, finalvalues, linestyle='-', color=color, linewidth=line_linewidth, label=label)

        plt.title(label)
        plt.xlabel(x_axislabel)
        plt.ylabel('{} ({})'.format(y_axislabel,finalunit))
        plt.savefig(f'{filename}.{imageformat}', format=imageformat, dpi=dpi)
        plt.legend(shadow=True, fontsize='small')
        print(f"Created file: {filename}.{imageformat}")
    else:
        #OO style
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        ax.set_xlabel(x_axislabel)  # Add an x-label to the axes.
        ax.set_ylabel('{} ({})'.format(y_axislabel,finalunit))  # Add a y-label to the axes.
        ax.set_title(label)  # Add a title to the axes.
        ax.legend(shadow=True, fontsize='small');  # Add a legend.


#contourplot
#Input: dictionary of (X,Y): energy   entries 
# Can also be other property than energy. Use RelativeEnergy=False
#Good colormaps: viridis, viridis_r, inferno, inferno_r, plasma, plasma_r, magma, magma_r
# Less recommended: jet, jet_r
def contourplot(surfacedictionary, label='Label',x_axislabel='Coord', y_axislabel='Coord', finalunit='kcal/mol', interpolation='Cubic', 
                interpolparameter=10, colormap='inferno_r', dpi=200, imageformat='png', RelativeEnergy=True, numcontourlines=500,
                contour_alpha=0.75, contourline_color='black', clinelabels=False, contour_values=None, title=""):
    print_line_with_mainheader("contourplot")
    #Relative energy conversion (if RelativeEnergy is True)
    conversionfactor = { 'kcal/mol' : 627.50946900, 'kcal/mol' : 627.50946900, 'kJ/mol' : 2625.499638, 'kJpermol' : 2625.499638, 
                        'eV' : 27.211386245988, 'cm-1' : 219474.6313702 }
    e=[]
    coords=[]
    x_c=[]
    y_c=[]
    print("")
    print("Read surfacedictionary:", surfacedictionary)
    print("Number of entries:", len(surfacedictionary))
    for i in surfacedictionary:
        coords.append(i)
        x_c.append(i[0])
        y_c.append(i[1])
        e.append(surfacedictionary[i])
    
    #X and Y ranges
    x = sorted(set(x_c))
    y = sorted(set(y_c))

    relsurfacedictionary={}
    #Creating relative-energy array here. Unmodified property is used if False
    if RelativeEnergy is True:
        print("RelativeEnergy option. Using finalunit:", finalunit)
        refenergy=float(min(e))
        relsurfacedictionary={}
        for i in surfacedictionary:
            relsurfacedictionary[(i[0],i[1])] = (surfacedictionary[i]-refenergy)*conversionfactor[finalunit]
        print("Relative energy surfacedictionary ({}): {}".format(finalunit,relsurfacedictionary))
        print("")
        rele=[]
        for numb in e:
            rele.append((numb-refenergy)*conversionfactor[finalunit])

        #Creating structured array
        st=[]
        for ind in range(0,len(x_c)):
            st.append((x_c[ind], y_c[ind], rele[ind]))

        rele_new=[]
        curr=[]

        for yitem in y:
            for xitem in x:
                for nz in range(0,len(st)):
                    if (xitem,yitem) == st[nz][0:2]:
                        curr.append(st[nz][2])
            rele_new.append(curr)
            curr=[]
        Z = rele_new
    else:
        #Creating structured array
        st=[]
        for ind in range(0,len(x_c)):
            st.append((x_c[ind], y_c[ind], e[ind]))

        e_new=[]
        curr=[]

        for yitem in y:
            for xitem in x:
                for nz in range(0,len(st)):
                    if (xitem,yitem) == st[nz][0:2]:
                        curr.append(st[nz][2])

            e_new.append(curr)
            curr=[]
        Z = e_new
    #Now we create contour plot

    X, Y = np.meshgrid(x, y)


    print("Interpolation:", interpolation)
    print("Number of contour lines:", numcontourlines)
    print("Contourf alpha parameter:", contour_alpha)
    print("Colormap:", colormap)
    print("Contour line color:", contourline_color)
    if interpolation is not None:
        print("Using cubic interpolation")
        try:
            from scipy.ndimage import zoom
        except:
            print("Problem importing scipy.ndimage. Make sure scipy is installed.")
            print("Cubic interpolation not possible. Switching off")
            interpolation=None
    if interpolation == 'Cubic':
        print("Using cubic power:", interpolparameter)
        #Cubic interpolation. Default power is 10 (should be generally good)
        pw = interpolparameter #power of the smoothing function
        X = zoom(X, pw, mode='nearest')
        #print(X[0])
        if X[0][-1] == 0.0:
            print(X[0])
            #Zero value sometimes when cubic power is 10 or larger?
            #Related to bug or ill-defined behaviour of zoom. mode='nearest' seems to solve issue
            #Probably related to what happens when interpolated value is right on edge of range. If it falls right on then a zero may get added.
            # mode='nearest' seems to give the value instead
            print("problem. zero value exiting")
            ashexit()
            
        Y = zoom(Y, pw, mode='nearest')
        Z = zoom(Z, pw, mode='nearest')
    #Filled contours. 
    print("Using {} numcontourlines for colormap".format(numcontourlines))
    

    #Loading matplotlib
    print("Loading Matplotlib")
    plt = load_matplotlib()


    #Clearing plt object in case previous plot
    plt.clf()
    
    #Fille contourplot
    contour_surface=plt.contourf(X, Y, Z, numcontourlines, alpha=contour_alpha, cmap=colormap)
    
    #Contour lines. numcontourlines is 50 by default 
    #Or if contourvalues provided then should be ascending list
    if contour_values is not None:
        print("Using set contour_values for contourlines:", contour_values)
        Clines = plt.contour(X, Y, Z, contour_values, colors=contourline_color)
    else:
        #By default using value/10 as used for colormap
        print("Using {} numcontourlines ".format(numcontourlines/10))
        Clines = plt.contour(X, Y, Z, int(numcontourlines/10), colors=contourline_color)
    
    # Contour-line labels
    if clinelabels is True: 
        plt.clabel(Clines, inline=True, fontsize=10)
    
    plt.title(title)
    plt.colorbar(contour_surface)
    plt.xlabel(x_axislabel)
    plt.ylabel(y_axislabel)
    plt.savefig('Surface{}.{}'.format(label,imageformat), format=imageformat, dpi=dpi)
    print("Created PNG file: Surface{}.{}".format(label,imageformat))
    
    
#plot_Spectrum reads stick-values (e.g. absorption energie or IPs) and intensities, broadens spectrum (writes out dat and stk files) and then creates image-file using Matplotlib.
#Default Gaussian broadening, Lorentz and Gaussian-Lorentzian (Voigt) also possible.
def plot_Spectrum(xvalues=None, yvalues=None, plotname='Spectrum', range=None, unit='eV', broadening=0.1, points=10000, 
    imageformat='png', dpi=200, matplotlib=True, CSV=True, color='blue', plot_sticks=True, lineshape='Gaussian', voigt_broadening=None):
    
    print_line_with_mainheader("plot_Spectrum")
    if xvalues is None or yvalues is None or range is None:
        print("plot_Spectrum requires xvalues, yvalues and range variables")
        ashexit()
    assert len(xvalues) == len(yvalues), "List of yvalues not same size as list of xvalues." 

    start=range[0]
    finish=range[1]

    print("")
    print(BC.OKGREEN,"-------------------------------------------------------------------",BC.END)
    print(BC.OKGREEN,"plot_Spectrum: Plotting broadened spectrum",BC.END)
    print(BC.OKGREEN,"-------------------------------------------------------------------",BC.END)
    print("")
    print("xvalues ({}): {}".format(len(xvalues),xvalues))
    print("yvalues ({}): {}".format(len(yvalues),yvalues))
    if len(xvalues) == 0:
        print("X-values list length zero. Exiting.")
        ashexit()

    #########################
    # Broaden spectra
    ########################
    print(BC.OKGREEN, "Plotting-range chosen:", start, "-", finish, unit, "with ", points, "points and ",
            broadening, "{} broadening.".format(unit), BC.END)

    # X-range
    x = np.linspace(start, finish, points)

    #Lineshape function
    if lineshape == 'Gaussian':
        print("Gaussian lineshape chosen")
        print("Setting sigma equal to broadening value:", broadening)
        sigma=broadening
    elif lineshape == 'Lorentz':
        print("Lorentz lineshape chosen")
        print("Setting gamma equal to broadening value:", broadening)
        gamma=broadening
    elif lineshape == 'Voigt':
        print("Voigt lineshape chosen")
        if voigt_broadening is None:
            print("voigt_broadening (list of sigma and gamma values) not provided. Using broadening value for both sigma and gamma")
            sigma=broadening
            gamma=broadening
            print(f"Setting sigma={sigma} and gamma={gamma}")
        else:
            print("voigt_broadening values provided")
            sigma=voigt_broadening[0]
            gamma=voigt_broadening[1]
            print(f"Setting sigma={sigma} and gamma={gamma}")

    spectrum = 0
    for peak, strength in zip(xvalues, yvalues):
        if lineshape == 'Gaussian':
            broadenedpeak = Gaussian(x, peak, strength, sigma)
        elif lineshape == 'Lorentz':
            broadenedpeak = Lorentz(x, peak, strength, gamma)
        elif lineshape == 'Voigt':
            broadenedpeak = Voigt(x, peak, strength, sigma,gamma)
        spectrum += broadenedpeak

    #Save dat file
    with open(plotname+".dat", 'w') as tdatfile:
        for i,j in zip(x,spectrum):
            if CSV is True:
                tdatfile.write("{:13.10f}, {:13.10f} \n".format(i,j))            
            else:
                tdatfile.write("{:13.10f} {:13.10f} \n".format(i,j))
    #Save stk file
    with open(plotname+".stk", 'w') as tstkfile:
        for b,c in zip(xvalues,yvalues):
            if CSV is True:
                tstkfile.write("{:13.10f}, {:13.10f} \n".format(b,c))                
            else:
                tstkfile.write("{:13.10f} {:13.10f} \n".format(b,c))

    print("Wrote file:", plotname+".dat")
    print("Wrote file:", plotname+".stk")
    
    ##################################
    # Plot with Matplotlib
    ####################################
    if matplotlib is True:
        print("Creating plot with Matplotlib")
        plt = load_matplotlib()
        fig, ax = plt.subplots()

        ax.plot(x, spectrum, 'C3', label=plotname, color=color)
        if plot_sticks is True:
            ax.stem(xvalues, yvalues, label=plotname, markerfmt=' ', basefmt=' ', linefmt=color, use_line_collection=True)
        plt.xlabel(unit)
        plt.ylabel('Intensity')
        #################################
        plt.xlim(start, finish)
        #plt.legend(shadow=True, fontsize='small')
        #label='_nolegend_'
        plt.savefig(plotname + '.'+imageformat, format=imageformat, dpi=dpi)
        # plt.show()
        print("Wrote file:", plotname+"."+imageformat)
    else:
        print("Skipped Matplotlib part.")
    print(BC.OKGREEN,"ALL DONE!", BC.END)


#Note: Input: dict containing occ/unoccuped alpha/beta orbitals . Created by MolecularOrbitagrab in ORCA interface
#MOdict= {"occ_alpha":bands_alpha, "occ_beta":bands_alpha, "unocc_alpha":virtbands_a, "unocc_beta":virtbands_b, "Openshell":Openshell}
def MOplot_vertical(mos_dict, pointsize=4000, linewidth=2, label="Label", yrange=[-30,3], imageformat='png'):
    print_line_with_mainheader("MOplot_vertical")

    plt = load_matplotlib()

    bands_alpha=mos_dict["occ_alpha"]
    bands_beta=mos_dict["occ_beta"]
    virtbands_alpha=mos_dict["unocc_alpha"]
    virtbands_beta=mos_dict["unocc_beta"]
    Openshell=mos_dict["Openshell"]
    fig, ax = plt.subplots()

    #Alpha MOs
    ax.scatter([1]*len(bands_alpha), bands_alpha, color='blue', marker = '_',  s=pointsize, linewidth=linewidth )
    ax.scatter([1]*len(virtbands_alpha), virtbands_alpha, color='cyan', marker = '_',  s=pointsize, linewidth=linewidth )

    #Beta MOs
    if Openshell == True:
        ax.scatter([1.05]*len(bands_beta), bands_beta, color='red', marker = '_',  s=pointsize, linewidth=linewidth )
        ax.scatter([1.05]*len(virtbands_beta), virtbands_beta, color='pink', marker = '_',  s=pointsize, linewidth=linewidth )

    plt.xlim(0.98,1.07)
    plt.ylim(yrange[0],yrange[1])
    plt.xticks([])
    plt.ylabel('MO energy (eV)')
    plt.title(label)

    #Vertical line
    plt.axhline(y=0.0, color='black', linestyle='--')
    plt.savefig(label+"."+imageformat, format=imageformat, dpi=200)

    print("Created plot:", label+"."+imageformat)

