import numpy as np
import matplotlib
#from functions_general import *
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

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

def Gaussian(x, mu, strength, sigma):
    "Produces a Gaussian curve"
    #print("x:", x,)
    #print( "mu:", mu, "strength:", strength, "sigma:", sigma)
    #print(strength / (sigma*np.sqrt(2*np.pi)))
    #bandshape = (strength / (sigma*np.sqrt(2*np.pi)))  * np.exp(-1*((x-mu))**2/(2*(sigma**2)))
    bandshape = (strength)  * np.exp(-1*((x-mu))**2/(2*(sigma**2)))
    return bandshape


#reactionprofile_plot
#Input: dictionary of (X,Y): energy   entries 
def reactionprofile_plot(surfacedictionary, finalunit='',label='Label', x_axislabel='Coord', y_axislabel='Energy', dpi=200, 
                         imageformat='png', RelativeEnergy=True, pointsize=40, scatter_linewidth=2, line_linewidth=1, color='blue' ):
    
    
    conversionfactor = { 'kcal/mol' : 627.50946900, 'kcalpermol' : 627.50946900, 'kJ/mol' : 2625.499638, 'kJpermol' : 2625.499638, 
                        'eV' : 27.211386245988, 'cm-1' : 219474.6313702 }
    e=[]
    coords=[]
    for i in surfacedictionary:
        coords.append(i)
        e.append(surfacedictionary[i])
    
    if RelativeEnergy is True:
        #List of energies and relenergies here
        refenergy=float(min(e))
        rele=[]
        for numb in e:
            rele.append((numb-refenergy)*conversionfactor[finalunit])
        finalvalues=rele
    else:
        finalvalues=e
    
    
    plt.scatter(coords, finalvalues, color=color, marker = 'o',  s=pointsize, linewidth=scatter_linewidth )
    plt.plot(coords, finalvalues, linestyle='-', color=color, linewidth=line_linewidth)

    plt.title(label)
    plt.xlabel(x_axislabel)
    plt.ylabel('{} ({})'.format(y_axislabel,finalunit))
    plt.savefig('Plot{}.png'.format(label), format=imageformat, dpi=dpi)
    print("Created PNG file: Plot{}.png".format(label))



#contourplot
#Input: dictionary of (X,Y): energy   entries 
# Can also be other property than energy. Use RelativeEnergy=False
#Good colormaps: viridis, viridis_r, inferno, inferno_r, plasma, plasma_r, magma, magma_r
# Less recommended: jet, jet_r
def contourplot(surfacedictionary, label='Label',x_axislabel='Coord', y_axislabel='Coord', finalunit=None, interpolation='Cubic', 
                interpolparameter=10, colormap='inferno_r', dpi=200, imageformat='png', RelativeEnergy=True, numcontourlines=500,
                contour_alpha=0.75, contourline_color='black', clinelabels=False, contour_values=None):
    
    #Relative energy conversion (if RelativeEnergy is True)
    conversionfactor = { 'kcal/mol' : 627.50946900, 'kcalpermol' : 627.50946900, 'kJ/mol' : 2625.499638, 'kJpermol' : 2625.499638, 
                        'eV' : 27.211386245988, 'cm-1' : 219474.6313702 }
    e=[]
    coords=[]
    x_c=[]
    y_c=[]
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
        refenergy=float(min(e))
        relsurfacedictionary={}
        for i in surfacedictionary:
            relsurfacedictionary[(i[0],i[1])] = (surfacedictionary[i]-refenergy)*conversionfactor[finalunit]
        print("relsurfacedictionary ({}): {}".format(finalunit,relsurfacedictionary))
        
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
            exit()
            
        Y = zoom(Y, pw, mode='nearest')
        Z = zoom(Z, pw, mode='nearest')
    #Filled contours. 
    print("Using {} numcontourlines for colormap".format(numcontourlines))
    
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
    
    plt.colorbar(contour_surface)
    plt.xlabel(x_axislabel)
    plt.ylabel(y_axislabel)
    plt.savefig('Surface{}.{}'.format(label,imageformat), format=imageformat, dpi=dpi)
    print("Created PNG file: Surface{}.{}".format(label,imageformat))
    
    
#plot_Spectrum reads stick-values (e.g. absorption energie or IPs) and intensities, broadens spectrum (writes out dat and stk files) and then creates image-file using Matplotlib.
#TODO: Currently only Gaussian broadening. Add Lorentzian and Voight
def plot_Spectrum(xvalues=None, yvalues=None, plotname='Spectrum', range=None, unit='eV', broadening=0.1, points=10000, imageformat='png', dpi=200, matplotlib=True):
    if xvalues is None or yvalues is None:
        print("plot_Spectrum requires xvalues and yvalues variables")
        exit(1)

    assert len(xvalues) == len(yvalues), "List of Dysonnorms not same size as list of IPs." 

    start=range[0]
    finish=range[1]

    print("")
    print(BC.OKGREEN,"-------------------------------------------------------------------",BC.END)
    print(BC.OKGREEN,"plot_Spectrum: Plotting broadened spectrum",BC.END)
    print(BC.OKGREEN,"-------------------------------------------------------------------",BC.END)
    print("")
    print("xvalues ({}): {}".format(len(xvalues),xvalues))
    print("yvalues ({}): {}".format(len(yvalues),yvalues))


    #########################
    # Plot spectra.
    ########################
    print(BC.OKGREEN, "Plotting-range chosen:", start, "-", finish, unit, "with ", points, "points and ",
            broadening, "{} broadening.".format(unit), BC.END)

    # X-range
    x = np.linspace(start, finish, points)
    stkheight = 0.5
    strength = 1.0

    spectrum = 0
    for peak, strength in zip(xvalues, yvalues):
        broadenedpeak = Gaussian(x, peak, strength, broadening)
        spectrum += broadenedpeak

    #Save dat file
    with open(plotname+".dat", 'w') as tdatfile:
        for i,j in zip(x,spectrum):
            tdatfile.write("{:13.10f} {:13.10f} \n".format(i,j))
    #Save stk file
    with open(plotname+".stk", 'w') as tstkfile:
        for b,c in zip(xvalues,yvalues):
            tstkfile.write("{:13.10f} {:13.10f} \n".format(b,c))

    print("Wrote file:", plotname+".dat")
    print("Wrote file:", plotname+".stk")
    
    ##################################
    # Plot with Matplotlib
    ####################################
    if matplotlib is True:
        print("Creating plot with Matplotlib")
        
        fig, ax = plt.subplots()

        ax.plot(x, spectrum, 'C3', label=plotname)
        ax.stem(xvalues, yvalues, label=plotname, markerfmt=' ', basefmt=' ', linefmt='C3-', use_line_collection=True)
        plt.xlabel(unit)
        plt.ylabel('Intensity')
        #################################
        plt.xlim(start, finish)
        plt.legend(shadow=True, fontsize='small')
        plt.savefig(plotname + '.'+imageformat, format=imageformat, dpi=dpi)
        # plt.show()
        print("Wrote file:", plotname+"."+imageformat)
    else:
        print("Skipped Matplotlib part.")
    print(BC.OKGREEN,"ALL DONE!", BC.END)