import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

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
                interpolparameter=10, colormap='inferno_r', dpi=200, imageformat='png', RelativeEnergy=True, numcontourlines=50,
                contour_alpha=0.75):
    
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

    #Creating relative-energy array here. Unmodified property is used if False
    if RelativeEnergy is True:
        refenergy=float(min(e))
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


    print("interpolation:", interpolation)
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
        #print(X)
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
    cp=plt.contourf(X, Y, Z, numcontourlines, alpha=contour_alpha, cmap=colormap)
    
    #Contour lines. numcontourlines is 50 by default 
    C = plt.contour(X, Y, Z, numcontourlines, colors='black')
    
    
    plt.colorbar(cp)
    plt.xlabel(x_axislabel)
    plt.ylabel(y_axislabel)
    plt.savefig('Surface{}.png'.format(label), format=imageformat, dpi=dpi)
    print("Created PNG file: Surface{}.png".format(label))