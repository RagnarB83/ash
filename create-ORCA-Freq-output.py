#!/usr/bin/env python3

#Create fake ORCA-outputfile containing geometry and Freq output from ORCA-style Hessian file
#TODO: Script previously read orca_vib output to get normal modes. We need to calculate this ourselves from Hessian and then read it out in ORCA format.

# For visualization in Chemcraft

#Read in Hessian file as first argument and outputfile from orca_vib as second

import os
import sys

hessfile=sys.argv[1]

orca_header="""                                 *****************
                                 * O   R   C   A *
                                 *****************

           --- An Ab Initio, DFT and Semiempirical electronic structure package ---

                  #######################################################
                  #                        -***-                        #
                  #  Department of molecular theory and spectroscopy    #
                  #              Directorship: Frank Neese              #
                  # Max Planck Institute for Chemical Energy Conversion #
                  #                  D-45470 Muelheim/Ruhr              #
                  #                       Germany                       #
                  #                                                     #
                  #                  All rights reserved                #
                  #                        -***-                        #
                  #######################################################


                         Program Version 3.0.3 - RELEASE   -




                       *****************************
                       * Geometry Optimization Run *
                       *****************************

         *************************************************************
         *                GEOMETRY OPTIMIZATION CYCLE   1            *
         *************************************************************
---------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------"""

#Function to grab masses and elements
def masselemgrab(hessfile):
    grab=False
    elems=[]; masses=[]
    with open(hessfile) as hfile:
        for line in hfile:
            if '$actual_temperature' in line:
                grab=False
            if grab==True and len(line.split()) == 1:
                numatoms=int(line.split()[0])
            if grab==True and len(line.split()) == 5 :
                elems.append(line.split()[0])
                masses.append(float(line.split()[1]))
            if '$atoms' in line:
                grab=True
    return masses, elems,numatoms

def massweight(matrix,masses,numatoms):
    mass_mat = np.zeros( (3*numatoms,3*numatoms), dtype = float )
    molwt = [ masses[int(i)] for i in range(numatoms) for j in range(3) ]
    for i in range(len(molwt)):
        mass_mat[i,i] = molwt[i] ** -0.5
    mwhessian = np.dot((np.dot(mass_mat,matrix)),mass_mat)
    return mwhessian,mass_mat

#
def calcfreq(evalues):
    hartree2j = 4.3597438e-18
    bohr2m = 5.29177210903e-11
    #amu2kg = 1.66054e-27
    amu2kg = 1.66053906660e-27
    #speed of light in cm/s
    c = 2.99792458e10
    pi = np.pi
    evalues_si = [val*hartree2j/bohr2m/bohr2m/amu2kg for val in evalues]
    vfreq_hz = [1/(2*pi)*np.sqrt(np.complex_(val)) for val in evalues_si]
    vfreq = [val/c for val in vfreq_hz]
    return vfreq

#Function to grab Hessian from ORCA-Hessian file
def Hessgrab(hessfile):
    hesstake=False
    j=0
    orcacoldim=5
    shiftpar=0
    lastchunk=False
    grabsize=False
    with open(hessfile) as hfile:
        for line in hfile:
            if '$vibrational_frequencies' in line:
                hesstake=False
                continue
            if hesstake==True and len(line.split()) == 1 and grabsize==True:
                grabsize=False
                hessdim=int(line.split()[0])

                hessarray2d=np.zeros((hessdim, hessdim))
            if hesstake==True and len(line.split()) == 5:
                continue
                #Headerline
            if hesstake==True and lastchunk==True:
                if len(line.split()) == hessdim - shiftpar +1:
                    for i in range(0,hessdim - shiftpar):
                        hessarray2d[j,i+shiftpar]=line.split()[i+1]
                    j+=1
            if hesstake==True and len(line.split()) == 6:
                # Hessianline
                for i in range(0, orcacoldim):
                    hessarray2d[j, i + shiftpar] = line.split()[i + 1]
                j += 1
                if j == hessdim:
                    shiftpar += orcacoldim
                    j = 0
                    if hessdim - shiftpar < orcacoldim:
                        lastchunk = True
            if '$hessian' in line:
                hesstake = True
                grabsize = True
            return hessarray2d

def grabcoordsfromhessfile(hessfile):
    #Grab coordinates from hessfile
    numatomgrab=False
    cartgrab=False
    elements=[]
    #x_coord=[]
    #y_coord=[]
    #z_coord=[]
    coords=[]
    count=0
    bohrang=0.529177249
    with open(hessfile) as hfile:
        for line in hfile:
            if cartgrab==True:
                count=count+1
                #print(line)
                #print(count)
                elem=line.split()[0]; x_c=bohrang*float(line.split()[2]);y_c=bohrang*float(line.split()[3]);z_c=bohrang*float(line.split()[4])
                elements.append(elem)
                #x_coord.append(x_c);y_coord.append(y_c);z_coord.append(z_c)
                coords.append([x_c,y_c,z_c])
                if count == numatoms:
                    break
            if numatomgrab==True:
                numatoms=int(line.split()[0])
                numatomgrab=False
                cartgrab=True
            if "$atoms" in line:
                numatomgrab=True

    return elements,coords

def clean_number(number):
    return np.real_if_close(number)

elements,coords=grabcoordsfromhessfile(hessfile)
print("coords:", coords)
print("elements:", elements)

# Grab masses, elements and numatoms from Hessianfile
masses, elems, numatoms = masselemgrab(hessfile)
#atomlist = []
#for i, j in enumerate(elems):
#    atomlist.append(str(j) + '-' + str(i))

# Grab Hessian from Hessianfile
hessian = Hessgrab(hessfile)

# Massweight Hessian
mwhessian, massmatrix = massweight(hessian, masses, numatoms)

# Diagonalize mass-weighted Hessian
evalues, evectors = la.eigh(mwhessian)
evectors = np.transpose(evectors)

# Calculate frequencies from eigenvalues

vfreq = calcfreq(evalues)
print("")
# Unweight eigenvectors to get normal modes
nmodes = np.dot(evectors, massmatrix)

filename='bla'
outfile = open(filename+'fake.out', 'w')

outfile.write(orca_header+'\n')
for el,coord in zip(coords):
    x=coord[0];y=coord[1];coord[2]
    line = "  {0:2s} {1:11.6f} {2:12.6f} {3:13.6f}".format(el, x, y, z)
    #print(line)
    #print('  S     51.226907   65.512868  106.021030')
    #exit()
    outfile.write(line+'\n')

outfile.write('-----------------------\n')
outfile.write('VIBRATIONAL FREQUENCIES\n')
outfile.write('-----------------------\n')
outfile.write('\n')
outfile.write('Scaling factor for frequencies =  1.000000000 (Found in file - NOT applied to frequencies read from HESS file)\n')
outfile.write('\n')
for mode in range(3*numatoms):
    smode=str(mode)+':'
    freq=clean_number(vfreq[mode])
    line= "{0:2s} {1:15.8f} {2:15.8f} {3:15.8f}".format(smode, freq, "cm**-1")
    outfile.write(str(mode)+':')
#    for i in range(N):
#        line = "{0:2s} {1:15.8f} {2:15.8f} {3:15.8f}".format(atoms[i], V[i, 0], V[i, 1], V[i, 2])
#        coords.append(line)
        #print line


#TODO: Finish write normal mode output in ORCA format from nmodes so that Chemcraft can read it

outfile.write('\n')
outfile.write('-----------------------\n')
exit()
for l in vibout:
    outfile.write(l)

#  S     51.226907   65.512868  106.021030
#  C         6.618918       -8.320782        6.930409
#C         6.618918       -8.320782        6.930409
outfile.close()

