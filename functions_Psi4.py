import numpy as np
#Psi4 interface.
# Inputfile version.
# Todo: replace with python-interface version

#create inputfile

#run Psi4 input


#check if Psi4 finished

#grab energy from output

#Slightly ugly
def grabPsi4EandG(outfile, numatoms, Grad):
    energy=None
    gradient = np.zeros((numatoms, 3))
    row=0
    gradgrab=False
    with open(outfile) as ofile:
        for line in ofile:
            if 'FINAL TOTAL ENERGY' in line:
                energy = float(line.split()[-1])
            if Grad == True:
                if gradgrab==True:
                    if len(line) < 2:
                        gradgrab=False
                        break
                    if '--' not in line:
                        if 'Atom' not in line:
                            val=line.split()
                            gradient[row] = [float(val[1]),float(val[2]),float(val[3])]
                            row+=1
                if '  -Total Gradient:' in line:
                    gradgrab = True
    if energy == None:
        print("Found no energy in Psi4 outputfile:", outfile)
        exit()
    return energy, gradient
