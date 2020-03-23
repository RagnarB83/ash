import numpy as np
#Psi4 interface.
# Inputfile version.
# Todo: replace with python-interface version

#create inputfile

#run Psi4 input


#check if Psi4 finished

#grab energy from output

def grabPsi4EandG(outfile, numatoms, Grad):
    energy=None
    gradient = np.zeros((numatoms, 3))
    row=0
    gradgrab=False
    print("here")
    with open(outfile) as ofile:
        for line in ofile:
            print("line:", line)
            if '    Total Energy =' in line:
                energy = float(line.split()[-1])
                print("energy:", energy)
            if Grad == True:

                if gradgrab==True:
                    print("len(line)", len(line))
                    if len(line) < 1:
                        gradgrab=False
                    if '--' not in line:
                        if 'Atom' not in line:
                            val=line.split()
                            gradient[row] = [float(val[1]),float(val[2]),float(val[3])]
                            row+=1
                if '  -Total Gradient:' in line:
                    gradgrab = True
    print("energy:", energy)
    print("gradient:", gradient)
    if energy == None:
        print("Found no energy in Psi4 outputfile:", outfile)
        exit()
    return energy, gradient
