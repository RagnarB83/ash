from functions_coords import *



# TODO: Create function to do Coulomb and LJ terms together in one loop.
#will be necessary for efficiency for large system

#TODO: For really fast COulomb and LJ. Probably have to write in Fortran/C/C++.
#Can steal from ABIN maybe: force_mm.f90  and interface ??

def LennardJones(coords,atomtypes, LJPairpotentials, connectivity=[]):
    print("Inside Lennard_jones function")
    print("Calculating LJ pairs based on connectivity:", connectivity)
    print("")
    if len(connectivity)==0:
        print("Warning!. No connectivity list present. Will treat all LJ pairs.")

    atomlist=list(range(0, len(coords)))
    #LJ energy
    energy=0
    #LJ gradient
    gradient = np.zeros((len(coords), 3))
    exit()
    #Iterating over atom i
    for i in atomlist:
        #Iterating over atom j
        for j in atomlist:
            #Skipping if same atom
            if i != j:
                #Skipping identical pairs
                if i < j:
                    for l in LJPairpotentials:
                        #print("l:", l)
                        #This checks if i-j pair exists in LJPairpotentials list:
                        if set([atomtypes[i], atomtypes[j]]) == set([l[0],l[1]]):
                        #if atomtypes[i] in l and atomtypes[j] in l:
                            #print("COUNTING!!! unless...")
                            #Now checking connectivity for whether we should calculate LJ energy for pair or not
                            skip=False
                            for conn in connectivity:
                                #print("conn:", conn)
                                #If i,j in same list
                                if all(x in conn for x in [i, j]) == True:
                                    #print("Atoms connected. skipping ")
                                    skip=True
                                    continue
                            if skip == False:
                                #print("atomtype_i : {}  and atomtype_j : {}".format(atomtypes[i],atomtypes[j]))
                                sigma=l[2]
                                eps=l[3]
                                pairdistance = distance(coords[i], coords[j])
                                V_LJ=4*eps*((sigma/pairdistance)**12-(sigma/pairdistance)**6)
                                #print("V_LJ", V_LJ)
                                energy+=V_LJ
                                #print("energy:", energy)
                                #Typo in http://localscf.com/localscf.com/LJPotential.aspx.html ??
                                #Using http://www.courses.physics.helsinki.fi/fys/moldyn/lectures/L4.pdf
                                LJgrad_const=(24*eps*((sigma/pairdistance)**6-2*(sigma/pairdistance)**12))*(1/(pairdistance**2))
                                gr=np.array([(coords[i][0] - coords[j][0])*LJgrad_const, (coords[i][1] - coords[j][1])*LJgrad_const,
                                     (coords[i][2] - coords[j][2])*LJgrad_const])
                                gradient[i] += gr
                                gradient[j] -= gr
                            blankline()
    #Convert gradient from kcal/mol per Å to hartree/Bohr
    final_gradient=gradient * (1/constants.harkcal) / constants.ang2bohr
    #Converg energy from kcal/mol to hartree
    final_energy=energy*(1/constants.harkcal)
    return final_energy,final_gradient

#TODO: Do we always calculate charge if atoms are connected? Need connectivity for CHARMM/Amber expressions
#Coulomb energy and gradient in Bohrs
def coulombcharge(charges, coords):
    print("MM charges:", charges)
    print("Num MM charges:", len(charges))
    print("Num coords:", len(coords))
    #Converting list to numpy array and converting to bohr
    coords_b=np.array(coords)*constants.ang2bohr
    #Coulomb energy
    energy=0
    #Initialize Coulomb gradient
    gradient = np.zeros((len(coords_b), 3))
    blankline()
    #Iterating over atom i
    for count_i,charge_i in enumerate(charges):
        #Iterating over atom j
        for count_j,charge_j in enumerate(charges):
            #Skipping if same atom
            if count_i != count_j:
                #Skipping identical pairs
                if count_i < count_j:
                    pairdistance=distance(coords_b[count_i],coords_b[count_j])
                    pairenergy=(charge_i*charge_j)/pairdistance
                    energy+=pairenergy
                    #Using electric field expression from: http://www.physnet.org/modules/pdf_modules/m115.pdf
                    Efield_pair_hat=np.array([(coords_b[count_i][0]-coords_b[count_j][0])/pairdistance,
                                              (coords_b[count_i][1]-coords_b[count_j][1])/pairdistance,
                                              (coords_b[count_i][2]-coords_b[count_j][2])/pairdistance ])
                    #Doing ij pair and storing contribution for each
                    Efield_pair_j=(Efield_pair_hat*charge_j)/(pairdistance**2)
                    Efield_pair_i = (Efield_pair_hat * charge_i) / (pairdistance**2)
                    gradient[count_i] += -1 * Efield_pair_j*charge_i
                    gradient[count_j] -= -1 * Efield_pair_i*charge_j
    return energy,gradient


