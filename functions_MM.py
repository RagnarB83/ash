from functions_coords import *

#UFF dictionary with parameters
#Taken from oldmolcrys/old-solvshell and originally from Chemshell
# Element: [R0,eps]. R0 in Angstrom (I think) and eps in kcal/mol
UFFdict={'H': [2.886, 0.044], 'He': [2.362, 0.056], 'Li': [2.451, 0.025], 'Be': [2.745, 0.085], 'B': [4.083, 0.18],
         'C': [3.851, 0.105], 'N': [3.66, 0.069], 'O': [3.5, 0.06], 'F': [3.364, 0.05], 'Ne': [3.243, 0.042],
         'Na': [2.983, 0.03], 'Mg': [3.021, 0.111], 'Al': [4.499, 0.505], 'Si': [4.295, 0.402], 'P': [4.147, 0.305],
         'S': [4.035, 0.274], 'Cl': [3.947, 0.227], 'Ar': [3.868, 0.185], 'K': [3.812, 0.035], 'Ca': [3.399, 0.238],
         'Sc': [3.295, 0.019], 'Ti': [3.175, 0.017], 'V': [3.144, 0.016], 'Cr': [3.023, 0.015], 'Mn': [2.961, 0.013],
         'Fe': [2.912, 0.013], 'Co': [2.872, 0.014], 'Ni': [2.834, 0.015], 'Cu': [3.495, 0.005], 'Zn': [2.763, 0.124],
         'Ga': [4.383, 0.415], 'Ge': [4.28, 0.379], 'As': [4.23, 0.309], 'Se': [4.205, 0.291], 'Br': [4.189, 0.251],
         'Kr': [4.141, 0.22], 'Rb': [4.114, 0.04], 'Sr': [3.641, 0.235], 'Y': [3.345, 0.072], 'Zr': [3.124, 0.069],
         'Nb': [3.165, 0.059], 'Mo': [3.052, 0.056], 'Tc': [2.998, 0.048], 'Ru': [2.963, 0.056], 'Rh': [2.929, 0.053],
         'Pd': [2.899, 0.048], 'Ag': [3.148, 0.036], 'Cd': [2.848, 0.228], 'In': [4.463, 0.599], 'Sn': [4.392, 0.567],
         'Sb': [4.42, 0.449], 'Te': [4.47, 0.398], 'I': [4.5, 0.339], 'Xe': [4.404, 0.332], 'Cs': [4.517, 0.045],
         'Ba': [3.703, 0.364], 'La': [3.522, 0.017], 'Ce': [3.556, 0.013], 'Pr': [3.606, 0.01], 'Nd': [3.575, 0.01],
         'Pm': [3.547, 0.009], 'Sm': [3.52, 0.008], 'Eu': [3.493, 0.008], 'Gd': [3.368, 0.009], 'Tb': [3.451, 0.007],
         'Dy': [3.428, 0.007], 'Ho': [3.409, 0.007], 'Er': [3.391, 0.007], 'Tm': [3.374, 0.006], 'Yb': [3.355, 0.228],
         'Lu': [3.64, 0.041], 'Hf': [4.141, 0.072], 'Ta': [3.17, 0.081], 'W': [3.069, 0.067], 'Re': [2.954, 0.066],
         'Os': [3.12, 0.037], 'Ir': [2.84, 0.073], 'Pt': [2.754, 0.08], 'Au': [3.293, 0.039], 'Hg': [2.705, 0.385],
         'Tl': [4.347, 0.68], 'Pb': [4.297, 0.663], 'Bi': [4.37, 0.518], 'Po': [4.709, 0.325], 'At': [4.75, 0.284],
         'Rn': [4.765, 0.248], 'Fr': [4.9, 0.05], 'Ra': [3.677, 0.404], 'Ac': [3.478, 0.033], 'Th': [3.396, 0.026],
         'Pa': [3.424, 0.022], 'U': [3.395, 0.022], 'Np': [3.424, 0.019], 'Pu': [3.424, 0.016], 'Am': [3.381, 0.014],
         'Cm': [3.326, 0.013], 'Bk': [3.339, 0.013], 'Cf': [3.313, 0.013], 'Es': [3.299, 0.012], 'Fm': [3.286, 0.012],
         'Md': [3.274, 0.011], 'No': [3.248, 0.011], 'Lr': [3.236, 0.011]}

#Modified UFF dictionary with no LJ on H.
UFF_modH_dict={'H': [0.000, 0.000], 'He': [2.362, 0.056], 'Li': [2.451, 0.025], 'Be': [2.745, 0.085], 'B': [4.083, 0.18],
         'C': [3.851, 0.105], 'N': [3.66, 0.069], 'O': [3.5, 0.06], 'F': [3.364, 0.05], 'Ne': [3.243, 0.042],
         'Na': [2.983, 0.03], 'Mg': [3.021, 0.111], 'Al': [4.499, 0.505], 'Si': [4.295, 0.402], 'P': [4.147, 0.305],
         'S': [4.035, 0.274], 'Cl': [3.947, 0.227], 'Ar': [3.868, 0.185], 'K': [3.812, 0.035], 'Ca': [3.399, 0.238],
         'Sc': [3.295, 0.019], 'Ti': [3.175, 0.017], 'V': [3.144, 0.016], 'Cr': [3.023, 0.015], 'Mn': [2.961, 0.013],
         'Fe': [2.912, 0.013], 'Co': [2.872, 0.014], 'Ni': [2.834, 0.015], 'Cu': [3.495, 0.005], 'Zn': [2.763, 0.124],
         'Ga': [4.383, 0.415], 'Ge': [4.28, 0.379], 'As': [4.23, 0.309], 'Se': [4.205, 0.291], 'Br': [4.189, 0.251],
         'Kr': [4.141, 0.22], 'Rb': [4.114, 0.04], 'Sr': [3.641, 0.235], 'Y': [3.345, 0.072], 'Zr': [3.124, 0.069],
         'Nb': [3.165, 0.059], 'Mo': [3.052, 0.056], 'Tc': [2.998, 0.048], 'Ru': [2.963, 0.056], 'Rh': [2.929, 0.053],
         'Pd': [2.899, 0.048], 'Ag': [3.148, 0.036], 'Cd': [2.848, 0.228], 'In': [4.463, 0.599], 'Sn': [4.392, 0.567],
         'Sb': [4.42, 0.449], 'Te': [4.47, 0.398], 'I': [4.5, 0.339], 'Xe': [4.404, 0.332], 'Cs': [4.517, 0.045],
         'Ba': [3.703, 0.364], 'La': [3.522, 0.017], 'Ce': [3.556, 0.013], 'Pr': [3.606, 0.01], 'Nd': [3.575, 0.01],
         'Pm': [3.547, 0.009], 'Sm': [3.52, 0.008], 'Eu': [3.493, 0.008], 'Gd': [3.368, 0.009], 'Tb': [3.451, 0.007],
         'Dy': [3.428, 0.007], 'Ho': [3.409, 0.007], 'Er': [3.391, 0.007], 'Tm': [3.374, 0.006], 'Yb': [3.355, 0.228],
         'Lu': [3.64, 0.041], 'Hf': [4.141, 0.072], 'Ta': [3.17, 0.081], 'W': [3.069, 0.067], 'Re': [2.954, 0.066],
         'Os': [3.12, 0.037], 'Ir': [2.84, 0.073], 'Pt': [2.754, 0.08], 'Au': [3.293, 0.039], 'Hg': [2.705, 0.385],
         'Tl': [4.347, 0.68], 'Pb': [4.297, 0.663], 'Bi': [4.37, 0.518], 'Po': [4.709, 0.325], 'At': [4.75, 0.284],
         'Rn': [4.765, 0.248], 'Fr': [4.9, 0.05], 'Ra': [3.677, 0.404], 'Ac': [3.478, 0.033], 'Th': [3.396, 0.026],
         'Pa': [3.424, 0.022], 'U': [3.395, 0.022], 'Np': [3.424, 0.019], 'Pu': [3.424, 0.016], 'Am': [3.381, 0.014],
         'Cm': [3.326, 0.013], 'Bk': [3.339, 0.013], 'Cf': [3.313, 0.013], 'Es': [3.299, 0.012], 'Fm': [3.286, 0.012],
         'Md': [3.274, 0.011], 'No': [3.248, 0.011], 'Lr': [3.236, 0.011]}



#Fast LJ-Coulomb via Fortran and f2PY
def LJCoulomb(coords,epsij, sigmaij, charges, connectivity=None):
    print("Inside LJCoulomb")
    #Todo: Avoid calling import everytime in the future...
    import LJCoulombv1
    print(LJCoulombv1.__doc__)
    numatoms=len(coords)
    rc=9999.5
    grad = np.zeros((numatoms,3))
    penergy, LJenergy, coulenergy, grad = LJCoulombv1.ljcoulegrad(coords, rc, epsij, sigmaij, charges, grad, dim=3, natom=numatoms)
    return penergy, grad, LJenergy, coulenergy

#Fast LJ-Coulomb via Fortran and f2PY
def LJCoulombv2(coords,epsij, sigmaij, charges, connectivity=None):
    print("Inside LJCoulomb")
    #Todo: Avoid calling import everytime in the future...
    import LJCoulombv2
    print(LJCoulombv2.__doc__)
    numatoms=len(coords)
    #rc: threshold for ignoring LJ interaction
    rc=9999.5
    grad = np.zeros((numatoms,3))
    #Calling Fortran function
    penergy, LJenergy, coulenergy, grad = LJCoulombv2.ljcoulegrad(coords, rc, epsij, sigmaij, charges, grad, dim=3, natom=numatoms)
    return penergy, grad, LJenergy, coulenergy

def LennardJones(coords, epsij, sigmaij, connectivity=None, qmatoms=None):
    print("Inside Python Lennard-Jones function")
    #print("qmatoms:", qmatoms)
    #print("QM atom pairs are skipped if qmatoms list provided")
    #print("connectivity: ", connectivity)
    #print("Calculating LJ pairs based on connectivity if present")
    #print("Note: This means that if two LJ sites are part of same molecular fragment then LJ is not calculated")
    #print("Note: Not correct behaviour for CHARMM/Amber etc")
    #print("Note: Will give correct behaviour for molcrys-QM/MM as QM fragment will not interact with itself")


    #if len(connectivity)==0:
    #    print("Warning!. No connectivity list present. Will treat all LJ pairs.")
    #else:
    #    print(len(connectivity)," connectivity lists present")

    atomlist=list(range(0, len(coords)))
    #LJ energy
    energy=0
    #LJ gradient
    gradient = np.zeros((len(coords), 3))
    #Iterating over atom i
    for i in atomlist:
        #Iterating over atom j
        for j in atomlist:
            #Skipping if same atom
            if i != j:
                #Skipping identical pairs
                if i < j:
                    #Skipping if atom pair in qmatoms list. I.e. not calculate QM-QM LJ terms
                    #if all(x in qmatoms for x in [i, j]) == True:
                    #    print("Skipping QM-pair:", i,j)
                    #    continue
                    #for l in LJPairpotentials:
                        #print("l:", l)
                        #This checks if i-j pair exists in LJPairpotentials list:
                    #    if set([atomtypes[i], atomtypes[j]]) == set([l[0],l[1]]):
                        #if atomtypes[i] in l and atomtypes[j] in l:
                            #print("COUNTING!!! unless...")
                            #Now checking connectivity for whether we should calculate LJ energy for pair or not
                            #Todo: This only makes sense in a QM/MM scheme with frozen MM?
                    #        skip=False
                    #        for conn in connectivity:
                                #print("conn:", conn)
                                #If i,j in same list
                    #            if all(x in conn for x in [i, j]) == True:
                                    #print("Atoms connected. skipping ")
                    #                skip=True
                    #                continue
                    #if skip == False:
                    #print("i : {}  and j : {}".format(i,j))
                    #print("atomtype_i : {}  and atomtype_j : {}".format(atomtypes[i],atomtypes[j]))
                    #sigma=l[2]
                    #eps=l[3]
                    pairdistance = distance(coords[i], coords[j])
                    #print("sigma, eps, pairdistance", sigma,eps,pairdistance)
                    V_LJ=4*epsij[i,j]*((sigmaij[i,j]/pairdistance)**12-(sigmaij[i,j]/pairdistance)**6)
                    #print("V_LJ: {} kcal/mol  V_LJ: {} au:".format(V_LJ,V_LJ/constants.harkcal))
                    energy+=V_LJ
                    #print("energy: {} kcal/mol  energy: {} au:".format(energy, energy / constants.harkcal))
                    #print("------------------------------")
                    #Typo in http://localscf.com/localscf.com/LJPotential.aspx.html ??
                    #Using http://www.courses.physics.helsinki.fi/fys/moldyn/lectures/L4.pdf
                    #TODO: Equation needs to be double-checked for correctness. L4.pdf equation ambiguous
                    #Check this: http://people.virginia.edu/~lz2n/mse627/notes/Potentials.pdf
                    LJgrad_const=(24*epsij[i,j]*((sigmaij[i,j]/pairdistance)**6-2*(sigmaij[i,j]/pairdistance)**12))*(1/(pairdistance**2))
                    #print("LJgrad_const:", LJgrad_const)
                    gr=np.array([(coords[i][0] - coords[j][0])*LJgrad_const, (coords[i][1] - coords[j][1])*LJgrad_const,
                                     (coords[i][2] - coords[j][2])*LJgrad_const])
                    #print("gr:", gr)
                    gradient[i] += gr
                    gradient[j] -= gr
                    #print("gradient[i]:", gradient[i])
                    #print("gradient[j]:", gradient[j])
                    #print("gradients in hartree/Bohr:")
                    #print("gradient[i]:", gradient[i]* (1/constants.harkcal) / constants.ang2bohr)
                    #print("gradient[j]:", gradient[j]* (1/constants.harkcal) / constants.ang2bohr)
    #Convert gradient from kcal/mol per Å to hartree/Bohr
    final_gradient=gradient * (1/constants.harkcal) / constants.ang2bohr
    print("LJ gradient (hartree/Bohr):", final_gradient)
    #Converg energy from kcal/mol to hartree
    final_energy=energy*(1/constants.harkcal)
    print("LJ energy (hartree)", final_energy)

    return final_energy,final_gradient

#TODO: Do we always calculate charge if atoms are connected? Need connectivity for CHARMM/Amber expressions
#Coulomb energy and gradient in Bohrs
def coulombcharge(charges, coords):
    #print("MM charges:", charges)
    #print("Num MM charges:", len(charges))
    #print("Num coords:", len(coords))
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

#Combined Lennard-Jones and Coulomb
#Terribly written

def LJCoulpy(coords,atomtypes, charges, LJPairpotentials, connectivity=None):
    print("Inside LJCoulpy function")
    print("Calculating LJ pairs based on connectivity")
    if len(connectivity)==0:
        print("Warning!. No connectivity list present. Will treat all LJ pairs.")
    else:
        print(len(connectivity)," connectivity lists present")

    atomlist=list(range(0, len(coords)))
    #LJ energy
    energy=0
    #LJ gradient
    gradient = np.zeros((len(coords), 3))
    #Iterating over atom i
    for i in atomlist:
        #Iterating over atom j
        for j in atomlist:
            #Skipping if same atom
            if i != j:
                #Skipping identical pairs
                if i < j:

                    #Coulomb part
                    pairdistance_b=distance(coords_b[i],coords_b[j])
                    pairenergy=(charges[i]*charges[j])/pairdistance_b
                    Coulenergy+=pairenergy
                    #Using electric field expression from: http://www.physnet.org/modules/pdf_modules/m115.pdf
                    Efield_pair_hat=np.array([(coords_b[i][0]-coords_b[j][0])/pairdistance_b,
                                              (coords_b[i][1]-coords_b[j][1])/pairdistance_b,
                                              (coords_b[i][2]-coords_b[j][2])/pairdistance_b ])
                    #Doing ij pair and storing contribution for each
                    Efield_pair_j=(Efield_pair_hat*charges[j])/(pairdistance_b**2)
                    Efield_pair_i = (Efield_pair_hat * charges[i]) / (pairdistance_b**2)
                    Coulgradient[i] += -1 * Efield_pair_j*charges[i]
                    Coulgradient[j] -= -1 * Efield_pair_i*charges[j]


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
                                #print("i : {}  and j : {}".format(i,j))
                                #print("atomtype_i : {}  and atomtype_j : {}".format(atomtypes[i],atomtypes[j]))
                                sigma=l[2]
                                eps=l[3]
                                #pairdistance = distance(coords[i], coords[j])
                                pairdistance=pairdistance_b*constants.bohr2ang
                                #print("sigma, eps, pairdistance", sigma,eps,pairdistance)
                                V_LJ=4*eps*((sigma/pairdistance)**12-(sigma/pairdistance)**6)
                                #print("V_LJ: {} kcal/mol  V_LJ: {} au:".format(V_LJ,V_LJ/constants.harkcal))
                                LJenergy+=V_LJ
                                #print("energy: {} kcal/mol  energy: {} au:".format(energy, energy / constants.harkcal))
                                #print("------------------------------")
                                #Typo in http://localscf.com/localscf.com/LJPotential.aspx.html ??
                                #Using http://www.courses.physics.helsinki.fi/fys/moldyn/lectures/L4.pdf
                                #TODO: Equation needs to be double-checked for correctness. L4.pdf equation ambiguous
                                #Check this: http://people.virginia.edu/~lz2n/mse627/notes/Potentials.pdf
                                LJgrad_const=(24*eps*((sigma/pairdistance)**6-2*(sigma/pairdistance)**12))*(1/(pairdistance**2))
                                #print("LJgrad_const:", LJgrad_const)
                                gr=np.array([(coords[i][0] - coords[j][0])*LJgrad_const, (coords[i][1] - coords[j][1])*LJgrad_const,
                                     (coords[i][2] - coords[j][2])*LJgrad_const])
                                #print("gr:", gr)
                                LJgradient[i] += gr
                                LJgradient[j] -= gr
                                #print("gradient[i]:", gradient[i])
                                #print("gradient[j]:", gradient[j])
                                #print("gradients in hartree/Bohr:")
                                #print("gradient[i]:", gradient[i]* (1/constants.harkcal) / constants.ang2bohr)
                                #print("gradient[j]:", gradient[j]* (1/constants.harkcal) / constants.ang2bohr)
    #Convert gradient from kcal/mol per Å to hartree/Bohr
    LJfinal_gradient=LJgradient * (1/constants.harkcal) / constants.ang2bohr
    print("LJ gradient (hartree/Bohr):", LJfinal_gradient)
    #Converg energy from kcal/mol to hartree
    LJfinal_energy=LJenergy*(1/constants.harkcal)
    print("LJ energy (hartree)", LJfinal_energy)

    #Coulomb
    print("Coulenergy : ", Coulenergy)

    final_energy = LJfinal_energy+Coulenergy

    final_gradient = LJfinal_gradient + Coulgradient

    return final_energy,final_gradient