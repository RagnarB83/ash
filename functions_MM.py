from functions_coords import *

#MMAtomobject used to store LJ parameter and possibly charge for MM atom with atomtype, e.g. OT
class AtomMMobject:
    def __init__(self, atomcharge=None, LJparameters=[]):
        sf="dsf"
        self.atomcharge = atomcharge
        self.LJparameters = LJparameters
    def add_charge(self, atomcharge=None):
        self.atomcharge = atomcharge
    def add_LJparameters(self, LJparameters=None):
        self.LJparameters=LJparameters

# TODO: Create function to do Coulomb and LJ terms together in one loop.
#will be necessary for efficiency for large system
def LennardJones(coords,atomtypes, LJPairpotentials, connectivity=[]):
    print("Inside Lennard_jones function")
    print("Calculating LJ pairs based on connectivity:", connectivity)
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

def MMforcefield_read(file):
    print("Reading forcefield file:", file)
    MM_forcefield = {}
    atomtypes=[]
    with open(file) as f:
        for line in f:
            if 'combination_rule' in line:
                combrule=line.split()[-1]
                print("Found combination rule defintion in forcefield file:", combrule)
                MM_forcefield["combination_rule"]=combrule
            if 'charge' in line:
                print("Found charge definition in forcefield file:", ' '.join(line.split()[:]))
                atomtype=line.split()[1]
                if atomtype not in MM_forcefield.keys():
                    MM_forcefield[atomtype]=AtomMMobject()
                charge=float(line.split()[2])
                MM_forcefield[atomtype].add_charge(atomcharge=charge)
                # TODO: Charges defined are currently not used I think
            if 'LennardJones_i_sigma' in line:
                print("Found LJ single-atom sigma definition in forcefield file:", ' '.join(line.split()[:]))
                atomtype=line.split()[1]
                if atomtype not in MM_forcefield.keys():
                    MM_forcefield[atomtype] = AtomMMobject()
                sigma_i=float(line.split()[2])
                eps_i=float(line.split()[3])
                MM_forcefield[atomtype].add_LJparameters(LJparameters=[sigma_i,eps_i])
            if 'LennardJones_i_R0' in line:
                print("Found LJ single-atom R0 definition in forcefield file:", ' '.join(line.split()[:]))
                atomtype=line.split()[1]
                R0tosigma=0.5**(1/6)
                print("R0tosigma conversion", R0tosigma)
                if atomtype not in MM_forcefield.keys():
                    MM_forcefield[atomtype] = AtomMMobject()
                sigma_i=float(line.split()[2]*R0tosigma)
                eps_i=float(line.split()[3])
                MM_forcefield[atomtype].add_LJparameters(LJparameters=[sigma_i,eps_i])
            if 'LennardJones_ij' in line:
                print("Found LJ pair definition in forcefield file")
                atomtype_i=line.split()[1]
                atomtype_j=line.split()[2]
                sigma_ij=float(line.split()[3])
                eps_ij=float(line.split()[4])
                print("This is incomplete. Exiting")
                exit()
                # TODO: Need to finish this. Should replace LennardJonespairpotentials later
    return MM_forcefield
