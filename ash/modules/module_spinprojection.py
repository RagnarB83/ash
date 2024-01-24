"""
Spinprojection module:

class SpinProjectionTheory

"""
from ash.functions.functions_general import BC, ashexit, print_time_rel
import ash.interfaces.interface_ORCA
import ash.constants
import ash.functions.functions_elstructure
import time

class SpinProjectionTheory:
    """ASH SpinProjection theory.
    Combines two theory levels (different spins) to give one final spin-projected energy.
    Spinprojections: Noodleman, Yamaguchi, Bencini
    """
    def __init__(self, theory1=None, theory2=None, printlevel=2, reuseorbs=True,
                 label=None, jobtype=None, localspins=None, charge_1=None, mult_1=None, charge_2=None, mult_2=None):
        print("Creating SpinProjectionTheory object. Jobtype: ", jobtype)
        self.theorytype="QM"
        self.theory1=theory1
        self.theory2=theory2
        self.printlevel=printlevel
        self.label=label
        self.reuseorbs=reuseorbs
        #This is an inputfilename that may be set externally (Singlepoint_par)
        self.filename="X"
        
        #This is the exception where charge/mult is part of the theory
        self.charge_1=charge_1
        self.charge_2=charge_2
        self.mult_1=mult_1
        self.mult_2=mult_2

        self.jobtype=jobtype
        if self.jobtype == "Yamaguchi" or self.jobtype =="Noodleman" or self.jobtype=="Bencini":
            if localspins == None:
                print("Yamaguchi/Noodleman/Bencini spin projection requires localspins keyword (list of local spins). Exiting.")
                ashexit()
            else:
                self.Spin_A=localspins[0]
                self.Spin_B=localspins[1]
                self.Spin_HS=self.Spin_A+self.Spin_B
                self.Spin_LS=abs(self.Spin_A-self.Spin_B)
        else:
            print("Unknown option")
            ashexit()

    
    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, elems=None, Grad=False, Hessian=False, PC=False, numcores=None, label=None, charge=None, mult=None ):
        module_init_time=time.time()
        print(BC.OKBLUE,BC.BOLD, "------------RUNNING SPINPROJECTIONTHEORY INTERFACE-------------", BC.END)

        #Only ti


        #Changing inputfilename of theory1 and theory2. Must be done here
        self.theory1.filename=self.filename+"spinprojtheory1"
        self.theory2.filename=self.filename+"spinprojtheory2"
        #theory2 will read MOs from theory1 by default
        if self.reuseorbs is True:
            self.theory2.moreadfile=self.theory1.filename+".gbw"
        
        #RUNNING both theories
        HSenergy = self.theory1.run(current_coords=current_coords, elems=elems, PC=PC, numcores=numcores, Grad=Grad, charge=self.charge_1, mult=self.mult_1)
        BSenergy = self.theory2.run(current_coords=current_coords, elems=elems, PC=PC, numcores=numcores, Grad=Grad, charge=self.charge_2, mult=self.mult_2)

        #Some theories like CC_CBS_Theory may return both energy and energy componentsdict as a tuple
        #TODO: avoid this nasty fix
        if type(HSenergy) is tuple:
            componentsdict1=HSenergy[1]
            HSenergy=HSenergy[0]
        if type(BSenergy) is tuple:
            componentsdict2=BSenergy[1]
            BSenergy=BSenergy[0]

        #Grab S2 expectation values. Used by Yamaguchi
        if self.theory1.__class__.__name__ == "ORCATheory":
            HS_S2=ash.interfaces.interface_ORCA.grab_spin_expect_values_ORCA(self.theory1.filename+'.out')
        if self.theory2.__class__.__name__ == "ORCATheory":
            BS_S2=ash.interfaces.interface_ORCA.grab_spin_expect_values_ORCA(self.theory2.filename+'.out')
        #ONly problem is if we grab S2 values in CCSD(T) job we get the (wrong?) CCSD(T) S2 values instead of CCSD S2 values (as used in paper by Stanton,Chan)
        if self.theory1.__class__.__name__ == "CFourTheory":
            HS_S2=self.theory1.cfour_grab_spinexpect()
            print("HS_S2:", HS_S2)
        if self.theory2.__class__.__name__ == "CFourTheory":
            BS_S2=self.theory2.cfour_grab_spinexpect()
            print("BS_S2:", BS_S2)

        print("Spin coupling analysis")
        print("Spin Hamiltonian form: H= -2J*S_A*S_B")
        print("Local spins are: S_A = {}  S_B = {}".format(self.Spin_A,self.Spin_B))
        print("Assuming theory1 is High-spin state and theory2 is low-spin Broken-symmetry state.")
        print("High-spin state (M_S = {}) energy: {}".format((self.mult_1-1)/2, HSenergy))
        print("Broken-symmetry state (M_S = {}) energy: {}".format((self.mult_2-1)/2, BSenergy))
        print("Direct naive energy difference: {} Eh, {} kcal/mol, {} cm-1".format(HSenergy-BSenergy,(HSenergy-BSenergy)*ash.constants.harkcal,(HSenergy-BSenergy)*ash.constants.hartocm))
        print("<S**2>(High-Spin):", HS_S2)
        print("<S**2>(BS):", BS_S2)
        if BSenergy < HSenergy:
            print("System is ANTIFERROMAGNETIC")
        else:
            print("System is FERROMAGNETIC")
        print("")
        
        #Calculate J using the HS/BS energies and either spin-expectation values or total spin
        if self.jobtype == "Yamaguchi":
            J=ash.functions.functions_elstructure.Jcoupling_Yamaguchi(HSenergy,BSenergy,HS_S2,BS_S2)
        #Strong-interaction limit (bond-formation)
        elif self.jobtype == "Bencini":
            J=ash.functions.functions_elstructure.Jcoupling_Bencini(HSenergy,BSenergy,self.Spin_HS)
        #Weak-interaction limit (little overlap betwen orbitals)
        elif self.jobtype == "Noodleman":
            J=ash.functions.functions_elstructure.Jcoupling_Noodleman(HSenergy,BSenergy,self.Spin_HS)

        #Now  calculate new E of LS state from J
        #Projected energy of low-spin state
        #Lande formula: E(S) = -J[S(S+1)-SA(SA+1)-SB(SB+1)]
        #Multiple of J for HS and LS states
        Jspinmultiple_HS=self.Spin_HS*(self.Spin_HS+1)-self.Spin_A*(self.Spin_A+1)-self.Spin_B*(self.Spin_B+1)
        Jspinmultiple_LS=self.Spin_LS*(self.Spin_LS+1)-self.Spin_A*(self.Spin_A+1)-self.Spin_B*(self.Spin_B+1)
        #Energy difference between HS and LS in multiples of J
        Jmultiple_HSLS=Jspinmultiple_HS-Jspinmultiple_LS
        print("Jmultiple_HSLS:", Jmultiple_HSLS)
        print("{}J : {}".format(Jmultiple_HSLS,Jmultiple_HSLS*J))
        
        #Final energy of LS state by using J-multiple and energy of HS state
        E_proj=HSenergy+Jmultiple_HSLS*J
        print("Projected energy of state S={} (label: {}) state : {}".format(self.Spin_LS,label,E_proj))
        finalE=E_proj
        print_time_rel(module_init_time, modulename='SpinProjectionTheory run')
        return finalE
    

