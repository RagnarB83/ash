import numpy as np
from modules.module_singlepoint import Singlepoint
#Interface to ASE

#Function to load whole AS
def load_ASE():
    try:
        import ase
    except:
        print("problem importing ASE")
        exit()

def Dynamics_ASE(fragment=None, theory=None, temperature=300, timestep=None):
    
    try:
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet
        from ase import units
    except:
        print("problem importing ase library. is ase installed?")
        exit()


    timestep_fs=timestep*1000



    #Create ASE atoms object from ASH fragment
    #Associate theory with it
    #atoms= #
    #Option 1: import actual ASE class and use?
    #Problem: requires modifying ASE as we have to create a separate calculator etc.
    #Option 2: Dummy ASE class where we create the attributes and methods we want
    class ASEatoms:
        def __init__(self, fragment=None, theory=None):
            self.fragment=fragment
            self.theory=theory
            self.numatoms=fragment.numatoms
            self.masses=np.array(fragment.list_of_masses)
            self.momenta=[]
            #self.current_positions=np.array(fragment.coords)
            self.constraints=[]
        def get_positions(self):
            print("Called get positions")
            print(fragment.coords)
            #exit()
            #Convert from Angstrom to something else?
            return self.fragment.coords
        def set_positions(self,pos):
            print("Called set_positions")
            print("pos:", pos)
            #exit()
            self.fragment.coords=pos
        def get_potential_energy(self):
            print("Calling get pot energy")
            return self.potenergy
        def get_kinetic_energy(self):
            print("Called get_kinetic_energy")
            #TODO: Calculate from momenta
            exit()
            self.kinenergy=6.00
            return self.kinenergy
        def get_forces(self, md=False):
            print("Called get_forces")
            print(self.fragment.coords)
            self.potenergy, self.gradient = Singlepoint(theory=self.theory, fragment=self.fragment, Grad=True)
            self.forces=self.gradient*-1
            print("self.potenergy:", self.potenergy)
            print("self.forces:", self.forces)
            if md == False:
                print("Called md false")
            else:
                print("Calling md true")

            return self.forces
        def get_masses(self):
            print("Called get_masses")
            return self.masses
        def set_momenta(self, momenta, apply_constraint=False):
            print("Called set_momenta")
            if apply_constraint ==  True:
                print("appcon true")
                exit()
            
            self.momenta=momenta
        def get_momenta(self):
            print("Called get_momenta")
            return self.momenta
        def has(self,bla):
            print("called has")
            if len(self.momenta)==0:
                return False
            else:
                return True
        def __len__(self):
            return self.numatoms

    #Creating ASE-style atoms object
    atoms = ASEatoms(fragment=fragment, theory=theory)
    
    print(atoms)
    print(atoms.__dict__)
    
    # Set the momenta corresponding to T=300K
    print("Called Maxwell")
    MaxwellBoltzmannDistribution(atoms, temp=None, temperature_K=temperature)
    print("Maxwell done")
    # We want to run MD with constant energy using the VelocityVerlet algorithm.
    print("Called VelocityVerlet")
    dyn = VelocityVerlet(atoms, timestep_fs * units.fs)

    #https://wiki.fysik.dtu.dk/ase/tutorials/md/md.html

    def printenergy(a):
        """Function to print the potential, kinetic and total energy"""
        epot = a.get_potential_energy() / len(a)
        ekin = a.get_kinetic_energy() / len(a)
        print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
            'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))


    # Now run the dynamics
    #printenergy(atoms)
    dyn.run(steps=3)
    #for i in range(5):
    #    dyn.run()
    #    printenergy(atoms)