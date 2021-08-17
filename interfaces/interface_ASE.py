import numpy as np
import time
import os
import copy


import constants
from functions.functions_general import print_line_with_mainheader, print_time_rel
from modules.module_singlepoint import Singlepoint


#Interface to limited parts of ASE


def Dynamics_ASE(fragment=None, theory=None, temperature=300, timestep=None, thermostat=None, simulation_steps=None, simulation_time=None,
                 barostat=None, trajectoryname="Trajectory_ASE", traj_frequency=1, coupling_freq=0.002, frozen_atoms=None, frozen_bonds=None,
                 frozen_angles=None, frozen_dihedrals=None, plumed_object=None):
    module_init_time = time.time()
    print_line_with_mainheader("ASE MOLECULAR DYNAMICS")
    
    #Delete old
    try:
        os.remove('md.log')
    except:
        pass
    
    
    
    
    
    if frozen_atoms==None: frozen_atoms=[]
    if frozen_bonds==None: frozen_bonds=[]
    if frozen_angles==None: frozen_angles=[]
    if frozen_dihedrals==None: frozen_dihedrals=[]
    try:
        from ase.constraints import FixAtoms, FixBondLengths, FixInternals
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet
        from ase.md.andersen import Andersen
        from ase.md.nvtberendsen import NVTBerendsen
        from ase.md.langevin import Langevin
        from ase.calculators.calculator import Calculator, FileIOCalculator
        from ase import units
        import ase.atoms
        print("Imported ASE")
    except:
        print("problem importing ase library. is ase installed?")
        print("Try: pip install ase")
        exit()

    #Print simulation info
    if simulation_steps == None and simulation_time == None:
        print("Either simulation_steps or simulation_time needs to be set")
        exit()
    if fragment == None:
        print("No fragment object. Exiting")
        exit()
    if simulation_time != None:
        simulation_steps=simulation_time/timestep
    if simulation_steps != None:
        simulation_time=simulation_steps*timestep
    timestep_fs=timestep*1000
    print("Simulation time: {} ps".format(simulation_time))
    print("Timestep: {} ps".format(timestep))
    print("Simulation steps: {}".format(simulation_steps))
    print("Temperature: {} K".format(temperature))
    print("")
    print("Trajectory write frequency:", traj_frequency)
    print("Number of frozen atoms:", len(frozen_atoms))
    print("Number of frozen bonds:", len(frozen_bonds))
    if len(frozen_atoms) < 50:
        print("Frozen atoms", frozen_atoms)
    if len(frozen_bonds) < 50:
        print("Frozen bonds", frozen_bonds)
    #Delete old stuff
    print("Removing old trajectory file if present")
    try:
        os.remove(trajectoryname+'.xyz')
        os.remove(trajectoryname+'.traj')
        os.remove("md.log")
    except:
        pass

    class ASHcalc(Calculator):
        def __init__(self, fragment=None, theory=None, plumed=None):
            self.gradientcalls=0
            self.fragment=fragment
            self.theory=theory
            self.results={}
            self.name='ash'
            self.parameters={}
            self.atoms=None
            self.forces=[]
            self.plumedobj=plumed
        def get_potential_energy(self, atomsobj):
            return self.potenergy
        def get_forces(self, atomsobj):
            print("")
            print("---------------------------------")
            print("")
            print("called ASHcalc get_forces")
            #print("atomsobj:", atomsobj)
            #print(atoms.__dict__)
            #silly. TODO: replace with coords comparison instead
            print("atomsobj.get_positions():", atomsobj.get_positions())
            print("fragment.coords:", fragment.coords)
            # Check if coordinates have changed. If not, return old forces
            if np.array_equal(atomsobj.get_positions(), fragment.coords) == True:
                #coordinates have not changed
                print("Same coords.")
                if len(self.forces)==0:
                    print("No forces available (1st step?). Will do calulation")
                else:
                    print("Returning old forces")
                    return self.forces
            print("Will calculate new forces")
            
            self.gradientcalls+=1

            #Copy ASE coords into ASH fragment
            self.fragment.coords=copy.copy(atomsobj.positions)
            #print("Current coordinates:", self.fragment.coords)
            #Calculate E+G
            energy, gradient = Singlepoint(theory=self.theory, fragment=self.fragment, Grad=True)
            self.potenergy=energy*constants.hartoeV
            self.forces=-gradient* units.Hartree / units.Bohr
            print("potenergy:", self.potenergy)
            print("self.forces:", self.forces)
            #DO PLUMED-STEP HERE
            if self.plumedobj!=None:
                print("Plumed active.")
                print("Calling Plumed")
                energy,forces=self.plumedobj.run(coords=fragment.coords, forces=self.forces, step=self.gradientcalls)
                print("energy:", energy)
                print("forces:", forces)
                #self.potenergy, self.forces = plumed_ash(energy,forces)
                #energy, forces = plumedlib.cv_calculation(istep, pos, vel, box, jobforces, jobenergy)
            
            
            print("Done with ASHcalc get_forces")
            return self.forces
        
    #Option 2: Dummy ASE class where we create the attributes and methods we want
    #Too complicated

    #Creating ASE-style atoms object
    #atoms = ASEatoms(fragment=fragment, theory=theory)
    print("Creating ASE atoms object")
    atoms = ase.atoms.Atoms(fragment.elems,positions=fragment.coords)

    #ASH calculator for ASE
    print("Creating ASH-ASE calculator")
    calc= ASHcalc(fragment=fragment, theory=theory, plumed=plumed_object)
    atoms.calc = calc

    print(atoms)
    print(atoms.__dict__)

    #CONSTRAINTS AND FROZEN ATOMS
    #Frozen atoms
    print("Adding possible constraints")
    all_constraints=[]
    if len(frozen_atoms) > 0:
        print("Freezing atoms")
        frozenatom_cons = FixAtoms(indices=frozen_atoms)
        all_constraints.append(frozenatom_cons)
    #Constraints
    
    if len(frozen_bonds) > 0:
        print("Freezing bonds")
        frozenbondlength_cons = FixBondLengths(frozen_bonds)
        all_constraints.append(frozenbondlength_cons)
    #NOTE: Angles and dihedrals are not tested!
    if len(frozen_angles) > 0:
        print("Freezing angles")
        for angle in frozen_angles:
            angle1 = [atoms.get_angle(*angle), angle]
            frozenangle_cons = FixInternals(angles_deg=[angle1])
            all_constraints.append(frozenangle_cons)
    if len(frozen_dihedrals) > 0:
        print("Freezing dihedrals")
        for dihedral in frozen_dihedrals:
            dihedral1 = [atoms.get_angle(*dihedral), dihedral]
            frozendihedral_cons = FixInternals(angles_deg=[dihedral1])
            all_constraints.append(frozendihedral_cons)
    #Adding all constraints
    atoms.set_constraint(all_constraints)
    print("Printing ASE atoms object:", atoms.__dict__)
    
    
    
    # Set the momenta corresponding to T=300K
    print("Calling MaxwellBoltzmannDistribution")
    #MaxwellBoltzmannDistribution(atoms, temp=None, temperature_K=temperature)
    MaxwellBoltzmannDistribution(atoms, temp=None, temperature_K=temperature)
    
    #NVE VV:
    if thermostat==None and barostat==None:
        print("Setting up VelocityVerlet")
        dyn = VelocityVerlet(atoms, timestep_fs * units.fs, trajectory=trajectoryname+'.traj', logfile='md.log')
    elif thermostat=="Langevin":
        print("Setting up Langevin thermostat")
        friction_coeff=coupling_freq
        dyn = Langevin(atoms, timestep_fs*units.fs, temperature*units.kB, friction_coeff, trajectory=trajectoryname+'.traj', logfile='md.log')
    elif thermostat=="Andersen":
        collision_prob=coupling_freq
        dyn = Andersen(atoms, timestep_fs*units.fs, temperature*units.kB, collision_prob, trajectory=trajectoryname+'.traj', logfile='md.log')
    elif thermostat=="Berendsen":
        dyn = NVTBerendsen(atoms, timestep_fs*units.fs, temperature*units.kB, taut=coupling_freq*1000*units.fs, trajectory=trajectoryname+'.traj', logfile='md.log')
    elif thermostat=="NoseHoover":
        print("nosehoover")
    else:
        print("Unknown thermostat/barostat. Exiting")
        exit()


    def printenergy(a):
        """Function to print the potential, kinetic and total energy"""
        epot = a.get_potential_energy() / len(a)
        ekin = a.get_kinetic_energy() / len(a)
        print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
            'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

    def print_step(a=atoms):
        print("-"*30)
        print("Step ", a.calc.gradientcalls)
        print("Pot. Energy (eV):", a.get_potential_energy())
        print("Kin. Energy (eV):", a.get_kinetic_energy())
        print("Total energy (eV):", a.get_total_energy())
        print("Temperature:", a.get_temperature(), "K" )
        print("-"*30)
    def write_traj(a=atoms, trajname=trajectoryname):
        print("Writing trajectory")
        fragment.write_xyzfile(xyzfilename=trajname+'.xyz', writemode='a')

    dyn.attach(print_step, interval=1)
    dyn.attach(write_traj, interval=traj_frequency)
    print("")
    print("")
    print("Running dynamics")
    print("simulation_steps:", simulation_steps)
    dyn.run(simulation_steps)

    print_time_rel(module_init_time, modulename='Dynamics_ASE', moduleindex=1)