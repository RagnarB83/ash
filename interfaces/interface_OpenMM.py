import time
import numpy as np
import constants
import os
from sys import stdout

from functions_general import BC,print_time_rel,listdiff,printdebug,print_line_with_mainheader
from module_coords import write_pdbfile

class OpenMMTheory:
    def __init__(self, printlevel=2, platform='CPU', numcores=None, 
                 CHARMMfiles=False, psffile=None, charmmtopfile=None, charmmprmfile=None,
                 GROMACSfiles=False, gromacstopfile=None, grofile=None, gromacstopdir=None,
                 Amberfiles=False, amberprmtopfile=None,
                 xmlfile=None, pdbfile=None, use_parmed=False,
                 do_energy_decomposition=False,
                 periodic=False, charmm_periodic_cell_dimensions=None, customnonbondedforce=False,
                 periodic_nonbonded_cutoff=12, dispersion_correction=True, 
                 switching_function_distance=10,
                 ewalderrortolerance=1e-5, PMEparameters=None,
                 delete_QM1_MM1_bonded=False, applyconstraints=False):
        
        module_init_time = time.time()
        # OPEN MM load
        try:
            import simtk.openmm.app
            import simtk.unit
            print("Imported OpenMM library version:", simtk.openmm.__version__)
            #import simtk.openmm
        except ImportError:
            raise ImportError(
                "OpenMM requires installing the OpenMM package. Try: conda install -c omnia openmm  \
                Also see http://docs.openmm.org/latest/userguide/application.html")

        print(BC.WARNING, BC.BOLD, "------------Defining OpenMM object-------------", BC.END)
        #Printlevel
        self.printlevel=printlevel

        #Load Parmed if requested
        if use_parmed == True:
            print("Using Parmed to read topologyfiles")
            try:
                import parmed
            except:
                print("Problem importing parmed Python library")
                print("Make sure parmed is present in your Python.")
                print("Parmed can be installed using pip: pip install parmed")
                exit()


        # Setting for controlling whether QM1-MM1 bonded terms are deleted or not in a QM/MM job
        #See modify_bonded_forces
        #TODO: Move option to module_QMMM instead
        self.delete_QM1_MM1_bonded=delete_QM1_MM1_bonded

        #Platform (CPU, CUDA, OpenCL) and Parallelization
        self.platform_choice=platform
        #CPU: Control either by provided numcores keyword, or by setting env variable: $OPENMM_CPU_THREADS in shell before running.
        self.properties= {}
        if self.platform_choice == 'CPU':
            print("Using platform: CPU")
            if numcores != None:
                print("Numcores variable provided to OpenMM object. Will use {} cores with OpenMM".format(numcores))
                self.properties["Threads"]=str(numcores)
            else:
                print("No numcores variable provided to OpenMM object")
                print("Checking if OPENMM_CPU_THREADS shell variable is presentþ")
                try:
                    print("OpenMM will use {} threads according to environment variable: OPENMM_CPU_THREADS".format(os.environ["OPENMM_CPU_THREADS"]))
                except:
                    print("OPENMM_CPU_THREADS environment variable not set. OpenMM will choose number of physical cores present.")
        else:
            print("Using platform:", self.platform_choice)
        
        #Whether to do energy decomposition of MM energy or not. Takes time. Can be turned off for MD runs
        self.do_energy_decomposition=do_energy_decomposition
        #Initializing
        self.coords=[]
        self.charges=[]
        self.Periodic = periodic
        self.ewalderrortolerance=ewalderrortolerance

        #Whether to apply constraints or not when calculating MM energy
        self.applyconstraints=applyconstraints

        #Switching function distance in Angstrom
        self.switching_function_distance=switching_function_distance

        #Residue names,ids,segments,atomtypes of all atoms of system.
        # Grabbed below from PSF-file. Information used to write PDB-file
        self.resnames=[]
        self.resids=[]
        self.segmentnames=[]
        self.atomtypes=[]
        self.atomnames=[]
            
        #OpenMM things
        self.openmm=simtk.openmm
        self.simulationclass=simtk.openmm.app.simulation.Simulation

        self.unit=simtk.unit
        self.Vec3=simtk.openmm.Vec3

        #Positions. Generally not used but can be if if e.g. grofile has been read in.
        #Purpose: set virtual sites etc.
        self.positions=None

        #TODO: Should we keep this? Probably not. Coordinates would be handled by ASH.
        #PDB_ygg_frag = Fragment(pdbfile=pdbfile, conncalc=False)
        #self.coords=PDB_ygg_frag.coords
        print_time_rel(module_init_time, modulename="import openMM")
        timeA = time.time()

        self.Forcefield=None
        #What type of forcefield files to read. Reads in different way.
        print("Now reading forcefield files")
        print("Note: OpenMM will fail in this step if parameters are missing in topology and parameter files (e.g. nonbonded entries)")
        # #Always creates object we call self.forcefield that contains topology attribute
        if CHARMMfiles is True:
            print("Reading CHARMM files")
            self.psffile=psffile
            if use_parmed == True:
                print("Using Parmed.")
                self.psf = parmed.charmm.CharmmPsfFile(psffile)
                self.params = parmed.charmm.CharmmParameterSet(charmmtopfile, charmmprmfile)
                #Grab resnames from psf-object. Different for parmed object
                #Note: OpenMM uses 0-indexing
                self.resnames=[self.psf.atoms[i].residue.name for i in range(0,len(self.psf.atoms))]
                self.resids=[self.psf.atoms[i].residue.idx for i in range(0,len(self.psf.atoms))]
                self.segmentnames=[self.psf.atoms[i].residue.segid for i in range(0,len(self.psf.atoms))]
                #self.atomtypes=[self.psf.atoms[i].attype for i in range(0,len(self.psf.atoms))]
                #TODO: Note: For atomnames it seems OpenMM converts atomnames to its own. Perhaps not useful
                self.atomnames=[self.psf.atoms[i].name for i in range(0,len(self.psf.atoms))]

            else:
                # Load CHARMM PSF files via native routine.
                self.psf = simtk.openmm.app.CharmmPsfFile(psffile)                
                self.params = simtk.openmm.app.CharmmParameterSet(charmmtopfile, charmmprmfile)
                #Grab resnames from psf-object
                #Note: OpenMM uses 0-indexing
                self.resnames=[self.psf.atom_list[i].residue.resname for i in range(0,len(self.psf.atom_list))]
                self.resids=[self.psf.atom_list[i].residue.idx for i in range(0,len(self.psf.atom_list))]
                self.segmentnames=[self.psf.atom_list[i].system for i in range(0,len(self.psf.atom_list))]
                self.atomtypes=[self.psf.atom_list[i].attype for i in range(0,len(self.psf.atom_list))]
                #TODO: Note: For atomnames it seems OpenMM converts atomnames to its own. Perhaps not useful
                self.atomnames=[self.psf.atom_list[i].name for i in range(0,len(self.psf.atom_list))]

            self.topology = self.psf.topology
            self.forcefield = self.psf


        elif GROMACSfiles is True:
            print("Reading Gromacs files")
            #Reading grofile, not for coordinates but for periodic vectors
            if use_parmed == True:
                print("Using Parmed.")
                print("GROMACS top dir:", gromacstopdir)
                parmed.gromacs.GROMACS_TOPDIR = gromacstopdir
                print("Reading GROMACS GRO file: ", grofile)
                gmx_gro = parmed.gromacs.GromacsGroFile.parse(grofile)
                
                print("Reading GROMACS topology file: ", gromacstopfile)
                gmx_top = parmed.gromacs.GromacsTopologyFile(gromacstopfile)

                #Getting PBC parameters
                gmx_top.box = gmx_gro.box
                gmx_top.positions = gmx_gro.positions
                self.positions = gmx_top.positions
                
                self.topology = gmx_top.topology
                self.forcefield = gmx_top
                
            else:
                print("Using built-in OpenMM routines to read GROMACS topology")
                print("Warning: may fail if virtual sites present (e.g. TIP4P residues)")
                print("Use parmed=True  to avoid")
                gro = simtk.openmm.app.GromacsGroFile(grofile)
                self.grotop = simtk.openmm.app.GromacsTopFile(gromacstopfile, periodicBoxVectors=gro.getPeriodicBoxVectors(),
                                    includeDir=gromacstopdir)

                self.topology = self.grotop.topology
                self.forcefield=self.grotop
            # Create an OpenMM system by calling createSystem on grotop
            #self.system = self.grotop.createSystem(nonbondedMethod=simtk.openmm.app.NoCutoff,
            #                                    nonbondedCutoff=1 * simtk.openmm.unit.nanometer)

        elif Amberfiles is True:
            print("Reading Amber files")
            print("Warning: Only new-style Amber7 prmtopfile will work")
            print("Warning: Will take periodic boundary conditions from PRMtop file.")
            if use_parmed == True:
                print("Using Parmed to read Amber files")
                self.prmtop = parmed.load_file(amberprmtopfile)
            else:
                print("Using built-in OpenMM routines to read Amber files.")
                #Note: Only new-style Amber7 prmtop files work
                self.prmtop = simtk.openmm.app.AmberPrmtopFile(amberprmtopfile)
            self.topology = self.prmtop.topology
            self.forcefield= self.prmtop
            # Create an OpenMM system by calling createSystem on prmtop
            #self.system = self.prmtop.createSystem(nonbondedMethod=simtk.openmm.app.NoCutoff,
            #                                    nonbondedCutoff=1 * simtk.openmm.unit.nanometer)
            
            #forces = {self.system.getForce(index).__class__.__name__: self.system.getForce(index) for index in range(self.system.getNumForces())}
            #self.nonbonded_force = forces['NonbondedForce']
        else:
            print("Reading OpenMM XML forcefield file and PDB file")
            #This would be regular OpenMM Forcefield definition requiring XML file
            #Topology from PDBfile annoyingly enough
            pdb = simtk.openmm.app.PDBFile(pdbfile)
            self.topology = pdb.topology
            #Todo: support multiple xml file here
            #forcefield = simtk.openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
            self.forcefield = simtk.openmm.app.ForceField(xmlfile)
            #self.system = self.forcefield.createSystem(nonbondedMethod=simtk.openmm.app.NoCutoff,
            #                                    nonbondedCutoff=1 * simtk.openmm.unit.nanometer)
            
            #forces = {self.system.getForce(index).__class__.__name__: self.system.getForce(index) for index in range(self.system.getNumForces())}
            #self.nonbonded_force = forces['NonbondedForce']

        # Deal with possible 4/5 site water model like TIP4P
        #NOTE: EXPERIMENTAL
        #NOTE: We have no positions here. Make separate callable function?????
        
        #if watermodel != None:
        #    print("watermodel:", watermodel)
        #    modeller = simtk.openmm.app.Modeller(self.topology, pdb.positions)
        #    modeller.addExtraParticles(self.forcefield)
        #    simtk.openmm.app.app.PDBFile.writeFile(modeller.topology, modeller.positions, open('test-water.pdb', 'w'))

        #Now after topology is defined we can create system
        #Get number of atoms
        self.numatoms=int(self.forcefield.topology.getNumAtoms())
        print("Number of atoms in OpenMM object", self.numatoms)
        
        #Setting active and frozen variables once topology is in place
        #NOTE: Is this actually used?
        #NOTE: Disabled for now
        #self.set_active_and_frozen_regions(active_atoms=active_atoms, frozen_atoms=frozen_atoms)


        #Periodic or non-periodic ystem
        if self.Periodic is True:
            print("System is periodic")
            print("Nonbonded cutoff is {} Angstrom".format(periodic_nonbonded_cutoff))
            #Parameters here are based on OpenMM DHFR example
            
            if CHARMMfiles is True:
                print("Using CHARMM files")

                if charmm_periodic_cell_dimensions == None:
                    print("Error: When using CHARMMfiles and Periodic=True, charmm_periodic_cell_dimensions keyword needs to be supplied")
                    print("Example: periodic_cell_dimensions= [200, 200, 200, 90, 90, 90]  in Angstrom and degrees")
                    exit()
                self.charmm_periodic_cell_dimensions = charmm_periodic_cell_dimensions
                print("Periodic cell dimensions:", charmm_periodic_cell_dimensions)
                self.a = charmm_periodic_cell_dimensions[0] * self.unit.angstroms
                self.b = charmm_periodic_cell_dimensions[1] * self.unit.angstroms
                self.c = charmm_periodic_cell_dimensions[2] * self.unit.angstroms
                if use_parmed == True:
                    self.forcefield.box=[self.a, self.b, self.c, charmm_periodic_cell_dimensions[3], charmm_periodic_cell_dimensions[4], charmm_periodic_cell_dimensions[5]]
                    print("Set box vectors:", self.forcefield.box)
                else:
                    self.forcefield.setBox(self.a, self.b, self.c, alpha=self.unit.Quantity(value=charmm_periodic_cell_dimensions[3], unit=self.unit.degree), 
                    beta=self.unit.Quantity(value=charmm_periodic_cell_dimensions[3], unit=self.unit.degree), gamma=self.unit.Quantity(value=charmm_periodic_cell_dimensions[3], unit=self.unit.degree))
                    #self.forcefield.setBox(self.a, self.b, self.c)
                    #print(self.forcefield.__dict__)
                    print("Set box vectors:", self.forcefield.box_vectors)
                #NOTE: SHould this be made more general??
                #print("a,b,c:", self.a, self.b, self.c)
                #print("box in self.forcefield", self.forcefield.get_box())

                #exit()
                
                self.system = self.forcefield.createSystem(self.params, nonbondedMethod=simtk.openmm.app.PME,
                                            nonbondedCutoff=periodic_nonbonded_cutoff * self.unit.angstroms, switchDistance=switching_function_distance*self.unit.angstroms)
            elif GROMACSfiles is True:
                #NOTE: Gromacs has read PBC info from Gro file already
                print("Ewald Error tolerance:", self.ewalderrortolerance)
                #Note: Turned off switchDistance. Not available for GROMACS?
                #
                self.system = self.forcefield.createSystem(nonbondedMethod=simtk.openmm.app.PME,
                                            nonbondedCutoff=periodic_nonbonded_cutoff * self.unit.angstroms, ewaldErrorTolerance=self.ewalderrortolerance)
            elif Amberfiles is True:
                #NOTE: Amber-interface has read PBC info from prmtop file already
                self.system = self.forcefield.createSystem(nonbondedMethod=simtk.openmm.app.PME,
                                            nonbondedCutoff=periodic_nonbonded_cutoff * self.unit.angstroms)
            else:
                self.system = self.forcefield.createSystem(nonbondedMethod=simtk.openmm.app.PME,
                                            nonbondedCutoff=periodic_nonbonded_cutoff * self.unit.angstroms, switchDistance=switching_function_distance*self.unit.angstroms)
                

            #TODO: Customnonbonded force option. Currently disabled
            print("OpenMM system created")
            print("Periodic vectors:", self.system.getDefaultPeriodicBoxVectors())
            #Force modification here
            print("OpenMM Forces defined:", self.system.getForces())


            #PRINTING PROPERTIES OF NONBONDED FORCE BELOW
            for i,force in enumerate(self.system.getForces()):
                if isinstance(force, simtk.openmm.CustomNonbondedForce):
                    #NOTE: THIS IS CURRENTLY NOT USED
                    pass
                    #print('CustomNonbondedForce: %s' % force.getUseSwitchingFunction())
                    #print('LRC? %s' % force.getUseLongRangeCorrection())
                    #force.setUseLongRangeCorrection(False)
                elif isinstance(force, simtk.openmm.NonbondedForce):
                    #Turn Dispersion correction on/off depending on user
                    #NOTE: Default: False   To be revisited

                    #NOte:
                    force.setUseDispersionCorrection(dispersion_correction)

                    #Modify PME Parameters if desired
                    #force.setPMEParameters(1.0/0.34, fftx, ffty, fftz)
                    if PMEparameters != None:
                        print("Changing PME parameters")
                        force.setPMEParameters(PMEparameters[0], PMEparameters[1], PMEparameters[2], PMEparameters[3])
                    #force.setSwitchingDistance(switching_function_distance)
                    #if switching_function == True:
                    #    force.setUseSwitchingFunction(switching_function)
                    #    #Switching distance in nm. To be looked at further
                    #   force.setSwitchingDistance(switching_function_distance)
                    #    print('SwitchingFunction distance: %s' % force.getSwitchingDistance())
                    print("Nonbonded force settings (after all modifications):")
                    print("Periodic cutoff distance: {}".format(force.getCutoffDistance()))
                    print('Use SwitchingFunction: %s' % force.getUseSwitchingFunction())
                    print('SwitchingFunction distance: {}'.format(force.getSwitchingDistance()))
                    print('Use Long-range Dispersion correction: %s' % force.getUseDispersionCorrection())

                    print("PME Parameters:", force.getPMEParameters())

                    # Set PME Parameters if desired
                    #force.setPMEParameters(3.285326106/self.unit.nanometers,60, 64, 60) 
                    #Keeping default for now
                    
                    self.nonbonded_force=force
                    # NOTE: These are hard-coded!
                    
            #Set charges in OpenMMobject by taking from Force
            print("Setting charges")
            self.getatomcharges(self.nonbonded_force)
                    
            
        #Non-Periodic
        else:
            print("System is non-periodic")

            if CHARMMfiles is True:
                self.system = self.forcefield.createSystem(self.params, nonbondedMethod=simtk.openmm.app.NoCutoff,
                                            nonbondedCutoff=1000 * simtk.openmm.unit.angstroms)
            else:
                self.system = self.forcefield.createSystem(nonbondedMethod=simtk.openmm.app.NoCutoff,
                                            nonbondedCutoff=1000 * simtk.openmm.unit.angstroms)

            print("OpenMM system created")
            print("OpenMM Forces defined:", self.system.getForces())
            print("")
            for i,force in enumerate(self.system.getForces()):
                if isinstance(force, simtk.openmm.NonbondedForce):
                    self.getatomcharges(force)
                    self.nonbonded_force=force

            #print("original forces: ", forces)
            # Get charges from OpenMM object into self.charges
            #self.getatomcharges(forces['NonbondedForce'])
            #print("self.system.getForces():", self.system.getForces())
            #self.getatomcharges(self.system.getForces()[6])
            

            #CASE CUSTOMNONBONDED FORCE
            #REPLACING REGULAR NONBONDED FORCE
            if customnonbondedforce is True:

                #Create CustomNonbonded force
                for i,force in enumerate(self.system.getForces()):
                    if isinstance(force, self.openmm.NonbondedForce):
                        custom_nonbonded_force,custom_bond_force = create_cnb(self.system.getForces()[i])
                print("1custom_nonbonded_force:", custom_nonbonded_force)
                print("num exclusions in customnonb:", custom_nonbonded_force.getNumExclusions())
                print("num 14 exceptions in custom_bond_force:", custom_bond_force.getNumBonds())
                
                #TODO: Deal with frozen regions. NOT YET DONE
                #Frozen-Act interaction
                #custom_nonbonded_force.addInteractionGroup(self.frozen_atoms,self.active_atoms)
                #Act-Act interaction
                #custom_nonbonded_force.addInteractionGroup(self.active_atoms,self.active_atoms)
                #print("2custom_nonbonded_force:", custom_nonbonded_force)
            
                #Pointing self.nonbonded_force to CustomNonBondedForce instead of Nonbonded force
                self.nonbonded_force = custom_nonbonded_force
                print("self.nonbonded_force:", self.nonbonded_force)
                self.custom_bondforce = custom_bond_force
                
                #Update system with new forces and delete old force
                self.system.addForce(self.nonbonded_force) 
                self.system.addForce(self.custom_bondforce) 
                
                #Remove oldNonbondedForce
                for i,force in enumerate(self.system.getForces()):
                    if isinstance(force, self.openmm.NonbondedForce):
                        self.system.removeForce(i)



        print_time_rel(timeA, modulename="system create")
        timeA = time.time()

        #constraints=simtk.openmm.app.HBonds, AllBonds, HAngles
        # Remove Frozen-Frozen interactions
        #Todo: Will be requested by QMMM object so unnecessary unless during pure MM??
        #if frozen_atoms is not None:
        #    print("Removing Frozen-Frozen interactions")
        #    self.addexceptions(frozen_atoms)


        #Modify particle masses in system object. For freezing atoms
        #for i in self.frozen_atoms:
        #    self.system.setParticleMass(i, 0 * simtk.openmm.unit.dalton)
        #print_time_rel(timeA, modulename="frozen atom setup")
        #timeA = time.time()

        #Modifying constraints after frozen-atom setting
        #print("Constraints:", self.system.getNumConstraints())

        #Finding defined constraints that involved frozen atoms. add to remove list
        #removelist=[]
        #for i in range(0,self.system.getNumConstraints()):
        #    constraint=self.system.getConstraintParameters(i)
        #    if constraint[0] in self.frozen_atoms or constraint[1] in self.frozen_atoms:
        #        #self.system.removeConstraint(i)
        #        removelist.append(i)

        #print("removelist:", removelist)
        #print("length removelist", len(removelist))
        #Remove constraints
        #removelist.reverse()
        #for r in removelist:
        #    self.system.removeConstraint(r)

        #print("Constraints:", self.system.getNumConstraints())
        #print_time_rel(timeA, modulename="constraint fix")
        timeA = time.time()
    
        #Platform
        print("Hardware platform:", self.platform_choice)
        self.platform = simtk.openmm.Platform.getPlatformByName(self.platform_choice)


        #Create simulation
        self.create_simulation()

        #Old:
        #NOTE: If self.system is modified then we have to remake self.simulation
        #self.simulation = simtk.openmm.app.simulation.Simulation(self.topology, self.system, self.integrator,self.platform)
        #self.simulation = self.simulationclass(self.topology, self.system, self.integrator,self.platform)




        print_time_rel(timeA, modulename="simulation setup")
        timeA = time.time()
        print_time_rel(module_init_time, modulename="OpenMM object creation")
        
    def set_active_and_frozen_regions(self, active_atoms=None, frozen_atoms=None):
        #FROZEN AND ACTIVE ATOMS
        self.allatoms=list(range(0,self.numatoms))
        if active_atoms is None and frozen_atoms is None:
            print("All {} atoms active, no atoms frozen".format(len(self.allatoms)))
            self.frozen_atoms = []
        elif active_atoms is not None and frozen_atoms is None:
            self.active_atoms=active_atoms
            self.frozen_atoms=listdiff(self.allatoms,self.active_atoms)
            print("{} active atoms, {} frozen atoms".format(len(self.active_atoms),len(self.frozen_atoms)))
            #listdiff
        elif frozen_atoms is not None and active_atoms is None:
            self.frozen_atoms = frozen_atoms
            self.active_atoms = listdiff(self.allatoms, self.frozen_atoms)
            print("{} active atoms, {} frozen atoms".format(len(self.active_atoms),len(self.frozen_atoms)))
        else:
            print("active_atoms and frozen_atoms can not be both defined")
            exit(1)


    #This removes interactions between particles in a region (e.g. QM-QM or frozen-frozen pairs)
    # Give list of atom indices for which we will remove all pairs
    #Todo: Way too slow to do for big list of e.g. frozen atoms but works well for qmatoms list size
    # Alternative: Remove force interaction and then add in the interaction of active atoms to frozen atoms
    # should be reasonably fast
    # https://github.com/openmm/openmm/issues/2124
    #https://github.com/openmm/openmm/issues/1696
    def addexceptions(self,atomlist):
        timeA=time.time()
        import itertools
        print("Add exceptions/exclusions. Removing i-j interactions for list :", len(atomlist), "atoms")

        #Has duplicates
        #[self.nonbonded_force.addException(i,j,0, 0, 0, replace=True) for i in atomlist for j in atomlist]
        #https://stackoverflow.com/questions/942543/operation-on-every-pair-of-element-in-a-list
        #[self.nonbonded_force.addException(i,j,0, 0, 0, replace=True) for i,j in itertools.combinations(atomlist, r=2)]
        numexceptions=0
        printdebug("self.system.getForces() ", self.system.getForces())
        #print("self.nonbonded_force:", self.nonbonded_force)
        
        for force in self.system.getForces():
            printdebug("force:", force)
            if isinstance(force, self.openmm.NonbondedForce):
                print("Case Nonbondedforce. Adding Exception for ij pair")
                for i in atomlist:
                    for j in atomlist:
                        printdebug("i,j : {} and {} ".format(i,j))
                        force.addException(i,j,0, 0, 0, replace=True)

                        #NOTE: Case where there is also a CustomNonbonded force present (GROMACS interface). 
                        # Then we have to add exclusion there too to avoid this issue: https://github.com/choderalab/perses/issues/357
                        #Basically both nonbonded forces have to have same exclusions (or exception where chargepro=0, eps=0)
                        #TODO: This leads to : Exception: CustomNonbondedForce: Multiple exclusions are specified for particles
                        #Basically we have to inspect what is actually present in CustomNonbondedForce
                        #for force in self.system.getForces():
                        #    if isinstance(force, self.openmm.CustomNonbondedForce):
                        #        force.addExclusion(i,j)

                        numexceptions+=1
            elif isinstance(force, self.openmm.CustomNonbondedForce):
                print("Case CustomNonbondedforce. Adding Exclusion for kl pair")
                for k in atomlist:
                    for l in atomlist:
                        #print("k,l : ", k,l)
                        force.addExclusion(k,l)
                        numexceptions+=1
        print("Number of exceptions/exclusions added: ", numexceptions)
        printdebug("self.system.getForces() ", self.system.getForces())
        #Seems like updateParametersInContext does not reliably work here so we have to remake the simulation instead
        #Might be bug (https://github.com/openmm/openmm/issues/2709). Revisit
        #self.nonbonded_force.updateParametersInContext(self.simulation.context)
        self.create_simulation()
        
        print_time_rel(timeA, modulename="add exception")
    #Run: coords or framents can be given (usually coords). qmatoms in order to avoid QM-QM interactions (TODO)
    #Probably best to do QM-QM exclusions etc. in a separate function though as we want run to be as simple as possible
    #qmatoms list provided for generality of MM objects. Not used here for now
    
    # Create/update simulation from scratch or after system has been modified (force modification or even deletion)
    def create_simulation(self, timestep=0.001, integrator='VerletIntegrator', coupling_frequency=None, temperature=None):
        timeA=time.time()
        print("Creating/updating OpenMM simulation object")
        printdebug("self.system.getForces() ", self.system.getForces())
        #self.integrator = self.langevinintegrator(0.0000001 * self.unit.kelvin,  # Temperature of heat bath
        #                                1 / self.unit.picosecond,  # Friction coefficient
        #                                0.002 * self.unit.picoseconds)  # Time step

        #Integrators: LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator, BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
        if integrator == 'VerletIntegrator':
            print("Defining verlet")
            self.integrator = self.openmm.VerletIntegrator(timestep*self.unit.picoseconds)
        elif integrator == 'VariableVerletIntegrator':
            self.integrator = self.openmm.VariableVerletIntegrator(timestep*self.unit.picoseconds)
        elif integrator == 'LangevinIntegrator':
            self.integrator = self.openmm.LangevinIntegrator(temperature*self.unit.kelvin, coupling_frequency/self.unit.picosecond, timestep*self.unit.picoseconds)
        elif integrator == 'LangevinMiddleIntegrator':
            self.integrator = self.openmm.LangevinMiddleIntegrator(temperature*self.unit.kelvin, coupling_frequency/self.unit.picosecond, timestep*self.unit.picoseconds)
        elif integrator == 'NoseHooverIntegrator':
            self.integrator = self.openmm.NoseHooverIntegrator(temperature*self.unit.kelvin, coupling_frequency/self.unit.picosecond, timestep*self.unit.picoseconds)
        #NOTE: Problem with Brownian, disabling
        #elif integrator == 'BrownianIntegrator':
        #    self.integrator = self.openmm.BrownianIntegrator(temperature*self.unit.kelvin, coupling_frequency/self.unit.picosecond, timestep*self.unit.picoseconds)
        elif integrator == 'VariableLangevinIntegrator':
            self.integrator = self.openmm.VariableLangevinIntegrator(temperature*self.unit.kelvin, coupling_frequency/self.unit.picosecond, timestep*self.unit.picoseconds)
        else:
            print(BC.FAIL,"Unknown integrator.\n Valid integrator keywords are: VerletIntegrator, VariableVerletIntegrator, LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VariableLangevinIntegrator ", BC.END)
            exit()
        self.simulation = self.simulationclass(self.topology, self.system, self.integrator,self.platform, self.properties)
        print_time_rel(timeA, modulename="creating simulation")
    
    #Functions for energy decompositions
    def forcegroupify(self):
        self.forcegroups = {}
        print("inside forcegroupify")
        print("self.system.getForces() ", self.system.getForces())
        print("Number of forces:", self.system.getNumForces())
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            force.setForceGroup(i)
            self.forcegroups[force] = i
        #print("self.forcegroups :", self.forcegroups)
        #exit()
    def getEnergyDecomposition(self,context):
        #Call and set force groups
        self.forcegroupify()
        energies = {}
        print("self.forcegroups:", self.forcegroups)
        for f, i in self.forcegroups.items():
            energies[f] = context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()
        return energies
    
    def printEnergyDecomposition(self):
        timeA=time.time()
        #Energy composition
        #TODO: Calling this is expensive (seconds)as the energy has to be recalculated.
        # Only do for cases: a) single-point b) First energy-step in optimization and last energy-step
        # OpenMM energy components
        openmm_energy = dict()
        energycomp = self.getEnergyDecomposition(self.simulation.context)
        print("energycomp: ", energycomp)
        print("self.forcegroups:", self.forcegroups)
        #print("len energycomp", len(energycomp))
        #print("openmm_energy: ", openmm_energy)
        print("")
        bondterm_set=False
        extrafcount=0
        #This currently assumes CHARMM36 components, More to be added
        for comp in energycomp.items():
            print("comp: ", comp)
            if 'HarmonicBondForce' in str(type(comp[0])):
                #Not sure if this works in general.
                if bondterm_set is False:
                    openmm_energy['Bond'] = comp[1]
                    bondterm_set=True
                else:
                    openmm_energy['Urey-Bradley'] = comp[1]
            elif 'HarmonicAngleForce' in str(type(comp[0])):
                openmm_energy['Angle'] = comp[1]
            elif 'PeriodicTorsionForce' in str(type(comp[0])):
                #print("Here")
                openmm_energy['Dihedrals'] = comp[1]
            elif 'CustomTorsionForce' in str(type(comp[0])):
                openmm_energy['Impropers'] = comp[1]
            elif 'CMAPTorsionForce' in str(type(comp[0])):
                openmm_energy['CMAP'] = comp[1]
            elif 'NonbondedForce' in str(type(comp[0])):
                openmm_energy['Nonbonded'] = comp[1]
            elif 'CMMotionRemover' in str(type(comp[0])):
                openmm_energy['CMM'] = comp[1]
            elif 'CustomBondForce' in str(type(comp[0])):
                openmm_energy['14-LJ'] = comp[1]
            else:
                extrafcount+=1
                openmm_energy['Otherforce'+str(extrafcount)] = comp[1]
                
        
        print_time_rel(timeA, modulename="energy decomposition")
        timeA = time.time()
        
        #The force terms to print in the ordered table.
        # Deprecated. Better to print everything.
        #Missing terms in force_terms will be printed separately
        #if self.Forcefield == 'CHARMM':
        #    force_terms = ['Bond', 'Angle', 'Urey-Bradley', 'Dihedrals', 'Impropers', 'CMAP', 'Nonbonded', '14-LJ']
        #else:
        #    #Modify...
        #    force_terms = ['Bond', 'Angle', 'Urey-Bradley', 'Dihedrals', 'Impropers', 'CMAP', 'Nonbonded']

        #Sum all force-terms
        sumofallcomponents=0.0
        for val in openmm_energy.values():
            sumofallcomponents+=val._value
        


        #Print energy table       
        print('%-20s | %-15s | %-15s' % ('Component', 'kJ/mol', 'kcal/mol'))
        print('-'*56)
        #TODO: Figure out better sorting of terms
        for name in sorted(openmm_energy):
            print('%-20s | %15.2f | %15.2f' % (name, openmm_energy[name] / self.unit.kilojoules_per_mole, openmm_energy[name] / self.unit.kilocalorie_per_mole))
        print('-'*56)
        print('%-20s | %15.2f | %15.2f' % ('Sumcomponents', sumofallcomponents, sumofallcomponents / 4.184))
        print("")
        print('%-20s | %15.2f | %15.2f' % ('Total', self.energy * constants.hartokj , self.energy * constants.harkcal))
        
        print("")
        print("")
        #Adding sum to table
        openmm_energy['Sum'] = sumofallcomponents
        self.energy_components=openmm_energy
    
    def run(self, current_coords=None, elems=None, Grad=False, fragment=None, qmatoms=None):
        module_init_time=time.time()
        timeA = time.time()
        print(BC.OKBLUE, BC.BOLD, "------------RUNNING OPENMM INTERFACE-------------", BC.END)
        #If no coords given to run then a single-point job probably (not part of Optimizer or MD which would supply coords).
        #Then try if fragment object was supplied.
        #Otherwise internal coords if they exist
        if current_coords is None:
            if fragment is None:
                if len(self.coords) != 0:
                    print("Using internal coordinates (from OpenMM object)")
                    current_coords=self.coords
                else:
                    print("Found no coordinates!")
                    exit(1)
            else:
                current_coords=fragment.coords

        #Making sure coords is np array and not list-of-lists
        current_coords=np.array(current_coords)
        ##  unit conversion for energy
        #eqcgmx = 2625.5002
        ## unit conversion for force
        #TODO: Check this.
        #fqcgmx = -49614.75258920567
        #fqcgmx = -49621.9
        #Convert from kj/(nm *mol) = kJ/(10*Ang*mol)
        #factor=2625.5002/(10*1.88972612546)
        #factor=-138.93548724479302
        #Correct:
        factor=-49614.752589207

        #pos = [Vec3(coords[:,0]/10,coords[:,1]/10,coords[:,2]/10)] * u.nanometer
        #Todo: Check speed on this
        print("Updating coordinates")
        timeA = time.time()
        pos = [self.Vec3(current_coords[i, 0] / 10, current_coords[i, 1] / 10, current_coords[i, 2] / 10) for i in range(len(current_coords))] * self.unit.nanometer

        self.simulation.context.setPositions(pos)
        print_time_rel(timeA, modulename="context: set positions")
        timeA = time.time()
        #While these distance constraints should not matter, applying them makes the energy function agree with previous benchmarking for bonded and nonbonded
        #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5549999/
        #Using 1e-6 hardcoded value since how used in paper
        if self.applyconstraints == True:
            print("Applying constraints before calculating MM energy")
            self.simulation.context.applyConstraints(1e-6)
            print_time_rel(timeA, modulename="context: apply constraints")
            timeA = time.time()

        print("Calculating MM state")
        
        print("forces")
        print(self.system.getForces())
        
        if Grad == True:
            state = self.simulation.context.getState(getEnergy=True, getForces=True)
            self.energy = state.getPotentialEnergy().value_in_unit(self.unit.kilojoule_per_mole) / constants.hartokj
            self.gradient = np.array(state.getForces(asNumpy=True)/factor)
        else:
            state = self.simulation.context.getState(getEnergy=True, getForces=False)
            self.energy = state.getPotentialEnergy().value_in_unit(self.unit.kilojoule_per_mole) / constants.hartokj

        print_time_rel(timeA, modulename="state")
        timeA = time.time()
        print("OpenMM Energy:", self.energy, "Eh")
        print("OpenMM Energy:", self.energy*constants.harkcal, "kcal/mol")
        
        #Do energy components or not. Can be turned off for e.g. MM MD simulation
        if self.do_energy_decomposition is True:
            self.printEnergyDecomposition()
        
        print("self.energy : ", self.energy, "Eh")
        print("Energy:", self.energy*constants.harkcal, "kcal/mol")
        #print("Grad is", Grad)
        #print("self.gradient:", self.gradient)

        print(BC.OKBLUE, BC.BOLD, "------------ENDING OPENMM INTERFACE-------------", BC.END)
        print_time_rel(module_init_time, modulename="OpenMM run", moduleindex=2)
        if Grad == True:
            return self.energy, self.gradient
        else:
            return self.energy
    #Get list of charges from chosen force object (usually original nonbonded force object)
    def getatomcharges(self,force):
        chargelist = []
        for i in range( force.getNumParticles() ):
            charge = force.getParticleParameters( i )[0]
            if isinstance(charge, self.unit.Quantity):
                charge = charge / self.unit.elementary_charge
                chargelist.append(charge)
        self.charges=chargelist
        return chargelist

    # Delete selected exceptions. Only for Coulomb.
    #Used to delete Coulomb interactions involving QM-QM and QM-MM atoms
    def delete_exceptions(self,atomlist):
        timeA=time.time()
        print("Deleting Coulombexceptions for atomlist:", atomlist)
        for force in self.system.getForces():
            if isinstance(force, self.openmm.NonbondedForce):
                for exc in range(force.getNumExceptions()):
                    #print(force.getExceptionParameters(exc))
                    #force.getExceptionParameters(exc)
                    p1,p2,chargeprod,sigmaij,epsilonij = force.getExceptionParameters(exc)
                    if p1 in atomlist or p2 in atomlist:
                        #print("p1: {} and p2: {}".format(p1,p2))
                        #print("chargeprod:", chargeprod)
                        #print("sigmaij:", sigmaij)
                        #print("epsilonij:", epsilonij)
                        chargeprod._value=0.0
                        force.setExceptionParameters(exc, p1, p2, chargeprod, sigmaij, epsilonij)
                        #print("New:", force.getExceptionParameters(exc))
        self.create_simulation()
        print_time_rel(timeA, modulename="delete_exceptions")
        

    #Function to
    def zero_nonbondedforce(self,atomlist, zeroCoulomb=True, zeroLJ=True):
        timeA=time.time()
        print("Zero-ing nonbondedforce")
        def charge_sigma_epsilon(charge,sigma,epsilon):
            if zeroCoulomb ==  True:
                newcharge=charge
                newcharge._value=0.0

            else:
                newcharge=charge
            if zeroLJ == True:
                newsigma=sigma
                newsigma._value=0.0
                newepsilon=epsilon
                newepsilon._value=0.0
            else:
                newsigma=sigma
                newepsilon=epsilon
            return [newcharge,newsigma,newepsilon]
        #Zero all nonbonding interactions for atomlist
        for force in self.system.getForces():
            if isinstance(force, self.openmm.NonbondedForce):
                #Setting single particle parameters
                for atomindex in atomlist:
                    oldcharge, oldsigma, oldepsilon = force.getParticleParameters(atomindex)
                    newpars = charge_sigma_epsilon(oldcharge,oldsigma,oldepsilon)
                    print(newpars)
                    force.setParticleParameters(atomindex, newpars[0],newpars[1],newpars[2])
                print("force.getNumExceptions() ", force.getNumExceptions())
                print("force.getNumExceptionParameterOffsets() ", force.getNumExceptionParameterOffsets())
                print("force.getNonbondedMethod():", force.getNonbondedMethod())
                print("force.getNumGlobalParameters() ", force.getNumGlobalParameters())
                #Now doing exceptions
                for exc in range(force.getNumExceptions()):
                    print(force.getExceptionParameters(exc))
                    force.getExceptionParameters(exc)
                    p1,p2,chargeprod,sigmaij,epsilonij = force.getExceptionParameters(exc)
                    #chargeprod._value=0.0
                    #sigmaij._value=0.0
                    #epsilonij._value=0.0
                    newpars2 = charge_sigma_epsilon(chargeprod,sigmaij,epsilonij)
                    force.setExceptionParameters(exc, p1, p2, newpars2[0], newpars2[1], newpars2[2])
                    #print("New:", force.getExceptionParameters(exc))
                #force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.CustomNonbondedForce):
                print("customnonbondedforce not implemented")
                exit()
        self.create_simulation()
        print_time_rel(timeA, modulename="zero_nonbondedforce")
        #self.create_simulation()
    #Updating charges in OpenMM object. Used to set QM charges to 0 for example
    #Taking list of atom-indices and list of charges (usually zero) and setting new charge
    #Note: Exceptions also needs to be dealt with (see delete_exceptions)
    def update_charges(self,atomlist,atomcharges):
        timeA=time.time()
        print("Updating charges in OpenMM object.")
        assert len(atomlist) == len(atomcharges)
        newcharges=[]
        #print("atomlist:", atomlist)
        for atomindex,newcharge in zip(atomlist,atomcharges):
            #Updating big chargelist of OpenMM object.
            #TODO: Is this actually used?
            self.charges[atomindex]=newcharge
            #print("atomindex: ", atomindex)
            #print("newcharge: ",newcharge)
            oldcharge, sigma, epsilon = self.nonbonded_force.getParticleParameters(atomindex)
            #Different depending on type of NonbondedForce
            if isinstance(self.nonbonded_force, self.openmm.CustomNonbondedForce):
                self.nonbonded_force.setParticleParameters(atomindex, [newcharge,sigma,epsilon])
                #bla1,bla2,bla3 = self.nonbonded_force.getParticleParameters(i)
                #print("bla1,bla2,bla3", bla1,bla2,bla3)
            elif isinstance(self.nonbonded_force, self.openmm.NonbondedForce):
                self.nonbonded_force.setParticleParameters(atomindex, newcharge,sigma,epsilon)
                #bla1,bla2,bla3 = self.nonbonded_force.getParticleParameters(atomindex)
                #print("bla1,bla2,bla3", bla1,bla2,bla3)

        #Instead of recreating simulation we can just update like this:
        print("Updating simulation object for modified Nonbonded force")
        printdebug("self.nonbonded_force:", self.nonbonded_force)
        #Making sure that there still is a nonbonded force present in system (in case deleted)
        for i,force in enumerate(self.system.getForces()):
            printdebug("i is {} and force is {}".format(i,force))
            if isinstance(force, self.openmm.NonbondedForce):
                printdebug("here")
                self.nonbonded_force.updateParametersInContext(self.simulation.context)
            if isinstance(force, self.openmm.CustomNonbondedForce):
                self.nonbonded_force.updateParametersInContext(self.simulation.context)
        self.create_simulation()
        printdebug("done here")
        print_time_rel(timeA, modulename="update_charges")

    def modify_bonded_forces(self,atomlist):
        timeA=time.time()
        print("Modifying bonded forces")
        print("")
        #This is typically used by QM/MM object to set bonded forces to zero for qmatoms (atomlist) 
        #Mimicking: https://github.com/openmm/openmm/issues/2792
        
        numharmbondterms_removed=0
        numharmangleterms_removed=0
        numpertorsionterms_removed=0
        numcustomtorsionterms_removed=0
        numcmaptorsionterms_removed=0
        numcmmotionterms_removed=0
        numcustombondterms_removed=0
        
        for force in self.system.getForces():
            if isinstance(force, self.openmm.HarmonicBondForce):
                printdebug("HarmonicBonded force")
                printdebug("There are {} HarmonicBond terms defined.".format(force.getNumBonds()))
                printdebug("")
                #REVISIT: Neglecting QM-QM and sQM1-MM1 interactions. i.e if one atom in bond-pair is QM we neglect
                for i in range(force.getNumBonds()):
                    #print("i:", i)
                    p1, p2, length, k = force.getBondParameters(i)
                    #print("p1: {} p2: {} length: {} k: {}".format(p1,p2,length,k))
                    #or: delete QM-QM and QM-MM
                    #and: delete QM-QM
                    
                    if self.delete_QM1_MM1_bonded == True:
                        exclude = (p1 in atomlist or p2 in atomlist)
                    else:
                        exclude = (p1 in atomlist and p2 in atomlist)
                    #print("exclude:", exclude)
                    if exclude is True:
                        printdebug("exclude True")
                        printdebug("atomlist:", atomlist)
                        printdebug("i:", i)
                        printdebug("Before p1: {} p2: {} length: {} k: {}".format(p1,p2,length,k))
                        force.setBondParameters(i, p1, p2, length, 0)
                        numharmbondterms_removed+=1
                        p1, p2, length, k = force.getBondParameters(i)
                        printdebug("After p1: {} p2: {} length: {} k: {}".format(p1,p2,length,k))
                        printdebug("")
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.HarmonicAngleForce):
                printdebug("HarmonicAngle force")
                printdebug("There are {} HarmonicAngle terms defined.".format(force.getNumAngles()))
                for i in range(force.getNumAngles()):
                    p1, p2, p3, angle, k = force.getAngleParameters(i)
                    #Are angle-atoms in atomlist? 
                    presence=[i in atomlist for i in [p1,p2,p3]]
                    #Excluding if 2 or 3 QM atoms. i.e. a QM2-QM1-MM1 or QM3-QM2-QM1 term
                    #Originally set to 2
                    if presence.count(True) >= 2:
                        printdebug("presence.count(True):", presence.count(True))
                        printdebug("exclude True")
                        printdebug("atomlist:", atomlist)
                        printdebug("i:", i)
                        printdebug("Before p1: {} p2: {} p3: {} angle: {} k: {}".format(p1,p2,p3,angle,k))
                        force.setAngleParameters(i, p1, p2, p3, angle, 0)
                        numharmangleterms_removed+=1
                        p1, p2, p3, angle, k = force.getAngleParameters(i)
                        printdebug("After p1: {} p2: {} p3: {} angle: {} k: {}".format(p1,p2,p3,angle,k))
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.PeriodicTorsionForce):
                printdebug("PeriodicTorsionForce force")
                printdebug("There are {} PeriodicTorsionForce terms defined.".format(force.getNumTorsions()))
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                    #Are torsion-atoms in atomlist? 
                    presence=[i in atomlist for i in [p1,p2,p3,p4]]
                    #Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    #print("Before p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1,p2,p3,p4,periodicity, phase,k))
                    #Originally set to 3
                    if presence.count(True) >= 3:
                        printdebug("Found torsion in QM-region")
                        printdebug("presence.count(True):", presence.count(True))
                        printdebug("exclude True")
                        printdebug("atomlist:", atomlist)
                        printdebug("i:", i)
                        printdebug("Before p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1,p2,p3,p4,periodicity, phase,k))
                        force.setTorsionParameters(i, p1, p2, p3, p4, periodicity, phase, 0)
                        numpertorsionterms_removed+=1
                        p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                        printdebug("After p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1,p2,p3,p4,periodicity, phase,k))
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.CustomTorsionForce):
                printdebug("CustomTorsionForce force")
                printdebug("There are {} CustomTorsionForce terms defined.".format(force.getNumTorsions()))
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                    #Are torsion-atoms in atomlist? 
                    presence=[i in atomlist for i in [p1,p2,p3,p4]]
                    #Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    #print("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                    #print("pars:", pars)
                    if presence.count(True) >= 3:
                        printdebug("Found torsion in QM-region")
                        printdebug("presence.count(True):", presence.count(True))
                        printdebug("exclude True")
                        printdebug("atomlist:", atomlist)
                        printdebug("i:", i)
                        printdebug("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                        force.setTorsionParameters(i, p1, p2, p3, p4, (0.0,0.0))
                        numcustomtorsionterms_removed+=1
                        p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                        print("After p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.CMAPTorsionForce):
                printdebug("CMAPTorsionForce force")
                printdebug("There are {} CMAP terms defined.".format(force.getNumTorsions()))
                printdebug("There are {} CMAP maps defined".format(force.getNumMaps()))
                #print("Assuming no CMAP terms in QM-region. Continuing")
                # Note (RB). CMAP is between pairs of backbone dihedrals.
                # Not sure if we can delete the terms:
                #http://docs.openmm.org/latest/api-c++/generated/OpenMM.CMAPTorsionForce.html
                #  
                #print("Map num 0", force.getMapParameters(0))
                #print("Map num 1", force.getMapParameters(1))
                #print("Map num 2", force.getMapParameters(2))
                for i in range(force.getNumTorsions()):
                    jj, p1, p2, p3, p4,v1,v2,v3,v4 = force.getTorsionParameters(i)
                    #Are torsion-atoms in atomlist? 
                    presence=[i in atomlist for i in [p1,p2,p3,p4,v1,v2,v3,v4]]
                    #NOTE: Not sure how to use count properly here when dealing with torsion atoms in QM-region
                    if presence.count(True) >= 4:
                        printdebug("jj: {} p1: {} p2: {} p3: {} p4: {}      v1: {} v2: {} v3: {} v4: {}".format(jj,p1,p2,p3,p4,v1,v2,v3,v4))
                        printdebug("presence:", presence)
                        printdebug("Found CMAP torsion partner in QM-region")
                        printdebug("Not deleting. To be revisited...")
                        #print("presence.count(True):", presence.count(True))
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                        #force.setTorsionParameters(i, p1, p2, p3, p4, (0.0,0.0))
                        #numcustomtorsionterms_removed+=1
                        #p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                        #print("After p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                #force.updateParametersInContext(self.simulation.context)
            
            elif isinstance(force, self.openmm.CustomBondForce):
                printdebug("CustomBondForce")
                printdebug("There are {} force terms defined.".format(force.getNumBonds()))
                #Neglecting QM1-MM1 interactions. i.e if one atom in bond-pair is QM we neglect
                for i in range(force.getNumBonds()):
                    #print("i:", i)
                    p1, p2, vars = force.getBondParameters(i)
                    #print("p1: {} p2: {}".format(p1,p2))
                    exclude = (p1 in atomlist and p2 in atomlist)
                    #print("exclude:", exclude)
                    if exclude is True:
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before")
                        #print("p1: {} p2: {}")
                        force.setBondParameters(i, p1, p2, [0.0,0.0,0.0])
                        numcustombondterms_removed+=1
                        p1, p2, vars = force.getBondParameters(i)
                        #print("p1: {} p2: {}")
                        #print("vars:", vars)
                        #exit()
                force.updateParametersInContext(self.simulation.context)
            
            elif isinstance(force, self.openmm.CMMotionRemover):
                pass
                #print("CMMotionRemover ")
                #print("nothing to be done")
            elif isinstance(force, self.openmm.CustomNonbondedForce):
                pass
                #print("CustomNonbondedForce force")
                #print("nothing to be done")
            elif isinstance(force, self.openmm.NonbondedForce):
                pass
                #print("NonbondedForce force")
                #print("nothing to be done")
            else:
                pass
                #print("Other force: ", force)
                #print("nothing to be done")

        print("")
        print("Number of bonded terms removed:", )
        print("Harmonic Bond terms:", numharmbondterms_removed)
        print("Harmonic Angle terms:", numharmangleterms_removed)
        print("Periodic Torsion terms:", numpertorsionterms_removed)
        print("Custom Torsion terms:", numcustomtorsionterms_removed)
        print("CMAP Torsion terms:", numcmaptorsionterms_removed)
        print("CustomBond terms", numcustombondterms_removed)
        print("")
        self.create_simulation()
        print_time_rel(timeA, modulename="modify_bonded_forces")
#For frozen systems we use Customforce in order to specify interaction groups
#if len(self.frozen_atoms) > 0:
    
    #Two possible ways.
    #https://github.com/openmm/openmm/issues/2698
    #1. Use CustomNonbondedForce  with interaction groups. Could be slow
    #2. CustomNonbondedForce but with scaling


#https://ahy3nz.github.io/posts/2019/30/openmm2/
#http://www.maccallumlab.org/news/2015/1/23/testing

#Comes close to NonbondedForce results (after exclusions) but still not correct
#The issue is most likely that the 1-4 LJ interactions should not be excluded but rather scaled.
#See https://github.com/openmm/openmm/issues/1200
#https://github.com/openmm/openmm/issues/1696
#How to do:
#1. Keep nonbonded force for only those interactions and maybe also electrostatics?
#Mimic this??: https://github.com/openmm/openmm/blob/master/devtools/forcefield-scripts/processCharmmForceField.py
#Or do it via Parmed? Better supported for future??
#2. Go through the 1-4 interactions and not exclude but scale somehow manually. But maybe we can't do that in CustomNonbonded Force?
#Presumably not but maybe can add a special force object just for 1-4 interactions. We
def create_cnb(original_nbforce):
    """Creates a CustomNonbondedForce object that mimics the original nonbonded force
    and also a Custombondforce to handle 14 exceptions
    """
    #Next, create a CustomNonbondedForce with LJ and Coulomb terms
    ONE_4PI_EPS0 = 138.935456
    #ONE_4PI_EPS0=1.0
    #TODO: Not sure whether sqrt should be present or not in epsilon???
    energy_expression  = "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod/r;"
    #sqrt ??
    energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"
    energy_expression += "sigma = 0.5*(sigma1+sigma2);"
    energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units
    energy_expression += "chargeprod = charge1*charge2;"
    custom_nonbonded_force = simtk.openmm.CustomNonbondedForce(energy_expression)
    custom_nonbonded_force.addPerParticleParameter('charge')
    custom_nonbonded_force.addPerParticleParameter('sigma')
    custom_nonbonded_force.addPerParticleParameter('epsilon')
    # Configure force
    custom_nonbonded_force.setNonbondedMethod(simtk.openmm.CustomNonbondedForce.NoCutoff)
    #custom_nonbonded_force.setCutoffDistance(9999999999)
    custom_nonbonded_force.setUseLongRangeCorrection(False)
    #custom_nonbonded_force.setUseSwitchingFunction(True)
    #custom_nonbonded_force.setSwitchingDistance(99999)
    print('adding particles to custom force')
    for index in range(self.system.getNumParticles()):
        [charge, sigma, epsilon] = original_nbforce.getParticleParameters(index)
        custom_nonbonded_force.addParticle([charge, sigma, epsilon])
    #For CustomNonbondedForce we need (unlike NonbondedForce) to create exclusions that correspond to the automatic exceptions in NonbondedForce
    #These are interactions that are skipped for bonded atoms
    numexceptions = original_nbforce.getNumExceptions()
    print("numexceptions in original_nbforce: ", numexceptions)
    
    #Turn exceptions from NonbondedForce into exclusions in CustombondedForce
    # except 1-4 which are not zeroed but are scaled. These are added to Custombondforce
    exceptions_14=[]
    numexclusions=0
    for i in range(0,numexceptions):
        #print("i:", i)
        #Get exception parameters (indices)
        p1,p2,charge,sigma,epsilon = original_nbforce.getExceptionParameters(i)
        #print("p1,p2,charge,sigma,epsilon:", p1,p2,charge,sigma,epsilon)
        #If 0.0 then these are CHARMM 1-2 and 1-3 interactions set to zero
        if charge._value==0.0 and epsilon._value==0.0:
            #print("Charge and epsilons are 0.0. Add proper exclusion")
            #Set corresponding exclusion in customnonbforce
            custom_nonbonded_force.addExclusion(p1,p2)
            numexclusions+=1
        else:
            #print("This is not an exclusion but a scaled interaction as it is is non-zero. Need to keep")
            exceptions_14.append([p1,p2,charge,sigma,epsilon])
            #[798, 801, Quantity(value=-0.0684, unit=elementary charge**2), Quantity(value=0.2708332103146632, unit=nanometer), Quantity(value=0.2672524882578271, unit=kilojoule/mole)]
    
    print("len exceptions_14", len(exceptions_14))
    #print("exceptions_14:", exceptions_14)
    print("numexclusions:", numexclusions)
    
    
    #Creating custombondforce to handle these special exceptions
    #Now defining pair parameters
    #https://github.com/openmm/openmm/issues/2698
    energy_expression  = "(4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod/r);"
    energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units
    custom_bond_force = self.openmm.CustomBondForce(energy_expression)
    custom_bond_force.addPerBondParameter('chargeprod')
    custom_bond_force.addPerBondParameter('sigma')
    custom_bond_force.addPerBondParameter('epsilon')
    
    for exception in exceptions_14:
        idx=exception[0];jdx=exception[1];c=exception[2];sig=exception[3];eps=exception[4]
        custom_bond_force.addBond(idx, jdx, [c, sig, eps])
    
    print('Number of defined 14 bonds in custom_bond_force:', custom_bond_force.getNumBonds())
    
    
    return custom_nonbonded_force,custom_bond_force

#TODO: Look into: https://github.com/ParmEd/ParmEd/blob/7e411fd03c7db6977e450c2461e065004adab471/parmed/structure.py#L2554
    
#myCustomNBForce= simtk.openmm.CustomNonbondedForce("4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)")
#myCustomNBForce.setNonbondedMethod(simtk.openmm.app.NoCutoff)
#myCustomNBForce.setCutoffDistance(1000*simtk.openmm.unit.angstroms)
#Frozen-Act interaction
#myCustomNBForce.addInteractionGroup(self.frozen_atoms,self.active_atoms)
#Act-Act interaction
#myCustomNBForce.addInteractionGroup(self.active_atoms,self.active_atoms)


#Simple Molecular Dynamics using the OpenMM  object
#Integrators: LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator, BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
#Additional thermostat: AndersenThermostat (use with Verlet)
#Barostat: MonteCarloBarostat (not yet supported: MonteCarloAnisotropicBarostat, MonteCarloMembraneBarostat)

#Note: enforcePeriodicBox=True/False/None option. False works for current test system to get trajectory without PBC problem. Other systems may require True or None.
#see https://github.com/openmm/openmm/issues/2688, https://github.com/openmm/openmm/pull/1895
#Also should we add: https://github.com/mdtraj/mdtraj ?
def OpenMM_MD(fragment=None, openmmobject=None, timestep=0.001, simulation_steps=None, simulation_time=None, traj_frequency=1000, temperature=300, integrator=None,
    barostat=None, trajectory_file_option='PDB', coupling_frequency=None, anderson_thermostat=False, enforcePeriodicBox=False):
    
    print_line_with_mainheader("OpenMM MOLECULAR DYNAMICS")

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
    print("Simulation time: {} ps".format(simulation_time))
    print("Timestep: {} ps".format(timestep))
    print("Simulation steps: {}".format(simulation_steps))
    print("Temperature: {} K".format(temperature))
    print("Integrator:", integrator)
    print("Anderon Thermostat:", anderson_thermostat)
    print("coupling_frequency: {} ps^-1 (for Nose-Hoover and Langevin integrators)".format(coupling_frequency))
    print("Barostat:", barostat)

    print("")
    print("Will write trajectory in format:", trajectory_file_option)
    print("Trajectory write frequency:", traj_frequency)
    print("enforcePeriodicBox option:", enforcePeriodicBox)
    print("")
    if barostat != None:
        print("Adding barostat")
        openmmobject.system.addForce(openmmobject.openmm.MonteCarloBarostat(1*openmmobject.openmm.unit.bar, temperature*openmmobject.openmm.unit.kelvin))
        integrator="LangevinMiddleIntegrator"
        print("Barostat requires using integrator:", integrator)
        openmmobject.create_simulation(timestep=0.001, temperature=temperature, integrator=integrator, coupling_frequency=coupling_frequency)
    elif anderson_thermostat == True:
        print("Anderson thermostat is on")
        openmmobject.system.addForce(openmmobject.openmm.AndersenThermostat(temperature*openmmobject.openmm.unit.kelvin, 1/openmmobject.openmm.unit.picosecond))
        integrator="VerletIntegrator"
        print("Now using integrator:", integrator)
        openmmobject.create_simulation(timestep=0.001, temperature=temperature, integrator=integrator, coupling_frequency=coupling_frequency)
    else:
        #Regular thermostat or integrator without barostat
        #Integrators: LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator, BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
        openmmobject.create_simulation(timestep=0.001, temperature=temperature, integrator=integrator, coupling_frequency=coupling_frequency)
    

    print("Simulation created. Adding coordinates")
    #Context: settings positions
    coords=np.array(fragment.coords)
    pos = [openmmobject.Vec3(coords[i, 0] / 10, coords[i, 1] / 10, coords[i, 2] / 10) for i in range(len(coords))] * openmmobject.openmm.unit.nanometer

    openmmobject.simulation.context.setPositions(pos)

    if trajectory_file_option == 'PDB':
        openmmobject.simulation.reporters.append(openmmobject.openmm.app.PDBReporter('output_traj.pdb', traj_frequency, enforcePeriodicBox=enforcePeriodicBox))
    elif trajectory_file_option == 'DCD':
        #NOTE: Safer option might be to use PDB-writer from OpenMM. We shall see.
        write_pdbfile(fragment,outputname="initial_frag", openmmobject=openmmobject)
        openmmobject.simulation.reporters.append(openmmobject.openmm.app.DCDReporter('output_traj.dcd', traj_frequency, enforcePeriodicBox=enforcePeriodicBox))
    openmmobject.simulation.reporters.append(openmmobject.openmm.app.StateDataReporter(stdout, traj_frequency, step=True, time=True,
            potentialEnergy=True, temperature=True, kineticEnergy=True,  separator='     '))

    #Run simulation
    openmmobject.simulation.step(simulation_steps)
