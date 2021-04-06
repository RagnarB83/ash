import time
import numpy as np
import constants

from functions_general import BC,print_time_rel,listdiff,printdebug

class OpenMMTheory:
    def __init__(self, pdbfile=None, platform='CPU', active_atoms=None, frozen_atoms=None,
                 CHARMMfiles=False, psffile=None, charmmtopfile=None, charmmprmfile=None,
                 GROMACSfiles=False, gromacstopfile=None, grofile=None, gromacstopdir=None,
                 Amberfiles=False, amberprmtopfile=None, printlevel=2, do_energy_composition=True,
                 xmlfile=None, periodic=False, periodic_cell_dimensions=None, customnonbondedforce=False):
        
        timeA = time.time()
        # OPEN MM load
        try:
            import simtk.openmm.app
            import simtk.unit
            #import simtk.openmm
        except ImportError:
            raise ImportError(
                "OpenMM requires installing the OpenMM package. Try: conda install -c omnia openmm  \
                Also see http://docs.openmm.org/latest/userguide/application.html")

        print(BC.WARNING, BC.BOLD, "------------Defining OpenMM object-------------", BC.END)
        #Printlevel
        self.printlevel=printlevel

        #Parallelization
        #Control by setting env variable: $OPENMM_CPU_THREADS in shell before running.
        #Don't think it's possible to change variable inside Python environment
        try:
            print("OpenMM will use {} threads according to environment variable: OPENMM_CPU_THREADS".format(os.environ["OPENMM_CPU_THREADS"]))
        except:
            print("OPENMM_CPU_THREADS environment variable not set. OpenMM will choose number of physical cores present.")
        #Whether to do energy composition of MM energy or not. Takes time. Can be turned off for MD runs
        self.do_energy_composition=do_energy_composition
        #Initializing
        self.coords=[]
        self.charges=[]
        self.Periodic = periodic
        #Residue names,ids,segments,atomtypes of all atoms of systme.
        # Grabbed below from PSF-file. Information used to write PDB-file
        self.resnames=[]
        self.resids=[]
        self.segmentnames=[]
        self.atomtypes=[]
        self.atomnames=[]
            
        #OpenMM things
        self.openmm=simtk.openmm
        self.simulationclass=simtk.openmm.app.simulation.Simulation
        self.langevinintegrator=simtk.openmm.LangevinIntegrator
        self.platform_choice=platform
        self.unit=simtk.unit
        self.Vec3=simtk.openmm.Vec3


        #TODO: Should we keep this? Probably not. Coordinates would be handled by ASH.
        #PDB_ygg_frag = Fragment(pdbfile=pdbfile, conncalc=False)
        #self.coords=PDB_ygg_frag.coords
        print_time_rel(timeA, modulename="import openMM")
        timeA = time.time()

        self.Forcefield=None
        #What type of forcefield files to read. Reads in different way.
        # #Always creates object we call self.forcefield that contains topology attribute
        if CHARMMfiles is True:
            self.Forcefield='CHARMM'
            print("Reading CHARMM files")
            # Load CHARMM PSF files. Both CHARMM-style and XPLOR allowed I believe. Todo: Check
            self.psffile=psffile
            self.psf = simtk.openmm.app.CharmmPsfFile(psffile)
            
            #Grab resnames from psf-object
            #Note: OpenMM uses 0-indexing
            self.resnames=[self.psf.atom_list[i].residue.resname for i in range(0,len(self.psf.atom_list))]
            self.resids=[self.psf.atom_list[i].residue.idx for i in range(0,len(self.psf.atom_list))]
            self.segmentnames=[self.psf.atom_list[i].system for i in range(0,len(self.psf.atom_list))]
            self.atomtypes=[self.psf.atom_list[i].attype for i in range(0,len(self.psf.atom_list))]
            #TODO: Note: For atomnames it seems OpenMM converts atomnames to its own. Perhaps not useful
            self.atomnames=[self.psf.atom_list[i].name for i in range(0,len(self.psf.atom_list))]
            
            self.params = simtk.openmm.app.CharmmParameterSet(charmmtopfile, charmmprmfile)
            self.topology = self.psf.topology
            self.forcefield = self.psf

        elif GROMACSfiles is True:
            print("Warning: Gromacs-files interface not tested")
            #Reading grofile, not for coordinates but for periodic vectors
            gro = simtk.openmm.app.GromacsGroFile(grofile)
            self.grotop = simtk.openmm.app.GromacsTopFile(gromacstopfile, periodicBoxVectors=gro.getPeriodicBoxVectors(),
                                 includeDir=gromacstopdir)
            #self.forcefield=self.grotop
            self.topology = self.grotop.topology
            self.forcefield=self.grotop
            # Create an OpenMM system by calling createSystem on grotop
            #self.system = self.grotop.createSystem(nonbondedMethod=simtk.openmm.app.NoCutoff,
            #                                    nonbondedCutoff=1 * simtk.openmm.unit.nanometer)
            
            #forces = {self.system.getForce(index).__class__.__name__: self.system.getForce(index) for index in range(self.system.getNumForces())}
            #self.nonbonded_force = forces['NonbondedForce']
        elif Amberfiles is True:
            self.Forcefield='Amber'
            print("Warning: Amber-files interface not well tested. Be careful")
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


        #Now after topology is defined we can create system

        #Setting active and frozen variables once topology is in place
        #NOTE: Is this actually used?
        self.set_active_and_frozen_regions(active_atoms=active_atoms, frozen_atoms=frozen_atoms)



        #Periodic or non-periodic ystem
        if self.Periodic is True:
            print("System is periodic")
            self.periodic_cell_dimensions = periodic_cell_dimensions
            self.a = periodic_cell_dimensions[0] * self.unit.angstroms
            self.b = periodic_cell_dimensions[1] * self.unit.angstroms
            self.c = periodic_cell_dimensions[2] * self.unit.angstroms
            
            #Parameters here are based on OpenMM DHFR example
            self.forcefield.setBox(self.a, self.b, self.c)
            if CHARMMfiles is True:
                self.system = self.forcefield.createSystem(self.params, nonbondedMethod=simtk.openmm.app.PME,
                                            nonbondedCutoff=12 * self.unit.angstroms, switchDistance=10*self.unit.angstroms)
            else:
                self.system = self.forcefield.createSystem(nonbondedMethod=simtk.openmm.app.PME,
                                            nonbondedCutoff=12 * self.unit.angstroms, switchDistance=10*self.unit.angstroms)
            #TODO: Customnonbonded force option here
            print("self.system.getForces() :", self.system.getForces())
            for i,force in enumerate(self.system.getForces()):
                if isinstance(force, simtk.openmm.CustomNonbondedForce):
                    print('CustomNonbondedForce: %s' % force.getUseSwitchingFunction())
                    print('LRC? %s' % force.getUseLongRangeCorrection())
                    force.setUseLongRangeCorrection(False)
                elif isinstance(force, simtk.openmm.NonbondedForce):
                    print('NonbondedForce: %s' % force.getUseSwitchingFunction())
                    print('LRC? %s' % force.getUseDispersionCorrection())
                    force.setUseDispersionCorrection(False)
                    force.setPMEParameters(1.0/0.34, periodic_cell_dimensions[3], periodic_cell_dimensions[4], periodic_cell_dimensions[5]) 
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

            print("system created")
            print("Number of forces:", self.system.getNumForces())
            print(self.system.getForces())
            print("")                
            print("")
            #print("original forces: ", forces)
            # Get charges from OpenMM object into self.charges
            self.getatomcharges(forces['NonbondedForce'])
            print("self.system.getForces():", self.system.getForces())
            self.getatomcharges(self.system.getForces()[6])
            

            #CASE CUSTOMNONBONDED FORCE
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

            else:
                #Regular Nonbonded force
                self.nonbonded_force=self.system.getForce(6)


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
        
        self.forcegroupify()
        print_time_rel(timeA, modulename="forcegroupify")
        timeA = time.time()
        
        #Dummy integrator
        self.integrator = self.langevinintegrator(300 * self.unit.kelvin,  # Temperature of heat bath
                                        1 / self.unit.picosecond,  # Friction coefficient
                                        0.002 * self.unit.picoseconds)  # Time step
        self.platform = simtk.openmm.Platform.getPlatformByName(self.platform_choice)

        #Defined first here. 
        #NOTE: If self.system is modified then we have to remake self.simulation
        #self.simulation = simtk.openmm.app.simulation.Simulation(self.topology, self.system, self.integrator,self.platform)
        self.simulation = self.simulationclass(self.topology, self.system, self.integrator,self.platform)


        print_time_rel(timeA, modulename="simulation setup")
        timeA = time.time()
        
    def set_active_and_frozen_regions(self, active_atoms=None, frozen_atoms=None):
        #FROZEN AND ACTIVE ATOMS
        self.numatoms=int(self.forcefield.topology.getNumAtoms())
        print("self.numatoms:", self.numatoms)
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
    #This removes interactions between particles (e.g. QM-QM or frozen-frozen pairs)
    # list of atom indices for which we will remove all pairs

    #Todo: Way too slow to do for all frozen atoms but works well for qmatoms list size
    # Alternative: Remove force interaction and then add in the interaction of active atoms to frozen atoms
    # should be reasonably fast
    # https://github.com/openmm/openmm/issues/2124
    #https://github.com/openmm/openmm/issues/1696
    def addexceptions(self,atomlist):
        import itertools
        print("Add exceptions/exclusions. Removing i-j interactions for list :", len(atomlist), "atoms")
        timeA=time.time()
        #Has duplicates
        #[self.nonbonded_force.addException(i,j,0, 0, 0, replace=True) for i in atomlist for j in atomlist]
        #https://stackoverflow.com/questions/942543/operation-on-every-pair-of-element-in-a-list
        #[self.nonbonded_force.addException(i,j,0, 0, 0, replace=True) for i,j in itertools.combinations(atomlist, r=2)]
        numexceptions=0
        if isinstance(self.nonbonded_force, self.openmm.NonbondedForce):
            print("Case Nonbondedforce. Adding Exception for ij pair")
            for i in atomlist:
                for j in atomlist:
                    #print("i,j : ", i,j)
                    self.nonbonded_force.addException(i,j,0, 0, 0, replace=True)
                    numexceptions+=1
        elif isinstance(self.nonbonded_force, self.openmm.CustomNonbondedForce):
            print("Case CustomNonbondedforce. Adding Exclusion for ij pair")
            for i in atomlist:
                for j in atomlist:
                    #print("i,j : ", i,j)
                    self.nonbonded_force.addExclusion(i,j)
                    numexceptions+=1
        print("Number of exceptions/exclusions added: ", numexceptions)
        
        #Seems like updateParametersInContext does not reliably work here so we have to remake the simulation instead
        #Might be bug (https://github.com/openmm/openmm/issues/2709). Revisit
        #self.nonbonded_force.updateParametersInContext(self.simulation.context)
        self.integrator = self.langevinintegrator(300 * self.unit.kelvin,  # Temperature of heat bath
                                        1 / self.unit.picosecond,  # Friction coefficient
                                        0.002 * self.unit.picoseconds)  # Time step
        self.simulation = self.simulationclass(self.topology, self.system, self.integrator,self.platform)
        
        print_time_rel(timeA, modulename="add exception")
    #Run: coords or framents can be given (usually coords). qmatoms in order to avoid QM-QM interactions (TODO)
    #Probably best to do QM-QM exclusions etc. in a separate function though as we want run to be as simple as possible
    #qmatoms list provided for generality of MM objects. Not used here for now
    
    
    #Functions for energy compositions
    def forcegroupify(self):
        self.forcegroups = {}
        #print("inside forcegroupify")
        #print("self.system.getForces() ", self.system.getForces())
        #print("Number of forces:", self.system.getNumForces())
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            force.setForceGroup(i)
            self.forcegroups[force] = i
        #print("self.forcegroups :", self.forcegroups)
        #exit()
    def getEnergyDecomposition(self,context, forcegroups):
        energies = {}
        for f, i in forcegroups.items():
            energies[f] = context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()
        return energies
    
    def printEnergyDecomposition(self):
        timeA=time.time()
        #Energy composition
        #TODO: Calling this is expensive (seconds)as the energy has to be recalculated.
        # Only do for cases: a) single-point b) First energy-step in optimization and last energy-step
        # OpenMM energy components
        openmm_energy = dict()
        energycomp = self.getEnergyDecomposition(self.simulation.context, self.forcegroups)
        #print("energycomp: ", energycomp)
        #print("len energycomp", len(energycomp))
        #print("openmm_energy: ", openmm_energy)
        print("")
        bondterm_set=False
        extrafcount=0
        #This currently assumes CHARMM36 components, More to be added
        for comp in energycomp.items():
            #print("comp: ", comp)
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
                
        
        print_time_rel(timeA, modulename="energy composition")
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
        print_time_rel(timeA, modulename="print table")
        
        
        timeA = time.time()
    
    def run(self, current_coords=None, elems=None, Grad=False, fragment=None, qmatoms=None):
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
        print_time_rel(timeA, modulename="context pos")
        timeA = time.time()
        print("Calculating MM state")
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
        if self.do_energy_composition is True:
            self.printEnergyDecomposition()
        
        print("self.energy : ", self.energy, "Eh")
        print("Energy:", self.energy*constants.harkcal, "kcal/mol")
        #print("Grad is", Grad)
        #print("self.gradient:", self.gradient)

        print(BC.OKBLUE, BC.BOLD, "------------ENDING OPENMM INTERFACE-------------", BC.END)
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
    #Updating charges in OpenMM object. Used to set QM charges to 0 for example
    #Taking list of atom-indices and list of charges (usually zero) and setting new charge
    def update_charges(self,atomlist,atomcharges):
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
        self.nonbonded_force.updateParametersInContext(self.simulation.context)
        
    def modify_bonded_forces(self,atomlist):
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
                #print("HarmonicBonded force")
                #print("There are {} HarmonicBond terms defined.".format(force.getNumBonds()))
                print("")
                #REVISIT: Neglecting QM1-MM1 interactions. i.e if one atom in bond-pair is QM we neglect
                #CURRENT BEHAVIOUR. Keeping QM1-MM1 interaction but removing all QM-QM interactions
                for i in range(force.getNumBonds()):
                    #print("i:", i)
                    p1, p2, length, k = force.getBondParameters(i)
                    #print("p1: {} p2: {} length: {} k: {}".format(p1,p2,length,k))
                    exclude = (p1 in atomlist and p2 in atomlist)
                    #print("exclude:", exclude)
                    if exclude is True:
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before p1: {} p2: {} length: {} k: {}".format(p1,p2,length,k))
                        force.setBondParameters(i, p1, p2, length, 0)
                        numharmbondterms_removed+=1
                        #p1, p2, length, k = force.getBondParameters(i)
                        #print("After p1: {} p2: {} length: {} k: {}".format(p1,p2,length,k))
                        #exit()
                #print("Updating force")
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.HarmonicAngleForce):
                #print("HarmonicAngle force")
                #print("There are {} HarmonicAngle terms defined.".format(force.getNumAngles()))
                for i in range(force.getNumAngles()):
                    p1, p2, p3, angle, k = force.getAngleParameters(i)
                    #Are angle-atoms in atomlist? 
                    presence=[i in atomlist for i in [p1,p2,p3]]
                    #Excluding if 2 or 3 QM atoms. i.e. a QM2-QM1-MM1 or QM3-QM2-QM1 term
                    if presence.count(True) >= 2:
                        #print("presence.count(True):", presence.count(True))
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before p1: {} p2: {} p3: {} angle: {} k: {}".format(p1,p2,p3,angle,k))
                        force.setAngleParameters(i, p1, p2, p3, angle, 0)
                        numharmangleterms_removed+=1
                        #p1, p2, p3, angle, k = force.getAngleParameters(i)
                        #print("After p1: {} p2: {} p3: {} angle: {} k: {}".format(p1,p2,p3,angle,k))
                #print("Updating force")
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.PeriodicTorsionForce):
                #print("PeriodicTorsionForce force")
                #print("There are {} PeriodicTorsionForce terms defined.".format(force.getNumTorsions()))
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                    #Are torsion-atoms in atomlist? 
                    presence=[i in atomlist for i in [p1,p2,p3,p4]]
                    #Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    #print("Before p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1,p2,p3,p4,periodicity, phase,k))
                    if presence.count(True) >= 3:
                        #print("Found torsion in QM-region")
                        #print("presence.count(True):", presence.count(True))
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1,p2,p3,p4,periodicity, phase,k))
                        force.setTorsionParameters(i, p1, p2, p3, p4, periodicity, phase, 0)
                        numpertorsionterms_removed+=1
                        p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                        #print("After p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1,p2,p3,p4,periodicity, phase,k))
                #print("Updating force")
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.CustomTorsionForce):
                #print("CustomTorsionForce force")
                #print("There are {} CustomTorsionForce terms defined.".format(force.getNumTorsions()))
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                    #Are torsion-atoms in atomlist? 
                    presence=[i in atomlist for i in [p1,p2,p3,p4]]
                    #Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    #print("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                    #print("pars:", pars)
                    if presence.count(True) >= 3:
                        #print("Found torsion in QM-region")
                        #print("presence.count(True):", presence.count(True))
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                        force.setTorsionParameters(i, p1, p2, p3, p4, (0.0,0.0))
                        numcustomtorsionterms_removed+=1
                        #p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                        #print("After p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                #print("Updating force")
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.CMAPTorsionForce):
                #print("CMAPTorsionForce force")
                #print("There are {} CMAP terms defined.".format(force.getNumTorsions()))
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, a,b,c,d,e = force.getTorsionParameters(i)
                    #Are torsion-atoms in atomlist? 
                    presence=[i in atomlist for i in [p1,p2,p3,p4]]
                    #Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    #print("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                    if presence.count(True) >= 3:
                        #print("Found torsion in QM-region")
                        #print("presence.count(True):", presence.count(True))
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                        force.setTorsionParameters(i, p1, p2, p3, p4, (0.0,0.0))
                        numcustomtorsionterms_removed+=1
                        #p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                        #print("After p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                #print("Updating force")
                force.updateParametersInContext(self.simulation.context)
            
            elif isinstance(force, self.openmm.CustomBondForce):
                #print("CustomBondForce")
                #print("There are {} force terms defined.".format(force.getNumBonds()))
                #Neglecting QM1-MM1 interactions. i.e if one atom in bond-pair is QM we neglect
                for i in range(force.getNumBonds()):
                    #print("i:", i)
                    p1, p2, vars = force.getBondParameters(i)
                    #print("p1: {} p2: {}".format(p1,p2))
                    charge=vars[0];sigma=vars[1];epsilon=vars[2]
                    #print("charge: {} sigma: {} epsilon: {}".format(charge,sigma,epsilon))
                    exclude = (p1 in atomlist or p2 in atomlist)
                    #print("exclude:", exclude)
                    if exclude is True:
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before")
                        #print("p1: {} p2: {}")
                        #print("charge: {} sigma: {} epsilon: {}".format(charge,sigma,epsilon))
                        force.setBondParameters(i, p1, p2, [0.0,0.0,0.0])
                        numcustombondterms_removed+=1
                        #p1, p2, vars = force.getBondParameters(i)
                        #charge=vars[0];sigma=vars[1];epsilon=vars[2]
                        #print("p1: {} p2: {}")
                        #print("charge: {} sigma: {} epsilon: {}".format(charge,sigma,epsilon))
                        #exit()
                #print("Updating force")
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