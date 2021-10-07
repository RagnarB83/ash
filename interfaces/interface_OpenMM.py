import os
from sys import stdout
import time
import traceback

import numpy as np

import ash
import constants

ashpath = os.path.dirname(ash.__file__)
from functions.functions_general import BC, print_time_rel, listdiff, printdebug, print_line_with_mainheader, isint
from functions.functions_elstructure import DDEC_calc, DDEC_to_LJparameters
from modules.module_coords import Fragment, write_pdbfile, distance_between_atoms, list_of_masses, write_xyzfile, \
    change_origin_to_centroid
from modules.module_MM import UFF_modH_dict, MMforcefield_read
from interfaces.interface_xtb import xTBTheory, grabatomcharges_xTB
from interfaces.interface_ORCA import ORCATheory, grabatomcharges_ORCA, chargemodel_select
from modules.module_singlepoint import Singlepoint


class OpenMMTheory:
    def __init__(self, printlevel=2, platform='CPU', numcores=None, Modeller=False, forcefield=None, topology=None,
                 CHARMMfiles=False, psffile=None, charmmtopfile=None, charmmprmfile=None,
                 GROMACSfiles=False, gromacstopfile=None, grofile=None, gromacstopdir=None,
                 Amberfiles=False, amberprmtopfile=None,
                 cluster_fragment=None, ASH_FF_file=None, PBCvectors=None,
                 xmlfiles=None, pdbfile=None, use_parmed=False,
                 xmlsystemfile=None,
                 do_energy_decomposition=False,
                 periodic=False, charmm_periodic_cell_dimensions=None, customnonbondedforce=False,
                 periodic_nonbonded_cutoff=12, dispersion_correction=True,
                 switching_function_distance=10,
                 ewalderrortolerance=1e-5, PMEparameters=None,
                 delete_QM1_MM1_bonded=False, applyconstraints=False,
                 autoconstraints=None, hydrogenmass=None, rigidwater=True):

        module_init_time = time.time()
        # OPEN MM load
        try:
            import openmm
            import openmm.app
            import openmm.unit
            print("Imported OpenMM library version:", openmm.__version__)
        except ImportError:
            raise ImportError(
                "OpenMMTheory requires installing the OpenMM library. Try: conda install -c conda-forge openmm  \
                Also see http://docs.openmm.org/latest/userguide/application.html")

        # OpenMM variables
        self.openmm = openmm
        self.simulationclass = openmm.app.simulation.Simulation

        self.unit = openmm.unit
        self.Vec3 = openmm.Vec3

        print(BC.WARNING, BC.BOLD, "------------Defining OpenMM object-------------", BC.END)
        # Printlevel
        self.printlevel = printlevel

        # Initialize system
        self.system = None

        # Load Parmed if requested
        if use_parmed is True:
            print("Using Parmed to read topologyfiles")
            try:
                import parmed
            except ImportError:
                print("Problem importing parmed Python library")
                print("Make sure parmed is present in your Python.")
                print("Parmed can be installed using pip: pip install parmed")
                exit()

        # Autoconstraints when creating MM system: Default: None,  Options: Hbonds, AllBonds, HAng
        if autoconstraints == 'HBonds':
            print("HBonds option: X-H bond lengths will automatically be constrained")
            self.autoconstraints = self.openmm.app.HBonds
        elif autoconstraints == 'AllBonds':
            print("AllBonds option: All bond lengths will automatically be constrained")
            self.autoconstraints = self.openmm.app.AllBonds
        elif autoconstraints == 'HAngles':
            print("HAngles option: All bond lengths and H-X-H and H-O-X angles will automatically be constrained")
            self.autoconstraints = self.openmm.app.HAngles
        elif autoconstraints is None or autoconstraints == 'None':
            print("No automatic constraints")
            self.autoconstraints = None
        else:
            print("Unknown autoconstraints option")
            exit()
        print("AutoConstraint setting:", self.autoconstraints)
        # Rigidwater constraints are on by default. Can be turned off
        self.rigidwater = rigidwater
        print("Rigidwater constraints:", self.rigidwater)
        # Modify hydrogenmass or not
        if hydrogenmass is not None:
            self.hydrogenmass = hydrogenmass * self.unit.amu
        else:
            self.hydrogenmass = None
        print("Hydrogenmass option:", self.hydrogenmass)

        # Setting for controlling whether QM1-MM1 bonded terms are deleted or not in a QM/MM job
        # See modify_bonded_forces
        # TODO: Move option to module_QMMM instead
        self.delete_QM1_MM1_bonded = delete_QM1_MM1_bonded
        # Initialize system_masses. Only used when freezing/unfreezing atoms
        self.system_masses = []
        # Platform (CPU, CUDA, OpenCL) and Parallelization
        self.platform_choice = platform
        # CPU: Control either by provided numcores keyword, or by setting env variable: $OPENMM_CPU_THREADS in shell before running.
        self.properties = {}
        if self.platform_choice == 'CPU':
            print("Using platform: CPU")
            if numcores is not None:
                print("Numcores variable provided to OpenMM object. Will use {} cores with OpenMM".format(numcores))
                self.properties["Threads"] = str(numcores)
            else:
                print("No numcores variable provided to OpenMM object")
                print("Checking if OPENMM_CPU_THREADS shell variable is present")
                try:
                    print("OpenMM will use {} threads according to environment variable: OPENMM_CPU_THREADS".format(
                        os.environ["OPENMM_CPU_THREADS"]))
                except KeyError:
                    print(
                        "OPENMM_CPU_THREADS environment variable not set. OpenMM will choose number of physical cores present.")
        else:
            print("Using platform:", self.platform_choice)

        # Whether to do energy decomposition of MM energy or not. Takes time. Can be turned off for MD runs
        self.do_energy_decomposition = do_energy_decomposition

        # Initializing
        self.coords = []
        self.charges = []
        self.Periodic = periodic
        self.ewalderrortolerance = ewalderrortolerance

        # Whether to apply constraints or not when calculating MM energy via ASH (does not apply to OpenMM MD)
        # NOTE: Should be False in general. Only True for special cases
        self.applyconstraints = applyconstraints

        # Switching function distance in Angstrom
        self.switching_function_distance = switching_function_distance

        # Residue names,ids,segments,atomtypes of all atoms of system.
        # Grabbed below from PSF-file. Information used to write PDB-file
        self.resnames = []
        self.resids = []
        self.segmentnames = []
        self.atomtypes = []
        self.atomnames = []

        # Positions. Generally not used but can be if e.g. grofile has been read in.
        # Purpose: set virtual sites etc.
        self.positions = None

        # TODO: Should we keep this? Probably not. Coordinates would be handled by ASH.
        # PDB_ygg_frag = Fragment(pdbfile=pdbfile, conncalc=False)
        # self.coords=PDB_ygg_frag.coords
        print_time_rel(module_init_time, modulename="import openMM")
        timeA = time.time()

        self.Forcefield = None
        # What type of forcefield files to read. Reads in different way.
        print("Now reading forcefield files")
        print(
            "Note: OpenMM will fail in this step if parameters are missing in topology and parameter files (e.g. nonbonded entries)")

        # #Always creates object we call self.forcefield that contains topology attribute
        if CHARMMfiles is True:
            print("Reading CHARMM files")
            self.psffile = psffile
            if use_parmed is True:
                print("Using Parmed.")
                self.psf = parmed.charmm.CharmmPsfFile(psffile)
                self.params = parmed.charmm.CharmmParameterSet(charmmtopfile, charmmprmfile)
                # Grab resnames from psf-object. Different for parmed object
                # Note: OpenMM uses 0-indexing
                self.resnames = [self.psf.atoms[i].residue.name for i in range(0, len(self.psf.atoms))]
                self.resids = [self.psf.atoms[i].residue.idx for i in range(0, len(self.psf.atoms))]
                self.segmentnames = [self.psf.atoms[i].residue.segid for i in range(0, len(self.psf.atoms))]
                # self.atomtypes=[self.psf.atoms[i].attype for i in range(0,len(self.psf.atoms))]
                # TODO: Note: For atomnames it seems OpenMM converts atomnames to its own. Perhaps not useful
                self.atomnames = [self.psf.atoms[i].name for i in range(0, len(self.psf.atoms))]

            else:
                # Load CHARMM PSF files via native routine.
                self.psf = openmm.app.CharmmPsfFile(psffile)
                self.params = openmm.app.CharmmParameterSet(charmmtopfile, charmmprmfile)
                # Grab resnames from psf-object
                # Note: OpenMM uses 0-indexing
                self.resnames = [self.psf.atom_list[i].residue.resname for i in range(0, len(self.psf.atom_list))]
                self.resids = [self.psf.atom_list[i].residue.idx for i in range(0, len(self.psf.atom_list))]
                self.segmentnames = [self.psf.atom_list[i].system for i in range(0, len(self.psf.atom_list))]
                self.atomtypes = [self.psf.atom_list[i].attype for i in range(0, len(self.psf.atom_list))]
                # TODO: Note: For atomnames it seems OpenMM converts atomnames to its own. Perhaps not useful
                self.atomnames = [self.psf.atom_list[i].name for i in range(0, len(self.psf.atom_list))]

            self.topology = self.psf.topology
            self.forcefield = self.psf

        elif GROMACSfiles is True:
            print("Reading Gromacs files")
            # Reading grofile, not for coordinates but for periodic vectors
            if use_parmed is True:
                print("Using Parmed.")
                print("GROMACS top dir:", gromacstopdir)
                parmed.gromacs.GROMACS_TOPDIR = gromacstopdir
                print("Reading GROMACS GRO file: ", grofile)
                gmx_gro = parmed.gromacs.GromacsGroFile.parse(grofile)

                print("Reading GROMACS topology file: ", gromacstopfile)
                gmx_top = parmed.gromacs.GromacsTopologyFile(gromacstopfile)

                # Getting PBC parameters
                gmx_top.box = gmx_gro.box
                gmx_top.positions = gmx_gro.positions
                self.positions = gmx_top.positions

                self.topology = gmx_top.topology
                self.forcefield = gmx_top

            else:
                print("Using built-in OpenMM routines to read GROMACS topology")
                print("Warning: may fail if virtual sites present (e.g. TIP4P residues)")
                print("Use parmed=True  to avoid")
                gro = openmm.app.GromacsGroFile(grofile)
                self.grotop = openmm.app.GromacsTopFile(gromacstopfile, periodicBoxVectors=gro.getPeriodicBoxVectors(),
                                                        includeDir=gromacstopdir)

                self.topology = self.grotop.topology
                self.forcefield = self.grotop

            # TODO: Define resnames, resids, segmentnames, atomtypes, atomnames??

            # Create an OpenMM system by calling createSystem on grotop
            # self.system = self.grotop.createSystem(nonbondedMethod=simtk.openmm.app.NoCutoff,
            #                                    nonbondedCutoff=1 * simtk.openmm.unit.nanometer)

        elif Amberfiles is True:
            print("Reading Amber files")
            print("Warning: Only new-style Amber7 prmtop-file will work")
            print("Warning: Will take periodic boundary conditions from prmtop file.")
            if use_parmed is True:
                print("Using Parmed to read Amber files")
                self.prmtop = parmed.load_file(amberprmtopfile)
            else:
                print("Using built-in OpenMM routines to read Amber files.")
                # Note: Only new-style Amber7 prmtop files work
                self.prmtop = openmm.app.AmberPrmtopFile(amberprmtopfile)
            self.topology = self.prmtop.topology
            self.forcefield = self.prmtop

            # TODO: Define resnames, resids, segmentnames, atomtypes, atomnames??
        elif Modeller is True:
            print("Using forcefield info from Modeller")
            self.topology = topology
            self.forcefield = forcefield

        elif ASH_FF_file is not None:
            print("Reading ASH cluster fragment file and ASH Forcefield file")

            # Converting ASH FF file to OpenMM XML file
            MM_forcefield = MMforcefield_read(ASH_FF_file)

            atomtypes_res = []
            atomnames_res = []
            elements_res = []
            atomcharges_res = []
            sigmas_res = []
            epsilons_res = []
            residue_types = []
            masses_res = []

            for resid, residuetype in enumerate(MM_forcefield['residues']):
                residue_types.append("RS" + str(resid))
                atypelist = MM_forcefield[residuetype + "_atomtypes"]
                # atypelist needs to be more unique due to different charges
                atomtypes_res.append(["R" + residuetype[-1] + str(j) for j, i in enumerate(atypelist)])
                elements_res.append(MM_forcefield[residuetype + "_elements"])
                atomcharges_res.append(MM_forcefield[residuetype + "_charges"])
                # Atomnames, have to be unique and 4 letters, adding number
                atomnames_res.append(["R" + residuetype[-1] + str(j) for j, i in enumerate(atypelist)])
                sigmas_res.append([MM_forcefield[atomtype].LJparameters[0] / 10 for atomtype in
                                   MM_forcefield[residuetype + "_atomtypes"]])
                epsilons_res.append([MM_forcefield[atomtype].LJparameters[1] * 4.184 for atomtype in
                                     MM_forcefield[residuetype + "_atomtypes"]])
                masses_res.append(list_of_masses(elements_res[-1]))

            xmlfile = write_xmlfile_nonbonded(resnames=residue_types, atomnames_per_res=atomnames_res,
                                              atomtypes_per_res=atomtypes_res,
                                              elements_per_res=elements_res, masses_per_res=masses_res,
                                              charges_per_res=atomcharges_res, sigmas_per_res=sigmas_res,
                                              epsilons_per_res=epsilons_res,
                                              filename="cluster_system.xml", coulomb14scale=1.0, lj14scale=1.0)

            # Creating lists for PDB-file
            # requires ffragmenttype_labels to be present in fragment.
            # NOTE: Hence will only work for molcrys-prepared files for now
            atomnames_full = []
            jindex = 0
            resid_index = 1
            residlabels = []
            residue_types_full = []
            for i, fragtypelabel in enumerate(cluster_fragment.fragmenttype_labels):
                atomnames_full.append(atomnames_res[fragtypelabel][jindex])
                residlabels.append(resid_index)
                jindex += 1
                residue_types_full.append("RS" + str(fragtypelabel))
                if jindex == len(atomnames_res[fragtypelabel]):
                    jindex = 0
                    resid_index += 1

            # Creating PDB-file, only for topology (not coordinates)
            write_pdbfile(cluster_fragment, outputname="cluster", resnames=residue_types_full, atomnames=atomnames_full,
                          residlabels=residlabels)
            pdb = openmm.app.PDBFile("cluster.pdb")
            self.topology = pdb.topology

            self.forcefield = openmm.app.ForceField([xmlfile])

        # Load XMLfile for whole system
        elif xmlsystemfile is None:
            print("Reading system XML file:", xmlsystemfile)
            xmlsystemfileobj = open(xmlsystemfile).read()
            # Deserialize the XML text to create a System object.
            self.system = openmm.XmlSerializer.deserializeSystem(xmlsystemfileobj)
            # We still need topology from somewhere to using pdbfile
            print("Reading topology from PDBfile:", pdbfile)
            pdb = openmm.app.PDBFile(pdbfile)
            self.topology = pdb.topology

        # Read topology from PDB-file and XML-forcefield files to define forcefield
        else:
            print("Reading OpenMM XML forcefield files and PDB file")
            print("xmlfiles:", xmlfiles)
            # This would be regular OpenMM Forcefield definition requiring XML file
            # Topology from PDBfile annoyingly enough
            pdb = openmm.app.PDBFile(pdbfile)
            self.topology = pdb.topology
            # Todo: support multiple xml file here
            # forcefield = simtk.openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
            self.forcefield = openmm.app.ForceField(*xmlfiles)

            # TODO: Define resnames, resids, segmentnames, atomtypes, atomnames??

        # Deal with possible 4/5 site water model like TIP4P
        # NOTE: EXPERIMENTAL
        # NOTE: We have no positions here. Make separate callable function?????

        # if watermodel is not None:
        #    print("watermodel:", watermodel)
        #    modeller = simtk.openmm.app.Modeller(self.topology, pdb.positions)
        #    modeller.addExtraParticles(self.forcefield)
        #    simtk.openmm.app.app.PDBFile.writeFile(modeller.topology, modeller.positions, open('test-water.pdb', 'w'))

        # NOW CREATE SYSTEM UNLESS already created (xmlsystemfile)
        if self.system is None:
            # Periodic or non-periodic ystem
            if self.Periodic is True:
                print("System is periodic")

                print("Nonbonded cutoff is {} Angstrom".format(periodic_nonbonded_cutoff))
                # Parameters here are based on OpenMM DHFR example

                if CHARMMfiles is True:
                    print("Using CHARMM files")

                    if charmm_periodic_cell_dimensions is None:
                        print(
                            "Error: When using CHARMMfiles and Periodic=True, charmm_periodic_cell_dimensions keyword needs to be supplied")
                        print(
                            "Example: charmm_periodic_cell_dimensions= [200, 200, 200, 90, 90, 90]  in Angstrom and degrees")
                        exit()
                    self.charmm_periodic_cell_dimensions = charmm_periodic_cell_dimensions
                    print("Periodic cell dimensions:", charmm_periodic_cell_dimensions)
                    self.a = charmm_periodic_cell_dimensions[0] * self.unit.angstroms
                    self.b = charmm_periodic_cell_dimensions[1] * self.unit.angstroms
                    self.c = charmm_periodic_cell_dimensions[2] * self.unit.angstroms
                    if use_parmed is True:
                        self.forcefield.box = [self.a, self.b, self.c, charmm_periodic_cell_dimensions[3],
                                               charmm_periodic_cell_dimensions[4], charmm_periodic_cell_dimensions[5]]
                        print("Set box vectors:", self.forcefield.box)
                    else:
                        self.forcefield.setBox(self.a, self.b, self.c,
                                               alpha=self.unit.Quantity(value=charmm_periodic_cell_dimensions[3],
                                                                        unit=self.unit.degree),
                                               beta=self.unit.Quantity(value=charmm_periodic_cell_dimensions[3],
                                                                       unit=self.unit.degree),
                                               gamma=self.unit.Quantity(value=charmm_periodic_cell_dimensions[3],
                                                                        unit=self.unit.degree))
                        # self.forcefield.setBox(self.a, self.b, self.c)
                        # print(self.forcefield.__dict__)
                        print("Set box vectors:", self.forcefield.box_vectors)
                    # NOTE: SHould this be made more general??
                    # print("a,b,c:", self.a, self.b, self.c)
                    # print("box in self.forcefield", self.forcefield.get_box())

                    # exit()
                    self.system = self.forcefield.createSystem(self.params, nonbondedMethod=openmm.app.PME,
                                                               constraints=self.autoconstraints,
                                                               hydrogenMass=self.hydrogenmass,
                                                               rigidWater=self.rigidwater,
                                                               nonbondedCutoff=periodic_nonbonded_cutoff * self.unit.angstroms,
                                                               switchDistance=switching_function_distance * self.unit.angstroms)
                elif GROMACSfiles is True:
                    # NOTE: Gromacs has read PBC info from Gro file already
                    print("Ewald Error tolerance:", self.ewalderrortolerance)
                    # Note: Turned off switchDistance. Not available for GROMACS?
                    #
                    self.system = self.forcefield.createSystem(nonbondedMethod=openmm.app.PME,
                                                               constraints=self.autoconstraints,
                                                               hydrogenMass=self.hydrogenmass,
                                                               rigidWater=self.rigidwater,
                                                               nonbondedCutoff=periodic_nonbonded_cutoff * self.unit.angstroms,
                                                               ewaldErrorTolerance=self.ewalderrortolerance)
                elif Amberfiles is True:
                    # NOTE: Amber-interface has read PBC info from prmtop file already
                    self.system = self.forcefield.createSystem(nonbondedMethod=openmm.app.PME,
                                                               constraints=self.autoconstraints,
                                                               hydrogenMass=self.hydrogenmass,
                                                               rigidWater=self.rigidwater,
                                                               nonbondedCutoff=periodic_nonbonded_cutoff * self.unit.angstroms)

                    # print("self.system num con", self.system.getNumConstraints())
                else:
                    print("Setting up periodic system here.")
                    # Modeller and manual xmlfiles
                    self.system = self.forcefield.createSystem(self.topology, nonbondedMethod=openmm.app.PME,
                                                               constraints=self.autoconstraints,
                                                               hydrogenMass=self.hydrogenmass,
                                                               rigidWater=self.rigidwater,
                                                               nonbondedCutoff=periodic_nonbonded_cutoff * self.unit.angstroms)
                    # switchDistance=switching_function_distance*self.unit.angstroms

                print("self.system dict", self.system.__dict__)

                # TODO: Customnonbonded force option. Currently disabled
                print("OpenMM system created")
                if PBCvectors != None:
                    pbcvectors_mod = PBCvectors
                    print("Setting PBC vectors by user request")
                    print("Assuming list of lists or list of Vec3 objects")
                    print("Assuming vectors in nanometers")
                    self.system.setDefaultPeriodicBoxVectors(*PBCvectors)
                print("Periodic vectors:", self.system.getDefaultPeriodicBoxVectors())
                # Force modification here
                print("OpenMM Forces defined:", self.system.getForces())

                # PRINTING PROPERTIES OF NONBONDED FORCE BELOW
                for i, force in enumerate(self.system.getForces()):
                    if isinstance(force, openmm.CustomNonbondedForce):
                        # NOTE: THIS IS CURRENTLY NOT USED
                        pass
                        # print('CustomNonbondedForce: %s' % force.getUseSwitchingFunction())
                        # print('LRC? %s' % force.getUseLongRangeCorrection())
                        # force.setUseLongRangeCorrection(False)
                    elif isinstance(force, openmm.NonbondedForce):
                        # Turn Dispersion correction on/off depending on user
                        # NOTE: Default: False   To be revisited

                        # NOte:
                        force.setUseDispersionCorrection(dispersion_correction)

                        # Modify PME Parameters if desired
                        # force.setPMEParameters(1.0/0.34, fftx, ffty, fftz)
                        if PMEparameters is not None:
                            print("Changing PME parameters")
                            force.setPMEParameters(PMEparameters[0], PMEparameters[1], PMEparameters[2],
                                                   PMEparameters[3])
                        # force.setSwitchingDistance(switching_function_distance)
                        # if switching_function is True:
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
                        # force.setPMEParameters(3.285326106/self.unit.nanometers,60, 64, 60)
                        # Keeping default for now

                        # self.nonbonded_force=force
                        # NOTE: These are hard-coded!


            # Non-Periodic
            else:
                print("System is non-periodic")

                if CHARMMfiles is True:
                    self.system = self.forcefield.createSystem(self.params, nonbondedMethod=openmm.app.NoCutoff,
                                                               nonbondedCutoff=1000 * openmm.unit.angstroms,
                                                               hydrogenMass=self.hydrogenmass)
                else:
                    self.system = self.forcefield.createSystem(self.topology, nonbondedMethod=openmm.app.NoCutoff,
                                                               constraints=self.autoconstraints,
                                                               rigidWater=self.rigidwater,
                                                               nonbondedCutoff=1000 * openmm.unit.angstroms,
                                                               hydrogenMass=self.hydrogenmass)

                print("OpenMM system created")
                print("OpenMM Forces defined:", self.system.getForces())
                print("")
                # for i,force in enumerate(self.system.getForces()):
                #    if isinstance(force, openmm.NonbondedForce):
                #        self.getatomcharges()
                #        self.nonbonded_force=force

                # print("original forces: ", forces)
                # Get charges from OpenMM object into self.charges
                # self.getatomcharges(forces['NonbondedForce'])
                # print("self.system.getForces():", self.system.getForces())
                # self.getatomcharges(self.system.getForces()[6])

                # CASE CUSTOMNONBONDED FORCE
                # REPLACING REGULAR NONBONDED FORCE
                if customnonbondedforce is True:
                    print("currently inactive")
                    exit()
                    # Create CustomNonbonded force
                    for i, force in enumerate(self.system.getForces()):
                        if isinstance(force, self.openmm.NonbondedForce):
                            custom_nonbonded_force, custom_bond_force = create_cnb(self.system.getForces()[i])
                    print("1custom_nonbonded_force:", custom_nonbonded_force)
                    print("num exclusions in customnonb:", custom_nonbonded_force.getNumExclusions())
                    print("num 14 exceptions in custom_bond_force:", custom_bond_force.getNumBonds())

                    # TODO: Deal with frozen regions. NOT YET DONE
                    # Frozen-Act interaction
                    # custom_nonbonded_force.addInteractionGroup(self.frozen_atoms,self.active_atoms)
                    # Act-Act interaction
                    # custom_nonbonded_force.addInteractionGroup(self.active_atoms,self.active_atoms)
                    # print("2custom_nonbonded_force:", custom_nonbonded_force)

                    # Pointing self.nonbonded_force to CustomNonBondedForce instead of Nonbonded force
                    self.nonbonded_force = custom_nonbonded_force
                    print("self.nonbonded_force:", self.nonbonded_force)
                    self.custom_bondforce = custom_bond_force

                    # Update system with new forces and delete old force
                    self.system.addForce(self.nonbonded_force)
                    self.system.addForce(self.custom_bondforce)

                    # Remove oldNonbondedForce
                    for i, force in enumerate(self.system.getForces()):
                        if isinstance(force, self.openmm.NonbondedForce):
                            self.system.removeForce(i)

        # Defining nonbonded force
        for i, force in enumerate(self.system.getForces()):
            if isinstance(force, openmm.NonbondedForce):
                # self.getatomcharges()
                self.nonbonded_force = force

        # Set charges in OpenMMobject by taking from Force (used by QM/MM)
        print("Setting charges")
        # self.getatomcharges(self.nonbonded_force)
        self.getatomcharges()

        # Storing numatoms and list of all atoms
        self.numatoms = int(self.system.getNumParticles())
        self.allatoms = list(range(0, self.numatoms))
        print("Number of atoms in OpenMM system:", self.numatoms)

        print("System constraints defined upon system creation:", self.system.getNumConstraints())
        print("Use printlevel =>2 to see list of all constraints")
        if self.printlevel >= 3:
            for i in range(0, self.system.getNumConstraints()):
                constraint = self.system.getConstraintParameters(i)
                print("constraint:", constraint)
        print("self.system dict", self.system.__dict__)
        print_time_rel(timeA, modulename="system create")
        timeA = time.time()

        # constraints=simtk.openmm.app.HBonds, AllBonds, HAngles
        # Remove Frozen-Frozen interactions
        # Todo: Will be requested by QMMM object so unnecessary unless during pure MM??
        # if frozen_atoms is not None:
        #    print("Removing Frozen-Frozen interactions")
        #    self.addexceptions(frozen_atoms)

        # Modify particle masses in system object. For freezing atoms
        # for i in self.frozen_atoms:
        #    self.system.setParticleMass(i, 0 * simtk.openmm.unit.dalton)
        # print_time_rel(timeA, modulename="frozen atom setup")
        # timeA = time.time()

        # Modifying constraints after frozen-atom setting
        # print("Constraints:", self.system.getNumConstraints())

        # Finding defined constraints that involved frozen atoms. add to remove list
        # removelist=[]
        # for i in range(0,self.system.getNumConstraints()):
        #    constraint=self.system.getConstraintParameters(i)
        #    if constraint[0] in self.frozen_atoms or constraint[1] in self.frozen_atoms:
        #        #self.system.removeConstraint(i)
        #        removelist.append(i)

        # print("removelist:", removelist)
        # print("length removelist", len(removelist))
        # Remove constraints
        # removelist.reverse()
        # for r in removelist:
        #    self.system.removeConstraint(r)

        # print("Constraints:", self.system.getNumConstraints())
        # print_time_rel(timeA, modulename="constraint fix")
        timeA = time.time()

        # Platform
        print("Hardware platform:", self.platform_choice)
        self.platform = openmm.Platform.getPlatformByName(self.platform_choice)

        # Create simulation
        self.create_simulation()

        # Old:
        # NOTE: If self.system is modified then we have to remake self.simulation
        # self.simulation = simtk.openmm.app.simulation.Simulation(self.topology, self.system, self.integrator,self.platform)
        # self.simulation = self.simulationclass(self.topology, self.system, self.integrator,self.platform)

        print_time_rel(timeA, modulename="simulation setup")
        timeA = time.time()
        print_time_rel(module_init_time, modulename="OpenMM object creation")

    # add force that restrains atoms to a fixed point:
    # https://github.com/openmm/openmm/issues/2568

    # To set positions in OpenMMobject (in nm) from np-array (Angstrom)
    def set_positions(self, coords):
        print("Setting coordinates of OpenMM object")
        coords_nm = coords * 0.1  # converting from Angstrom to nm
        pos = [self.Vec3(coords_nm[i, 0], coords_nm[i, 1], coords_nm[i, 2]) for i in
               range(len(coords_nm))] * self.unit.nanometer
        self.simulation.context.setPositions(pos)
        print("Coordinates set")

    # This is custom externa force that restrains group of atoms to center of system
    def add_center_force(self, center_coords=None, atomindices=None, forceconstant=1.0):
        print("Inside add_center_force")
        print("center_coords:", center_coords)
        print("atomindices:", atomindices)
        print("forceconstant:", forceconstant)
        centerforce = self.openmm.CustomExternalForce("k * (abs(x-x0) + abs(y-y0) + abs(z-z0))")
        centerforce.addGlobalParameter("k",
                                       forceconstant * 4.184 * self.unit.kilojoule / self.unit.angstrom / self.unit.mole)
        centerforce.addPerParticleParameter('x0')
        centerforce.addPerParticleParameter('y0')
        centerforce.addPerParticleParameter('z0')
        # Coordinates of system center
        center_x = center_coords[0] / 10
        center_y = center_coords[1] / 10
        center_z = center_coords[2] / 10
        for i in atomindices:
            # centerforce.addParticle(i, np.array([0.0, 0.0, 0.0]))
            centerforce.addParticle(i, self.Vec3(center_x, center_y, center_z))
        self.system.addForce(centerforce)
        self.create_simulation()
        print("Added center force")
        return centerforce

    def add_custom_external_force(self):
        # customforce=None
        # inspired by https://github.com/CCQC/janus/blob/ba70224cd7872541d279caf0487387104c8253e6/janus/mm_wrapper/openmm_wrapper.py
        customforce = self.openmm.CustomExternalForce("-x*fx -y*fy -z*fz")
        # customforce.addGlobalParameter('shift', 0.0)
        customforce.addPerParticleParameter('fx')
        customforce.addPerParticleParameter('fy')
        customforce.addPerParticleParameter('fz')
        for i in range(self.system.getNumParticles()):
            customforce.addParticle(i, np.array([0.0, 0.0, 0.0]))
        self.system.addForce(customforce)
        # self.externalforce=customforce
        # Necessary:
        self.create_simulation()
        # http://docs.openmm.org/latest/api-c++/generated/OpenMM.CustomExternalForce.html

        print("Added force")
        return customforce

    def update_custom_external_force(self, customforce, gradient):
        # print("Updating custom external force")
        # shiftpar_inkjmol=shiftparameter*2625.4996394799
        # Convert Eh/Bohr gradient to force in kj/mol nm
        # *49614.501681716106452
        forces = -gradient * 49614.752589207
        for i, f in enumerate(forces):
            customforce.setParticleParameters(i, i, f)
        # print("xx")
        # self.externalforce.X(shiftparameter)
        # NOTE: updateParametersInContext expensive. Avoid somehow???
        # https://github.com/openmm/openmm/issues/1892
        # print("Current value of global par 0:", self.externalforce.getGlobalParameterDefaultValue(0))
        # self.externalforce.setGlobalParameterDefaultValue(0, shiftpar_inkjmol)
        # print("Current value of global par 0:", self.externalforce.getGlobalParameterDefaultValue(0))

        customforce.updateParametersInContext(self.simulation.context)

    # Write XML-file for full system
    def saveXML(self, xmlfile="system_full.xml"):
        serialized_system = self.openmm.XmlSerializer.serialize(self.system)
        with open(xmlfile, 'w') as f:
            f.write(serialized_system)
        print("Wrote system XML file:", xmlfile)

    # Function to add bond constraints to system before MD
    def add_bondconstraints(self, constraints=None):
        #for i in range(0,self.system.getNumConstraints()):
        #    print("Constraint:", i)
        #    print(self.system.getConstraintParameters(i))
        #prevconstraints=[self.system.getConstraintParameters(i) for i in range(0,self.system.getNumConstraints())]
        #print("prevconstraints:", prevconstraints)
        
        for i, j, d in constraints:
            #Checking if constraints previously defined
            #if ()
            #else:
            print("Adding bond constraint between atoms {} and {} . Distance value: {} Å".format(i, j, d))
            self.system.addConstraint(i, j, d * self.unit.angstroms)

    def remove_constraints(self, constraints):
        todelete = []
        # Looping over all defined system constraints
        for i in range(0, self.system.getNumConstraints()):
            con = self.system.getConstraintParameters(i)
            for usercon in constraints:
                if all(elem in usercon for elem in [con[0], con[1]]):
                    todelete.append(i)
        for d in reversed(todelete):
            self.system.removeConstraint(d)

    # Function to add restraints to system before MD
    def add_bondrestraints(self, restraints=None):
        new_restraints = self.openmm.HarmonicBondForce()
        for i, j, d, k in restraints:
            print(
                "Adding bond restraint between atoms {} and {}. Distance value: {} Å. Force constant: {} kcal/mol*Å^-2".format(
                    i, j, d, k))
            new_restraints.addBond(i, j, d * self.unit.angstroms,
                                   k * self.unit.kilocalories_per_mole / self.unit.angstroms ** 2)
        self.system.addForce(new_restraints)

    # TODO: Angleconstraints and Dihedral restraints

    # Function to freeze atoms during OpenMM MD simulation. Sets masses to zero. Does not modify potential energy-function.
    def freeze_atoms(self, frozen_atoms=None):
        # Preserve original masses
        self.system_masses = [self.system.getParticleMass(i) for i in self.allatoms]
        # print("self.system_masses:", self.system_masses)
        print("Freezing {} atoms by setting particles masses to zero.".format(len(frozen_atoms)))

        # Modify particle masses in system object. For freezing atoms
        for i in frozen_atoms:
            self.system.setParticleMass(i, 0 * self.unit.daltons)

    def unfreeze_atoms(self):
        # Looping over system_masses if frozen, otherwise empty list
        for atom, mass in zip(self.allatoms, self.system_masses):
            self.system.setParticleMass(atom, mass)

    # Currently unused
    def set_active_and_frozen_regions(self, active_atoms=None, frozen_atoms=None):
        # FROZEN AND ACTIVE ATOMS
        self.allatoms = list(range(0, self.numatoms))
        if active_atoms is None and frozen_atoms is None:
            print("All {} atoms active, no atoms frozen".format(len(self.allatoms)))
            self.frozen_atoms = []
        elif active_atoms is not None and frozen_atoms is None:
            self.active_atoms = active_atoms
            self.frozen_atoms = listdiff(self.allatoms, self.active_atoms)
            print("{} active atoms, {} frozen atoms".format(len(self.active_atoms), len(self.frozen_atoms)))
            # listdiff
        elif frozen_atoms is not None and active_atoms is None:
            self.frozen_atoms = frozen_atoms
            self.active_atoms = listdiff(self.allatoms, self.frozen_atoms)
            print("{} active atoms, {} frozen atoms".format(len(self.active_atoms), len(self.frozen_atoms)))
        else:
            print("active_atoms and frozen_atoms can not be both defined")
            exit(1)

    # This removes interactions between particles in a region (e.g. QM-QM or frozen-frozen pairs)
    # Give list of atom indices for which we will remove all pairs
    # Todo: Way too slow to do for big list of e.g. frozen atoms but works well for qmatoms list size
    # Alternative: Remove force interaction and then add in the interaction of active atoms to frozen atoms
    # should be reasonably fast
    # https://github.com/openmm/openmm/issues/2124
    # https://github.com/openmm/openmm/issues/1696
    def addexceptions(self, atomlist):
        timeA = time.time()
        import itertools
        print("Add exceptions/exclusions. Removing i-j interactions for list :", len(atomlist), "atoms")

        # Has duplicates
        # [self.nonbonded_force.addException(i,j,0, 0, 0, replace=True) for i in atomlist for j in atomlist]
        # https://stackoverflow.com/questions/942543/operation-on-every-pair-of-element-in-a-list
        # [self.nonbonded_force.addException(i,j,0, 0, 0, replace=True) for i,j in itertools.combinations(atomlist, r=2)]
        numexceptions = 0
        printdebug("self.system.getForces() ", self.system.getForces())
        # print("self.nonbonded_force:", self.nonbonded_force)

        for force in self.system.getForces():
            printdebug("force:", force)
            if isinstance(force, self.openmm.NonbondedForce):
                print("Case Nonbondedforce. Adding Exception for ij pair")
                for i in atomlist:
                    for j in atomlist:
                        printdebug("i,j : {} and {} ".format(i, j))
                        force.addException(i, j, 0, 0, 0, replace=True)

                        # NOTE: Case where there is also a CustomNonbonded force present (GROMACS interface).
                        # Then we have to add exclusion there too to avoid this issue: https://github.com/choderalab/perses/issues/357
                        # Basically both nonbonded forces have to have same exclusions (or exception where chargepro=0, eps=0)
                        # TODO: This leads to : Exception: CustomNonbondedForce: Multiple exclusions are specified for particles
                        # Basically we have to inspect what is actually present in CustomNonbondedForce
                        # for force in self.system.getForces():
                        #    if isinstance(force, self.openmm.CustomNonbondedForce):
                        #        force.addExclusion(i,j)

                        numexceptions += 1
            elif isinstance(force, self.openmm.CustomNonbondedForce):
                print("Case CustomNonbondedforce. Adding Exclusion for kl pair")
                for k in atomlist:
                    for l in atomlist:
                        # print("k,l : ", k,l)
                        force.addExclusion(k, l)
                        numexceptions += 1
        print("Number of exceptions/exclusions added: ", numexceptions)
        printdebug("self.system.getForces() ", self.system.getForces())
        # Seems like updateParametersInContext does not reliably work here so we have to remake the simulation instead
        # Might be bug (https://github.com/openmm/openmm/issues/2709). Revisit
        # self.nonbonded_force.updateParametersInContext(self.simulation.context)
        self.create_simulation()

        print_time_rel(timeA, modulename="add exception")

    # Run: coords or framents can be given (usually coords). qmatoms in order to avoid QM-QM interactions (TODO)
    # Probably best to do QM-QM exclusions etc. in a separate function though as we want run to be as simple as possible
    # qmatoms list provided for generality of MM objects. Not used here for now

    # Create/update simulation from scratch or after system has been modified (force modification or even deletion)
    def create_simulation(self, timestep=0.001, integrator='VerletIntegrator', coupling_frequency=None,
                          temperature=None):
        timeA = time.time()
        print("Creating/updating OpenMM simulation object")
        printdebug("self.system.getForces() ", self.system.getForces())
        # self.integrator = self.langevinintegrator(0.0000001 * self.unit.kelvin,  # Temperature of heat bath
        #                                1 / self.unit.picosecond,  # Friction coefficient
        #                                0.002 * self.unit.picoseconds)  # Time step

        # Integrators: LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator, BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
        if integrator == 'VerletIntegrator':
            self.integrator = self.openmm.VerletIntegrator(timestep * self.unit.picoseconds)
        elif integrator == 'VariableVerletIntegrator':
            self.integrator = self.openmm.VariableVerletIntegrator(timestep * self.unit.picoseconds)
        elif integrator == 'LangevinIntegrator':
            self.integrator = self.openmm.LangevinIntegrator(temperature * self.unit.kelvin,
                                                             coupling_frequency / self.unit.picosecond,
                                                             timestep * self.unit.picoseconds)
        elif integrator == 'LangevinMiddleIntegrator':
            # openmm recommended with 4 fs timestep, Hbonds 1/ps friction
            self.integrator = self.openmm.LangevinMiddleIntegrator(temperature * self.unit.kelvin,
                                                                   coupling_frequency / self.unit.picosecond,
                                                                   timestep * self.unit.picoseconds)
        elif integrator == 'NoseHooverIntegrator':
            self.integrator = self.openmm.NoseHooverIntegrator(temperature * self.unit.kelvin,
                                                               coupling_frequency / self.unit.picosecond,
                                                               timestep * self.unit.picoseconds)
        # NOTE: Problem with Brownian, disabling
        # elif integrator == 'BrownianIntegrator':
        #    self.integrator = self.openmm.BrownianIntegrator(temperature*self.unit.kelvin, coupling_frequency/self.unit.picosecond, timestep*self.unit.picoseconds)
        elif integrator == 'VariableLangevinIntegrator':
            self.integrator = self.openmm.VariableLangevinIntegrator(temperature * self.unit.kelvin,
                                                                     coupling_frequency / self.unit.picosecond,
                                                                     timestep * self.unit.picoseconds)
        else:
            print(BC.FAIL,
                  "Unknown integrator.\n Valid integrator keywords are: VerletIntegrator, VariableVerletIntegrator, LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VariableLangevinIntegrator ",
                  BC.END)
            exit()
        self.simulation = self.simulationclass(self.topology, self.system, self.integrator, self.platform,
                                               self.properties)
        print_time_rel(timeA, modulename="creating simulation")

    # Functions for energy decompositions
    def forcegroupify(self):
        self.forcegroups = {}
        print("inside forcegroupify")
        print("self.system.getForces() ", self.system.getForces())
        print("Number of forces:\n", self.system.getNumForces())
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            force.setForceGroup(i)
            self.forcegroups[force] = i
        # print("self.forcegroups :", self.forcegroups)
        # exit()

    def getEnergyDecomposition(self, context):
        # Call and set force groups
        self.forcegroupify()
        energies = {}
        # print("self.forcegroups:", self.forcegroups)
        for f, i in self.forcegroups.items():
            energies[f] = context.getState(getEnergy=True, groups=2 ** i).getPotentialEnergy()
        return energies

    def printEnergyDecomposition(self):
        timeA = time.time()
        # Energy composition
        # TODO: Calling this is expensive (seconds)as the energy has to be recalculated.
        # Only do for cases: a) single-point b) First energy-step in optimization and last energy-step
        # OpenMM energy components
        openmm_energy = dict()
        energycomp = self.getEnergyDecomposition(self.simulation.context)
        # print("energycomp: ", energycomp)
        # print("self.forcegroups:", self.forcegroups)
        # print("len energycomp", len(energycomp))
        # print("openmm_energy: ", openmm_energy)
        print("")
        bondterm_set = False
        extrafcount = 0
        # This currently assumes CHARMM36 components, More to be added
        for comp in energycomp.items():
            # print("comp: ", comp)
            if 'HarmonicBondForce' in str(type(comp[0])):
                # Not sure if this works in general.
                if bondterm_set is False:
                    openmm_energy['Bond'] = comp[1]
                    bondterm_set = True
                else:
                    openmm_energy['Urey-Bradley'] = comp[1]
            elif 'HarmonicAngleForce' in str(type(comp[0])):
                openmm_energy['Angle'] = comp[1]
            elif 'PeriodicTorsionForce' in str(type(comp[0])):
                # print("Here")
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
                extrafcount += 1
                openmm_energy['Otherforce' + str(extrafcount)] = comp[1]

        print_time_rel(timeA, modulename="energy decomposition")
        timeA = time.time()

        # The force terms to print in the ordered table.
        # Deprecated. Better to print everything.
        # Missing terms in force_terms will be printed separately
        # if self.Forcefield == 'CHARMM':
        #    force_terms = ['Bond', 'Angle', 'Urey-Bradley', 'Dihedrals', 'Impropers', 'CMAP', 'Nonbonded', '14-LJ']
        # else:
        #    #Modify...
        #    force_terms = ['Bond', 'Angle', 'Urey-Bradley', 'Dihedrals', 'Impropers', 'CMAP', 'Nonbonded']

        # Sum all force-terms
        sumofallcomponents = 0.0
        for val in openmm_energy.values():
            sumofallcomponents += val._value

        # Print energy table
        print('%-20s | %-15s | %-15s' % ('Component', 'kJ/mol', 'kcal/mol'))
        print('-' * 56)
        # TODO: Figure out better sorting of terms
        for name in sorted(openmm_energy):
            print('%-20s | %15.2f | %15.2f' % (name, openmm_energy[name] / self.unit.kilojoules_per_mole,
                                               openmm_energy[name] / self.unit.kilocalorie_per_mole))
        print('-' * 56)
        print('%-20s | %15.2f | %15.2f' % ('Sumcomponents', sumofallcomponents, sumofallcomponents / 4.184))
        print("")
        print('%-20s | %15.2f | %15.2f' % ('Total', self.energy * constants.hartokj, self.energy * constants.harkcal))

        print("")
        print("")
        # Adding sum to table
        openmm_energy['Sum'] = sumofallcomponents
        self.energy_components = openmm_energy

    def run(self, current_coords=None, elems=None, Grad=False, fragment=None, qmatoms=None):
        module_init_time = time.time()
        timeA = time.time()
        print(BC.OKBLUE, BC.BOLD, "------------RUNNING OPENMM INTERFACE-------------", BC.END)
        # If no coords given to run then a single-point job probably (not part of Optimizer or MD which would supply coords).
        # Then try if fragment object was supplied.
        # Otherwise internal coords if they exist
        if current_coords is None:
            if fragment is None:
                if len(self.coords) != 0:
                    print("Using internal coordinates (from OpenMM object)")
                    current_coords = self.coords
                else:
                    print("Found no coordinates!")
                    exit(1)
            else:
                current_coords = fragment.coords

        # Making sure coords is np array and not list-of-lists
        current_coords = np.array(current_coords)
        ##  unit conversion for energy
        # eqcgmx = 2625.5002
        ## unit conversion for force
        # TODO: Check this.
        # fqcgmx = -49614.75258920567
        # fqcgmx = -49621.9
        # Convert from kj/(nm *mol) = kJ/(10*Ang*mol)
        # factor=2625.5002/(10*1.88972612546)
        # factor=-138.93548724479302
        # Correct:
        factor = -49614.752589207

        # pos = [Vec3(coords[:,0]/10,coords[:,1]/10,coords[:,2]/10)] * u.nanometer
        # Todo: Check speed on this
        print("Updating coordinates")
        timeA = time.time()
        # print(type(current_coords))

        # for i in range(len(positions)):
        #    if isvsites[i]:
        #        pos[i] = vsfuncs[i](pos, vsidxs[i], vswts[i])
        # newpos = [self.Vec3(*i) for i in pos]*self.unit.nanometer

        # NOTE: THIS IS STILL RATHER SLOW

        current_coords_nm = current_coords * 0.1  # converting from Angstrom to nm
        pos = [self.Vec3(current_coords_nm[i, 0], current_coords_nm[i, 1], current_coords_nm[i, 2]) for i in
               range(len(current_coords_nm))] * self.unit.nanometer
        # pos = [self.Vec3(*v) for v in current_coords_nm] * self.unit.nanometer #slower
        print_time_rel(timeA, modulename="Creating pos array")
        timeA = time.time()
        # THIS IS THE SLOWEST PART. Probably nothing to be done
        self.simulation.context.setPositions(pos)

        print_time_rel(timeA, modulename="Updating MM positions")
        timeA = time.time()
        # While these distance constraints should not matter, applying them makes the energy function agree with previous benchmarking for bonded and nonbonded
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5549999/
        # Using 1e-6 hardcoded value since how used in paper
        # NOTE: Weirdly, applyconstraints is True result in constraints for TIP3P disappearing
        if self.applyconstraints is True:
            print("Applying constraints before calculating MM energy")
            self.simulation.context.applyConstraints(1e-6)
            print_time_rel(timeA, modulename="context: apply constraints")
            timeA = time.time()

        print("Calling OpenMM getState")
        if Grad is True:
            state = self.simulation.context.getState(getEnergy=True, getForces=True)
            self.energy = state.getPotentialEnergy().value_in_unit(self.unit.kilojoule_per_mole) / constants.hartokj
            self.gradient = np.array(state.getForces(asNumpy=True) / factor)
        else:
            state = self.simulation.context.getState(getEnergy=True, getForces=False)
            self.energy = state.getPotentialEnergy().value_in_unit(self.unit.kilojoule_per_mole) / constants.hartokj

        print_time_rel(timeA, modulename="OpenMM getState")
        timeA = time.time()
        print("OpenMM Energy:", self.energy, "Eh")
        print("OpenMM Energy:", self.energy * constants.harkcal, "kcal/mol")

        # Do energy components or not. Can be turned off for e.g. MM MD simulation
        if self.do_energy_decomposition is True:
            self.printEnergyDecomposition()

        print("self.energy : ", self.energy, "Eh")
        print("Energy:", self.energy * constants.harkcal, "kcal/mol")
        # print("Grad is", Grad)
        # print("self.gradient:", self.gradient)

        print(BC.OKBLUE, BC.BOLD, "------------ENDING OPENMM INTERFACE-------------", BC.END)
        print_time_rel(module_init_time, modulename="OpenMM run", moduleindex=2)
        if Grad is True:
            return self.energy, self.gradient
        else:
            return self.energy

    # Get list of charges from chosen force object (usually original nonbonded force object)
    def getatomcharges_old(self, force):
        chargelist = []
        for i in range(force.getNumParticles()):
            charge = force.getParticleParameters(i)[0]
            if isinstance(charge, self.unit.Quantity):
                charge = charge / self.unit.elementary_charge
                chargelist.append(charge)
        self.charges = chargelist
        return chargelist

    def getatomcharges(self):
        chargelist = []
        for force in self.system.getForces():
            if isinstance(force, self.openmm.NonbondedForce):
                for i in range(force.getNumParticles()):
                    charge = force.getParticleParameters(i)[0]
                    if isinstance(charge, self.unit.Quantity):
                        charge = charge / self.unit.elementary_charge
                        chargelist.append(charge)
                self.charges = chargelist
        return chargelist

    # Delete selected exceptions. Only for Coulomb.
    # Used to delete Coulomb interactions involving QM-QM and QM-MM atoms
    def delete_exceptions(self, atomlist):
        timeA = time.time()
        print("Deleting Coulombexceptions for atomlist:", atomlist)
        for force in self.system.getForces():
            if isinstance(force, self.openmm.NonbondedForce):
                for exc in range(force.getNumExceptions()):
                    # print(force.getExceptionParameters(exc))
                    # force.getExceptionParameters(exc)
                    p1, p2, chargeprod, sigmaij, epsilonij = force.getExceptionParameters(exc)
                    if p1 in atomlist or p2 in atomlist:
                        # print("p1: {} and p2: {}".format(p1,p2))
                        # print("chargeprod:", chargeprod)
                        # print("sigmaij:", sigmaij)
                        # print("epsilonij:", epsilonij)
                        chargeprod._value = 0.0
                        force.setExceptionParameters(exc, p1, p2, chargeprod, sigmaij, epsilonij)
                        # print("New:", force.getExceptionParameters(exc))
        self.create_simulation()
        print_time_rel(timeA, modulename="delete_exceptions")

    # Function to
    def zero_nonbondedforce(self, atomlist, zeroCoulomb=True, zeroLJ=True):
        timeA = time.time()
        print("Zero-ing nonbondedforce")

        def charge_sigma_epsilon(charge, sigma, epsilon):
            if zeroCoulomb is True:
                newcharge = charge
                newcharge._value = 0.0

            else:
                newcharge = charge
            if zeroLJ is True:
                newsigma = sigma
                newsigma._value = 0.0
                newepsilon = epsilon
                newepsilon._value = 0.0
            else:
                newsigma = sigma
                newepsilon = epsilon
            return [newcharge, newsigma, newepsilon]

        # Zero all nonbonding interactions for atomlist
        for force in self.system.getForces():
            if isinstance(force, self.openmm.NonbondedForce):
                # Setting single particle parameters
                for atomindex in atomlist:
                    oldcharge, oldsigma, oldepsilon = force.getParticleParameters(atomindex)
                    newpars = charge_sigma_epsilon(oldcharge, oldsigma, oldepsilon)
                    print(newpars)
                    force.setParticleParameters(atomindex, newpars[0], newpars[1], newpars[2])
                print("force.getNumExceptions() ", force.getNumExceptions())
                print("force.getNumExceptionParameterOffsets() ", force.getNumExceptionParameterOffsets())
                print("force.getNonbondedMethod():", force.getNonbondedMethod())
                print("force.getNumGlobalParameters() ", force.getNumGlobalParameters())
                # Now doing exceptions
                for exc in range(force.getNumExceptions()):
                    print(force.getExceptionParameters(exc))
                    force.getExceptionParameters(exc)
                    p1, p2, chargeprod, sigmaij, epsilonij = force.getExceptionParameters(exc)
                    # chargeprod._value=0.0
                    # sigmaij._value=0.0
                    # epsilonij._value=0.0
                    newpars2 = charge_sigma_epsilon(chargeprod, sigmaij, epsilonij)
                    force.setExceptionParameters(exc, p1, p2, newpars2[0], newpars2[1], newpars2[2])
                    # print("New:", force.getExceptionParameters(exc))
                # force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.CustomNonbondedForce):
                print("customnonbondedforce not implemented")
                exit()
        self.create_simulation()
        print_time_rel(timeA, modulename="zero_nonbondedforce")
        # self.create_simulation()

    # Updating charges in OpenMM object. Used to set QM charges to 0 for example
    # Taking list of atom-indices and list of charges (usually zero) and setting new charge
    # Note: Exceptions also needs to be dealt with (see delete_exceptions)
    def update_charges(self, atomlist, atomcharges):
        timeA = time.time()
        print("Updating charges in OpenMM object.")
        assert len(atomlist) == len(atomcharges)
        newcharges = []
        # print("atomlist:", atomlist)
        for atomindex, newcharge in zip(atomlist, atomcharges):
            # Updating big chargelist of OpenMM object.
            # TODO: Is this actually used?
            self.charges[atomindex] = newcharge
            # print("atomindex: ", atomindex)
            # print("newcharge: ",newcharge)
            oldcharge, sigma, epsilon = self.nonbonded_force.getParticleParameters(atomindex)
            # Different depending on type of NonbondedForce
            if isinstance(self.nonbonded_force, self.openmm.CustomNonbondedForce):
                self.nonbonded_force.setParticleParameters(atomindex, [newcharge, sigma, epsilon])
                # bla1,bla2,bla3 = self.nonbonded_force.getParticleParameters(i)
                # print("bla1,bla2,bla3", bla1,bla2,bla3)
            elif isinstance(self.nonbonded_force, self.openmm.NonbondedForce):
                self.nonbonded_force.setParticleParameters(atomindex, newcharge, sigma, epsilon)
                # bla1,bla2,bla3 = self.nonbonded_force.getParticleParameters(atomindex)
                # print("bla1,bla2,bla3", bla1,bla2,bla3)

        # Instead of recreating simulation we can just update like this:
        print("Updating simulation object for modified Nonbonded force")
        printdebug("self.nonbonded_force:", self.nonbonded_force)
        # Making sure that there still is a nonbonded force present in system (in case deleted)
        for i, force in enumerate(self.system.getForces()):
            printdebug("i is {} and force is {}".format(i, force))
            if isinstance(force, self.openmm.NonbondedForce):
                printdebug("here")
                self.nonbonded_force.updateParametersInContext(self.simulation.context)
            if isinstance(force, self.openmm.CustomNonbondedForce):
                self.nonbonded_force.updateParametersInContext(self.simulation.context)
        self.create_simulation()
        printdebug("done here")
        print_time_rel(timeA, modulename="update_charges")

    def modify_bonded_forces(self, atomlist):
        timeA = time.time()
        print("Modifying bonded forces")
        print("")
        # This is typically used by QM/MM object to set bonded forces to zero for qmatoms (atomlist)
        # Mimicking: https://github.com/openmm/openmm/issues/2792

        numharmbondterms_removed = 0
        numharmangleterms_removed = 0
        numpertorsionterms_removed = 0
        numcustomtorsionterms_removed = 0
        numcmaptorsionterms_removed = 0
        numcmmotionterms_removed = 0
        numcustombondterms_removed = 0

        for force in self.system.getForces():
            if isinstance(force, self.openmm.HarmonicBondForce):
                printdebug("HarmonicBonded force")
                printdebug("There are {} HarmonicBond terms defined.".format(force.getNumBonds()))
                printdebug("")
                # REVISIT: Neglecting QM-QM and sQM1-MM1 interactions. i.e if one atom in bond-pair is QM we neglect
                for i in range(force.getNumBonds()):
                    # print("i:", i)
                    p1, p2, length, k = force.getBondParameters(i)
                    # print("p1: {} p2: {} length: {} k: {}".format(p1,p2,length,k))
                    # or: delete QM-QM and QM-MM
                    # and: delete QM-QM

                    if self.delete_QM1_MM1_bonded is True:
                        exclude = (p1 in atomlist or p2 in atomlist)
                    else:
                        exclude = (p1 in atomlist and p2 in atomlist)
                    # print("exclude:", exclude)
                    if exclude is True:
                        printdebug("exclude True")
                        printdebug("atomlist:", atomlist)
                        printdebug("i:", i)
                        printdebug("Before p1: {} p2: {} length: {} k: {}".format(p1, p2, length, k))
                        force.setBondParameters(i, p1, p2, length, 0)
                        numharmbondterms_removed += 1
                        p1, p2, length, k = force.getBondParameters(i)
                        printdebug("After p1: {} p2: {} length: {} k: {}".format(p1, p2, length, k))
                        printdebug("")
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.HarmonicAngleForce):
                printdebug("HarmonicAngle force")
                printdebug("There are {} HarmonicAngle terms defined.".format(force.getNumAngles()))
                for i in range(force.getNumAngles()):
                    p1, p2, p3, angle, k = force.getAngleParameters(i)
                    # Are angle-atoms in atomlist?
                    presence = [i in atomlist for i in [p1, p2, p3]]
                    # Excluding if 2 or 3 QM atoms. i.e. a QM2-QM1-MM1 or QM3-QM2-QM1 term
                    # Originally set to 2
                    if presence.count(True) >= 2:
                        printdebug("presence.count(True):", presence.count(True))
                        printdebug("exclude True")
                        printdebug("atomlist:", atomlist)
                        printdebug("i:", i)
                        printdebug("Before p1: {} p2: {} p3: {} angle: {} k: {}".format(p1, p2, p3, angle, k))
                        force.setAngleParameters(i, p1, p2, p3, angle, 0)
                        numharmangleterms_removed += 1
                        p1, p2, p3, angle, k = force.getAngleParameters(i)
                        printdebug("After p1: {} p2: {} p3: {} angle: {} k: {}".format(p1, p2, p3, angle, k))
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.PeriodicTorsionForce):
                printdebug("PeriodicTorsionForce force")
                printdebug("There are {} PeriodicTorsionForce terms defined.".format(force.getNumTorsions()))
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                    # Are torsion-atoms in atomlist?
                    presence = [i in atomlist for i in [p1, p2, p3, p4]]
                    # Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    # print("Before p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1,p2,p3,p4,periodicity, phase,k))
                    # Originally set to 3
                    if presence.count(True) >= 3:
                        printdebug("Found torsion in QM-region")
                        printdebug("presence.count(True):", presence.count(True))
                        printdebug("exclude True")
                        printdebug("atomlist:", atomlist)
                        printdebug("i:", i)
                        printdebug(
                            "Before p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1, p2, p3, p4,
                                                                                                        periodicity,
                                                                                                        phase, k))
                        force.setTorsionParameters(i, p1, p2, p3, p4, periodicity, phase, 0)
                        numpertorsionterms_removed += 1
                        p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                        printdebug(
                            "After p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1, p2, p3, p4,
                                                                                                       periodicity,
                                                                                                       phase, k))
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.CustomTorsionForce):
                printdebug("CustomTorsionForce force")
                printdebug("There are {} CustomTorsionForce terms defined.".format(force.getNumTorsions()))
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                    # Are torsion-atoms in atomlist?
                    presence = [i in atomlist for i in [p1, p2, p3, p4]]
                    # Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    # print("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                    # print("pars:", pars)
                    if presence.count(True) >= 3:
                        printdebug("Found torsion in QM-region")
                        printdebug("presence.count(True):", presence.count(True))
                        printdebug("exclude True")
                        printdebug("atomlist:", atomlist)
                        printdebug("i:", i)
                        printdebug("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1, p2, p3, p4, pars))
                        force.setTorsionParameters(i, p1, p2, p3, p4, (0.0, 0.0))
                        numcustomtorsionterms_removed += 1
                        p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                        print("After p1: {} p2: {} p3: {} p4: {} pars {}".format(p1, p2, p3, p4, pars))
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.CMAPTorsionForce):
                printdebug("CMAPTorsionForce force")
                printdebug("There are {} CMAP terms defined.".format(force.getNumTorsions()))
                printdebug("There are {} CMAP maps defined".format(force.getNumMaps()))
                # print("Assuming no CMAP terms in QM-region. Continuing")
                # Note (RB). CMAP is between pairs of backbone dihedrals.
                # Not sure if we can delete the terms:
                # http://docs.openmm.org/latest/api-c++/generated/OpenMM.CMAPTorsionForce.html
                #  
                # print("Map num 0", force.getMapParameters(0))
                # print("Map num 1", force.getMapParameters(1))
                # print("Map num 2", force.getMapParameters(2))
                for i in range(force.getNumTorsions()):
                    jj, p1, p2, p3, p4, v1, v2, v3, v4 = force.getTorsionParameters(i)
                    # Are torsion-atoms in atomlist?
                    presence = [i in atomlist for i in [p1, p2, p3, p4, v1, v2, v3, v4]]
                    # NOTE: Not sure how to use count properly here when dealing with torsion atoms in QM-region
                    if presence.count(True) >= 4:
                        printdebug(
                            "jj: {} p1: {} p2: {} p3: {} p4: {}      v1: {} v2: {} v3: {} v4: {}".format(jj, p1, p2, p3,
                                                                                                         p4, v1, v2, v3,
                                                                                                         v4))
                        printdebug("presence:", presence)
                        printdebug("Found CMAP torsion partner in QM-region")
                        printdebug("Not deleting. To be revisited...")
                        # print("presence.count(True):", presence.count(True))
                        # print("exclude True")
                        # print("atomlist:", atomlist)
                        # print("i:", i)
                        # print("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                        # force.setTorsionParameters(i, p1, p2, p3, p4, (0.0,0.0))
                        # numcustomtorsionterms_removed+=1
                        # p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                        # print("After p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                # force.updateParametersInContext(self.simulation.context)

            elif isinstance(force, self.openmm.CustomBondForce):
                printdebug("CustomBondForce")
                printdebug("There are {} force terms defined.".format(force.getNumBonds()))
                # Neglecting QM1-MM1 interactions. i.e if one atom in bond-pair is QM we neglect
                for i in range(force.getNumBonds()):
                    # print("i:", i)
                    p1, p2, vars = force.getBondParameters(i)
                    # print("p1: {} p2: {}".format(p1,p2))
                    exclude = (p1 in atomlist and p2 in atomlist)
                    # print("exclude:", exclude)
                    if exclude is True:
                        # print("exclude True")
                        # print("atomlist:", atomlist)
                        # print("i:", i)
                        # print("Before")
                        # print("p1: {} p2: {}")
                        force.setBondParameters(i, p1, p2, [0.0, 0.0, 0.0])
                        numcustombondterms_removed += 1
                        p1, p2, vars = force.getBondParameters(i)
                        # print("p1: {} p2: {}")
                        # print("vars:", vars)
                        # exit()
                force.updateParametersInContext(self.simulation.context)

            elif isinstance(force, self.openmm.CMMotionRemover):
                pass
                # print("CMMotionRemover ")
                # print("nothing to be done")
            elif isinstance(force, self.openmm.CustomNonbondedForce):
                pass
                # print("CustomNonbondedForce force")
                # print("nothing to be done")
            elif isinstance(force, self.openmm.NonbondedForce):
                pass
                # print("NonbondedForce force")
                # print("nothing to be done")
            else:
                pass
                # print("Other force: ", force)
                # print("nothing to be done")

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


# For frozen systems we use Customforce in order to specify interaction groups
# if len(self.frozen_atoms) > 0:

# Two possible ways.
# https://github.com/openmm/openmm/issues/2698
# 1. Use CustomNonbondedForce  with interaction groups. Could be slow
# 2. CustomNonbondedForce but with scaling


# https://ahy3nz.github.io/posts/2019/30/openmm2/
# http://www.maccallumlab.org/news/2015/1/23/testing

# Comes close to NonbondedForce results (after exclusions) but still not correct
# The issue is most likely that the 1-4 LJ interactions should not be excluded but rather scaled.
# See https://github.com/openmm/openmm/issues/1200
# https://github.com/openmm/openmm/issues/1696
# How to do:
# 1. Keep nonbonded force for only those interactions and maybe also electrostatics?
# Mimic this??: https://github.com/openmm/openmm/blob/master/devtools/forcefield-scripts/processCharmmForceField.py
# Or do it via Parmed? Better supported for future??
# 2. Go through the 1-4 interactions and not exclude but scale somehow manually. But maybe we can't do that in CustomNonbonded Force?
# Presumably not but maybe can add a special force object just for 1-4 interactions. We
def create_cnb(original_nbforce):
    """Creates a CustomNonbondedForce object that mimics the original nonbonded force
    and also a Custombondforce to handle 14 exceptions
    """
    # Next, create a CustomNonbondedForce with LJ and Coulomb terms
    ONE_4PI_EPS0 = 138.935456
    # ONE_4PI_EPS0=1.0
    # TODO: Not sure whether sqrt should be present or not in epsilon???
    energy_expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod/r;"
    # sqrt ??
    energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"
    energy_expression += "sigma = 0.5*(sigma1+sigma2);"
    energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units
    energy_expression += "chargeprod = charge1*charge2;"
    custom_nonbonded_force = openmm.CustomNonbondedForce(energy_expression)
    custom_nonbonded_force.addPerParticleParameter('charge')
    custom_nonbonded_force.addPerParticleParameter('sigma')
    custom_nonbonded_force.addPerParticleParameter('epsilon')
    # Configure force
    custom_nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
    # custom_nonbonded_force.setCutoffDistance(9999999999)
    custom_nonbonded_force.setUseLongRangeCorrection(False)
    # custom_nonbonded_force.setUseSwitchingFunction(True)
    # custom_nonbonded_force.setSwitchingDistance(99999)
    print('adding particles to custom force')
    for index in range(self.system.getNumParticles()):
        [charge, sigma, epsilon] = original_nbforce.getParticleParameters(index)
        custom_nonbonded_force.addParticle([charge, sigma, epsilon])
    # For CustomNonbondedForce we need (unlike NonbondedForce) to create exclusions that correspond to the automatic exceptions in NonbondedForce
    # These are interactions that are skipped for bonded atoms
    numexceptions = original_nbforce.getNumExceptions()
    print("numexceptions in original_nbforce: ", numexceptions)

    # Turn exceptions from NonbondedForce into exclusions in CustombondedForce
    # except 1-4 which are not zeroed but are scaled. These are added to Custombondforce
    exceptions_14 = []
    numexclusions = 0
    for i in range(0, numexceptions):
        # print("i:", i)
        # Get exception parameters (indices)
        p1, p2, charge, sigma, epsilon = original_nbforce.getExceptionParameters(i)
        # print("p1,p2,charge,sigma,epsilon:", p1,p2,charge,sigma,epsilon)
        # If 0.0 then these are CHARMM 1-2 and 1-3 interactions set to zero
        if charge._value == 0.0 and epsilon._value == 0.0:
            # print("Charge and epsilons are 0.0. Add proper exclusion")
            # Set corresponding exclusion in customnonbforce
            custom_nonbonded_force.addExclusion(p1, p2)
            numexclusions += 1
        else:
            # print("This is not an exclusion but a scaled interaction as it is is non-zero. Need to keep")
            exceptions_14.append([p1, p2, charge, sigma, epsilon])
            # [798, 801, Quantity(value=-0.0684, unit=elementary charge**2), Quantity(value=0.2708332103146632, unit=nanometer), Quantity(value=0.2672524882578271, unit=kilojoule/mole)]

    print("len exceptions_14", len(exceptions_14))
    # print("exceptions_14:", exceptions_14)
    print("numexclusions:", numexclusions)

    # Creating custombondforce to handle these special exceptions
    # Now defining pair parameters
    # https://github.com/openmm/openmm/issues/2698
    energy_expression = "(4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod/r);"
    energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units
    custom_bond_force = self.openmm.CustomBondForce(energy_expression)
    custom_bond_force.addPerBondParameter('chargeprod')
    custom_bond_force.addPerBondParameter('sigma')
    custom_bond_force.addPerBondParameter('epsilon')

    for exception in exceptions_14:
        idx = exception[0]
        jdx = exception[1]
        c = exception[2]
        sig = exception[3]
        eps = exception[4]
        custom_bond_force.addBond(idx, jdx, [c, sig, eps])

    print('Number of defined 14 bonds in custom_bond_force:', custom_bond_force.getNumBonds())

    return custom_nonbonded_force, custom_bond_force


# TODO: Look into: https://github.com/ParmEd/ParmEd/blob/7e411fd03c7db6977e450c2461e065004adab471/parmed/structure.py#L2554

# myCustomNBForce= simtk.openmm.CustomNonbondedForce("4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)")
# myCustomNBForce.setNonbondedMethod(simtk.openmm.app.NoCutoff)
# myCustomNBForce.setCutoffDistance(1000*simtk.openmm.unit.angstroms)
# Frozen-Act interaction
# myCustomNBForce.addInteractionGroup(self.frozen_atoms,self.active_atoms)
# Act-Act interaction
# myCustomNBForce.addInteractionGroup(self.active_atoms,self.active_atoms)


# Clean up list of lists of constraint definition. Add distance if missing
def clean_up_constraints_list(fragment=None, constraints=None):
    print("Checking defined constraints")
    newconstraints = []
    for con in constraints:
        if len(con) == 3:
            newconstraints.append(con)
        elif len(con) == 2:
            distance = distance_between_atoms(fragment=fragment, atom1=con[0], atom2=con[1])
            print("Missing distance definition between atoms {} and {}. Adding distance {} Å".format(con[0], con[1],
                                                                                                     distance))
            newcon = [con[0], con[1], distance]
            newconstraints.append(newcon)
    return newconstraints


# Simple Molecular Dynamics using the OpenMM  object
# Integrators: LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator, BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
# Additional thermostat: AndersenThermostat (use with Verlet)
# Barostat: MonteCarloBarostat (not yet supported: MonteCarloAnisotropicBarostat, MonteCarloMembraneBarostat)

# Note: enforcePeriodicBox=True/False/None option. False works for current test system to get trajectory without PBC problem. Other systems may require True or None.
# see https://github.com/openmm/openmm/issues/2688, https://github.com/openmm/openmm/pull/1895
# Also should we add: https://github.com/mdtraj/mdtraj ?
def OpenMM_MD(fragment=None, theory=None, timestep=0.001, simulation_steps=None, simulation_time=None,
              traj_frequency=1000, temperature=300, integrator=None,
              barostat=None, pressure=1, trajectory_file_option='PDB', coupling_frequency=None, anderson_thermostat=False,
              enforcePeriodicBox=True, frozen_atoms=None, constraints=None, restraints=None,
              datafilename=None, dummy_MM=False, plumed_object=None, add_center_force=False,
              center_force_atoms=None, centerforce_constant=1.0):
    module_init_time = time.time()

    print_line_with_mainheader("OpenMM MOLECULAR DYNAMICS")

    # Distinguish between OpenMM theory or QM/MM theory
    if theory.__class__.__name__ == "OpenMMTheory":
        openmmobject = theory
        QM_MM_object = None
    elif theory.__class__.__name__ == "QMMMTheory":
        QM_MM_object = theory
        openmmobject = theory.mm_theory
    else:
        print("Unknown theory. Exiting")
        exit()

    if frozen_atoms is None:
        frozen_atoms = []
    if constraints is None:
        constraints = []
    if restraints is None:
        restraints = []

    if simulation_steps is None and simulation_time is None:
        print("Either simulation_steps or simulation_time needs to be set")
        exit()
    if fragment is None:
        print("No fragment object. Exiting")
        exit()
    if simulation_time is not None:
        simulation_steps = int(simulation_time / timestep)
    if simulation_steps is not None:
        simulation_time = simulation_steps * timestep
    print("Simulation time: {} ps".format(simulation_time))
    print("Timestep: {} ps".format(timestep))
    print("Simulation steps: {}".format(simulation_steps))
    print("Temperature: {} K".format(temperature))
    print("Number of frozen atoms:", len(frozen_atoms))
    if len(frozen_atoms) < 50:
        print("Frozen atoms", frozen_atoms)
    print("OpenMM autoconstraints:", openmmobject.autoconstraints)
    print("OpenMM hydrogenmass:",
          openmmobject.hydrogenmass)  # Note 1.5 amu mass is recommended for LangevinMiddle with 4fs timestep
    print("OpenMM rigidwater constraints:", openmmobject.rigidwater)
    print("Constraints:", constraints)
    print("Restraints:", restraints)
    print("Integrator:", integrator)

    print("Anderon Thermostat:", anderson_thermostat)
    print("coupling_frequency: {} ps^-1 (for Nosé-Hoover and Langevin integrators)".format(coupling_frequency))
    print("Barostat:", barostat)

    print("")
    print("Will write trajectory in format:", trajectory_file_option)
    print("Trajectory write frequency:", traj_frequency)
    print("enforcePeriodicBox option:", enforcePeriodicBox)
    print("")

    if openmmobject.autoconstraints is None:
        print(BC.WARNING, "Warning: Autoconstraints have not been set in OpenMMTheory object definition.")
        print(
            "This means that by default no bonds are constrained in the MD simulation. This usually requires a small timestep: 0.5 fs or so")
        print("autoconstraints='HBonds' is recommended for 1-2 fs timesteps with Verlet (4fs with Langevin).")
        print("autoconstraints='AllBonds' or autoconstraints='HAngles' allows even larger timesteps to be used", BC.END)
        print("Will continue...")
    if openmmobject.rigidwater is True and len(frozen_atoms) != 0 or (
            openmmobject.autoconstraints is not None and len(frozen_atoms) != 0):
        print(
            "Warning: Frozen_atoms options selected but there are general constraints defined in the OpenMM object (either rigidwater=True or autoconstraints != None")
        print("OpenMM will crash if constraints and frozen atoms involve the same atoms")
    print("")

    # createSystem(constraints=None), createSystem(constraints=HBonds), createSystem(constraints=All-Bonds), createSystem(constraints=HAngles)
    # HBonds constraints: timestep can be 2fs with Verlet and 4fs with Langevin
    # HAngles constraints: even larger timesteps
    # HAngles constraints: even larger timesteps

    print("Before adding constraints, system contains {} constraints".format(openmmobject.system.getNumConstraints()))

    # Freezing atoms in OpenMM object by setting particles masses to zero. Needs to be done before simulation creation
    if len(frozen_atoms) > 0:
        openmmobject.freeze_atoms(frozen_atoms=frozen_atoms)

    # Adding constraints/restraints between atoms
    if len(constraints) > 0:
        print("Constraints defined.")
        # constraints is a list of lists defining bond constraints: constraints = [[700,701], [802,803,1.04]]
        # Cleaning up constraint list. Adding distance if missing
        constraints = clean_up_constraints_list(fragment=fragment, constraints=constraints)
        print("Will enforce constrain definitions during MD:", constraints)
        openmmobject.add_bondconstraints(constraints=constraints)
    if len(restraints) > 0:
        print("Restraints defined")
        # restraints is a list of lists defining bond restraints: constraints = [[atom_i,atom_j, d, k ]]    Example: [[700,701, 1.05, 5.0 ]] Unit is Angstrom and kcal/mol * Angstrom^-2
        openmmobject.add_bondrestraints(restraints=restraints)

    print("After adding constraints, system contains {} constraints".format(openmmobject.system.getNumConstraints()))

    forceclassnames=[i.__class__.__name__ for i in openmmobject.system.getForces()]
    # Set up system with chosen barostat, thermostat, integrator
    if barostat is not None:
        print("Attempting to add barostat")
        if "MonteCarloBarostat" not in forceclassnames:
            print("Adding barostat")
            openmmobject.system.addForce(openmmobject.openmm.MonteCarloBarostat(pressure * openmmobject.openmm.unit.bar,
                                                                            temperature * openmmobject.openmm.unit.kelvin))
        else:
            print("Barostat already present. Skipping")
        print("after barostat added")

        integrator = "LangevinMiddleIntegrator"
        print("Barostat requires using integrator:", integrator)
        openmmobject.create_simulation(timestep=timestep, temperature=temperature, integrator=integrator,
                                       coupling_frequency=coupling_frequency)
    elif anderson_thermostat is True:
        print("Anderson thermostat is on")
        if "AndersenThermostat" not in forceclassnames:
            openmmobject.system.addForce(
                openmmobject.openmm.AndersenThermostat(temperature * openmmobject.openmm.unit.kelvin,
                                                    1 / openmmobject.openmm.unit.picosecond))
        integrator = "VerletIntegrator"
        print("Now using integrator:", integrator)
        openmmobject.create_simulation(timestep=timestep, temperature=temperature, integrator=integrator,
                                       coupling_frequency=coupling_frequency)
    else:
        #Deleting barostat and Andersen thermostat if present from previous sims
        for i,forcename in enumerate(forceclassnames):
            if forcename == "MonteCarloBarostat" or forcename == "AndersenThermostat":
                print("Removing old force:", forcename)
                openmmobject.system.removeForce(i)
        
        # Regular thermostat or integrator without barostat
        # Integrators: LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator,
        # BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
        openmmobject.create_simulation(timestep=timestep, temperature=temperature, integrator=integrator,
                                       coupling_frequency=coupling_frequency)
    print("Simulation created.")
    forceclassnames=[i.__class__.__name__ for i in openmmobject.system.getForces()]
    print("OpenMM System forces present:", forceclassnames)
    print("Checking Initial PBC vectors")
    state = openmmobject.simulation.context.getState()
    a, b, c = state.getPeriodicBoxVectors()
    print(f"A: ", a)
    print(f"B: ", b)
    print(f"C: ", c)

    # THIS DOES NOT APPLY TO QM/MM. MOVE ELSEWHERE??
    if trajectory_file_option == 'PDB':
        openmmobject.simulation.reporters.append(openmmobject.openmm.app.PDBReporter('output_traj.pdb', traj_frequency,
                                                                                     enforcePeriodicBox=enforcePeriodicBox))
    elif trajectory_file_option == 'DCD':
        # NOTE: Disabling for now
        # with open('initial_MDfrag_step1.pdb', 'w') as f: openmmobject.openmm.app.pdbfile.PDBFile
        # .writeModel(openmmobject.topology, openmmobject.simulation.context.getState(getPositions=True,
        # enforcePeriodicBox=enforcePeriodicBox).getPositions(), f)
        # print("Wrote PDB")
        openmmobject.simulation.reporters.append(openmmobject.openmm.app.DCDReporter('output_traj.dcd', traj_frequency,
                                                                                     enforcePeriodicBox=enforcePeriodicBox))
    elif trajectory_file_option == 'NetCDFReporter':
        print("NetCDFReporter traj format selected. This requires mdtraj. Importing.")
        mdtraj = MDtraj_import_()
        openmmobject.simulation.reporters.append(mdtraj.reporters.NetCDFReporter('output_traj.nc', traj_frequency))
    elif trajectory_file_option == 'HDF5Reporter':
        print("HDF5Reporter traj format selected. This requires mdtraj. Importing.")
        mdtraj = MDtraj_import_()
        openmmobject.simulation.reporters.append(
            mdtraj.reporters.HDF5Reporter('output_traj.lh5', traj_frequency, enforcePeriodicBox=enforcePeriodicBox))

    if barostat is not None:
        volume = density = True
    else:
        volume = density = False

    #If statedatareporter filename set:
    if datafilename != None:
        outputoption=datafilename
    #otherwise stdout:
    else:
        outputoption=stdout
    
    openmmobject.simulation.reporters.append(
        openmmobject.openmm.app.StateDataReporter(outputoption, traj_frequency, step=True, time=True,
                                                    potentialEnergy=True, kineticEnergy=True, volume=volume,
                                                    density=density, temperature=True, separator=','))


    # Run simulation
    kjmolnm_to_atomic_factor = -49614.752589207

    # NOTE: Better to use OpenMM-plumed interface instead??
    if plumed_object is not None:
        print("Plumed active")
        # Create new OpenMM custom external force
        print("Creating new OpenMM custom external force for Plumed")
        plumedcustomforce = openmmobject.add_custom_external_force()

    # QM/MM MD
    if QM_MM_object is not None:
        print("QM_MM_object provided. Switching to QM/MM loop")
        print("QM/MM requires enforcePeriodicBox to be False")
        enforcePeriodicBox = False
        # enforcePeriodicBox or not
        print("enforcePeriodicBox:", enforcePeriodicBox)

        # OpenMM_MD with QM/MM object does not make sense without openmm_externalforce
        # (it would calculate OpenMM energy twice) so turning on in case forgotten
        if QM_MM_object.openmm_externalforce is False:
            print("QM/MM object was not set to have openmm_externalforce=True.")
            print("Turning on externalforce option")
            QM_MM_object.openmm_externalforce = True
            QM_MM_object.openmm_externalforceobject = QM_MM_object.mm_theory.add_custom_external_force()
        # TODO:
        # Should we set parallelization of QM theory here also in case forgotten?

        centercoordinates = False
        # CENTER COORDINATES HERE on SOLUTE HERE ??
        # TODO: Deprecated I think
        if centercoordinates is True:
            # Solute atoms assumed to be QM-region
            fragment.write_xyzfile(xyzfilename="fragment-before-centering.xyz")
            soluteatoms = QM_MM_object.qmatoms
            solutecoords = fragment.get_coords_for_atoms(soluteatoms)[0]
            print("Changing origin to centroid")
            fragment.coords = change_origin_to_centroid(fragment.coords, subsetcoords=solutecoords)
            fragment.write_xyzfile(xyzfilename="fragment-after-centering.xyz")

        # Now adding center force acting on solute
        if add_center_force is True:
            print("add_center_force is True")
            print("Forceconstant is: {} kcal/mol/Ang^2".format(centerforce_constant))
            if center_force_atoms is None:
                print("center_force_atoms unset. Using QM/MM atoms :", QM_MM_object.qmatoms)
                center_force_atoms = QM_MM_object.qmatoms
            # Get geometric center of system (Angstrom)
            center = fragment.get_coordinate_center()
            print("center:", center)

            openmmobject.add_center_force(center_coords=center, atomindices=center_force_atoms,
                                          forceconstant=centerforce_constant)

        # Setting coordinates of OpenMM object from current fragment.coords
        openmmobject.set_positions(fragment.coords)

        # After adding QM/MM force, possible Plumed force, possible center force
        # Let's list all OpenMM object system forces
        print("OpenMM Forces defined:", openmmobject.system.getForces())
        print("Now starting QM/MM MD simulation")
        # Does step by step
        # Delete old traj
        try:
            os.remove("OpenMMMD_traj.xyz")
        # Crashes when permissions not present or file is folder. Should never occur.
        except FileNotFoundError:
            pass
        # Simulation loop
        for step in range(simulation_steps):
            checkpoint_begin_step = time.time()
            print("Step:", step)
            # Get current coordinates to use for QM/MM step
            current_coords = np.array(openmmobject.simulation.context.getState(getPositions=True,
                                                                               enforcePeriodicBox=enforcePeriodicBox).getPositions(
                asNumpy=True)) * 10
            # state =  openmmobject.simulation.context.getState(getPositions=True, enforcePeriodicBox=enforcePeriodicBox)
            # current_coords = np.array(state.getPositions(asNumpy=True))*10
            # Manual trajectory option (reporters do not work for manual dynamics steps)
            if step % traj_frequency == 0:
                write_xyzfile(fragment.elems, current_coords, "OpenMMMD_traj", printlevel=1, writemode='a')
            # Run QM/MM step to get full system QM+PC gradient.
            # Updates OpenMM object with QM-PC forces
            checkpoint = time.time()
            QM_MM_object.run(current_coords=current_coords, elems=fragment.elems, Grad=True,
                             exit_after_customexternalforce_update=True)
            print_time_rel(checkpoint, modulename="QM/MM run", moduleindex=2)
            # NOTE: Think about energy correction (currently skipped above)
            # Now take OpenMM step (E+G + displacement etc.)
            checkpoint = time.time()
            openmmobject.simulation.step(1)
            print_time_rel(checkpoint, modulename="openmmobject sim step", moduleindex=2)
            print_time_rel(checkpoint_begin_step, modulename="Total sim step", moduleindex=2)
            # NOTE: Better to use OpenMM-plumed interface instead??
            # After MM step, grab coordinates and forces
            if plumed_object is not None:
                print("Plumed active. Untested. Hopefully works")
                current_coords = np.array(
                    openmmobject.simulation.context.getState(getPositions=True).getPositions(asNumpy=True))  # in nm
                current_forces = np.array(
                    openmmobject.simulation.context.getState(getForces=True).getForces(asNumpy=True))  # in kJ/mol /nm
                energy, newforces = plumed_object.run(coords=current_coords, forces=current_forces,
                                                      step=step)  # Plumed object needs to be configured for OpenMM
                openmmobject.update_custom_external_force(plumedcustomforce, newforces)

    # Dummy MM step to check how MD is affected by additional steps
    # TODO: DELETE
    elif dummy_MM is True:
        # Testing whether any cost-penalty by running step-by-step
        print("Dummy MM option")
        # Adding a custom external force
        openmmobject.add_custom_external_force()

        # Setting coordinates
        openmmobject.set_positions(fragment.coords)
        current_coords = fragment.coords

        dummygradient = np.random.random((fragment.numatoms, 3)) * 0.001
        for i in range(simulation_steps):
            # Get current coordinates to use for QM/MM step
            current_coords = np.array(
                openmmobject.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)) * 10
            # Custom force update (slows things down a bit)
            openmmobject.update_custom_external_force(dummygradient)
            # Now take OpenMM step
            checkpoint = time.time()
            openmmobject.simulation.step(1)
            print_time_rel(checkpoint, modulename="openmmobject sim step", moduleindex=2)
    else:
        print("Regular classical OpenMM MD option chosen")
        # Setting coordinates
        openmmobject.set_positions(fragment.coords)
        # Running all steps in one go
        # TODO: If we wanted to support plumed then we would have to do step 1-by-1 here
        openmmobject.simulation.step(simulation_steps)
    print("OpenMM MD simulation finished!")
    # Close Plumed also if active. Flushes HILLS/COLVAR etc.
    if plumed_object is not None:
        plumed_object.close()

    # enforcePeriodicBox=True
    state = openmmobject.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
    print("Checking PBC vectors:")
    a, b, c = state.getPeriodicBoxVectors()
    print(f"A: ", a)
    print(f"B: ", b)
    print(f"C: ", c)

    # Set new PBC vectors since they may have changed
    print("Updating PBC vectors")
    #Context. Used?
    openmmobject.simulation.context.setPeriodicBoxVectors(*[a,b,c])
    #System. Necessary
    openmmobject.system.setDefaultPeriodicBoxVectors(*[a,b,c])
    
    # Writing final frame to disk as PDB
    with open('final_MDfrag_laststep.pdb', 'w') as f:
        openmmobject.openmm.app.pdbfile.PDBFile.writeHeader(openmmobject.topology, f)
    with open('final_MDfrag_laststep.pdb', 'a') as f:
        openmmobject.openmm.app.pdbfile.PDBFile.writeModel(openmmobject.topology,
                                                           state.getPositions(asNumpy=True).value_in_unit(
                                                               openmmobject.unit.angstrom), f)
    # Updating ASH fragment
    newcoords = state.getPositions(asNumpy=True).value_in_unit(openmmobject.unit.angstrom)
    print("Updating coordinates in ASH fragment")
    fragment.coords = newcoords

    # Remove frozen atom constraints in the end
    print("Removing frozen atoms from OpenMM object")
    openmmobject.unfreeze_atoms()
    
    #Remove user-applied constraints to avoid double-enforcing later
    print("Removing constraints")
    openmmobject.remove_constraints(constraints)
    
    print_time_rel(module_init_time, modulename="OpenMM_MD", moduleindex=1)


def OpenMM_Opt(fragment=None, theory=None, maxiter=1000, tolerance=1, frozen_atoms=None, constraints=None,
               restraints=None, trajectory_file_option='PDB', traj_frequency=1, enforcePeriodicBox=True):
    module_init_time = time.time()
    print_line_with_mainheader("OpenMM Optimization")
    if frozen_atoms is None:
        frozen_atoms = []
    if constraints is None:
        constraints = []
    if restraints is None:
        restraints = []

    if fragment is None:
        print("No fragment object. Exiting")
        exit()

    # Distinguish between OpenMM theory or QM/MM theory
    if theory.__class__.__name__ == "OpenMMTheory":
        openmmobject = theory
    else:
        print("Only OpenMMTheory allowed in OpenMM_Opt. Exiting")
        exit()

    print("Max iterations:", maxiter)
    print("Energy tolerance:", tolerance)
    print("Number of frozen atoms:", len(frozen_atoms))
    if len(frozen_atoms) < 50:
        print("Frozen atoms", frozen_atoms)
    print("OpenMM autoconstraints:", openmmobject.autoconstraints)
    print("OpenMM hydrogenmass:", openmmobject.hydrogenmass)
    print("OpenMM rigidwater constraints:", openmmobject.rigidwater)
    print("Constraints:", constraints)
    print("Restraints:", restraints)
    print("")

    if openmmobject.autoconstraints is None:
        print(BC.WARNING, "Warning: Autoconstraints have not been set in OpenMMTheory object definition.")
        print("This means that by default no bonds are constrained in the optimization.", BC.END)
        print("Will continue...")
    if openmmobject.rigidwater is True and len(frozen_atoms) != 0 or (
            openmmobject.autoconstraints is not None and len(frozen_atoms) != 0):
        print(
            "Warning: Frozen_atoms options selected but there are general constraints defined in the OpenMM object "
            "(either rigidwater=True or autoconstraints is not None")
        print("OpenMM will crash if constraints and frozen atoms involve the same atoms")
    # createSystem(constraints=None), createSystem(constraints=HBonds), createSystem(constraints=All-Bonds), createSystem(constraints=HAngles)
    # HBonds constraints: timestep can be 2fs with Verlet and 4fs with Langevin
    # HAngles constraints: even larger timesteps
    # HAngles constraints: even larger timesteps

    # Freezing atoms in OpenMM object by setting particles masses to zero. Needs to be done before simulation creation

    if len(frozen_atoms) > 0:
        print("Freezing atoms")
        openmmobject.freeze_atoms(frozen_atoms=frozen_atoms)
    # Adding constraints/restraints between atoms
    if len(constraints) > 0:
        print("Constraints defined:")
        print(
            "Before adding constraints, system contains {} constraints".format(openmmobject.system.getNumConstraints()))
        # constraints is a list of lists defining bond constraints: constraints = [[700,701], [802,803,1.04]]
        # Cleaning up constraint list. Adding distance if missing
        constraints = clean_up_constraints_list(fragment=fragment, constraints=constraints)
        # print("Will enforce constrain definitions during Opt:", constraints)
        openmmobject.add_bondconstraints(constraints=constraints)
        print(
            "After adding constraints, system contains {} constraints".format(openmmobject.system.getNumConstraints()))
    if len(restraints) > 0:
        print("Restraints defined:")
        # restraints is a list of lists defining bond restraints: constraints = [[atom_i,atom_j, d, k ]]
        # Example: [[700,701, 1.05, 5.0 ]] Unit is Angstrom and kcal/mol * Angstrom^-2
        openmmobject.add_bondrestraints(restraints=restraints)

    openmmobject.create_simulation(timestep=0.001, temperature=1, integrator='VerletIntegrator')
    print("Simulation created.")

    # Context: settings positions
    print("Now adding coordinates")
    openmmobject.set_positions(fragment.coords)
    # coords=np.array(fragment.coords)
    # pos = [openmmobject.Vec3(coords[i, 0] / 10, coords[i, 1] / 10, coords[i, 2] / 10) for i in range(len(coords))] * openmmobject.openmm.unit.nanometer
    # openmmobject.simulation.context.setPositions(pos)

    print("")
    state = openmmobject.simulation.context.getState(getEnergy=True, getForces=True,
                                                     enforcePeriodicBox=enforcePeriodicBox)
    print("Initial potential energy is: {} Eh".format(
        state.getPotentialEnergy().value_in_unit_system(openmmobject.unit.md_unit_system) / constants.hartokj))
    kjmolnm_to_atomic_factor = -49614.752589207
    forces_init = np.array(state.getForces(asNumpy=True)) / kjmolnm_to_atomic_factor
    rms_force = np.sqrt(sum(n * n for n in forces_init.flatten()) / len(forces_init.flatten()))
    print("RMS force: {} Eh/Bohr".format(rms_force))
    print("Max force component: {} Eh/Bohr".format(forces_init.max()))
    print("")
    print("Starting minimization")

    openmmobject.simulation.minimizeEnergy(maxIterations=maxiter, tolerance=tolerance)
    print("Minimization done")
    print("")
    state = openmmobject.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True,
                                                     enforcePeriodicBox=enforcePeriodicBox)
    print("Potential energy is: {} Eh".format(
        state.getPotentialEnergy().value_in_unit_system(openmmobject.unit.md_unit_system) / constants.hartokj))
    forces_final = np.array(state.getForces(asNumpy=True)) / kjmolnm_to_atomic_factor
    rms_force = np.sqrt(sum(n * n for n in forces_final.flatten()) / len(forces_final.flatten()))
    print("RMS force: {} Eh/Bohr".format(rms_force))
    print("Max force component: {} Eh/Bohr".format(forces_final.max()))

    # Get coordinates
    newcoords = state.getPositions(asNumpy=True).value_in_unit(openmmobject.unit.angstrom)
    print("")
    print("Updating coordinates in ASH fragment")
    fragment.coords = newcoords

    with open('frag-minimized.pdb', 'w') as f:
        openmmobject.openmm.app.pdbfile.PDBFile.writeHeader(openmmobject.topology, f)
    with open('frag-minimized.pdb', 'a') as f:
        openmmobject.openmm.app.pdbfile.PDBFile.writeModel(openmmobject.topology,
                                                           openmmobject.simulation.context.getState(getPositions=True,
                                                                                                    enforcePeriodicBox=enforcePeriodicBox).getPositions(),
                                                           f)

    # Remove frozen atom constraints in the end
    print("Removing frozen atoms from OpenMM object")
    openmmobject.unfreeze_atoms()

    #Remove user-applied constraints to avoid double-enforcing later
    print("Removing constraints")
    openmmobject.remove_constraints(constraints)

    print('All Done!')
    print_time_rel(module_init_time, modulename="OpenMM_Opt", moduleindex=1)
    # Now write a serialized state that has coordinates
    # print('Finished. Writing serialized XML restart file...')
    # with open('job.min.xml', 'w') as f:
    #    f.write(
    #            openmmobject.openmm.XmlSerializer.serialize(
    #                openmmobject.simulation.context.getState(getPositions=True, getVelocities=True,
    #                                    getForces=True, getEnergy=True,
    #                                    enforcePeriodicBox=True)
    #            )
    #    )

    # print('Loading the XML file and calculating energy')
    # openmmobject.simulation.context.setState(
    #        openmmobject.openmm.XmlSerializer.deserialize(open('job.min.xml').read())
    # )
    # state = openmmobject.simulation.context.getState(getEnergy=True)
    # print('After minimization. Potential energy is %.5f' %
    #        (state.getPotentialEnergy().value_in_unit_system(openmmobject.unit.md_unit_system))
    # )


def OpenMM_Modeller(pdbfile=None, forcefield=None, xmlfile=None, waterxmlfile=None, watermodel=None, pH=7.0,
                    solvent_padding=10.0, solvent_boxdims=None, extraxmlfile=None, residue_variants=None,
                    ionicstrength=0.1, iontype='K+'):
    module_init_time = time.time()
    print_line_with_mainheader("OpenMM Modeller")
    try:
        import openmm as openmm
        import openmm.app as openmm_app
        import openmm.unit as openmm_unit
        print("Imported OpenMM library version:", openmm.__version__)

    except ImportError:
        raise ImportError(
            "OpenMM requires installing the OpenMM package. Try: conda install -c conda-forge openmm  \
            Also see http://docs.openmm.org/latest/userguide/application.html")
    try:
        import pdbfixer
    except ImportError:
        print("Problem importing pdbfixer. Install first via conda:")
        print("conda install -c conda-forge pdbfixer")
        exit()

    def write_pdbfile_openMM(topology, positions, filename):
        openmm.app.PDBFile.writeFile(topology, positions, file=open(filename, 'w'))
        print("Wrote PDB-file:", filename)

    def print_systemsize():
        print("System size: {} atoms\n".format(len(modeller.getPositions())))

    # https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions#template

    # Water model. May be overridden by forcefield below
    if watermodel == "tip3p":
        # Possible Problem: this only has water, no ions.
        waterxmlfile = "tip3p.xml"
    elif waterxmlfile is not None:
        # Problem: we need to define watermodel also
        print("Using waterxmlfile:", waterxmlfile)
    # Forcefield options
    if forcefield is not None:
        if forcefield == 'Amber99':
            xmlfile = "amber99sb.xml"
        elif forcefield == 'Amber96':
            xmlfile = "amber96.xml"
        elif forcefield == 'Amber03':
            xmlfile = "amber03.xml"
        elif forcefield == 'Amber10':
            xmlfile = "amber10.xml"
        elif forcefield == 'Amber14':
            xmlfile = "amber14-all.xml"
            # Using specific Amber FB version of TIP3P
            if watermodel == "tip3p":
                waterxmlfile = "amber14/tip3pfb.xml"
        elif forcefield == 'Amber96':
            xmlfile = "amber96.xml"
        elif forcefield == 'CHARMM36':
            xmlfile = "charmm36.xml"
            # Using specific CHARMM36 version of TIP3P
            waterxmlfile = "charmm36/water.xml"
        elif forcefield == 'CHARMM2013':
            xmlfile = "charmm_polar_2013.xml"
        elif forcefield == 'Amoeba2013':
            xmlfile = "amoeba2013.xml"
        elif forcefield == 'Amoeba2009':
            xmlfile = "amoeba2009.xml"
    elif xmlfile is not None:
        print("Using xmlfile:", xmlfile)
    else:
        print("You must provide a forcefield or xmlfile keyword!")
        exit()

    print("Forcefield:", forcefield)
    print("XMfile:", xmlfile)
    print("Water model:", watermodel)
    print("Xmlfile:", waterxmlfile)
    print("pH:", pH)

    print("User-provided dictionary of residue_variants:", residue_variants)
    # Define a forcefield
    if extraxmlfile is None:
        forcefield = openmm_app.forcefield.ForceField([xmlfile, waterxmlfile])
    else:
        print("Using extra XML file:", extraxmlfile)
        forcefield = openmm_app.forcefield.ForceField([xmlfile, waterxmlfile, extraxmlfile])

    # Fix basic mistakes in PDB by PDBFixer
    # This will e.g. fix bad terminii
    print("Running PDBFixer")
    fixer = pdbfixer.PDBFixer(pdbfile)
    fixer.findMissingResidues()
    print("Found missing residues:", fixer.missingResidues)
    fixer.findNonstandardResidues()
    print("Found non-standard residues:", fixer.nonstandardResidues)
    # fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    print("Found missing atoms:", fixer.missingAtoms)
    print("Found missing terminals:", fixer.missingTerminals)
    # print(fixer.__dict__)
    fixer.addMissingAtoms()
    print("Added missing atoms")
    # fixer.removeHeterogens(True)
    # print(fixer.__dict__)

    openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, open('system_afterfixes.pdb', 'w'))
    print("PDBFixer done")
    print("Wrote PDBfile: system_afterfixes.pdb")

    # Load fixed PDB-file and create Modeller object
    pdb = openmm_app.PDBFile("system_afterfixes.pdb")
    print("Loading Modeller")
    modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
    numresidues = modeller.topology.getNumResidues()
    print("Modeller topology has {} residues".format(numresidues))

    # User provided dictionary of special residues e.g residue_variants={0:'LYN', 17:'CYX', 18:'ASH', 19:'HIE' }
    # Now creating list of all residues [None,None,None...] with added changes
    residue_states = [None for i in range(0, numresidues)]
    if residue_variants is not None:
        for resid, newstate in residue_variants.items():
            residue_states[resid] = newstate

    # Adding hydrogens.
    # This is were missing residue/atom errors will come
    print("")
    print("Adding hydrogens for pH:", pH)
    modeller.addHydrogens(forcefield, pH=pH, variants=residue_states)
    # try:
    #    modeller.addHydrogens(forcefield, pH=pH, variants=residue_variants)
    # except ValueError as err:
    #    traceback.print_tb(err.__traceback__)
    #    print("")
    #    print(BC.FAIL,"ASH: OpenMM exited with ValueError. Probably means that you are missing forcefield terms or you provided wrong residue info.")
    #    print("Please provide an extraxmlfile argument to OpenMM_Modeller or fix the residue info", BC.END)
    #    exit()
    # except KeyError as err:
    #    traceback.print_tb(err.__traceback__)
    #    print("ASH: Some key error")
    #    exit()
    # except Exception as err:
    #    traceback.print_tb(err.__traceback__)
    #    print("ASH: OpenMM failed with some error. Read the error message above carefully.")
    #    exit()

    write_pdbfile_openMM(modeller.topology, modeller.positions, "system_afterH.pdb")
    print_systemsize()

    # Solvent
    print("Adding solvent, watermodel:", watermodel)
    if solvent_boxdims is not None:
        print("Solvent boxdimension provided: {} Å".format(solvent_boxdims))
        modeller.addSolvent(forcefield, boxSize=openmm.Vec3(solvent_boxdims[0], solvent_boxdims[1],
                                                            solvent_boxdims[2]) * openmm_unit.angstrom)
    else:
        print("Using solvent padding (solvent_padding=X keyword): {} Å".format(solvent_padding))
        modeller.addSolvent(forcefield, padding=solvent_padding * openmm_unit.angstrom, model=watermodel)
    write_pdbfile_openMM(modeller.topology, modeller.positions, "system_aftersolvent.pdb")
    print_systemsize()

    # Ions
    print("Adding ionic strength: {} using ions {}".format(ionicstrength, iontype))
    modeller.addSolvent(forcefield, ionicStrength=ionicstrength * openmm_unit.molar, positiveIon=iontype)
    write_pdbfile_openMM(modeller.topology, modeller.positions, "system_afterions.pdb")
    print_systemsize()

    # Create ASH fragment
    fragment = Fragment(pdbfile="system_afterions.pdb")
    # Write to disk
    fragment.print_system(filename="fragment.ygg")
    fragment.write_xyzfile(xyzfilename="fragment.xyz")
    print_time_rel(module_init_time, modulename="OpenMM_Modeller", moduleindex=1)
    # Return forcefield object,  topology object and ASH fragment
    return forcefield, modeller.topology, fragment


def MDtraj_import_():
    print("Importing mdtraj (https://www.mdtraj.org)")
    try:
        import mdtraj
    except ImportError:
        print("Problem importing mdtraj. Try: pip install mdtraj or conda install -c conda-forge mdtraj")
        exit()
    return mdtraj


# anchor_molecules. Use if automatic guess fails
def MDtraj_imagetraj(trajectory, pdbtopology, format='DCD', unitcell_lengths=None, unitcell_angles=None,
                     solute_anchor=None):
    traj_basename = os.path.splitext(trajectory)[0]
    mdtraj = MDtraj_import_()

    # Load traj
    print("Loading trajecory using mdtraj")
    traj = mdtraj.load(trajectory, top=pdbtopology)
    numframes = len(traj._time)
    print("Found {} frames in trajectory".format(numframes))
    print("PBC information in trajectory:")
    print("traj.unitcell_lengths:", traj.unitcell_lengths)
    print("traj.unitcell_angles", traj.unitcell_angles)
    # If PBC information is missing from traj file (OpenMM: Charmmfiles, Amberfiles option etc) then provide this info
    if unitcell_lengths is not None:
        print("unitcell_lengths info provided by user.")
        unitcell_lengths_nm = [i / 10 for i in unitcell_lengths]
        traj.unitcell_lengths = np.array(unitcell_lengths_nm * numframes).reshape(numframes, 3)
        traj.unitcell_angles = np.array(unitcell_angles * numframes).reshape(numframes, 3)
    # else:
    #    print("Missing PBC info. This can be provided by unitcell_lengths and unitcell_angles keywords")

    # Manual anchor if needed
    # NOTE: not sure how well this works but it's something
    if solute_anchor is True:
        anchors = [set(traj.topology.residue(0).atoms)]
        print("anchors:", anchors)
        # Re-imaging trajectory
        imaged = traj.image_molecules(anchor_molecules=anchors)
    else:
        imaged = traj.image_molecules()

    # Save trajectory in format
    if format == 'DCD':
        imaged.save(traj_basename + '_imaged.dcd')
        print("Saved reimaged trajectory:", traj_basename + '_imaged.dcd')
    elif format == 'PDB':
        imaged.save(traj_basename + '_imaged.pdb')
        print("Saved reimaged trajectory:", traj_basename + '_imaged.pdb')
    else:
        print("Unknown traj format")

    # traj.save('file.h5')
    # traj.save('file.nc')
    # traj.save('file.xyz')
    # traj.save('file.pdb')


def MDAnalysis_transform(topfile, trajfile, solute_indices=None, trajoutputformat='PDB', trajname="MDAnalysis_traj"):
    # Load traj
    print("MDAnalysis interface: transform")

    try:
        import MDAnalysis as mda
        import MDAnalysis.transformations as trans
    except ImportError:
        print("Problem importing MDAnalysis library.")
        print("Install via: pip install mdtraj")
        exit()

    print("Loading trajecory using MDAnalysis")
    print("Topology file:", topfile)
    print("Trajectory file:", trajfile)
    print("Solute_indices:", solute_indices)
    print("Trajectory output format", trajoutputformat)
    print("Will unwrap solute and center in box")
    print("Will then wrap full system")

    # Load trajectory
    u = mda.Universe(topfile, trajfile, in_memory=True)
    print(u.trajectory.ts, u.trajectory.time)

    # Grab solute
    numatoms = len(u.atoms)
    solutenum = len(solute_indices)
    solute = u.atoms[:solutenum]
    solvent = u.atoms[solutenum:numatoms]
    fullsystem = u.atoms[:numatoms]
    elems_list = list(fullsystem.types)
    # Guess bonds. Could also read in vdW radii. Could also read in connectivity from ASH if this fails
    solute.guess_bonds()
    # Unwrap solute, center solute and wraps full system (or solvent)
    workflow = (trans.unwrap(solute),
                trans.center_in_box(solute, center='mass'),
                trans.wrap(fullsystem, compound='residues'))

    u.trajectory.add_transformations(*workflow)
    if trajoutputformat == 'PDB':
        fullsystem.write(trajname + ".pdb", frames='all')

    # TODO: Distinguish between transforming whole trajectory vs. single geometry
    # Maybe just read in single-frame trajectory so that things are general
    # Returning last frame. To be used in ASH workflow
    lastframe = u.trajectory[-1]

    return elems_list, lastframe._pos


# Assumes all atoms present (including hydrogens)
def solvate_small_molecule(fragment=None, charge=None, mult=None, watermodel=None, solvent_boxdims=[70.0, 70.0, 70.0],
                           nonbonded_pars="CM5_UFF", orcatheory=None, numcores=1):
    # , ionicstrength=0.1, iontype='K+'
    print_line_with_mainheader("SmallMolecule Solvator")
    try:
        import openmm as openmm
        import openmm.app as openmm_app
        import openmm.unit as openmm_unit
        from openmm import XmlSerializer
        print("Imported OpenMM library version:", openmm.__version__)

    except ImportError:
        raise ImportError(
            "OpenMM requires installing the OpenMM package. Try: conda install -c conda-forge openmm  \
            Also see http://docs.openmm.org/latest/userguide/application.html")

    def write_pdbfile_openMM(topology, positions, filename):
        openmm.app.PDBFile.writeFile(topology, positions, file=open(filename, 'w'))
        print("Wrote PDB-file:", filename)

    def print_systemsize():
        print("System size: {} atoms\n".format(len(modeller.getPositions())))

    # Defining simple atomnames and atomtypes to be used for solute
    atomnames = [el + "Y" + str(i) for i, el in enumerate(fragment.elems)]
    atomtypes = [el + "X" + str(i) for i, el in enumerate(fragment.elems)]

    # Take input ASH fragment and write a basic PDB file via ASH
    write_pdbfile(fragment, outputname="smallmol", dummyname='LIG', atomnames=atomnames)

    # Load PDB-file and create Modeller object
    pdb = openmm_app.PDBFile("smallmol.pdb")
    print("Loading Modeller")
    modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
    numresidues = modeller.topology.getNumResidues()
    print("Modeller topology has {} residues".format(numresidues))

    # Forcefield

    # TODO: generalize to other solvents.
    # Create local ASH library of XML files
    if watermodel == "tip3p":
        print("Using watermodel=TIP3P . Using parameters in:", ashpath + "/databases/forcefields")
        forcefieldpath = ashpath + "/databases/forcefields"
        waterxmlfile = forcefieldpath + "/tip3p_water_ions.xml"
        coulomb14scale = 1.0
        lj14scale = 1.0
    elif watermodel == "charmm_tip3p":
        coulomb14scale = 1.0
        lj14scale = 1.0
        # NOTE: Problem combining this and solute XML file.
        print("Using watermodel: CHARMM-TIP3P (has ion parameters also)")
        # This is the modified CHARMM-TIP3P (LJ parameters on H at least, maybe bonded parameters defined also)
        # Advantage: also contains ion parameters
        waterxmlfile = "charmm36/water.xml"
    else:
        print("unknown watermodel");
        exit()

    # Define nonbonded paramers
    if nonbonded_pars == "CM5_UFF":
        print("Using CM5 atomcharges and UFF-LJ parameters.")
        atompropdict = basic_atom_charges_ORCA(fragment=fragment, charge=charge, mult=mult,
                                               orcatheory=orcatheory, chargemodel="CM5", numcores=numcores)
        charges = atompropdict['charges']
        # Basic UFF LJ parameters
        # Converting r0 parameters from Ang to nm and to sigma
        sigmas = [UFF_modH_dict[el][0] * 0.1 / (2 ** (1 / 6)) for el in fragment.elems]
        # Convering epsilon from kcal/mol to kJ/mol
        epsilons = [UFF_modH_dict[el][1] * 4.184 for el in fragment.elems]
    elif nonbonded_pars == "DDEC3" or nonbonded_pars == "DDEC6":
        print("Using {} atomcharges and DDEC-derived parameters.".format(nonbonded_pars))
        atompropdict = basic_atom_charges_ORCA(fragment=fragment, charge=charge, mult=mult,
                                               orcatheory=orcatheory, chargemodel=nonbonded_pars, numcores=numcores)
        charges = atompropdict['charges']
        r0 = atompropdict['r0s']
        eps = atompropdict['epsilons']
        sigmas = [s * 0.1 / (2 ** (1 / 6)) for s in r0]
        epsilons = [e * 4.184 for e in eps]
    elif nonbonded_pars == "xtb_UFF":
        print("Using xTB charges and UFF-LJ parameters")
        charges = basic_atomcharges_xTB(fragment=fragment, charge=charge, mult=mult, xtbmethod='GFN2')
        # Basic UFF LJ parameters
        # Converting r0 parameters from Ang to nm and to sigma
        sigmas = [UFF_modH_dict[el][0] * 0.1 / (2 ** (1 / 6)) for el in fragment.elems]
        # Convering epsilon from kcal/mol to kJ/mol
        epsilons = [UFF_modH_dict[el][1] * 4.184 for el in fragment.elems]
    else:
        print("unknown nonbonded_pars option")
        exit()

    print("sigmas:", sigmas)
    print("epsilons:", epsilons)

    # Creating XML-file for solute

    xmlfile = write_xmlfile_nonbonded(resnames=["LIG"], atomnames_per_res=[atomnames], atomtypes_per_res=[atomtypes],
                                      elements_per_res=[fragment.elems], masses_per_res=[fragment.masses],
                                      charges_per_res=[charges],
                                      sigmas_per_res=[sigmas], epsilons_per_res=[epsilons], filename="solute.xml",
                                      coulomb14scale=coulomb14scale, lj14scale=lj14scale)

    print("Creating forcefield using XML-files:", xmlfile, waterxmlfile)
    forcefield = openmm_app.forcefield.ForceField([xmlfile, waterxmlfile])

    # , waterxmlfile
    # if extraxmlfile == None:
    #    print("here")
    #    forcefield=openmm_app.forcefield.ForceField(xmlfile, waterxmlfile)
    # else:
    #    print("Using extra XML file:", extraxmlfile)
    #    forcefield=openmm_app.forcefield.ForceField(xmlfile, waterxmlfile, extraxmlfile)

    # Solvent+Ions
    print("Adding solvent, watermodel:", watermodel)
    # NOTE: modeller.addsolvent will automatically add ions to neutralize any excess charge
    # TODO: Replace with something simpler
    if solvent_boxdims is not None:
        print("Solvent boxdimension provided: {} Å".format(solvent_boxdims))
        modeller.addSolvent(forcefield, boxSize=openmm.Vec3(solvent_boxdims[0], solvent_boxdims[1],
                                                            solvent_boxdims[2]) * openmm_unit.angstrom)

    # Write out solvated system coordinates
    write_pdbfile_openMM(modeller.topology, modeller.positions, "system_aftersolvent.pdb")
    print_systemsize()
    # Create ASH fragment and write to disk
    newfragment = Fragment(pdbfile="system_aftersolvent.pdb")
    newfragment.print_system(filename="newfragment.ygg")
    newfragment.write_xyzfile(xyzfilename="newfragment.xyz")

    # Return forcefield object,  topology object and ASH fragment
    return forcefield, modeller.topology, newfragment


# Simple XML-writing function. Will only write nonbonded parameters
def write_xmlfile_nonbonded(resnames=None, atomnames_per_res=None, atomtypes_per_res=None, elements_per_res=None,
                            masses_per_res=None, charges_per_res=None, sigmas_per_res=None,
                            epsilons_per_res=None, filename="system.xml", coulomb14scale=0.833333, lj14scale=0.5):
    print("Inside write_xml file")
    # resnames=["MOL1", "MOL2"]
    # atomnames_per_res=[["CM1","CM2","HX1","HX2"],["OT1","HT1","HT2"]]
    # atomtypes_per_res=[["CM","CM","H","H"],["OT","HT","HT"]]
    # sigmas_per_res=[[1.2,1.2,1.3,1.3],[1.25,1.17,1.17]]
    # epsilons_per_res=[[0.2,0.2,0.3,0.3],[0.25,0.17,0.17]]
    # etc.
    # Always list of lists now

    assert len(resnames) == len(atomnames_per_res) == len(atomtypes_per_res)
    # Get list of all unique atomtypes, elements, masses
    # all_atomtypes=list(set([item for sublist in atomtypes_per_res for item in sublist]))
    # all_elements=list(set([item for sublist in elements_per_res for item in sublist]))
    # all_masses=list(set([item for sublist in masses_per_res for item in sublist]))

    # Create list of all AtomTypelines (unique)
    atomtypelines = []
    for resname, atomtypelist, elemlist, masslist in zip(resnames, atomtypes_per_res, elements_per_res, masses_per_res):
        for atype, elem, mass in zip(atomtypelist, elemlist, masslist):
            atomtypeline = "<Type name=\"{}\" class=\"{}\" element=\"{}\" mass=\"{}\"/>\n".format(atype, atype, elem,
                                                                                                  str(mass))
            if atomtypeline not in atomtypelines:
                atomtypelines.append(atomtypeline)
    # Create list of all nonbonded lines (unique)
    nonbondedlines = []
    for resname, atomtypelist, chargelist, sigmalist, epsilonlist in zip(resnames, atomtypes_per_res, charges_per_res,
                                                                         sigmas_per_res, epsilons_per_res):
        for atype, charge, sigma, epsilon in zip(atomtypelist, chargelist, sigmalist, epsilonlist):
            nonbondedline = "<Atom type=\"{}\" charge=\"{}\" sigma=\"{}\" epsilon=\"{}\"/>\n".format(atype, charge,
                                                                                                     sigma, epsilon)
            if nonbondedline not in nonbondedlines:
                nonbondedlines.append(nonbondedline)

    with open(filename, 'w') as xmlfile:
        xmlfile.write("<ForceField>\n")
        xmlfile.write("<AtomTypes>\n")
        for atomtypeline in atomtypelines:
            xmlfile.write(atomtypeline)
        xmlfile.write("</AtomTypes>\n")
        xmlfile.write("<Residues>\n")
        for resname, atomnamelist, atomtypelist in zip(resnames, atomnames_per_res, atomtypes_per_res):
            xmlfile.write("<Residue name=\"{}\">\n".format(resname))
            for i, (atomname, atomtype) in enumerate(zip(atomnamelist, atomtypelist)):
                xmlfile.write("<Atom name=\"{}\" type=\"{}\"/>\n".format(atomname, atomtype))
            # All other atoms
            xmlfile.write("</Residue>\n")
        xmlfile.write("</Residues>\n")

        xmlfile.write("<NonbondedForce coulomb14scale=\"{}\" lj14scale=\"{}\">\n".format(coulomb14scale, lj14scale))
        for nonbondedline in nonbondedlines:
            xmlfile.write(nonbondedline)
        xmlfile.write("</NonbondedForce>\n")
        xmlfile.write("</ForceField>\n")
    print("Wrote XML-file:", filename)
    return filename


# TODO: Move elsewhere?
def basic_atomcharges_xTB(fragment=None, charge=None, mult=None, xtbmethod='GFN2'):
    print("Now calculating atom charges for fragment")
    print("Using default xTB charges")
    calc = xTBTheory(fragment=fragment, charge=charge, runmode='inputfile',
                     mult=mult, xtbmethod=xtbmethod)

    Singlepoint(theory=calc, fragment=fragment)
    atomcharges = grabatomcharges_xTB()
    print("atomcharges:", atomcharges)
    print("fragment elems:", fragment.elems)
    return atomcharges


# TODO: Move elsewhere?
def basic_atom_charges_ORCA(fragment=None, charge=None, mult=None, orcatheory=None, chargemodel=None, numcores=1):
    atompropdict = {}
    print("Will calculate charges using ORCA")

    # Define default ORCA object if notprovided
    if orcatheory is None:
        print("orcatheory not provided. Will do r2SCAN/def2-TZVP single-point calculation")
        orcasimpleinput = "! r2SCAN def2-TZVP tightscf "
        orcablocks = "%scf maxiter 300 end"
        orcatheory = ORCATheory(fragment=fragment, charge=charge, mult=mult, orcasimpleinput=orcasimpleinput,
                                orcablocks=orcablocks, numcores=numcores)
    if chargemodel == 'CM5':
        orcatheory.extraline = chargemodel_select(chargemodel)
    # Run ORCA calculation
    Singlepoint(theory=orcatheory, fragment=fragment)
    if 'DDEC' not in chargemodel:
        atomcharges = grabatomcharges_ORCA(chargemodel, orcatheory.filename + '.out')
        atompropdict['charges'] = atomcharges
    else:
        atomcharges, molmoms, voldict = DDEC_calc(elems=fragment.elems, theory=orcatheory,
                                                  gbwfile=orcatheory.filename + '.gbw', numcores=numcores,
                                                  DDECmodel='DDEC3', calcdir='DDEC', molecule_charge=charge,
                                                  molecule_spinmult=mult)
        atompropdict['charges'] = atomcharges
        r0list, epsilonlist = DDEC_to_LJparameters(fragment.elems, molmoms, voldict)
        print("r0list:", r0list)
        print("epsilonlist:", epsilonlist)
        atompropdict['r0s'] = r0list
        atompropdict['epsilons'] = epsilonlist

    print("atomcharges:", atomcharges)
    print("fragment elems:", fragment.elems)
    return atompropdict


def read_NPT_statefile(npt_output):
    import csv
    from collections import defaultdict
    # Read in CSV file of last NPT simulation and store in lists
    columns = defaultdict(list)

    with open(npt_output, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k, v) in row.items():
                columns[k].append(v)

    # Extract step number, volume and density and cast as floats
    steps = np.array(columns['#"Step"'])
    volume = np.array(columns["Box Volume (nm^3)"]).astype(float)
    # volume = volume[-1:-10:-1]
    density = np.array(columns["Density (g/mL)"]).astype(float)
    # density = density[-1:-10:-1]

    # Calculate standard deviations
    # volume_std = np.std(volume)
    # density_std = np.std(density)

    # resultdict = {"volume_std": volume_std, "density_std": density_std, "density": density[-1], "volume": volume[-1]}
    resultdict = {"steps": steps, "volume": volume, "density": density}
    return resultdict

#############################
#  Multi-step MD protocols  #
#############################


def OpenMM_box_relaxation(fragment=None, theory=None, datafilename="nptsim.csv", numsteps_per_NPT=10000,
                          volume_threshold=1.0, density_threshold=0.001, temperature=300, timestep=0.001,
                          traj_frequency=100, trajectory_file_option='DCD', coupling_frequency=1,
                          frozen_atoms=None, constraints=None, restraints=None):
    """OpenMM_box_relaxation: NPT simulations until volume and density stops changing

    Args:
        fragment ([type], optional): [description]. Defaults to None.
        theory ([type], optional): [description]. Defaults to None.
        datafilename (str, optional): [description]. Defaults to "nptsim.csv".
        numsteps_per_NPT (int, optional): [description]. Defaults to 2000.
        volume_threshold (float, optional): [description]. Defaults to 1.0.
        density_threshold (float, optional): [description]. Defaults to 0.001.
        temperature (int, optional): [description]. Defaults to 300.
        timestep (float, optional): [description]. Defaults to 0.001.
        traj_frequency (int, optional): [description]. Defaults to 100.
        trajectory_file_option (str, optional): [description]. Defaults to 'DCD'.
        coupling_frequency (int, optional): [description]. Defaults to 1.
    """

    print("Starting relaxation of periodic box size\n")

    if fragment is None or theory is None:
        print("Fragment and theory required")
        exit()

    # Starting parameters
    steps = 0
    volume_std = 10
    density_std = 1

    print("Density threshold:", density_threshold)
    print("Volume threshold:", volume_threshold)

    while volume_std >= volume_threshold and density_std >= density_threshold:
        OpenMM_MD(fragment=fragment, theory=theory, timestep=timestep, simulation_steps=numsteps_per_NPT,
                  traj_frequency=traj_frequency, temperature=temperature, integrator="LangevinMiddleIntegrator",
                  coupling_frequency=coupling_frequency, barostat='MonteCarloBarostat', datafilename=datafilename,
                  trajectory_file_option=trajectory_file_option, frozen_atoms=frozen_atoms, constraints=constraints,
                  restraints=restraints)
        steps += numsteps_per_NPT

        # Read reporter file and calculate stdev
        NPTresults = read_NPT_statefile(datafilename)
        volume = NPTresults["volume"]
        density = NPTresults["density"]
        volume_std = np.std(volume)
        density_std = np.std(density)

        print("{} steps taken in total. Volume : {} stdev: {}\tDensity: {} stdev: {}".format(steps,
                                                                                             volume[-1],
                                                                                             volume_std,
                                                                                             density[-1],
                                                                                             density_std))

    print("Relaxation of periodic box size finished!\n")








###########################
# CLASS-BASED OpenMM_MD
# Under development
###########################

class OpenMM_MDclass:
    def __init__(self,fragment=None, theory=None, timestep=0.001, 
              traj_frequency=1000, temperature=300, integrator=None,
              barostat=None, pressure=1, trajectory_file_option='PDB', coupling_frequency=None, anderson_thermostat=False,
              enforcePeriodicBox=True, frozen_atoms=None, constraints=None, restraints=None,
              datafilename=None, dummy_MM=False, plumed_object=None, add_center_force=False,
              center_force_atoms=None, centerforce_constant=1.0):
        module_init_time = time.time()

        print_line_with_mainheader("OpenMM MOLECULAR DYNAMICS INITIALIZATION")

        # Distinguish between OpenMM theory or QM/MM theory
        if theory.__class__.__name__ == "OpenMMTheory":
            self.openmmobject = theory
            self.QM_MM_object = None
        elif theory.__class__.__name__ == "QMMMTheory":
            self.QM_MM_object = theory
            self.openmmobject = theory.mm_theory
        else:
            print("Unknown theory. Exiting")
            exit()

        if frozen_atoms is None:
            self.frozen_atoms = []
        else:
            self.frozen_atoms=frozen_atoms
        if constraints is None:
            self.constraints = []
        else:
            self.constraints=constraints
        if restraints is None:
            self.restraints = []
        else:
            self.restraints=restraints

        if fragment is None:
            print("No fragment object. Exiting")
            exit()
        else:
            self.fragment=fragment
        #various
        self.temperature=temperature
        self.pressure=pressure
        self.integrator=integrator
        self.coupling_frequency=coupling_frequency
        self.timestep=timestep
        self.traj_frequency=traj_frequency
        self.plumed_object=plumed_object
        print("Temperature: {} K".format(self.temperature))
        print("Number of frozen atoms:", len(self.frozen_atoms))
        if len(self.frozen_atoms) < 50:
            print("Frozen atoms", self.frozen_atoms)
        print("OpenMM autoconstraints:", self.openmmobject.autoconstraints)
        print("OpenMM hydrogenmass:",
            self.openmmobject.hydrogenmass)  # Note 1.5 amu mass is recommended for LangevinMiddle with 4fs timestep
        print("OpenMM rigidwater constraints:", self.openmmobject.rigidwater)
        print("Constraints:", self.constraints)
        print("Restraints:", self.restraints)
        print("Integrator:", self.integrator)

        print("Anderon Thermostat:", anderson_thermostat)
        print("coupling_frequency: {} ps^-1 (for Nosé-Hoover and Langevin integrators)".format(self.coupling_frequency))
        print("Barostat:", barostat)

        print("")
        print("Will write trajectory in format:", trajectory_file_option)
        print("Trajectory write frequency:", self.traj_frequency)
        print("enforcePeriodicBox option:", enforcePeriodicBox)
        print("")

        if self.openmmobject.autoconstraints is None:
            print(BC.WARNING, "Warning: Autoconstraints have not been set in OpenMMTheory object definition.")
            print(
                "This means that by default no bonds are constrained in the MD simulation. This usually requires a small timestep: 0.5 fs or so")
            print("autoconstraints='HBonds' is recommended for 1-2 fs timesteps with Verlet (4fs with Langevin).")
            print("autoconstraints='AllBonds' or autoconstraints='HAngles' allows even larger timesteps to be used", BC.END)
            print("Will continue...")
        if self.openmmobject.rigidwater is True and len(self.frozen_atoms) != 0 or (
                self.openmmobject.autoconstraints is not None and len(self.frozen_atoms) != 0):
            print(
                "Warning: Frozen_atoms options selected but there are general constraints defined in the OpenMM object (either rigidwater=True or autoconstraints != None")
            print("OpenMM will crash if constraints and frozen atoms involve the same atoms")
        print("")

        # createSystem(constraints=None), createSystem(constraints=HBonds), createSystem(constraints=All-Bonds), createSystem(constraints=HAngles)
        # HBonds constraints: timestep can be 2fs with Verlet and 4fs with Langevin
        # HAngles constraints: even larger timesteps
        # HAngles constraints: even larger timesteps

        print("Before adding constraints, system contains {} constraints".format(self.openmmobject.system.getNumConstraints()))

        # Freezing atoms in OpenMM object by setting particles masses to zero. Needs to be done before simulation creation
        if len(self.frozen_atoms) > 0:
            self.openmmobject.freeze_atoms(frozen_atoms=self.frozen_atoms)

        # Adding constraints/restraints between atoms
        if len(self.constraints) > 0:
            print("Constraints defined.")
            # constraints is a list of lists defining bond constraints: constraints = [[700,701], [802,803,1.04]]
            # Cleaning up constraint list. Adding distance if missing
            self.constraints = clean_up_constraints_list(fragment=self.fragment, constraints=self.constraints)
            print("Will enforce constrain definitions during MD:", self.constraints)
            self.openmmobject.add_bondconstraints(constraints=self.constraints)
        if len(self.restraints) > 0:
            print("Restraints defined")
            # restraints is a list of lists defining bond restraints: constraints = [[atom_i,atom_j, d, k ]]    Example: [[700,701, 1.05, 5.0 ]] Unit is Angstrom and kcal/mol * Angstrom^-2
            self.openmmobject.add_bondrestraints(restraints=self.restraints)

        print("After adding constraints, system contains {} constraints".format(self.openmmobject.system.getNumConstraints()))

        forceclassnames=[i.__class__.__name__ for i in self.openmmobject.system.getForces()]
        # Set up system with chosen barostat, thermostat, integrator
        if barostat is not None:
            print("Attempting to add barostat")
            if "MonteCarloBarostat" not in forceclassnames:
                print("Adding barostat")
                self.openmmobject.system.addForce(self.openmmobject.openmm.MonteCarloBarostat(self.pressure * self.openmmobject.openmm.unit.bar,
                                                                                self.temperature * self.openmmobject.openmm.unit.kelvin))
            else:
                print("Barostat already present. Skipping")
            print("after barostat added")

            self.integrator = "LangevinMiddleIntegrator"
            print("Barostat requires using integrator:", integrator)
            self.openmmobject.create_simulation(timestep=self.timestep, temperature=self.temperature, integrator=self.integrator,
                                        coupling_frequency=self.coupling_frequency)
        elif anderson_thermostat is True:
            print("Anderson thermostat is on")
            if "AndersenThermostat" not in forceclassnames:
                self.openmmobject.system.addForce(
                    self.openmmobject.openmm.AndersenThermostat(self.temperature * self.openmmobject.openmm.unit.kelvin,
                                                        1 / self.openmmobject.openmm.unit.picosecond))
            self.integrator = "VerletIntegrator"
            print("Now using integrator:", integrator)
            self.openmmobject.create_simulation(timestep=self.timestep, temperature=self.temperature, integrator=self.integrator,
                                        coupling_frequency=coupling_frequency)
        else:
            #Deleting barostat and Andersen thermostat if present from previous sims
            for i,forcename in enumerate(forceclassnames):
                if forcename == "MonteCarloBarostat" or forcename == "AndersenThermostat":
                    print("Removing old force:", forcename)
                    self.openmmobject.system.removeForce(i)
            
            # Regular thermostat or integrator without barostat
            # Integrators: LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator,
            # BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
            self.openmmobject.create_simulation(timestep=self.timestep, temperature=self.temperature, integrator=self.integrator,
                                        coupling_frequency=self.coupling_frequency)
        print("Simulation created.")
        forceclassnames=[i.__class__.__name__ for i in self.openmmobject.system.getForces()]
        print("OpenMM System forces present:", forceclassnames)
        print("Checking Initial PBC vectors")
        self.state = self.openmmobject.simulation.context.getState()
        a, b, c = self.state.getPeriodicBoxVectors()
        print(f"A: ", a)
        print(f"B: ", b)
        print(f"C: ", c)

        # THIS DOES NOT APPLY TO QM/MM. MOVE ELSEWHERE??
        if trajectory_file_option == 'PDB':
            self.openmmobject.simulation.reporters.append(self.openmmobject.openmm.app.PDBReporter('output_traj.pdb', self.traj_frequency,
                                                                                        enforcePeriodicBox=enforcePeriodicBox))
        elif trajectory_file_option == 'DCD':
            # NOTE: Disabling for now
            # with open('initial_MDfrag_step1.pdb', 'w') as f: openmmobject.openmm.app.pdbfile.PDBFile
            # .writeModel(openmmobject.topology, openmmobject.simulation.context.getState(getPositions=True,
            # enforcePeriodicBox=enforcePeriodicBox).getPositions(), f)
            # print("Wrote PDB")
            self.openmmobject.simulation.reporters.append(openmmobject.openmm.app.DCDReporter('output_traj.dcd', self.traj_frequency,
                                                                                        enforcePeriodicBox=enforcePeriodicBox))
        elif trajectory_file_option == 'NetCDFReporter':
            print("NetCDFReporter traj format selected. This requires mdtraj. Importing.")
            mdtraj = MDtraj_import_()
            self.openmmobject.simulation.reporters.append(mdtraj.reporters.NetCDFReporter('output_traj.nc', self.traj_frequency))
        elif trajectory_file_option == 'HDF5Reporter':
            print("HDF5Reporter traj format selected. This requires mdtraj. Importing.")
            mdtraj = MDtraj_import_()
            self.openmmobject.simulation.reporters.append(
                mdtraj.reporters.HDF5Reporter('output_traj.lh5', self.traj_frequency, enforcePeriodicBox=enforcePeriodicBox))

        if barostat is not None:
            volume = density = True
        else:
            volume = density = False

        #If statedatareporter filename set:
        if datafilename != None:
            outputoption=datafilename
        #otherwise stdout:
        else:
            outputoption=stdout
        
        self.openmmobject.simulation.reporters.append(
            self.openmmobject.openmm.app.StateDataReporter(outputoption, self.traj_frequency, step=True, time=True,
                                                        potentialEnergy=True, kineticEnergy=True, volume=volume,
                                                        density=density, temperature=True, separator=','))

        # NOTE: Better to use OpenMM-plumed interface instead??
        if plumed_object is not None:
            print("Plumed active")
            # Create new OpenMM custom external force
            print("Creating new OpenMM custom external force for Plumed")
            plumedcustomforce = openmmobject.add_custom_external_force()

        # QM/MM MD
        if self.QM_MM_object is not None:
            print("QM_MM_object provided. Switching to QM/MM loop")
            print("QM/MM requires enforcePeriodicBox to be False")
            enforcePeriodicBox = False
            # enforcePeriodicBox or not
            print("enforcePeriodicBox:", enforcePeriodicBox)

            # OpenMM_MD with QM/MM object does not make sense without openmm_externalforce
            # (it would calculate OpenMM energy twice) so turning on in case forgotten
            if self.QM_MM_object.openmm_externalforce is False:
                print("QM/MM object was not set to have openmm_externalforce=True.")
                print("Turning on externalforce option")
                self.QM_MM_object.openmm_externalforce = True
                self.QM_MM_object.openmm_externalforceobject = self.QM_MM_object.mm_theory.add_custom_external_force()
            # TODO:
            # Should we set parallelization of QM theory here also in case forgotten?

            centercoordinates = False
            # CENTER COORDINATES HERE on SOLUTE HERE ??
            # TODO: Deprecated I think
            if centercoordinates is True:
                # Solute atoms assumed to be QM-region
                self.fragment.write_xyzfile(xyzfilename="fragment-before-centering.xyz")
                soluteatoms = self.QM_MM_object.qmatoms
                solutecoords = self.fragment.get_coords_for_atoms(soluteatoms)[0]
                print("Changing origin to centroid")
                self.fragment.coords = change_origin_to_centroid(fragment.coords, subsetcoords=solutecoords)
                self.fragment.write_xyzfile(xyzfilename="fragment-after-centering.xyz")

            # Now adding center force acting on solute
            if add_center_force is True:
                print("add_center_force is True")
                print("Forceconstant is: {} kcal/mol/Ang^2".format(centerforce_constant))
                if center_force_atoms is None:
                    print("center_force_atoms unset. Using QM/MM atoms :", self.QM_MM_object.qmatoms)
                    center_force_atoms = self.QM_MM_object.qmatoms
                # Get geometric center of system (Angstrom)
                center = self.fragment.get_coordinate_center()
                print("center:", center)

                self.openmmobject.add_center_force(center_coords=center, atomindices=center_force_atoms,
                                            forceconstant=centerforce_constant)

            # Setting coordinates of OpenMM object from current fragment.coords
            self.openmmobject.set_positions(self.fragment.coords)

            # After adding QM/MM force, possible Plumed force, possible center force
            # Let's list all OpenMM object system forces
            print("OpenMM Forces defined:", self.openmmobject.system.getForces())
            print("Now starting QM/MM MD simulation")
            # Does step by step
            # Delete old traj
            try:
                os.remove("OpenMMMD_traj.xyz")
            # Crashes when permissions not present or file is folder. Should never occur.
            except FileNotFoundError:
                pass
            print_time_rel(module_init_time, modulename="OpenMM_MD setup", moduleindex=1)
            
    # Simulation loop
    def run(self, simulation_steps=None, simulation_time=None):
        module_init_time = time.time()
        print_line_with_mainheader("OpenMM MOLECULAR DYNAMICS RUN")

        if simulation_steps is None and simulation_time is None:
            print("Either simulation_steps or simulation_time needs to be set")
            exit()
        if simulation_time is not None:
            simulation_steps = int(simulation_time / self.timestep)
        if simulation_steps is not None:
            simulation_time = simulation_steps * self.timestep
            
        print("Simulation time: {} ps".format(simulation_time))
        print("Simulation steps: {}".format(simulation_steps))
        print("Timestep: {} ps".format(self.timestep))
        
        # Run simulation
        kjmolnm_to_atomic_factor = -49614.752589207

        if self.QM_MM_object is not None:
            for step in range(simulation_steps):
                checkpoint_begin_step = time.time()
                print("Step:", step)
                # Get current coordinates to use for QM/MM step
                current_coords = np.array(self.openmmobject.simulation.context.getState(getPositions=True,
                                                                                enforcePeriodicBox=self.enforcePeriodicBox).getPositions(
                    asNumpy=True)) * 10
                # state =  openmmobject.simulation.context.getState(getPositions=True, enforcePeriodicBox=enforcePeriodicBox)
                # current_coords = np.array(state.getPositions(asNumpy=True))*10
                # Manual trajectory option (reporters do not work for manual dynamics steps)
                if step % self.traj_frequency == 0:
                    write_xyzfile(self.fragment.elems, current_coords, "OpenMMMD_traj", printlevel=1, writemode='a')
                # Run QM/MM step to get full system QM+PC gradient.
                # Updates OpenMM object with QM-PC forces
                checkpoint = time.time()
                self.QM_MM_object.run(current_coords=current_coords, elems=self.fragment.elems, Grad=True,
                                exit_after_customexternalforce_update=True)
                print_time_rel(checkpoint, modulename="QM/MM run", moduleindex=2)
                # NOTE: Think about energy correction (currently skipped above)
                # Now take OpenMM step (E+G + displacement etc.)
                checkpoint = time.time()
                self.openmmobject.simulation.step(1)
                print_time_rel(checkpoint, modulename="openmmobject sim step", moduleindex=2)
                print_time_rel(checkpoint_begin_step, modulename="Total sim step", moduleindex=2)
                # NOTE: Better to use OpenMM-plumed interface instead??
                # After MM step, grab coordinates and forces
                if self.plumed_object is not None:
                    print("Plumed active. Untested. Hopefully works")
                    current_coords = np.array(
                        self.openmmobject.simulation.context.getState(getPositions=True).getPositions(asNumpy=True))  # in nm
                    current_forces = np.array(
                        self.openmmobject.simulation.context.getState(getForces=True).getForces(asNumpy=True))  # in kJ/mol /nm
                    energy, newforces = self.plumed_object.run(coords=current_coords, forces=current_forces,
                                                        step=step)  # Plumed object needs to be configured for OpenMM
                    self.openmmobject.update_custom_external_force(self.plumedcustomforce, newforces)

        else:
            print("Regular classical OpenMM MD option chosen")
            # Setting coordinates
            self.openmmobject.set_positions(self.fragment.coords)
            # Running all steps in one go
            # TODO: If we wanted to support plumed then we would have to do step 1-by-1 here
            self.openmmobject.simulation.step(simulation_steps)

        print("OpenMM MD simulation finished!")
        # Close Plumed also if active. Flushes HILLS/COLVAR etc.
        if self.plumed_object is not None:
            self.plumed_object.close()

        # enforcePeriodicBox=True
        self.state = self.openmmobject.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        print("Checking PBC vectors:")
        a, b, c = self.state.getPeriodicBoxVectors()
        print(f"A: ", a)
        print(f"B: ", b)
        print(f"C: ", c)

        # Set new PBC vectors since they may have changed
        print("Updating PBC vectors")
        #Context. Used?
        self.openmmobject.simulation.context.setPeriodicBoxVectors(*[a,b,c])
        #System. Necessary
        self.openmmobject.system.setDefaultPeriodicBoxVectors(*[a,b,c])
        
        # Writing final frame to disk as PDB
        with open('final_MDfrag_laststep.pdb', 'w') as f:
            self.openmmobject.openmm.app.pdbfile.PDBFile.writeHeader(self.openmmobject.topology, f)
        with open('final_MDfrag_laststep.pdb', 'a') as f:
            self.openmmobject.openmm.app.pdbfile.PDBFile.writeModel(self.openmmobject.topology,
                                                            self.state.getPositions(asNumpy=True).value_in_unit(
                                                                self.openmmobject.unit.angstrom), f)
        # Updating ASH fragment
        newcoords = self.state.getPositions(asNumpy=True).value_in_unit(self.openmmobject.unit.angstrom)
        print("Updating coordinates in ASH fragment")
        self.fragment.coords = newcoords

        # Remove frozen atom constraints in the end
        print("Removing frozen atoms from OpenMM object")
        self.openmmobject.unfreeze_atoms()
        
        #Remove user-applied constraints to avoid double-enforcing later
        print("Removing constraints")
        self.openmmobject.remove_constraints(self.constraints)
        
        print_time_rel(module_init_time, modulename="OpenMM_MD run", moduleindex=1)