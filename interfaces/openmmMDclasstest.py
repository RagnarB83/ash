


class OpenMM_MDclass:
    def __init__(self,fragment=None, theory=None, timestep=0.001, 
              traj_frequency=1000, temperature=300, integrator=None,
              barostat=None, pressure=1, trajectory_file_option='PDB', coupling_frequency=None, anderson_thermostat=False,
              enforcePeriodicBox=True, frozen_atoms=None, constraints=None, restraints=None,
              datafilename=None, dummy_MM=False, plumed_object=None, add_center_force=False,
              center_force_atoms=None, centerforce_constant=1.0):
        module_init_time = time.time()

        print_line_with_mainheader("OpenMM MOLECULAR DYNAMICS CLASS")

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
        if constraints is None:
            self.constraints = []
        if restraints is None:
            self.restraints = []

        if fragment is None:
            print("No fragment object. Exiting")
            exit()

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
            # Simulation loop

    def run(self, simulation_steps=None, simulation_time=None):

        print_line_with_mainheader("OpenMM MOLECULAR DYNAMICS RUN")

        if simulation_steps is None and simulation_time is None:
            print("Either simulation_steps or simulation_time needs to be set")
            exit()
        if simulation_time is not None:
            simulation_steps = int(simulation_time / timestep)
        if simulation_steps is not None:
            simulation_time = simulation_steps * timestep
            
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
            openmmobject.set_positions(fragment.coords)
            # Running all steps in one go
            # TODO: If we wanted to support plumed then we would have to do step 1-by-1 here
            openmmobject.simulation.step(simulation_steps)

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
                                                            state.getPositions(asNumpy=True).value_in_unit(
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
        self.openmmobject.remove_constraints(constraints)
        
        print_time_rel(module_init_time, modulename="OpenMM_MD run", moduleindex=1)