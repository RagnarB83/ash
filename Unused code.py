Part of Fragment class
def calc_connectivity(self, conndepth=99, scale=None, tol=None):
    self.atomlist = list(range(0, self.numatoms))
    self.connectivity = []
    # Going through each atom and getting recursively connected atoms
    testlist = self.atomlist
    # Removing atoms from atomlist until empty
    while len(testlist) > 0:
        for index in testlist:
            wholemol = get_molecule_members_loop_np2(self.coords, self.elems, conndepth, scale, tol, atomindex=index)
            if wholemol in self.connectivity:
                continue
            else:
                self.connectivity.append(wholemol)
                for i in wholemol:
                    testlist.remove(i)
    # Calculate number of atoms in connectivity list of lists
    conn_number_sum = 0
    for l in self.connectivity:
        conn_number_sum += len(l)
    if self.numatoms != conn_number_sum:
        print("Connectivity problem")
        exit()
    self.connected_atoms_number = conn_number_sum
    print("self.connected_atoms_number:", self.connected_atoms_number)
    print("self.connectivity:", self.connectiv

    print("PSI4 Run Mode: Inputfile based")
    print("Not complete yet...")
    exit()
    # Create Psi4 inputfile with generic name
    self.inputfilename = "orca-input"
    print("Creating inputfile:", self.inputfilename + '.inp')
    print("ORCA input:")
    print(self.orcasimpleinput)
    print(self.extraline)
    print(self.orcablocks)
    if PC == True:
        print("Pointcharge embedding is on!")
    create_psi4_pcfile(self.inputfilename, mm_elems, current_MM_coords, MMcharges)
    create_psi4_input_pc(self.inputfilename, qm_elems, current_coords, self.psi4settings,
                         self.charge, self.mult)
    else:
    create_psi4_input_plain(self.inputfilename, qm_elems, current_coords, self.psi4settings,
                            self.charge, self.mult)

    # Run inputfile using Psi4 parallelization. Take nprocs argument.
    print(BC.OKGREEN, "Psi4 Calculation started.", BC.END)
    # Doing gradient or not.
    if Grad == True:
        run_orca_SP_Psi4par(self.psi4dir, self.inputfilename + '.inp', nprocs=nprocs, Grad=True)
    else:
        run_orca_SP_Psi4par(self.psi4dir, self.inputfilename + '.inp', nprocs=nprocs)
    # print(BC.OKGREEN, "------------ORCA calculation done-------------", BC.END)
    print(BC.OKGREEN, "Psi4 Calculation done.", BC.END)

    # Check if finished. Grab energy and gradient
    outfile = self.inputfilename + '.out'
    engradfile = self.inputfilename + '.engrad'
    pcgradfile = self.inputfilename + '.pcgrad'
    if checkPsi4finished(outfile) == True:
        self.energy = finalenergygrab(outfile)

    if Grad == True:
        self.grad = gradientgrab(engradfile)
    if PC == True:
    # Grab pointcharge gradient. i.e. gradient on MM atoms from QM-MM elstat interaction.
        self.pcgrad = pcgradientgrab(pcgradfile)
    print(BC.OKBLUE, BC.BOLD, "------------ENDING PSI4-INTERFACE-------------", BC.END)
    return self.energy, self.grad, self.pcgrad
    else:
    print(BC.OKBLUE, BC.BOLD, "------------ENDING PSI4-INTERFACE-------------", BC.END)
    return self.energy, self.grad






#OLD Optimizer object class. Replaced with function

#Yggdrasill Optimizer class for basic usage
class Optimizer:
    def __init__(self, fragment=None, theory='', optimizer='', maxiter=50, frozen_atoms=[], RMSGtolerance=0.0001, MaxGtolerance=0.0003):
        if fragment is not None:
            self.fragment=fragment
        else:
            print("No fragment provided to optimizer")
            print("Checking if theory object contains defined fragment")
            try:
                self.fragment=theory.fragment
                print("Found theory fragment")
            except:
                print("Found no fragment in theory object either.")
                print("Exiting...")
                exit()
        self.theory=theory
        self.optimizer=optimizer
        self.maxiter=maxiter
        self.RMSGtolerance=RMSGtolerance
        self.MaxGtolerance=MaxGtolerance
        #Frozen atoms: List of atoms that should be frozen
        self.frozen_atoms=frozen_atoms
        #List of active vs. frozen labels
        self.actfrozen_labels=[]
        for i in range(self.fragment.numatoms):
            print("i", i)
            if i in self.frozen_atoms:
                self.actfrozen_labels.append('Frozen')
            else:
                self.actfrozen_labels.append('Active')

    def run(self):
        beginTime = time.time()
        print(BC.OKMAGENTA, BC.BOLD, "------------STARTING OPTIMIZER-------------", BC.END)
        print_option='Big'
        print("Running Optimizer")
        print("Optimization algorithm:", self.optimizer)
        if len(self.frozen_atoms)> 0:
            print("Frozen atoms:", self.frozen_atoms)
        blankline()
        #Printing info and basic initalization of parameters
        if self.optimizer=="SD":
            print("Using very basic stupid Steepest Descent algorithm")
            sdscaling=0.85
            print("SD Scaling parameter:", sdscaling)
        elif self.optimizer=="KNARR-LBFGS":
            print("Using LBFGS optimizer from Knarr by Vilhjálmur Ásgeirsson")
            print("LBFGS parameters (currently hardcoded)")
            print(LBFGS_parameters)
            reset_opt = False
        elif self.optimizer=="SD2":
            sdscaling = 0.01
            print("Using different SD optimizer")
            print("SD Scaling parameter:", sdscaling)
        elif self.optimizer=="KNARR-FIRE":
            time_step=0.01
            was_scaled=False
            print("FIRE Parameters for timestep:", timestep)
            print(GetFIREParam(time_step))

        print("Tolerances:  RMSG: {}  MaxG: {}  Eh/Bohr".format(self.RMSGtolerance, self.MaxGtolerance))
        #Name of trajectory file
        trajname="opt-trajectory.xyz"
        print("Writing XYZ trajectory file: ", trajname)
        #Remove old trajectory file if present
        try:
            os.remove(trajname)
        except:
            pass
        #Current coordinates
        current_coords=self.fragment.coords
        elems=self.fragment.elems

        #OPTIMIZATION LOOP
        #TODO: think about whether we should switch to fragment object for geometry handling
        for step in range(1,self.maxiter):
            CheckpointTime = time.time()
            blankline()
            print("GEOMETRY OPTIMIZATION STEP", step)
            print("Current geometry (Å):")
            if self.theory.__class__.__name__ == "QMMMTheory":
                print_coords_all(current_coords,elems, indices=self.fragment.allatoms, labels=self.theory.hybridatomlabels, labels2=self.actfrozen_labels)
            else:
                print_coords_all(current_coords, elems, indices=self.fragment.allatoms, labels=self.actfrozen_labels)
            blankline()

            #Running E+G theory job.
            E, Grad = self.theory.run(current_coords=current_coords, elems=self.fragment.elems, Grad=True)
            print("E,", E)
            print("Grad,", Grad)

            #Applying frozen atoms constraint. Setting gradient to zero on atoms
            if len(self.frozen_atoms) > 0:
                print("Applying frozen-atom constraints")
                #Setting gradient for atoms to zero
                for num,gradcomp in enumerate(Grad):
                    if num in self.frozen_atoms:
                        Grad[num]=[0.0,0.0,0.0]
            print("Grad (after frozen constraints)", Grad)
            #Converting to atomic forces in eV/Angstrom. Used by Knarr
            forces_evAng=Grad * (-1) * constants.hartoeV / constants.bohr2ang
            blankline()
            print("Step: {}    Energy: {} Eh.".format(step, E))
            if print_option=='Big':
                blankline()
                print("Gradient (Eh/Bohr): \n{}".format(Grad))
                blankline()
            RMSG=RMS_G(Grad)
            MaxG=Max_G(Grad)

            #Write geometry to trajectory (with energy)
            write_xyz_trajectory(trajname, current_coords, elems, E)

            # Convergence threshold check
            if RMSG < self.RMSGtolerance and MaxG < self.MaxGtolerance:
                print("RMSG: {:3.9f}       Tolerance: {:3.9f}    YES".format(RMSG, self.RMSGtolerance))
                print("MaxG: {:3.9f}       Tolerance: {:3.9f}    YES".format(MaxG, self.MaxGtolerance))
                print(BC.OKGREEN,"Geometry optimization Converged!",BC.END)
                write_xyz_trajectory(trajname, current_coords, elems, E)

                # Updating energy and coordinates of Yggdrasill fragment before ending
                self.fragment.set_energy(E)
                print("Final optimized energy:", self.fragment.energy)
                self.fragment.replace_coords(elems, current_coords, conn=False)

                # Writing out fragment file and XYZ file
                self.fragment.print_system(filename='Fragment-optimized.ygg')
                self.fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')

                blankline()
                print_time_rel_and_tot(CheckpointTime, beginTime, 'Opt Step')
                return
            elif RMSG > self.RMSGtolerance and MaxG < self.MaxGtolerance:
                print("RMSG: {:3.9f}       Tolerance: {:3.9f}    NO".format(RMSG, self.RMSGtolerance))
                print("MaxG: {:3.9f}       Tolerance: {:3.9f}    YES".format(MaxG, self.MaxGtolerance))
                print(BC.WARNING,"Not converged",BC.END)
            elif RMSG < self.RMSGtolerance and MaxG > self.MaxGtolerance:
                print("RMSG: {:3.9f}       Tolerance: {:3.9f}    YES".format(RMSG, self.RMSGtolerance))
                print("MaxG: {:3.9f}       Tolerance: {:3.9f}    NO".format(MaxG, self.MaxGtolerance))
                print(BC.WARNING,"Not converged",BC.END)
            else:
                print("RMSG: {:3.9f}       Tolerance: {:3.9f}    NO".format(RMSG, self.RMSGtolerance))
                print("MaxG: {:3.9f}       Tolerance: {:3.9f}    NO".format(MaxG, self.MaxGtolerance))
                print(BC.WARNING,"Not converged",BC.END)

            blankline()
            if self.optimizer=='SD':
                print("Using Basic Steepest Descent optimizer")
                print("Scaling parameter:", sdscaling)
                current_coords=steepest_descent(current_coords,Grad,sdscaling)
            elif self.optimizer=='SD2':
                print("Using Basic Steepest Descent optimizer, SD2 with norm")
                print("Scaling parameter:", sdscaling)
                current_coords=steepest_descent2(current_coords,Grad,sdscaling)
            elif self.optimizer=="KNARR-FIRE":
                print("Taking FIRE step")
                # FIRE
                if step == 1 or reset_opt:
                    reset_opt = False
                    fire_param = GetFIREParam(time_step)
                    ZeroVel=np.zeros( (3*len(current_coords),1))
                    CurrentVel=ZeroVel
                if was_scaled:
                    time_step *= 0.95
                velo, time_step, fire_param = GlobalFIRE(forces_evAng, CurrentVel, time_step, fire_param)
                CurrentVel=velo
                step, velo = EulerStep(CurrentVel, forces_evAng, time_step)
                CurrentVel=velo

            elif self.optimizer=='NR':
                print("disabled")
                exit()
                #Identity matrix
                Hess_approx=np.identity(3*len(current_coords))
                #TODO: Not active
                current_coords = newton_raphson(current_coords, Grad, Hess_approx)
            elif self.optimizer=='KNARR-LBFGS':
                if step == 1 or reset_opt:
                    if reset_opt == True:
                        print("Resetting optimizer")
                    print("Taking SD-like step")
                    reset_opt = False
                    sk = []
                    yk = []
                    rhok = []
                    #Store original atomic forces (in eV/Å)
                    keepf=np.copy(forces_evAng)
                    keepr = np.copy(current_coords)
                    step = TakeFDStep(self.theory, current_coords, LBFGS_parameters["fd_step"], forces_evAng, self.fragment.elems)

                else:
                    print("Doing LBFGS Update")
                    sk, yk, rhok = LBFGSUpdate(current_coords, keepr, forces_evAng, keepf,
                                               sk, yk, rhok, LBFGS_parameters["lbfgs_memory"])
                    keepf=np.copy(forces_evAng)
                    keepr = np.copy(current_coords)
                    print("Taking LBFGS Step")
                    step, negativecurv = LBFGSStep(forces_evAng, sk, yk, rhok)
                    step *= LBFGS_parameters["lbfgs_damping"]
                    if negativecurv:
                        reset_opt = True
            else:
                print("Optimizer option not supported.")
                exit()
            #Take the actual step
            #Todo: Implement maxmove-scaling here if step too large
            current_coords=current_coords+step
            #Write current geometry (after step) to disk as 'Current_geometry.xyz'.
            # Can be used if optimization failed, SCF convergence problemt etc.
            write_xyzfile(elems, current_coords, 'Current_geometry')
            blankline()
            print_time_rel_and_tot(CheckpointTime, beginTime, 'Opt Step')
        print(BC.FAIL,"Optimization did not converge in {} iteration".format(self.maxiter),BC.END)

