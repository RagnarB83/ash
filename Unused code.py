#Distance for 2D arrays of coords
#Probably won't be used much. Slower than
function distance_array(x::Array{Float64, 2}, y::Array{Float64, 2})
    nx = size(x, 1)
    ny = size(y, 1)
    r=zeros(nx,ny)

        for j = 1:ny
            @fastmath for i = 1:nx
                @inbounds dx = y[j, 1] - x[i, 1]
                @inbounds dy = y[j, 2] - x[i, 2]
                @inbounds dz = y[j, 3] - x[i, 3]
                rSq = dx*dx + dy*dy + dz*dz
                @inbounds r[i, j] = sqrt(rSq)
            end
        end
    return r
end


#Connectivity entirely via Julia
#Old: Delete
function old_calc_connectivity(coords,elems,conndepth,scale, tol,eldict_covrad)
    # Calculate connectivity by looping over all atoms
	found_atoms = Int64[]
	#List of lists
	fraglist = Array{Int64}[]
	#println(typeof(fraglist))
    #Looping over atoms
	for atom in 1:length(elems)
		if length(found_atoms) == length(elems)
			println("All atoms accounted for. Exiting...")
			return fraglist
		end
		if atom-1 ∉ found_atoms
			members = get_molecule_members_julia(coords, elems, conndepth, scale, tol, eldict_covrad, atomindex=atom-1)
			if members ∉ fraglist
				push!(fraglist,members)
				found_atoms = [found_atoms;members]
			end
		end
	end
	return fraglist
end

# Numpy clever loop test with Julia
@timefn
def bget_molecule_members_loop_np2_jul(coords, elems, loopnumber, scale, tol, atomindex=None, membs=None):
    if membs is None:
        membs = []
        membs.append(atomindex)
        # print("membs:", membs)
        # get_connected_atoms_julia(coords, elems,eldict_covrad, scale,tol, atomindex)
        # print("eldict_covrad:", eldict_covrad)
        # print(type(coords))
        # print(type(elems))
        # print(type(eldict_covrad))
        # print(type(scale))
        # print(type(tol))
        # print(type(atomindex))
        timestampA = time.time()
        membs = list(Main.Juliafunctions.get_connected_atoms_julia(coords, elems, eldict_covrad, scale, tol, atomindex))

        # exit()
        # print("membs:", membs)

    finalmembs = membs
    for i in range(loopnumber):
        # print("loop i:", i)
        # Get list of lists of connatoms for each member
        newmembers = [Main.Juliafunctions.get_connected_atoms_julia(coords, elems, eldict_covrad, scale, tol, k) for k
                      in membs]
        # print("newmembers:", newmembers)
        # print(type(newmembers))
        # PROBLEM: Julia returns newmembers as list of numpy arrays
        # exit()
        # Get a unique flat list
        trimmed_flat = np.unique([item for sublist in newmembers for item in sublist]).tolist()
        # print("trimmed_flat:", trimmed_flat)
        # print("finalmembs ", finalmembs)
        # exit()
        # Check if new atoms not previously found
        membs = listdiff(trimmed_flat, finalmembs)
        # print("membs ", membs)
        # Exit loop if nothing new found
        if len(membs) == 0:
            # print("exiting")
            return finalmembs
        # print("type of membs:", type(membs))
        # print("type of finalmembs:", type(finalmembs))
        finalmembs += membs
        # print("finalmembs ", finalmembs)
        finalmembs = np.unique(finalmembs).tolist()
        # print("finalmembs ", finalmembs)
        # exit()
        # print("finalmembs:", finalmembs)
        # print_time_rel(timestampA, modulename='finalmembs  julia')
        # exit()
    return finalmembs





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










#OLD NUMFREQ class

#Numerical frequencies class
#Todo: Change to function?
class NumericalFrequencies:
    def __init__(self, fragment, theory, npoint=2, displacement=0.0005, hessatoms=None, numcores=1, runmode='serial' ):
        self.runmode=runmode
        self.numcores=numcores
        self.fragment=fragment
        self.theory=theory
        self.coords=fragment.coords
        self.elems=fragment.elems
        self.numatoms=len(self.elems)
        #Hessatoms list is allatoms (if not defined), otherwise the atoms provided and thus a partial Hessian is calculated.
        self.allatoms=list(range(0,self.numatoms))
        if hessatoms is None:
            self.hessatoms=self.allatoms
        else:
            self.hessatoms=hessatoms
        self.npoint = npoint
        self.displacement=displacement
        self.displacement_bohr = self.displacement *constants.ang2bohr


    def run(self):
        print("Starting Numerical Frequencies job for fragment")
        print("System size:", self.numatoms)
        print("Hessian atoms:", self.hessatoms)
        if self.hessatoms != self.allatoms:
            print("This is a partial Hessian.")
        if self.npoint ==  1:
            print("One-point formula used (forward difference)")
        elif self.npoint == 2:
            print("Two-point formula used (central difference)")
        else:
            print("Unknown npoint option. npoint should be set to 1 (one-point) or 2 (two-point formula).")
            exit()
        print("Displacement: {:5.4f} Å ({:5.4f} Bohr)".format(self.displacement,self.displacement_bohr))
        blankline()
        print("Starting geometry:")
        #Converting to numpy array
        #TODO: get rid list->np-array conversion
        current_coords_array=np.array(self.coords)
        print_coords_all(current_coords_array, self.elems)
        blankline()

        #Looping over each atom and each coordinate to create displaced geometries
        #Only displacing atom if in hessatoms list. i.e. possible partial Hessian
        list_of_displaced_geos=[]
        list_of_displacements=[]
        for atom_index in range(0,len(current_coords_array)):
            if atom_index in self.hessatoms:
                for coord_index in range(0,3):
                    val=current_coords_array[atom_index,coord_index]
                    #Displacing in + direction
                    current_coords_array[atom_index,coord_index]=val+self.displacement
                    y = current_coords_array.copy()
                    list_of_displaced_geos.append(y)
                    list_of_displacements.append([atom_index, coord_index, '+'])
                    if self.npoint == 2:
                        #Displacing  - direction
                        current_coords_array[atom_index,coord_index]=val-self.displacement
                        y = current_coords_array.copy()
                        list_of_displaced_geos.append(y)
                        list_of_displacements.append([atom_index, coord_index, '-'])
                    #Displacing back
                    current_coords_array[atom_index, coord_index] = val

        # Original geo added here if onepoint
        if self.npoint == 1:
            list_of_displaced_geos.append(current_coords_array)
            list_of_displacements.append('Originalgeo')

        if self.runmode == 'serial':
            #Looping over geometries and running
            freqinputfiles=[]

            #Dictionary for each displacement:
            #   key: AtomNCoordPDirectionm   where N=atomnumber, P=x,y,z and direction m: + or -
            #   value: gradient
            displacement_dictionary={}
            print("List of displacements:", list_of_displacements)

            for disp, geo in zip(list_of_displacements,list_of_displaced_geos):
                if disp == 'Originalgeo':
                    calclabel = 'Originalgeo'
                else:
                    atom_disp=disp[0]
                    if disp[1] == 0:
                        crd='x'
                    elif disp[1] == 1:
                        crd = 'y'
                    elif disp[1] == 2:
                        crd = 'z'
                    drection=disp[2]
                    #displacement_jobname='Numfreq-Disp-'+'Atom'+str(atom_disp)+crd+drection
                    print("Displacing Atom: {} Coordinate: {} Direction: {}".format(atom_disp, crd, drection))
                    calclabel='Atom{}Coord{}Direction{}'.format(atom_disp,crd,drection)
                if type(self.theory)==ORCATheory:
                    #create_orca_input_plain(displacement_jobname, self.elems, geo, self.theory.orcasimpleinput,
                    #                    self.theory.orcablocks, self.theory.charge, self.theory.mult, Grad=True)
                    energy, gradient = self.theory.run(current_coords=geo, elems=self.elems, Grad=True,
                                                                     nprocs=self.numcores)
                    print("gradient:", gradient)
                    #Adding gradient to dictionary for AtomNCoordPDirectionm
                    displacement_dictionary[calclabel] = gradient
                elif type(self.theory)==QMMMTheory:
                    print("QM/MM Theory for Numfreq in progress")
                    energy, gradient = self.theory.run(current_coords=geo, elems=self.elems, Grad=True, nprocs=self.numcores)
                    displacement_dictionary[calclabel] = gradient
                elif type(self.theory)==xTBTheory:
                    energy, gradient = self.theory.run(current_coords=geo, elems=self.elems, Grad=True, nprocs=self.numcores)
                    displacement_dictionary[calclabel] = gradient
                else:
                    print("theory not implemented for numfreq yet")
                    exit()
                #freqinputfiles.append(displacement_jobname)
        elif self.runmode == 'parallel':
            print("parallel not ready")
            exit(1)


        print("Calculations are done.")

        #If partial Hessian remove non-hessatoms part of gradient:
        #Get partial matrix by deleting atoms not present in list.
        if self.npoint == 1:
            original_grad=get_partial_matrix(self.allatoms, self.hessatoms, displacement_dictionary['Originalgeo'])
            original_grad_1d = np.ravel(original_grad)
        #Initialize Hessian
        hesslength=3*len(self.hessatoms)
        hessian=np.zeros((hesslength,hesslength))


        #Onepoint-formula Hessian
        if self.npoint == 1:
            #Starting index for Hessian array
            index=0
            #Getting displacements as keys from dictionary and sort
            dispkeys = list(displacement_dictionary.keys())
            #Sort seems to sort it correctly w.r.t. atomnumber,x,y,z and +/-
            dispkeys.sort()
            #for displacement, grad in displacement_dictionary.items():
            for dispkey in dispkeys:
                grad=displacement_dictionary[dispkey]
                #Skipping original geo
                if dispkey != 'Originalgeo':
                    #Getting grad as numpy matrix and converting to 1d
                    # If partial Hessian remove non-hessatoms part of gradient:
                    grad = get_partial_matrix(self.allatoms, self.hessatoms, grad)
                    grad_1d = np.ravel(grad)
                    Hessrow=(grad_1d - original_grad_1d)/self.displacement_bohr
                    hessian[index,:]=Hessrow
                    index+=1
        #Twopoint-formula Hessian. pos and negative directions come in order
        elif self.npoint == 2:
            count=0; hessindex=0
            #Getting displacements as keys from dictionary and sort
            dispkeys = list(displacement_dictionary.keys())
            #Sort seems to sort it correctly w.r.t. atomnumber,x,y,z and +/-
            dispkeys.sort()
            #for file in freqinputfiles:
            #for displacement, grad in testdict.items():
            for dispkey in dispkeys:
                if dispkey != 'Originalgeo':
                    count+=1
                    if count == 1:
                        grad_pos=displacement_dictionary[dispkey]
                        #print("pos I hope")
                        #print("dispkey:", dispkey)
                        # If partial Hessian remove non-hessatoms part of gradient:
                        grad_pos = get_partial_matrix(self.allatoms, self.hessatoms, grad_pos)
                        grad_pos_1d = np.ravel(grad_pos)
                    elif count == 2:
                        grad_neg=displacement_dictionary[dispkey]
                        #print("neg I hope")
                        #print("dispkey:", dispkey)
                        #Getting grad as numpy matrix and converting to 1d
                        # If partial Hessian remove non-hessatoms part of gradient:
                        grad_neg = get_partial_matrix(self.allatoms, self.hessatoms, grad_neg)
                        grad_neg_1d = np.ravel(grad_neg)
                        Hessrow=(grad_pos_1d - grad_neg_1d)/(2*self.displacement_bohr)
                        hessian[hessindex,:]=Hessrow
                        grad_pos_1d=0
                        grad_neg_1d=0
                        count=0
                        hessindex+=1
                    else:
                        print("Something bad happened")
                        exit()
                blankline()


        #Symmetrize Hessian by taking average of matrix and transpose
        symm_hessian=(hessian+hessian.transpose())/2
        self.hessian=symm_hessian
        #Write Hessian to file
        with open("Hessian", 'w') as hfile:
            hfile.write(str(hesslength)+' '+str(hesslength)+'\n')
            for row in self.hessian:
                rowline=' '.join(map(str, row))
                hfile.write(str(rowline)+'\n')
            blankline()
            print("Wrote Hessian to file: Hessian")
        #Write ORCA-style Hessian file
        write_ORCA_Hessfile(self.hessian, self.coords, self.elems, self.fragment.list_of_masses, self.hessatoms)

        #Project out Translation+Rotational modes
        #TODO

        #Diagonalize mass-weighted Hessian
        # Get partial matrix by deleting atoms not present in list.
        hesselems = get_partial_list(self.allatoms, self.hessatoms, self.elems)
        hessmasses = get_partial_list(self.allatoms, self.hessatoms, self.fragment.list_of_masses)
        print("Elements:", hesselems)
        print("Masses used:", hessmasses)
        self.frequencies=diagonalizeHessian(self.hessian,hessmasses,hesselems)[0]

        #Print out normal mode output. Like in Chemshell or ORCA
        blankline()
        print("Normal modes:")
        #TODO: Eigenvectors print here
        print("Eigenvectors to be  be printed here")
        blankline()
        #Print out Freq output. Maybe print normal mode compositions here instead???
        printfreqs(self.frequencies,len(self.hessatoms))

        #Print out thermochemistry
        thermochemcalc(self.frequencies,self.hessatoms, self.fragment, self.theory.mult, temp=298.18,pressure=1)

        #TODO: https://pages.mtu.edu/~msgocken/ma5630spring2003/lectures/diff/diff/node6.html
        print("Numerical frequencies done!")



#EVEN OLDER BAD CODE
#Numerical frequencies class
class NumericalFrequencies:
    def __init__(self, fragment, theory, npoint=2, displacement=0.0005, hessatoms=None, numcores=1 ):
        self.numcores=numcores
        self.fragment=fragment
        self.theory=theory
        self.coords=fragment.coords
        self.elems=fragment.elems
        self.numatoms=len(self.elems)
        #Hessatoms list is allatoms (if not defined), otherwise the atoms provided and thus a partial Hessian is calculated.
        self.allatoms=list(range(0,self.numatoms))
        if hessatoms==[]:
            self.hessatoms=self.allatoms
        else:
            self.hessatoms=hessatoms
        self.npoint = npoint
        self.displacement=displacement
        #self.displacement_bohr = self.displacement *constants.bohr2ang
        print("self.displacement_bohr:", self.displacement_bohr)
        self.displacement_bohr = 1
        print("self.displacement_bohr:", self.displacement_bohr)
        exit()
    def run(self):
        print("Starting Numerical Frequencies job for fragment")
        print("System size:", self.numatoms)
        print("Hessian atoms:", self.hessatoms)
        exit(1)
        if self.hessatoms != self.allatoms:
            print("This is a partial Hessian.")
        if self.npoint ==  1:
            print("One-point formula used (forward difference)")
        elif self.npoint == 2:
            print("Two-point formula used (central difference)")
        else:
            print("Unknown npoint option. npoint should be set to 1 (one-point) or 2 (two-point formula).")
            exit()
        print("Displacement: {:3.3f} Å ({:3.3f} Bohr)".format(self.displacement,self.displacement_bohr))
        blankline()
        print("Starting geometry:")
        #Converting to numpy array
        #TODO: get rid list->np-array conversion
        current_coords_array=np.array(self.coords)
        print_coords_all(current_coords_array, self.elems)
        blankline()

        #Looping over each atom and each coordinate to create displaced geometries
        #Only displacing atom if in hessatoms list. i.e. possible partial Hessian
        list_of_displaced_geos=[]
        list_of_displacements=[]
        for atom_index in range(0,len(current_coords_array)):
            if atom_index in self.hessatoms:
                for coord_index in range(0,3):
                    val=current_coords_array[atom_index,coord_index]
                    #Displacing in + direction
                    current_coords_array[atom_index,coord_index]=val+self.displacement
                    y = current_coords_array.copy()
                    list_of_displaced_geos.append(y)
                    list_of_displacements.append([atom_index, coord_index, '+'])
                    if self.npoint == 2:
                        #Displacing  - direction
                        current_coords_array[atom_index,coord_index]=val-self.displacement
                        y = current_coords_array.copy()
                        list_of_displaced_geos.append(y)
                        list_of_displacements.append([atom_index, coord_index, '-'])
                    #Displacing back
                    current_coords_array[atom_index, coord_index] = val

        #Looping over geometries and creating inputfiles.
        freqinputfiles=[]
        for disp, geo in zip(list_of_displacements,list_of_displaced_geos):
            atom_disp=disp[0]
            if disp[1] == 0:
                crd='x'
            elif disp[1] == 1:
                crd = 'y'
            elif disp[1] == 2:
                crd = 'z'
            drection=disp[2]
            displacement_jobname='Numfreq-Disp-'+'Atom'+str(atom_disp)+crd+drection
            print("Displacing Atom: {} Coordinate: {} Direction: {}".format(atom_disp, crd, drection))

            if type(self.theory)==ORCATheory:
                create_orca_input_plain(displacement_jobname, self.elems, geo, self.theory.orcasimpleinput,
                                    self.theory.orcablocks, self.theory.charge, self.theory.mult, Grad=True)
            elif type(self.theory)==QMMMTheory:
                print("QM/MM Theory for Numfreq in progress")
                exit()
            elif type(self.theory)==xTBTheory:
                print("xtb for Numfreq not implemented yet")
                exit()
            else:
                print("theory not implemented for numfreq yet")
                exit()
            freqinputfiles.append(displacement_jobname)
        #freqinplist is in order of atom, x/y/z-coordinates, direction etc.
        #e.g. Numfreq-Disp-Atom0x+,  Numfreq-Disp-Atom0x-, Numfreq-Disp-Atom0y+  etc.

        #Adding initial geometry to freqinputfiles list
        if type(self.theory) == ORCATheory:
            create_orca_input_plain('Originalgeo', self.elems, current_coords_array, self.theory.orcasimpleinput,
                                self.theory.orcablocks, self.theory.charge, self.theory.mult, Grad=True)
        elif type(self.theory) == QMMMTheory:
            print("QM/MM theory for Numfreq in progress")
        else:
            print("theory not implemented for numfreq yet")
            exit()
        freqinputfiles.append('Originalgeo')

        #Run all inputfiles in parallel by multiprocessing
        blankline()
        print("Starting Displacement calculations.")
        if type(self.theory) == ORCATheory:
            run_inputfiles_in_parallel(self.theory.orcadir, freqinputfiles, self.numcores)
        elif type(self.theory) == QMMMTheory:
            print("QM/MM theory for Numfreq in progress")
        else:
            print("theory not implemented for numfreq yet")
            exit()

        #Grab energy and gradient of original geometry. Only used for onepoint formula
        original_grad = ORCAgradientgrab('Originalgeo' + '.engrad')

        #If partial Hessian remove non-hessatoms part of gradient:
        #Get partial matrix by deleting atoms not present in list.
        original_grad=get_partial_matrix(self.allatoms, self.hessatoms, original_grad)
        original_grad_1d = np.ravel(original_grad)

        #Initialize Hessian
        hesslength=3*len(self.hessatoms)
        hessian=np.zeros((hesslength,hesslength))

        #Twopoint-formula Hessian. pos and negative directions come in order
        if self.npoint == 2:
            count=0; hessindex=0
            for file in freqinputfiles:
                if file != 'Originalgeo':
                    count+=1
                    if count == 1:
                        if type(self.theory) == ORCATheory:
                            grad_pos = ORCAgradientgrab(file + '.engrad')
                        else:
                            print("theory not implemented for numfreq yet")
                            exit()
                        # If partial Hessian remove non-hessatoms part of gradient:
                        grad_pos = get_partial_matrix(self.allatoms, self.hessatoms, grad_pos)
                        grad_pos_1d = np.ravel(grad_pos)
                    elif count == 2:
                        #Getting grad as numpy matrix and converting to 1d
                        if type(self.theory) == ORCATheory:
                            grad_neg=ORCAgradientgrab(file+'.engrad')
                        else:
                            print("theory not implemented for numfreq yet")
                            exit()
                        # If partial Hessian remove non-hessatoms part of gradient:
                        grad_neg = get_partial_matrix(self.allatoms, self.hessatoms, grad_neg)
                        grad_neg_1d = np.ravel(grad_neg)
                        Hessrow=(grad_pos_1d - grad_neg_1d)/(2*self.displacement_bohr)
                        hessian[hessindex,:]=Hessrow
                        grad_pos_1d=0
                        grad_neg_1d=0
                        count=0
                        hessindex+=1
                    else:
                        print("Something bad happened")
                        exit()
                blankline()

        #Onepoint-formula Hessian
        elif self.npoint == 1:
            for index,file in enumerate(freqinputfiles):
                #Skipping original geo
                if file != 'Originalgeo':
                    #Getting grad as numpy matrix and converting to 1d
                    if type(self.theory) == ORCATheory:
                        grad=ORCAgradientgrab(file+'.engrad')
                    else:
                        print("theory not implemented for numfreq yet")
                        exit()
                    # If partial Hessian remove non-hessatoms part of gradient:
                    grad = get_partial_matrix(self.allatoms, self.hessatoms, grad)
                    grad_1d = np.ravel(grad)
                    Hessrow=(grad_1d - original_grad_1d)/self.displacement_bohr
                    hessian[index,:]=Hessrow

        #Symmetrize Hessian by taking average of matrix and transpose
        symm_hessian=(hessian+hessian.transpose())/2
        self.hessian=symm_hessian

        #Write Hessian to file
        with open("Hessian", 'w') as hfile:
            hfile.write(str(hesslength)+' '+str(hesslength)+'\n')
            for row in self.hessian:
                rowline=' '.join(map(str, row))
                hfile.write(str(rowline)+'\n')
            blankline()
            print("Wrote Hessian to file: Hessian")
        #Write ORCA-style Hessian file
        write_ORCA_Hessfile(self.hessian, self.coords, self.elems, self.fragment.list_of_masses, self.hessatoms)

        #Project out Translation+Rotational modes
        #TODO

        #Diagonalize Hessian
        print("self.fragment.list_of_masses:", self.fragment.list_of_masses)
        print("self.elems:", self.elems)
        # Get partial matrix by deleting atoms not present in list.
        hesselems = get_partial_list(self.allatoms, self.hessatoms, self.elems)
        hessmasses = get_partial_list(self.allatoms, self.hessatoms, self.fragment.list_of_masses)
        print("hesselems", hesselems)
        print("hessmasses:", hessmasses)
        self.frequencies=diagonalizeHessian(self.hessian,hessmasses,hesselems)[0]

        #Print out normal mode output. Like in Chemshell or ORCA
        blankline()
        print("Normal modes:")
        print("Eigenvectors will be printed here")
        blankline()
        #Print out Freq output. Maybe print normal mode compositions here instead???
        printfreqs(self.frequencies,len(self.hessatoms))

        #Print out thermochemistry
        thermochemcalc(self.frequencies,self.hessatoms, self.fragment, self.theory.mult, temp=298.18,pressure=1)

        #TODO: https://pages.mtu.edu/~msgocken/ma5630spring2003/lectures/diff/diff/node6.html
