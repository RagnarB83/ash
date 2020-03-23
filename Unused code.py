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
