for ShellRegion in ['ShellRegion2', 'ShellRegion1']:
    # TODO: This is silly.
    if ShellRegion == "ShellRegion1":
        shell = ShellRegion1
    elif ShellRegion == "ShellRegion2":
        shell = ShellRegion2
    print("This is  {} with a shell radius of {} Å".format(ShellRegion, shell))

    # Looping over snapshot
    for snapshot in totrepsnaps:
        # Get elems and coords from each Chemshell frament file
        # Todo: Change to XYZ-file read-in instead (if snapfiles have been converted)
        elems, coords = read_fragfile_xyz(snapshot)
        # create Yggdrasill fragment
        snap_frag = yggdrasill.Fragment(elems=elems, coords=coords)
        # QM and PE regions
        solute_elems = [elems[i] for i in solvsphere.soluteatomsA]
        solute_coords = [coords[i] for i in solvsphere.soluteatomsA]
        # Defining QM and PE regions
        solvshell = get_solvshell(solvsphere, snap_frag.elems, snap_frag.coords, shell, solute_elems, solute_coords,
                                  settings_solvation.scale, settings_solvation.tol)
        print("solvshell is", solvshell)
        # qmatoms = solvsphere.soluteatomsA + solvshell
        qmatoms = solvsphere.soluteatomsA
        peatoms = solvshell  # Polarizable atoms
        print("qmatoms+peatoms:", qmatoms + peatoms)
        mmatoms = listdiff(solvsphere.allatoms, qmatoms + peatoms)  # Nonpolarizable atoms
        print("qmatoms ({} atoms): {}".format(len(qmatoms), qmatoms))
        print("peatoms ({} atoms)".format(len(peatoms)))
        print("mmatoms ({} atoms)".format(len(mmatoms)))

        # Define Psi4 QMregion
        Psi4QMpart_A = yggdrasill.Psi4Theory(charge=solvsphere.ChargeA, mult=solvsphere.MultA, psi4settings=psi4dict,
                                             psi4functional=psi4_functional, runmode='library', printsetting=True)
        Psi4QMpart_B = yggdrasill.Psi4Theory(charge=solvsphere.ChargeB, mult=solvsphere.MultB, psi4settings=psi4dict,
                                             psi4functional=psi4_functional, runmode='library', printsetting=True)

        # Potential options: SEP (Standard Potential), TIP3P Todo: Other options: To be done!
        # PE Solvent-type label for PyFrame. For water, use: HOH, TIP3? WAT?
        PElabel_pyframe = 'HOH'
        # Create PolEmbed theory object. fragment always defined with it
        PolEmbed_SP_A = yggdrasill.PolEmbedTheory(fragment=snap_frag, qm_theory=Psi4QMpart_A,
                                                  qmatoms=qmatoms, peatoms=peatoms, mmatoms=mmatoms, pot_option=pot_option,
                                                  pyframe=True, pot_create=True, PElabel_pyframe=PElabel_pyframe)

        # Note: pot_create=False for B since the embedding potential is the same
        PolEmbed_SP_B = yggdrasill.PolEmbedTheory(fragment=snap_frag, qm_theory=Psi4QMpart_B,
                                                  qmatoms=qmatoms, peatoms=peatoms, mmatoms=mmatoms, pot_option=pot_option,
                                                  pyframe=True, pot_create=False, PElabel_pyframe=PElabel_pyframe)
        # Simple Energy SP calc. potfile needed for B run.
        blankline()
        print(BC.OKGREEN,
              "Starting PolEmbed job for snapshot {} with ShellRegion: {}. State A: Charge: {}  Mult: {}".format(snapshot,
                                                                                                                 ShellRegion,
                                                                                                                 solvsphere.ChargeA,
                                                                                                                 solvsphere.MultA),
              BC.END)
        PolEmbedEnergyA = PolEmbed_SP_A.run(potfile='System.pot', nprocs=NumCores)
        print(BC.OKGREEN,
              "Starting PolEmbed job for snapshot {} with ShellRegion: {}. State B: Charge: {}  Mult: {}".format(snapshot,
                                                                                                                 ShellRegion,
                                                                                                                 solvsphere.ChargeB,
                                                                                                                 solvsphere.MultB),
              BC.END)
        PolEmbedEnergyB = PolEmbed_SP_B.run(potfile='System.pot', nprocs=NumCores, restart=True)
        PolEmbedEnergyAB = (PolEmbedEnergyB - PolEmbedEnergyA) * constants.hartoeV
        # Deleting pot file. Todo: Delete other stuff?
        os.remove('System.pot')

        if shell == ShellRegion1:
            if 'snapA' in snapshot:
                LRPol_Arepsnaps_ABenergy_Region1.append(PolEmbedEnergyAB)
            if calctype == "redox":
                if 'snapB' in snapshot:
                    LRPol_Brepsnaps_ABenergy_Region1.append(PolEmbedEnergyAB)
                LRPol_Allrepsnaps_ABenergy_Region1.append(PolEmbedEnergyAB)
        elif shell == ShellRegion2:
            if 'snapA' in snapshot:
                LRPol_Arepsnaps_ABenergy_Region2.append(PolEmbedEnergyAB)
            if calctype == "redox":
                if 'snapB' in snapshot:
                    LRPol_Brepsnaps_ABenergy_Region2.append(PolEmbedEnergyAB)
                LRPol_Allrepsnaps_ABenergy_Region2.append(PolEmbedEnergyAB)