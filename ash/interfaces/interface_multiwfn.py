import subprocess as sp
import os
import re
import shutil

from ash.functions.functions_general import BC,ashexit, writestringtofile, pygrep, print_line_with_mainheader
from ash.dictionaries_lists import atom_core_electrons
import ash.settings_ash
"""
    Interface to the Multiwfn program
"""

#Use multiwfndir or just distribute multiwfn with ASH

#Multiwfn input:
 # ORCA: Molden file
 # Psi4: fchk file
 # MRCC: Molden and CCDENSITIES

#TODO: Support settings.ini file?

def multiwfn_run(moldenfile, fchkfile=None, option='density', num_frozen_orbs='Auto', mrccoutputfile=None, mrccdensityfile=None, multiwfndir=None, grid=3, numcores=1,
                 fragmentfiles=None, fockfile=None, openshell=False, printlevel=2):
    """Run the Multiwfn program on a Molden file and process the output.

    Args:
        moldenfile (str): Molden wavefunction file (from ORCA, MRCC etc.) used as Multiwfn input.
        fchkfile (str): fchk file (not yet supported).
        option (str): type of analysis. Available options:
            Cube-file generation: 'density', 'valence-density', 'spin-density'/'spindensity', 'elf'
            Atomic charges (main function 7): 'hirshfeld', 'adch', 'chelpg', 'mk'
            Bond orders (main function 9): 'mayerbondorder'/'MBO', 'fuzzy_bondorder'/'FBO',
                'laplacianbondorder'/'LBO'
            Other: 'nocv', 'mrcc-density'
        num_frozen_orbs (str/int): number of frozen orbitals for option='valence-density'
            ('Auto' determines it from the atom_core_electrons dictionary).
        mrccoutputfile (str): MRCC outputfile (for MRCC densities).
        mrccdensityfile (str): MRCC density file, e.g. CCDENSITIES.
        multiwfndir (str): directory containing the Multiwfn executable. If None, the
            multiwfndir setting or the PATH is used.
        grid (int): Multiwfn grid-quality setting used for Cube-file generation.
        numcores (int): number of cores used by Multiwfn.
        fragmentfiles (list): fragment files for option='nocv'.
        fockfile (str): file containing a Fock-matrix printout for option='nocv'.
        openshell (bool): whether the system is open-shell (option='nocv').
        printlevel (int): ASH printlevel.

    Returns:
        For Cube-file options: the name of the created Cube-file (str).
        For charge options: a list of atomic charges (in electrons), or None if no .chg
            file was created.
        For bond-order options: a dict {(atom_i, atom_j): bondorder} with 1-based atom
            indices, or None if nothing could be parsed.
        For other options: None.
    """
    print_line_with_mainheader("multiwfn_run")

    multiwfn_citation_string="""\nDon't forget to to cite Multiwfn if you use it in your work:
Tian Lu, Feiwu Chen, Multiwfn: A multifunctional wavefunction analyzer, J. Comput. Chem., 33, 580-592 (2012)
http://onlinelibrary.wiley.com/doi/10.1002/jcc.22885/abstract
    """
    if printlevel >= 2:
        print(multiwfn_citation_string)
        print()
        print("multiwfndir:", multiwfndir)
        print("Molden file:", moldenfile) #Inputfile is typically a Molden file
        print("Option:", option)
        print("Gridsetting:", grid)
        print("Numcores:", numcores)

    print("Copying ASH Multiwfn settings.ini file to current directory")
    shutil.copy(f"{ash.ashpath}/external/Multiwfn/settings.ini", "settings.ini")
    ############################
    #PREPARING MULTIWFN INPUT
    ############################
    if multiwfndir == None:
        print(BC.WARNING, "No multiwfndir argument passed to multiwfn_run. Attempting to find multiwfndir variable inside settings_ash", BC.END)
        try:
            multiwfndir=ash.settings_ash.settings_dict["multiwfndir"]
        except:
            print(BC.WARNING,"Found no multiwfndir variable in settings_ash module either.",BC.END)
            try:
                multiwfndir = os.path.dirname(shutil.which('Multiwfn'))
                print(BC.OKGREEN,"Found Multiwfn in path. Setting multiwfndir to:", multiwfndir, BC.END)
            except:
                print("Found no Multiwfn executable in path. Exiting... ")
                ashexit()

    #Basic reading of MOlden file
    if moldenfile != None:
        print("\nReading Molden file by ASH:", moldenfile)
        molden_dict = ash.functions.functions_elstructure.read_molden_file(moldenfile)
        numatoms = len(molden_dict["elems"])
        print(f"The file contains {numatoms} atoms and coordinates:")
        elems = molden_dict["elems"]
        coords = molden_dict["coords"]
        for el,c in zip(elems,coords):
            print(f"{el} {c[0]:10.5f} {c[1]:10.5f} {c[2]:10.5f}")

    #TODO: Update, once fchk files are supported
    if os.path.isfile(moldenfile) is False:
        print(f"The selected Moldenfile: {moldenfile} does not exist. Exiting")
        ashexit()

    #Rename MOLDEN-file.
    if moldenfile == "MOLDEN_NAT":
        if printlevel >= 2:
            print("Renaming MOLDEN_NAT to MOLDEN_NAT.molden")
        os.rename(moldenfile, "MOLDEN_NAT.molden")
        moldenfile="MOLDEN_NAT.molden"

    #Valence density requires num_frozen_orbs
    if option == 'valence-density':
        print("Valence density option chosen")
        print("Warning: Valence density option requires a number of frozen orbitals (orbitals to ignore)")
        print("Options: 'Auto' or integer (number of frozen orbitals)")
        print("num_frozen_orbs option:", num_frozen_orbs)
        if num_frozen_orbs == None:
            print("Option valence-density requires num_frozen_orbs")
            print("Specify how many orbitals are frozen")
            ashexit()
        elif num_frozen_orbs == 'Auto':
            print("Determining number of frozen orbitals automatically (used to define valence density)")
            print("Using atom_core_electrons dictionary:", atom_core_electrons)
            tot_num_frozen_orbs = 0
            for el in elems:
                core_els = atom_core_electrons[el]
                tot_num_frozen_orbs += int(core_els/2)
            print("tot_num_frozen_orbs:", tot_num_frozen_orbs)
            num_frozen_orbs = tot_num_frozen_orbs
            print("Number of frozen orbitals determined to be:", num_frozen_orbs)


    #MRCC density special case
    #if option=="mrcc-density":
    #    print("Option mrcc-density chosen")
    #    if mrccoutputfile == None:
    #        print("MRCC outputfile should also be provided")
    #        ashexit()
    #    core_electrons = int(pygrep("Number of core electrons:",mrccoutputfile)[-1])
    #    print("Core electrons found in outputfile:", core_electrons)
    #    frozen_orbs = int(core_electrons/2)
    #    print("Frozen orbitals:", frozen_orbs)
    #    #Rename MRCC Molden file to mrcc.molden
    #    shutil.copy(moldenfile, "mrcc.molden")
    #    #First Multiwfn call. Create new Moldenfile based on correlated density
    #    write_multiwfn_input_option(option=option, grid=grid, frozenorbitals=frozen_orbs, densityfile=mrccdensityfile, printlevel=printlevel)
    #    print("Now calling Multiwfn to process the MRCC, Molden and CCDENSITIES files")
    #    with open("mwfnoptions") as input:
    #        sp.run([multiwfndir+'/Multiwfn', "mrcc.molden"], stdin=input)
    #    print("Multiwfn is done with this part")
    #    #Writes: mrccnew.molden a proper Molden WF file for MRCC WF. Now we can proceed
    #    option="density"
    #    moldenfile="mrccnew.molden"
    #    #Now make new mwfnoptions file for the density generation
    #    write_multiwfn_input_option(option="density", grid=grid, printlevel=printlevel)

    ###########################
    #WRITING MULTIWFN INPUT
    ###########################
    if option == 'nocv':
        print("NOCV option")
        print("fragmentfiles:", fragmentfiles)
        print("Fockfile:", fockfile)
        if fragmentfiles == None:
            print("NOCV option requires fragmentfiles")
            ashexit()
        if fockfile == None:
            print("NOCV option requires fockfile option (can be ORCA output containing Fock-matrix printout)")
            ashexit()
        #Create dummy-input
        write_multiwfn_input_option(option="nocv", grid=grid, fragmentfiles=fragmentfiles, openshell=openshell,
                                    fockfile=fockfile, printlevel=printlevel)
    elif option == 'valence-density':
        write_multiwfn_input_option(option=option, grid=grid, printlevel=printlevel,frozenorbitals=num_frozen_orbs)
    #Density and other options (may or may not work)
    elif option == "elf":
        print("ELF analysis is chosen")
        write_multiwfn_input_option(option=option, grid=grid, printlevel=printlevel)
    else:
        #Writing input
        write_multiwfn_input_option(option=option, grid=grid, printlevel=printlevel)

    ############################
    #RUNNING MULTIWFN
    ############################
    #TODO: Use logging instead
    input=open("mwfnoptions")
    output=open("multiwfn.out",'w')
    if printlevel >= 2:
        print(f"Now calling Multiwfn (using {numcores} cores)")
    sp.run([multiwfndir+'/Multiwfn', moldenfile,'-nt', str(numcores)], stdin=input, stdout=output)
    input.close()
    output.close()
    if printlevel >= 2:
        print("Multiwfn is done!")
    print()

    ############################
    #POST-PROCESSING OUTPUT
    ############################
    #NOTE: For now only Molden-file as main inputfile is supported.
    #TODO: Generalize below
    originputbasename=os.path.splitext(moldenfile)[0]

    if option =="density":
        if printlevel >= 2:
            print("Density option chosen")
        outputfile="density.cub"
        finaloutputfile=originputbasename+'_mwfn.cube'
        os.rename(outputfile, finaloutputfile)
        if printlevel >= 2:
            print("Electron density outputfile written:", finaloutputfile)
        return finaloutputfile
    elif option =="valence-density":
        if printlevel >= 2:
            print("Valence density option chosen")
        outputfile="density.cub"
        finaloutputfile=originputbasename+'_mwfn_valdens.cube'
        os.rename(outputfile, finaloutputfile)
        if printlevel >= 2:
            print("Electron density outputfile written:", finaloutputfile)
        return finaloutputfile
    elif option == "elf":
        if printlevel >= 2:
            print("ELF option chosen")
        outputfile="ELF.cub"
        finaloutputfile=originputbasename+'_ELF.cube'
        os.rename(outputfile, finaloutputfile)
        if printlevel >= 2:
            print("ELF Cube-file written:", finaloutputfile)
        return finaloutputfile
    elif option =="nocv":
        print("NOCV option was chosen.")
        print("Relevant Cube-files were created and NOCV output can be found in: NOCV.txt")
        print("See multiwfn.out for a log-file of the Multiwfn call.")
    elif option in multiwfn_charge_options:
        if printlevel >= 2:
            print(f"Atomic-charge option ({option}) was chosen.")
        #Multiwfn writes the charges to [basename of inputfile].chg
        chgfile=originputbasename+'.chg'
        if os.path.isfile(chgfile) is False:
            print(BC.WARNING,f"No charge file ({chgfile}) was created by Multiwfn.",BC.END)
            print(BC.WARNING,"See multiwfn.out for details.",BC.END)
            return None
        elems, coords, charges = read_multiwfn_chgfile(chgfile)
        if printlevel >= 2:
            print(f"Charge file written by Multiwfn: {chgfile}")
            print(f"\n{option} atomic charges (electrons):")
            for i,(el,q) in enumerate(zip(elems,charges)):
                print(f"{i:<6d} {el:<4s} {q:12.6f}")
            print(f"{'Sum of charges:':<20s}{sum(charges):12.6f}")
        return charges
    elif option in multiwfn_bondorder_options:
        if printlevel >= 2:
            print(f"Bond-order option ({option}) was chosen.")
        bondorders=read_multiwfn_bondorders("multiwfn.out")
        if len(bondorders) == 0:
            print(BC.WARNING,"No bond orders could be parsed from multiwfn.out.",BC.END)
            #Mayer and Wiberg bond orders need basis-function information, which e.g. a
            #wfn/wfx file does not carry. Multiwfn then returns to its main menu
            if os.path.isfile("multiwfn.out"):
                with open("multiwfn.out") as f:
                    if "does not contain basis function information" in f.read():
                        print(BC.WARNING,f"Multiwfn reports that {moldenfile} contains no basis-function information.",BC.END)
                        print(BC.WARNING,f"The {option} model requires it. Use a file with basis-set information (e.g. a Molden or fch file),",BC.END)
                        print(BC.WARNING,"or a density-based model instead (fuzzy_bondorder/FBO or laplacianbondorder/LBO).",BC.END)
            return None
        if printlevel >= 2:
            print(f"\n{option} bond orders (only bonds above the Multiwfn print-threshold):")
            for (i,j),bo in bondorders.items():
                print(f"{i:<6d} {j:<6d} {bo:12.6f}")
        return bondorders


#This function creates an inputfile of numbers that defines what Multiwfn does
def write_multiwfn_input_option(option=None, grid=3, frozenorbitals=None, densityfile=None,
                                fragmentfiles=None,fockfile=None, file4=None, openshell=False, printlevel=2):
    """Write the file of Multiwfn menu-answers (mwfnoptions) that defines what Multiwfn does.

    Args:
        option (str): analysis type, see multiwfn_run for the list of options.
        grid (int): Multiwfn grid-quality setting.
        frozenorbitals (int): number of frozen orbitals (valence-density, mrcc-density).
        densityfile (str): correlated density file (mrcc-density).
        fragmentfiles (list): fragment files (nocv).
        fockfile (str): file containing a Fock-matrix printout (nocv).
        file4 (str): unused.
        openshell (bool): whether the system is open-shell (nocv).
        printlevel (int): ASH printlevel.

    Returns:
        None. Writes the file 'mwfnoptions' to the current directory.
    """
    #Create input formula as file
    if option == 'density':
        denstype=1
        #grid=3 #high-quality grid
        writeoutput=2 #Write Cubefile to current dir
        # 5 Output and plot specific property within a spatial region (calc. grid data)
        # 1 Electron density                 2 Gradient norm of electron density

        inputformula=f"""5
{denstype}
{grid}
{writeoutput}
0
q
        """
    elif option == 'valence-density':
        denstype=1
        #grid=3 #high-quality grid
        writeoutput=2 #Write Cubefile to current dir
        # 5 Output and plot specific property within a spatial region (calc. grid data)
        # 1 Electron density                 2 Gradient norm of electron density

        inputformula=f"""6
26
1-{frozenorbitals}
0
q
-1
5
{denstype}
{grid}
{writeoutput}
0
q
        """
    elif option == "elf":
        inputformula=f"""5
9
3
2
-1
0
17
1
2
2
1
-4
1
-4
3
-10
q
        """
    elif option == 'nocv':
        print("Writing Multiwfn inputfile for NOCV analysis")
        denstype=1
        numprodfrags=len(fragmentfiles)
        fragmentfilenames="\n".join(fragmentfiles)


        #If alpha/beta sets
        if openshell is True:
            openshell1="y"
            openshell2="y"
        else:
            openshell1=""
            openshell2=""
        #grid=3 #high-quality grid
        writeoutput=2 #Write Cubefile to current dir
        # 5 Output and plot specific property within a spatial region (calc. grid data)
        # 1 Electron density                 2 Gradient norm of electron density

        #-2 means generation of Fock matrix by information in file ? Is this valid. Output is not entirely correct
        #-1 means that we read Fock matrix externally. Can read ORCA output that contains Fock-matrix printout
        #%output Print[P_Iter_F] 1 end
        #Not perfect agreement with ORCA though, unclear why
        inputformula=f"""23
{numprodfrags}
{fragmentfilenames}
{openshell1}
{openshell2}
-1
{fockfile}
8
3
Pauli-deform.cube
9
orbdeform.cube
10
totdeform.cube
7
1
NOCVpair0.cube
2
NOCVpair1.cube
3
NOCVpair2.cube
4
NOCVpair3.cube
4
NOCVpair4.cube
q
-4

-10
q
        """
    elif option in multiwfn_charge_options:
        submenu, extra_answers = multiwfn_charge_options[option]
        extralines = "\n".join(extra_answers)
        inputformula=f"""7
{submenu}
{extralines}
y
0
q
        """
    #Special option for creating MRCC correlated WF moldenfile
    elif option =="mrcc-density":
        if frozenorbitals == None:
            print("mrccdensity requires frozenorbitals")
            ashexit()
        if densityfile == None:
            print("mrccdensity requires densityfile")
            ashexit()
        #ASSUMES presence of file CCDENSITIES
        #Will write file: mrccnew.molden
        inputformula=f"""1000
97
./{densityfile}
{frozenorbitals}
y
100
2
6
mrccnew.molden
0
q

        """
    elif option =="laplacianbondorder" or option =="LBO":
        inputformula=f"""9
8
y
0
q
        """
    elif option =="mayerbondorder" or option =="MBO":
        inputformula=f"""9
1
y
0
q
        """
    elif option =="fuzzy_bondorder" or option =="FBO":
        inputformula=f"""9
7
y
0
q
        """
    elif option =="spindensity" or option =="spin-density":
        inputformula=f"""5
5
{grid}
2
0
q
        """

    else:
        print("write_multiwfn_input_option: unknown option")
        ashexit()
    #Write inputformula to disk
    writestringtofile(inputformula,"mwfnoptions")
    if printlevel >= 2:
        print("Wrote Multiwfn inputfile to disk: mwfnoptions")


#Analyze CCSD(T) wavefunctions from Psi4
#See chapter 4.A.8 in Multiwfn manual

def read_fchkfile(file):

    #Load multiwfn with file

    #Enter main function 200 and subfunction 16

    #transform density matrix into natural orbitals

    #Choose CCSD or other ?

    #For open-shell choose different NOs

    pass


#Atomic-charge models supported via Multiwfn main function 7 (Population analysis).
#Key: ASH option name. Value: (Multiwfn submenu number, list of extra answers before the export prompt)
#Submenu numbers and prompt sequences follow the Multiwfn manual (Section 3.9 and examples 4.7.1/4.7.2):
# 1: Hirshfeld (asks for atomic-density method; 1 = built-in atomic densities)
# 11: ADCH (identical procedure to Hirshfeld)
# 12: CHELPG and 13: Merz-Kollman (ESP-fitting; a settings menu appears, 1 = start calculation)
multiwfn_charge_options = {
    "hirshfeld": (1, ["1"]),
    "adch": (11, ["1"]),
    "chelpg": (12, ["1"]),
    "mk": (13, ["1"]),
}

#Bond-order models supported via Multiwfn main function 9 (Bond order analysis)
multiwfn_bondorder_options = {
    "mayerbondorder": 1, "MBO": 1,
    "fuzzy_bondorder": 7, "FBO": 7,
    "laplacianbondorder": 8, "LBO": 8,
}


def read_multiwfn_chgfile(chgfile):
    """Read a Multiwfn .chg file (element, x, y, z, charge per line).

    Args:
        chgfile (str): path to the .chg file written by Multiwfn.

    Returns:
        tuple: (elems, coords, charges) where elems is a list of element symbols,
            coords a list of [x,y,z] in Angstrom and charges a list of floats.
    """
    elems = []
    coords = []
    charges = []
    with open(chgfile) as f:
        for line in f:
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                x, y, z, q = (float(i) for i in fields[1:5])
            except ValueError:
                continue
            elems.append(fields[0])
            coords.append([x, y, z])
            charges.append(q)
    return elems, coords, charges


#Matches e.g.:  #    1:       1(C )      2(H )     0.95583
bondorder_re = re.compile(r"#\s*\d+:\s*(\d+)\(\s*([A-Za-z]+)\s*\)\s*-*\s*(\d+)\(\s*([A-Za-z]+)\s*\)\s*:?\s*(-?\d+\.\d+)")


def read_multiwfn_bondorders(outputfile="multiwfn.out"):
    """Parse bond orders from a Multiwfn outputfile (main function 9).

    Only bonds printed by Multiwfn are returned (controlled by the bndordthres
    parameter in settings.ini, 0.05 by default).

    Args:
        outputfile (str): Multiwfn outputfile to parse.

    Returns:
        dict: {(atomindex_i, atomindex_j): bondorder}. Atom indices are 1-based,
            as printed by Multiwfn.
    """
    bondorders = {}
    with open(outputfile) as f:
        for line in f:
            match = bondorder_re.search(line)
            if match is not None:
                i = int(match.group(1))
                j = int(match.group(3))
                bondorders[(i, j)] = float(match.group(5))
    return bondorders
