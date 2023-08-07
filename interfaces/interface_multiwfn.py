import subprocess as sp
import os
import shutil

from ash.functions.functions_general import BC,ashexit, writestringtofile, pygrep, print_line_with_mainheader
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

def multiwfn_run(moldenfile, fchkfile=None, option='density', mrccoutputfile=None, mrccdensityfile=None, multiwfndir=None, grid=3, numcores=1,
                 fragmentfiles=None, fockfile=None, openshell=False, printlevel=2):
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
    
    #TODO: Update, once fchk files are supported 
    if os.path.isfile(moldenfile) is False:
        print(f"The selected Moldenfile: {moldenfile} does not exist. Exiting")
        ashexit()
    
    #Rename MOLDEN-file. Necessary for some reason. MOLDEN_NAT does not work 
    if moldenfile == "MOLDEN_NAT":
        if printlevel >= 2:
            print("Renaming MOLDEN_NAT to MOLDEN_NAT.molden")
        os.rename(moldenfile, "MOLDEN_NAT.molden")
        moldenfile="MOLDEN_NAT.molden"


    #MRCC density special case
    if option=="mrcc-density":
        print("Option mrcc-density chosen")
        if mrccoutputfile == None:
            print("MRCC outputfile should also be provided")
            ashexit()
        core_electrons = int(pygrep("Number of core electrons:",mrccoutputfile)[-1])
        print("Core electrons found in outputfile:", core_electrons)
        frozen_orbs = int(core_electrons/2)
        print("Frozen orbitals:", frozen_orbs)
        #Rename MRCC Molden file to mrcc.molden
        shutil.copy(moldenfile, "mrcc.molden")
        #First Multiwfn call. Create new Moldenfile based on correlated density
        write_multiwfn_input_option(option=option, grid=grid, frozenorbitals=frozen_orbs, densityfile=mrccdensityfile, printlevel=printlevel)
        print("Now calling Multiwfn to process the MRCC, Molden and CCDENSITIES files")
        with open("mwfnoptions") as input:
            sp.run([multiwfndir+'/Multiwfn', "mrcc.molden"], stdin=input)
        print("Multiwfn is done with this part")
        #Writes: mrccnew.molden a proper Molden WF file for MRCC WF. Now we can proceed
        option="density"
        moldenfile="mrccnew.molden"
        #Now make new mwfnoptions file for the density generation
        write_multiwfn_input_option(option="density", grid=grid, printlevel=printlevel)
    elif option == 'nocv':
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
    #Density and other options (may or may not work)
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
    elif option =="nocv":
        print("NOCV option was chosen.")
        print("Relevant Cube-files were created and NOCV output can be found in: NOCV.txt")
        print("See multiwfn.out for a log-file of the Multiwfn call.")
    elif option =="hirshfeld":
        print("Hirshfeld option was chosen.")

        #read file: 
        outputfile=moldenfile+'.chg'
        print(f"A file: {outputfile} was created")


#This function creates an inputfile of numbers that defines what Multiwfn does
def write_multiwfn_input_option(option=None, grid=3, frozenorbitals=None, densityfile=None,
                                fragmentfiles=None,fockfile=None, file4=None, openshell=False, printlevel=2):
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
    elif option == 'hirshfeld':        
        inputformula=f"""7
1
1
y
0
q
        """

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
    elif option =="mayerbondorder":
        pass
    elif option =="fuzzy_bondorder":
        pass
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

