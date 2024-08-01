import subprocess as sp
import os
import shutil
import time
import numpy as np
import math

import ash.settings_ash
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader,pygrep
from ash.interfaces.interface_multiwfn import write_multiwfn_input_option

MRCC_basis_dict={'DZ':'cc-pVDZ', 'TZ':'cc-pVTZ', 'QZ':'cc-pVQZ', '5Z':'cc-pV5Z', 'ADZ':'aug-cc-pVDZ', 'ATZ':'aug-cc-pVTZ', 'AQZ':'aug-cc-pVQZ',
            'A5Z':'aug-cc-pV5Z'}

#MRCC Theory object.
class MRCCTheory:
    def __init__(self, mrccdir=None, filename='mrcc', printlevel=2,
                mrccinput=None, numcores=1, parallelization='OMP-and-MKL', label="MRCC",
                keep_orientation=True, frozen_core_settings='Auto', no_basis_read_orbs=False):

        self.theorynamelabel="MRCC"
        self.theorytype="QM"
        self.analytic_hessian=False
        self.label=label

        print_line_with_mainheader("MRCCTheory initialization")

        if mrccinput is None:
            print("MRCCTheory requires a mrccinput keyword")
            ashexit()

        if mrccdir is None:
            print(BC.WARNING, "No mrccdir argument passed to MRCCTheory. Attempting to find mrccdir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.mrccdir=ash.settings_ash.settings_dict["mrccdir"]
            except KeyError:
                print(BC.WARNING,"Found no mrccdir variable in settings_ash module either.",BC.END)
                try:
                    self.mrccdir = os.path.dirname(shutil.which('dmrcc'))
                    print(BC.OKGREEN,"Found dmrcc in PATH. Setting mrccdir to:", self.mrccdir, BC.END)
                except:
                    print(BC.FAIL,"Found no dmrcc executable in PATH. Exiting... ", BC.END)
                    ashexit()
        else:
            self.mrccdir = mrccdir


        #Printlevel
        self.printlevel=printlevel
        self.filename=filename
        self.mrccinput=mrccinput
        self.numcores=numcores
        self.frozen_core_settings = frozen_core_settings
        #Parallelization strategy: 'OMP', 'OMP-and-MKL' or 'MPI'
        self.parallelization=parallelization
        self.keep_orientation=keep_orientation

        # No basis read orbs option If True, MRCC will read in fort.55 and fort.56 files (containing integrals)
        self.no_basis_read_orbs=no_basis_read_orbs
        if self.no_basis_read_orbs is True:
            if 'basis' in mrccinput:
                print("Error: basis keyword found in mrccinput. should not be used when no_basis_read_orbs option is active")
                ashexit()

        if self.keep_orientation is True:
            print("Warning: keep_orientation options is on (by default)! This means that the original input structure in an MRCC job is kept and symmetry is turned off")
            print("This is necessary for gradient calculations and also is you want the density for the original structure")
            print("Do keep_orientation=False if you want MRCC to use symmetry and its own standard orientation")
    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("MRCC cleanup not yet implemented.")

    #Determines Frozen core seetings to apply
    def determine_frozen_core(self,elems):
        print("Determining frozen core")
        print("frozen_core_settings options are: Auto, None or MRCC")
        print("Auto uses ASH frozen core settings (mimics ORCA settings)")
        print("MRCC uses default MRCC frozen core settings (not good for 3d metals)")
        #Frozen core settings
        FC_elems={'H':0,'He':0,'Li':0,'Be':0,'B':2,'C':2,'N':2,'O':2,'F':2,'Ne':2,
        'Na':2,'Mg':2,'Al':10,'Si':10,'P':10,'S':10,'Cl':10,'Ar':10,
        'K':10,'Ca':10,'Sc':10,'Ti':10,'V':10,'Cr':10,'Mn':10,'Fe':10,'Co':10,'Ni':10,'Cu':10,'Zn':10,
        'Ga':18,'Ge':18,'As':18,'Se':18, 'Br':18, 'Kr':18}

        if self.frozen_core_settings == None:
            print("Frozen core requested OFF. MRCC will run all-electron calculations")
            self.frozencore_string=f"0"
        else:
            print("Frozen core is ON!")
            if self.frozen_core_settings == 'Auto':
                print("Auto frozen-core settings requested")
                num_el=0
                for el in elems:
                    num_el+=FC_elems[el]
                frozen_core_el=num_el
                frozen_core_orbs=int(num_el/2)
                print("Total frozen electrons in system:", frozen_core_el)
                print("Total frozen orbitals in system:", frozen_core_orbs)
                self.frozencore_string=f"{frozen_core_orbs}"
            elif self.frozen_core_settings == 'MRCC':
                print("MRCC settings requested")
                self.frozencore_string=f"frozen"
            else:
                print("Unknown option for frozen_core_settings")
                ashexit()
    #Method to grab dipole moment from a MRCC outputfile (assumes run has been executed)
    def get_dipole_moment(self):
        return grab_dipole_moment(self.filename+'.out')
    #NOTE: Polarizability not available in MRCC
    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        module_init_time=time.time()
        if numcores == None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, "------------RUNNING MRCC INTERFACE-------------", BC.END)
        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for MRCCTheory.run method", BC.END)
            ashexit()

        print("Running MRCC object.")
        print("Job label:", label)
        print("Creating inputfile: MINP")
        print("MRCC input:")
        print(self.mrccinput)

        #Coords provided to run
        if current_coords is not None:
            pass
        else:
            print("no current_coords")
            ashexit()

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        #FROZEN CORE SETTINGS
        self.determine_frozen_core(qm_elems)

        #Grab energy and gradient
        #TODO: No qm/MM yet. need to check if possible in MRCC
        #Note: for gradient and QM/MM it is best to keep_orientation=True in write_mrcc_input

        if Grad==True:
            write_mrcc_input(self.mrccinput,charge,mult,qm_elems,current_coords,numcores,Grad=True, keep_orientation=self.keep_orientation,
                             PC_coords=current_MM_coords, PC_charges=MMcharges, frozen_core_option=self.frozencore_string, no_basis_read_orbs=self.no_basis_read_orbs)
            run_mrcc(self.mrccdir,self.filename+'.out',self.parallelization,numcores)
            self.energy=grab_energy_mrcc(self.filename+'.out')
            self.gradient = grab_gradient_mrcc(self.filename+'.out',len(qm_elems))

            #PC self energy
            if PC == True:
                pc_self_energy = grab_MRCC_PC_self_energy("mrcc_job.dat")
                print("PC self-energy:", pc_self_energy)
                self.energy = self.energy - pc_self_energy
            #Pointcharge-gradient
            self.pcgradient = grab_MRCC_pointcharge_gradient("mrcc_job.dat",MMcharges)

        else:
            write_mrcc_input(self.mrccinput,charge,mult,qm_elems,current_coords,numcores, keep_orientation=self.keep_orientation,
                             PC_coords=current_MM_coords, PC_charges=MMcharges, frozen_core_option=self.frozencore_string, no_basis_read_orbs=self.no_basis_read_orbs)
            run_mrcc(self.mrccdir,self.filename+'.out',self.parallelization,numcores)
            self.energy=grab_energy_mrcc(self.filename+'.out')

            #PC self energy
            if PC == True:
                pc_self_energy = grab_MRCC_PC_self_energy("mrcc_job.dat")
                print("PC self-energy:", pc_self_energy)
                self.energy = self.energy - pc_self_energy

        #TODO: write in error handling here
        print(BC.OKBLUE, BC.BOLD, "------------ENDING MRCC INTERFACE-------------", BC.END)
        if Grad == True:
            print("Single-point MRCC energy:", self.energy)
            print("MRCC gradient:", self.gradient)
            print_time_rel(module_init_time, modulename='MRCC run', moduleindex=2)
            if PC is True:
                return self.energy, self.gradient, self.pcgradient
            else:
                return self.energy, self.gradient
        else:
            print("Single-point MRCC energy:", self.energy)
            print_time_rel(module_init_time, modulename='MRCC run', moduleindex=2)
            return self.energy

def run_mrcc(mrccdir,filename,parallelization,numcores):
    with open(filename, 'w') as ofile:
        #process = sp.run([mrccdir + '/dmrcc'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

        if parallelization == 'OMP':
            print(f"OMP parallelization is active. Using OMP_NUM_THREADS={numcores}")
            os.environ['OMP_NUM_THREADS'] = str(numcores)
            os.environ['MKL_NUM_THREADS'] = str(1)
            process = sp.run([mrccdir + '/dmrcc'], env=os.environ, check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
        elif parallelization == 'OMP-and-MKL':
            print(f"OMP-and-MKL parallelization is active. Both OMP_NUM_THREADS and MKL_NUM_THREADS set to: {numcores}")
            os.environ['OMP_NUM_THREADS'] = str(numcores)
            os.environ['MKL_NUM_THREADS'] = str(numcores)
            process = sp.run([mrccdir + '/dmrcc'], env=os.environ, check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
        elif parallelization == 'MPI':
            print(f"MPI parallelization active. Will use {numcores} MPI processes. (OMP and MKL disabled)")
            os.environ['MKL_NUM_THREADS'] = str(1)
            os.environ['OMP_NUM_THREADS'] = str(1)
            process = sp.run([mrccdir + '/dmrcc'], env=os.environ, check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

#TODO: Gradient option
#NOTE: Now setting ccsdthreads and ptthreads to number of cores
def write_mrcc_input(mrccinput,charge,mult,elems,coords,numcores,Grad=False,keep_orientation=False, PC_coords=None,PC_charges=None,
                     frozen_core_option=None, no_basis_read_orbs=True):
    print("Writing MRCC inputfile")

    with open("MINP", 'w') as inpfile:


        # For case where no basis is defined, assumed that fort.55 and fort.56 files have been created (contaning integrals)
        if no_basis_read_orbs is True:
            print("Warning: no_basis_read_orbs is True. Adding iface=cfour option, means that integrals will be read from fort.55 and fort.56 files")
            inpfile.write("iface=cfour") #Activates CFour interface (just means that MRCC will read in fort.55 and fort.56 files)

        for m in mrccinput.split('\n'):
            if 'core' in m:
                print("Warning: ignoring user-defined core option. Using frozen_core_option instead")
            else:
                inpfile.write(str(m)+'\n')
        #inpfile.write(mrccinput + '\n')
        #Frozen core
        inpfile.write(f'core={frozen_core_option}\n')
        inpfile.write(f'ccsdthreads={numcores}\n')
        inpfile.write(f'ptthreads={numcores}\n')
        inpfile.write('unit=angs\n')
        inpfile.write('charge={}\n'.format(charge))
        inpfile.write('mult={}\n'.format(mult))

        #Preserve orientation by this hack
        if keep_orientation is True:
            print("keep_orientation is True. Turning off symmetry and doing dummy QM/MM calculation to preserve orientation")
            inpfile.write('symm=off\n')
            inpfile.write('qmmm=Amber\n')
        else:
            print("keep_orientation is False. MRCC will reorient the molecule and use symmetry")

        #If Grad true set density to first-order. Gives properties and gradient
        if Grad is True:
            inpfile.write('dens=2\n')
            #dens=2 for RHF
            if "calc=RHF" in mrccinput:
                inpfile.write('dens=2\n')
            #dens=2 needed for pc grad
            elif PC_charges != None:
                inpfile.write('dens=2\n')
            else:
                inpfile.write('dens=1\n')
        inpfile.write('geom=xyz\n')
        inpfile.write('{}\n'.format(len(elems)))
        inpfile.write('\n')
        for el,c in zip(elems,coords):
            inpfile.write('{}   {} {} {}\n'.format(el,c[0],c[1],c[2]))
        inpfile.write('\n')
        #Pointcharges for QM/MM

        #Or dummy PC for orientation
        if keep_orientation is True:
            inpfile.write('pointcharges\n')
            #Write PC charges and coords if given
            if PC_coords != None:
                inpfile.write(f'{len(PC_charges)}\n')
                for charge,coord in zip(PC_charges,PC_coords):
                    inpfile.write(f'{coord[0]} {coord[1]} {coord[2]} {charge}\n')
            #Else write 0
            else:
                inpfile.write('0\n')

def grab_energy_mrcc(outfile):
    #Option 1. Grabbing all lines containing energy in outputfile. Take last entry.
    # CURRENT Option 2: grab energy from iface file. Higher level WF entry should be last
    with open("iface") as f:
        for line in f:
            if 'ENERGY' in line:
                energy=float(line.split()[5])
    return energy


def grab_gradient_mrcc(file,numatoms):
    grab=False
    grab2=False
    atomcount=0
    gradient=np.zeros((numatoms,3))
    with open(file) as f:
        for line in f:
            if grab is True or grab2 is True:
                if '*******' in line:
                    grab=False
                    grab2=False
                if len(line.split())==5:
                    gradient[atomcount,0] = float(line.split()[-3])
                    gradient[atomcount,1] = float(line.split()[-2])
                    gradient[atomcount,2] = float(line.split()[-1])
                    atomcount+=1
            if ' Molecular gradient [au]:' in line:
                grab=True
            if ' Cartesian gradient [au]:' in line:
                grab2=True
    return gradient


#MRCC HLC correction on fragment. Either provide MRCCTheory object or use default settings
# Calculates HLC - CCSD(T) correction, e.g. CCSDT - CCSD(T) energy
#Either use fragment or provide coords and elems
def run_MRCC_HLC_correction(coords=None, elems=None, fragment=None, charge=None, mult=None, theory=None, method='CCSDT', basis='TZ',
                            ref='RHF', openshell=False, numcores=1):
    init_time=time.time()
    if fragment is None:
        fragment = ash.Fragment(coords=coords, elems=elems, charge=charge,mult=mult)
    if openshell is True:
        ref='UHF'
    print("\nNow running MRCC HLC correction")
    #MRCCTheory
    mrccinput_HL=f"""
    basis={MRCC_basis_dict[basis]}
    calc={method}
    scftype={ref}
    mem=9000MB
    scftype={ref}
    ccmaxit=150
    core=frozen
    """
    mrccinput_ccsd_t=f"""
    basis={MRCC_basis_dict[basis]}
    calc=CCSD(T)
    scftype={ref}
    mem=9000MB
    scftype={ref}
    ccmaxit=150
    core=frozen
    """
    if theory is None:
        #HL calculation
        theory_HL = MRCCTheory(mrccinput=mrccinput_HL, numcores=numcores, filename='MRCC_HLC_HL')
        print("Now running MRCC HLC calculation on fragment")
        result_HL = ash.Singlepoint(theory=theory_HL,fragment=fragment)

        #CCSD(T) calculation
        theory_ccsd_t = MRCCTheory(mrccinput=mrccinput_ccsd_t, numcores=numcores, filename='MRCC_HLC_ccsd_t')
        print("Changing method in MRCCTheory object to CCSD(T)")
        print("Now running MRCC CCSD(T) calculation on fragment")
        result_ccsd_t = ash.Singlepoint(theory=theory_ccsd_t,fragment=fragment)

        delta_corr = result_HL.energy - result_ccsd_t.energy

        print("High-level MRCC CCSD(T)-> Highlevel correction:", delta_corr, "au")
    else:
        #Running HL calculation provided
        theory.filename='MRCC_HLC_HL.out'
        print("Now running MRCC HLC calculation on fragment")
        result_big = ash.Singlepoint(theory=theory,fragment=fragment)

        #Changing method to CCSD(T)
        for i in theory.mrccinput.split():
            if 'calc=' in i:
                theory.mrccinput = theory.mrccinput.replace(i,"calc=CCSD(T)")
        theory.filename='MRCC_HLC_ccsd_t'
        print("Changing method in MRCCTheory object to CCSD(T)")
        print("Now running MRCC CCSD(T) calculation on fragment")
        result_ccsd_t = ash.Singlepoint(theory=theory,fragment=fragment)

        delta_corr = result_big.energy - result_ccsd_t.energy
        print("High-level MRCC correction:", delta_corr, "au")
    print_time_rel(init_time, modulename='run_MRCC_HLC_correction', moduleindex=2)
    return delta_corr


#Get MRCC pointcharge gradient via the electric field on pointcharges
def grab_MRCC_pointcharge_gradient(file,charges):
    num_charges=len(charges)
    pc_grad=np.zeros((num_charges,3))
    grab=False
    pccount=0
    with open(file) as f:
        for line in f:
            if grab is True:
                if len(line.split()) < 3:
                    grab=False
                if 'Magnitude of dipole mome' in line:
                    grab=False
                if len(line.split()) == 3:
                    #pcgradient is  -F = -q*E
                    charge = charges[pccount]
                    pc_grad[pccount,0] = -1*charge*float(line.split()[0])
                    pc_grad[pccount,1] = -1*charge*float(line.split()[1])
                    pc_grad[pccount,2] = -1*charge*float(line.split()[2])
                    pccount+=1
            if ' Electric field at MM atoms' in line:
                grab=True
    return pc_grad

#Get MRCC PC self energy
def grab_MRCC_PC_self_energy(file):
    grab=False
    with open(file) as f:
        for line in f:
            if grab is True:
                pc_self_energy=float(line.split()[-1])
                grab=False
            if 'Self energy of the point charges [AU]' in line:
                grab=True
    return pc_self_energy


#Function to create a correct correlated WF Molden file from a MRCC Molden file,
def convert_MRCC_Molden_file(mrccoutputfile=None, moldenfile=None, mrccdensityfile=None, multiwfndir=None, printlevel=2):
    print("convert_MRCC_Molden_file")

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

    if mrccoutputfile == None:
        print("MRCC outputfile should also be provided")
        ashexit()
    core_electrons = int(pygrep("Number of core electrons:",mrccoutputfile)[-1])
    print("Core electrons found in outputfile:", core_electrons)
    frozen_orbs = int(core_electrons/2)
    print("Frozen orbitals:", frozen_orbs)
    #Rename MRCC Molden file to mrcc.molden
    shutil.copy(moldenfile, "mrcc.molden")
    #Write Multiwfn input. Will new Moldenfile based on correlated density
    write_multiwfn_input_option(option="mrcc-density", frozenorbitals=frozen_orbs, densityfile=mrccdensityfile, printlevel=printlevel)
    print("Now calling Multiwfn to process the MRCC, Molden and CCDENSITIES files")
    with open("mwfnoptions") as input:
        sp.run([multiwfndir+'/Multiwfn', "mrcc.molden"], stdin=input)
    print("Multiwfn is done")
    if os.path.isfile("mrccnew.molden") is False:
        print("Error: Multiwfn failed to create new Molden file. Exiting.")
        ashexit()
    print("Created new Molden file: ")
    print("This file contains the natural orbitals of the correlated density from MRCC")


def grab_dipole_moment(outfile):
    dipole_moment = np.zeros(3)
    grab=False
    with open(outfile) as f:
        for line in f:
            if grab is True:
                if ' Dipole moment [Debye]:' in line:
                    grab=False
                if 'x=' in line:
                    dipole_moment[0] = float(line.split()[1])
                    dipole_moment[1] = float(line.split()[3])
                    dipole_moment[2] = float(line.split()[5])
            if ' Dipole moment [au]:' in line:
                grab=True
    return dipole_moment


# Write MRCC rudimentary inputfile (fort.56) file with list of occupations
#NOTE: For frozen-core calculations the occupations should only be for active electrons
def MRCC_write_basic_inputfile(occupations=None, filename="fort.56", scf_type="RHF",
                               ex_level=4, nsing=1, ntrip=0, rest=0, CC_CI=1, dens=0, CS=1,
                               spatial=1, HF=1, ndoub=0, nacto=0, nactv=0, tol=9, maxex=0,
                               sacc=0, freq=0.0000, symm=0, conver=0, diag=0, dboc=0, mem=1024):
    print("SCF_type:", scf_type)
    if scf_type == 'RHF':
        nsing=1
        ndoub=0
        CS=1
        spatial=1
        HF=1
    elif scf_type == 'ROHF':
        nsing=0
        ndoub=1
        CS=0
        spatial=1
        HF=0
    elif scf_type == 'UHF':
        nsing=0
        ndoub=1
        CS=0
        spatial=0
        HF=1

    occupation_string = ' '.join(str(x) for x in occupations)
    inputstring=f"""   {ex_level}    {nsing}    {ntrip}     {rest}    {CC_CI}    {dens}     {conver}     {symm}     {diag}    {CS}    {spatial}     {HF}      {ndoub}    {nacto}      {nactv}    {tol}      {maxex}     {sacc} {freq}     {dboc} {mem}
ex.lev, nsing, ntrip, rest, CC/CI, dens, conver, symm, diag, CS, spatial, HF, ndoub, nacto, nactv, tol, maxex, sacc, freq, dboc, mem
 {occupation_string}"""

    with open(filename, 'w') as f:
        f.write(inputstring)

    #Touch KEYWD file (otherwise MRCC crashes)
    sp.run(["touch", "KEYWD"])

