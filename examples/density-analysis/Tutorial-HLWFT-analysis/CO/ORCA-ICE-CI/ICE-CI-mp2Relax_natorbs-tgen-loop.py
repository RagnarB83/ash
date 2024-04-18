from ash import *

numcores=4

frag = Fragment(diatomic="CO", bondlength=1.1294, charge=0, mult=1)

#Settings
basis="cc-pVDZ"
nmin=1.999
nmax=0.00
initial_orbitals="RI-MP2"

#Call function
for tgen in [10,1,5e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6]:
    Auto_ICE_CAS(fragment=frag, basis=basis, nmin=nmin, nmax=nmax, numcores=numcores, CASCI=True, tgen=tgen, memory=10000,
        initial_orbitals=initial_orbitals, MP2_density='relaxed')
    #Grabbing dipole moment from each ICE ORCA outputfile
    dipole = pygrep("Total Dipole Moment", "orca.out")
    dipole_z = float(dipole[-1])
    print(f"ICE-CI tgen: {tgen} dipole Z (A.U.): {dipole_z}")
    #Make Moldenfile
    mfile = make_molden_file_ORCA('orca.gbw')
    os.rename("orca.molden", f"ICE_CI_mp2nat_tgen_{tgen}.molden")
