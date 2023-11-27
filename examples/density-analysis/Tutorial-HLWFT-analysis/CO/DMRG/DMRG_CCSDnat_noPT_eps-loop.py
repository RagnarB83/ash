from ash import *

numcores=3
actualcores=3

frag = Fragment(diatomic="CO", bondlength=1.1294, charge=0, mult=1)

pyscf = PySCFTheory(basis="cc-pVDZ", numcores=actualcores, scf_type='RHF', conv_tol=1e-9,memory=50000)

#Block2 DMRG settings
maxM=500
parmethod='OpenMP'
initial_orbitals='CCSD'
singlet_embedding=True

for maxM in [1,20,50,100,200,300,400,500,750,1000,1500,2000,3000]:
    blockcalc = BlockTheory(pyscftheoryobject=pyscf, cas_nmin=1.999, cas_nmax=0.0, macroiter=0, numcores=actualcores, memory=30000,
        initial_orbitals=initial_orbitals, block_parallelization=parmethod, maxM=maxM, singlet_embedding=singlet_embedding,
        DMRG_DoRDM=True)

    #Now calling Singlepoint_reaction with moreadfiles option
    result = Singlepoint(fragment=frag, theory=blockcalc)
    os.rename("DMRG_Final_nat_orbs.molden", f"DMRG_natorbs_maxM_{maxM}.molden")
