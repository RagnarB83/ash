from ash import *

numcores=1
actualcores=1

frag = Fragment(diatomic="CO", bondlength=1.1294, charge=0, mult=1)

eps_values=[4e-4,3e-4,2.5e-4,2e-4,1e-4,9e-5,8e-5,7e-5,6e-5,5e-5,4e-5,3e-5,2e-5,1e-5]

pyscf = PySCFTheory(basis="cc-pVDZ", numcores=actualcores, scf_type='RHF', conv_tol=1e-9,memory=50000)

#Dice
for eps in eps_values:
    dicecalc = DiceTheory(pyscftheoryobject=pyscf, numcores=actualcores, SHCI=True, memory=70000,
                SHCI_cas_nmin=1.999, SHCI_cas_nmax=0.0, SHCI_stochastic=True, SHCI_PTiter=400, SHCI_sweep_iter= [0,3,6],
                SHCI_sweep_epsilon = [ 4*eps,2*eps,eps ], SHCI_davidsonTol=1e-8, SHCI_epsilon2=1e-8, SHCI_epsilon2Large=1e-5, SHCI_macroiter=0,
                initial_orbitals='CCSD(T)',SHCI_DoRDM=True)
    #Now calling Singlepoint_reaction with moreadfiles option
    result = Singlepoint(fragment=frag, theory=dicecalc)
    os.rename("SHCI_Final_nat_orbs.molden", f"SHCI_natorbs_eps_{eps}.molden")
