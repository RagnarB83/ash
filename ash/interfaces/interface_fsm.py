from ash.functions.functions_general import ashexit, print_line_with_mainheader
from ash.modules.module_singlepoint import Singlepoint
from ash.modules.module_coords import Fragment
from ash.constants import hartoeV
import numpy as np
import copy

# Simpler ASH-ASE calculator
class ASH_ASE_calculator:
    def __init__(self, theory=None, fragment=None):
        self.theory = theory
        # Used for elems, charge and mult
        self.fragment = fragment
        self.calls=0
    def get_potential_energy(self, atomsobj):
        print("Called ASHcalc get_potential_energy")
        self.calls+=1
        print(atomsobj)
        #Copy ASE coords into ASH fragment
        coords = copy.copy(atomsobj.positions)
        energy,gradient = self.theory.run(current_coords=coords, elems=self.fragment.elems, 
                                 charge=self.fragment.charge, mult=self.fragment.mult, Grad=True)
        self.potenergy = energy*hartoeV
        self.forces = -gradient*51.4220674763
        return self.potenergy

    def get_forces(self, atomsobj):
        print("Called ASHcalc get_forces")
        return self.forces


def FSM(reactant=None, product=None, theory=None, method="L-BFGS-B", optcoords="ric",
        nnodes_min=10, interp="lst", ninterp=100, stepsize=0.0, interpolate=False, maxiter=1, maxls=3, dmax=0.3, outdir=".", verbose=True):
    fsm = FreezingString_class(reactant=reactant, product=product, theory=theory, method=method, optcoords=optcoords,
                                nnodes_min=nnodes_min, interp=interp, ninterp=ninterp, stepsize=stepsize,
                                interpolate=interpolate, maxiter=maxiter, maxls=maxls, dmax=dmax, outdir=outdir, verbose=verbose)
    fsm.run()


class FreezingString_class:

    def __init__(self,reactant=None, product=None, theory=None, method="L-BFGS-B", optcoords="cart",
                 nnodes_min=10, interp="lst", ninterp=100, stepsize=0.0, interpolate=False,
                 maxiter=1, maxls=3, dmax=0.3, outdir=".", verbose=True):
        print_line_with_mainheader("Freezing String calculation initialized")
        try:
            import ase
            from mlfsm.geom import project_trans_rot
        except:
            print("Error import ase or mlfsm. Check if installed")
            ashexit()

        # ASH Fragments (or ASE atoms)
        #self.reactant=reactant
        #self.product
        self.elems= reactant.elems
        self.reactant_ase = ase.atoms.Atoms(reactant.elems,positions=reactant.coords)
        self.product_ase = ase.atoms.Atoms(product.elems,positions=product.coords)

        # Align product to reactant structure
        _, aligned_product = project_trans_rot(self.reactant_ase.get_positions(), self.product_ase.get_positions())
        self.product_ase.set_positions(aligned_product.reshape(-1, 3))


        self.reactant_ase.info.update({"charge": reactant.charge, "spin": reactant.mult})
        self.product_ase.info.update({"charge": product.charge, "spin": product.mult})

        # Theory as ASE
        self.calc = ASH_ASE_calculator(fragment=reactant, theory=theory)
        self.method = method
        self.nnodes_min = nnodes_min
        self.interp = interp
        self.ninterp = ninterp
        self.stepsize = stepsize
        self.interpolate = interpolate
        self.maxiter = maxiter
        self.maxls = maxls
        self.dmax = dmax
        self.optcoords = optcoords
        self.outdir = outdir
        self.verbose=verbose

    def run(self):
        print_line_with_mainheader("Freezing String run")
        try:
            from mlfsm.cos import FreezingString
            from mlfsm.opt import CartesianOptimizer, InternalsOptimizer

        except:
            print("Error import mlfsm. Check if installed")
            ashexit()


        # Initialize FSM string
        #, stepsize=self.stepsize
        print("Creating Freezing String object")
        string = FreezingString(self.reactant_ase, self.product_ase, nnodes_min=self.nnodes_min, 
                                interp_method=self.interp, ninterp=self.ninterp)

        if self.interpolate:
            print("Interpolating...")
            string.interpolate(self.outdir)
            return
        #import mlfsm
        #optimizer: mlfsm.Optimizer
        # Choose optimizer
        if self.optcoords == "cart":
            print("Coordinates: Cartesian")
            optimizer = CartesianOptimizer(self.calc, self.method, self.maxiter, self.maxls, self.dmax)
        elif self.optcoords == "ric":
            print("Coordinates: Internal")
            optimizer = InternalsOptimizer(self.calc, self.method, self.maxiter, self.maxls, self.dmax)
        else:
            raise ValueError("Check optimizer coordinates")
        print("Starting FSM optimization")
        
        # Run FSM
        while string.growing:
            string.grow()
            string.optimize(optimizer)
            string.write(self.outdir)

        print(f"Gradient calls: {string.ngrad}")

        # Grab paths and energies
        all_atoms = string.r_string + string.p_string[::-1]
        all_tot_energies = np.array(string.r_energy + string.p_energy[::-1])
        all_rel_energies = all_tot_energies - min(all_tot_energies)
        ts_idx = all_rel_energies.argmax()
        print("ts_idx:", ts_idx)
        print("TS atom positions:", all_atoms[ts_idx].get_positions())
        print(("TS energy (eV):", all_tot_energies[ts_idx]))
        ts_atoms = all_atoms[ts_idx]

        SP = Fragment(elems=self.elems, coords=ts_atoms.get_positions(), 
                            charge=self.reactant_ase.info["charge"], 
                            mult=self.reactant_ase.info["spin"])
        SP.write_xyzfile(xyzfilename=f"TS_guess.xyz")
        SP.print_coords()
        SP.set_energy(all_tot_energies[ts_idx]/hartoeV)
        print(f"TS guess energy : {SP.energy} Eh")
        exit()


        print("Attempting to plot FSM path")
        path = [structure.get_positions() for structure in all_atoms]
        from mlfsm.geom import calculate_arc_length
        s = calculate_arc_length(np.array(path))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(s, all_energies, label="FSM Path")
        ax.scatter(s[ts_idx], all_energies[ts_idx], color="red", label="TS Guess")
        ax.scatter(s[0], all_energies[0], color="black", label="Reactant/Product")
        ax.scatter(s[-1], all_energies[-1], color="black")
        ax.set_xlabel("Arclength (Å)")
        ax.set_ylabel("Energy (eV)")
        _ = ax.legend()

        # Save figure
        fig.savefig(f"{self.outdir}/FSM_path.png", dpi=300)

        #except:
        #   print("Error importing matplotlib. Check if installed")