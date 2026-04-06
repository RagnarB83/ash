from ash.functions.functions_general import ashexit, print_line_with_mainheader
from ash.modules.module_singlepoint import Singlepoint
from ash.constants import hartoeV
import numpy as np
import copy

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


def FSM(reactant=None, product=None, theory=None, method="L-BFGS-B", optcoords="cart",
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
        except:
            print("Error import ase. Check if installed")
            ashexit()

        # ASH Fragments (or ASE atoms)
        #self.reactant=reactant
        #self.product
        self.reactant_ase = ase.atoms.Atoms(reactant.elems,positions=reactant.coords)
        self.product_ase = ase.atoms.Atoms(product.elems,positions=product.coords)

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
