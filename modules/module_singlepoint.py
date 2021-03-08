"""
    Singlepoint module:

    Function Singlepoint

    class ZeroTheory
    """
import numpy as np

import ash
from functions_general import BC,blankline

#Single-point energy function
def Singlepoint(fragment=None, theory=None, Grad=False):
    """Singlepoint function: runs a single-point energy calculation using ASH theory and ASH fragment.

    Args:
        fragment (ASH frgment, optional): An ASH fragment. Defaults to None.
        theory (ASH theory, optional): Any valid ASH theory. Defaults to None.
        Grad (bool, optional): Do gradient or not Defaults to False.

    Returns:
        float: Energy
        or
        float,np.array : Energy and gradient array
    """
    print("")
    if fragment is None or theory is None:
        print(BC.FAIL,"Singlepoint requires a fragment and a theory object",BC.END)
        exit(1)
    coords=fragment.coords
    elems=fragment.elems
    # Run a single-point energy job with gradient
    if Grad ==True:
        print(BC.WARNING,"Doing single-point Energy+Gradient job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
        # An Energy+Gradient calculation where we change the number of cores to 12
        energy,gradient= theory.run(current_coords=coords, elems=elems, Grad=True)
        print("Energy: ", energy)
        return energy,gradient
    # Run a single-point energy job without gradient (default)
    else:
        print(BC.WARNING,"Doing single-point Energy job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
        energy = theory.run(current_coords=coords, elems=elems)
        print("Energy: ", energy)
        #Now adding total energy to fragment
        fragment.energy=energy
        return energy

# Theory object that always gives zero energy and zero gradient. Useful for setting constraints
class ZeroTheory:
    def __init__(self, fragment=None, charge=None, mult=None, printlevel=None, nprocs=1, label=None):
        """Class Zerotheory: Simple dummy theory that gives zero energy and a zero-valued gradient array
            Note: includes unnecessary attributes for consistency.

        Args:
            fragment (ASH fragment, optional): A valid ASH fragment. Defaults to None.
            charge (int, optional): Charge. Defaults to None.
            mult (int, optional): Spin multiplicity. Defaults to None.
            printlevel (int, optional): Printlevel:0,1,2 or 3. Defaults to None.
            nprocs (int, optional): Number of cores. Defaults to 1.
            label (str, optional): String label. Defaults to None.
        """
        self.nprocs=nprocs
        self.charge=charge
        self.mult=mult
        self.printlevel=printlevel
        self.label=label
        self.fragment=fragment
        pass
    def run(self, current_coords=None, elems=None, Grad=False, PC=False, nprocs=None ):
        self.energy = 0.0
        #Gradient as np array 
        self.gradient = np.zeros((len(elems), 3))
        return self.energy,self.gradient