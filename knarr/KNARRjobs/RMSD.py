import time
import KNARRsettings

from KNARRio.system_print import PrintJob, PrintCallBack, PrintJobDone
from KNARRio.io import WriteXYZ, WriteCon
from KNARRatom.utilities import MinimizeRotation, RMS3


# Author: Vilhjalmur Asgeirsson, 2019

def DoRMSD(atoms, product):
    PrintJob('Minimization of RMSD')
    PrintCallBack('RMSD', atoms)

    if not atoms.setup:
        raise RuntimeError("Reactant atoms object is not properly initialized")

    if not product.setup:
        raise RuntimeError("Product atoms object is not properly initialized")

    # Check reactant and product

    if atoms.GetSymbols() != product.GetSymbols() or atoms.GetNDim() != product.GetNDim():
        raise RuntimeError("Input configurations do not match")

    if atoms.IsConstrained() or product.IsConstrained():
        raise RuntimeError("Configurations are constrained - unable to perform rotation and translation")

    if atoms.GetPBC() or product.GetPBC():
        print('Periodic boundary conditions are activated. Are you sure you want to '
              'rotate and translate these structures?')

    start_t = time.time()
    basename = atoms.GetOutputFile()

    # =======================================
    # Minimize RMSD
    # =======================================

    rmsdbefore = RMS3(atoms.GetNDim(), atoms.GetCoords() - product.GetCoords())
    atom_coords, prod_coords = MinimizeRotation(atoms.GetNDim(), atoms.GetCoords(),
                                                product.GetCoords(), fixcenter=True)
    product.SetCoords(prod_coords)
    atoms.SetCoords(atom_coords)
    rmsdafter = RMS3(atoms.GetNDim(), atoms.GetCoords() - product.GetCoords())

    print('RMSD: %6.3f -> %6.3f %s' % (rmsdbefore, rmsdafter, KNARRsettings.lengthstring))

    # =======================================
    # Write output
    # =======================================

    if KNARRsettings.extension == 0:
        WriteXYZ(basename + "_reactant.xyz", atoms.GetNDim(), atoms.GetCoords(), atoms.GetSymbols(),
                 energy=atoms.GetEnergy())
        WriteXYZ(basename + "_product.xyz", product.GetNDim(), product.GetCoords(), product.GetSymbols(),
                 energy=product.GetEnergy())

    else:
        WriteCon(basename + "_reactant.con", 1, atoms.GetCoords(), atoms.GetSymbols(),
                 atoms.GetCell(), atoms.GetConstraints())
        WriteCon(basename + "_product.con", 1, product.GetCoords(), product.GetSymbols(),
                 product.GetCell(), product.GetConstraints())

    # Execution done
    PrintJobDone('RMSD job', time.time() - start_t)

    return atoms.GetCoords(), product.GetCoords()
