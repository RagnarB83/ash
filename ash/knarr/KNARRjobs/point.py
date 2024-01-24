import time
import KNARRsettings

from KNARRio.system_print import PrintJob, PrintCallBack, PrintJobDone
from KNARRio.io import WriteForcesFile, WriteEnergyFile


# Author: Vilhjalmur Asgeirsson, 2019.

def DoPoint(atoms, calculator):
    PrintJob('Single Point Energy and Gradient Computation')
    PrintCallBack('pointjob', calculator, atoms)

    if not atoms.setup:
        raise RuntimeError("Atoms object is not properly initialized")
    if not calculator.setup:
        raise RuntimeError("Calculator is not properly initialized")

    atoms.PrintConfiguration('Input configuration:')

    basename = atoms.GetOutputFile()
    start_t = time.time()

    # Perform the actual computation
    calculator.Compute(atoms)

    print("Potential energy   : %6.6f %s\n" % (atoms.GetEnergy(), KNARRsettings.energystring))

    WriteEnergyFile(basename + '.energy', atoms.GetEnergy())
    WriteForcesFile(basename + '.forces', atoms.GetNDim(), atoms.GetSymbols(), atoms.GetForces())

    # Execution done
    PrintJobDone('Point job', time.time() - start_t)

    return atoms.GetEnergy(), atoms.GetForces()
