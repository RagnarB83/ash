import numpy as np


# Author: Vilhjalmur Asgeirsson, 2019.

class Atom(object):

    def __init__(self, name="unknwn_obj", ndim=0, ndof=0, coords=None, symbols=None,
                 constraints=None, cell=None, pbc=None, twodee=False):
        self.name = name  # name

        self.ndim = ndim  # Number of dimensions
        self.ndof = ndof  # Number of active DOF
        self.nim = 1  # Number of images
        self.mass = [] #Mass of atoms
        self.coords = coords  # Full configuration
        self.forces = None  # Full atomic forces
        self.energy = None  # Energy
        self.energy0 = 0.0  # Energy of previous iteration

        self.symbols = symbols  # Chemical elementsXL
        self.pbc = pbc  # Periodic system?
        self.cell = cell  # Cell dimenions
        self.constraints = constraints  # Cartesian constraints
        self.moveable = None  # List of moveable atoms

        self.r = None  # Active configuration
        self.f = None  # Forces for active configuration
        self.h = None  # Hessian matrix for active configuration
        self.v = None  # velocity for active configuration
        self.v0 = None  # velocity of previous step for active configuration
        self.a = None  # Acceleration
        self.a0 = None

        self.twodee = twodee
        self.globaldof = 0
        self.forcecalls = 0  # Number of forcecalls executed on this atoms object
        self.setup = False
        self.output = "knarr"

    # ==================================================================
    # Function methods
    # ==================================================================

    def TranslateToCenter(self):
        center = self.GetCenter()
        for i in range(self.GetNDim()):
            self.coords[i] -= center[0]
            self.coords[i + 1] -= center[1]
            self.coords[i + 2] -= center[2]
        return

    def Rotate(self, target):
        from KNARRatom.utilities import MinimizeRotation
        if self.IsChain():
            raise RuntimeError("Rotate method can not be used on a chain")

        if self.IsConstrained():
            raise RuntimeError("Configurations are constrained - unable to perform rotation and translation")

        if not target.setup:
            raise RuntimeError("target structure has not been setup")

        if target.GetNDim() != self.GetNDim():
            raise RuntimeError("Dimension mismatch")

        atom_coords, target_coords = MinimizeRotation(self.GetNDim(), target.GetCoords(), self.GetCoords())
        self.SetCoords(atom_coords)
        target.SetCoords(target_coords)
        return

    def GetFreeMass(self):
        return self.GetMass()[self.GetMoveableAtoms()]
    
    def ComputeMass(self):
        ind = 0
        mass = np.zeros(shape=(self.ndim, 1))
        elem = ['a', 'h', 'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne', 'na', 'mg', 'al', 'si', 'p', 's', 'cl', 'ar',
                'k', 'ca', 'sc', 'ti', 'v', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br',
                'kr', 'rb', 'sr', 'y', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te',
                'i', 'xe', 'cs', 'ba', 'hf', 'ta', 'w', 're', 'os', 'ir', 'pt', 'au' 'hg', 'ti', 'pb', 'bi', 'po',
                'at', 'rn']
        atm = [1.0, 1.00790, 4.002, 6.94, 9.012, 10.810, 12.011, 14.0067, 15.999, 18.998, 20.1797, 22.989, 24.305,
               26.981, 28.085, 30.973, 32.06, 35.45, 39.948, 39.0983, 40.078, 44.955, 47.867, 50.9415, 51.9961, 54.938,
               55.845, 58.933, 58.6934, 63.546, 65.38, 69.723, 72.63, 74.92160, 78.96, 79.904, 83.798, 85.4678, 87.62,
               88.90585, 91.224, 92.90638, 95.96, 98.0, 101.07, 102.90, 106.42, 107.8682, 112.411, 114.818, 118.710,
               121.760, 127.60, 126.90, 131.293, 132.91, 137.33, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08,
               196.97, 200.59, 204.30, 207.2, 208.98, 209.0, 210.0, 222.0]
        for i in self.symbols:
            for j in range(0, len(elem)):
                i = i.strip()
                if i.upper() == elem[j].upper():
                    mass[ind] = atm[j]
                    ind += 1
        return mass

    def ComputeA(self):
        assert self.GetF() is not None
        mass = self.GetM()
        forces=self.GetF()
        assert len(mass) == len(self.GetF())
        acc = np.divide(self.GetF(), mass)
        self.SetA(acc)
        return

    def MIC(self):
        from KNARRatom.utilities import MIC
        if self.pbc is None:
            raise RuntimeError("PBC have not been determined")
        if not self.pbc:
            return
        if self.cell is None:
            raise RuntimeError("No cell has been added to atoms object")

        x = MIC(self.GetNDim(), self.GetCoords(), self.GetPBC(), self.GetCell())
        self.SetCoords(x)
        return

    def ReadAtomsFromFile(self, fname):
        import os
        from KNARRio.io import ReadXYZ, ReadCon, ReadXYZF, ReadCellFile, ReadConstraintsFile
        basename = fname.split('.')[0]
        extension = (fname.split('.')[-1]).upper()
        cell = None
        cnstr = None

        if extension == 'CON':
            rxyz, ndim, symbols, cell, cnstr = ReadCon(fname)
        elif extension == 'XYZ':
            rxyz, ndim, symbols = ReadXYZ(fname)
        elif extension == 'XYZF':
            rxyz, fxyz, energy, ndim, symbols = ReadXYZF(fname)
        else:
            raise IOError("File extension: %s has not been implemented", extension)

        self.setup = True
        self.ndim = ndim
        self.SetCoords(rxyz)
        self.SetSymbols(symbols)
        self.SetMass()


        constr = np.zeros(shape=(self.ndim, 1))
        if extension == 'XYZ' or extension == 'XYFZ':
            fname_constraints = basename + '.constraints'
            if not os.path.isfile(fname_constraints):
                print('**Warning: .constraints file not found! All atoms assumed to be moveable')
            else:
                constr = ReadConstraintsFile(fname_constraints)

            if self.GetPBC():
                print('BLOCK BLOCK')
                fname_cell = basename + '.cell'
                if not os.path.isfile(fname_cell):
                    print('**Warning: .cell file not found! Unit cell dimensions set to zero.')
                    cell = [0.0, 0.0, 0.0]
                else:
                    cell = ReadCellFile(basename + '.cell')
                self.SetCell(cell)

        elif extension == 'CON':
            if cell is not None:
                self.SetCell(cell)

            if cnstr is not None:
                constr = cnstr

        # Make sure constraints are correct
        for i, val in enumerate(constr):
            if val > 0:
                constr[i] = 1
            else:
                constr[i] = 0

        self.SetConstraints(constr)

        # Get ndof and moveable atoms
        self.ndof = self.ndim - int(constr.sum())
        self.SetMoveableAtoms()

        if extension == 'XYZF':
            self.SetForces(fxyz)
            self.SetEnergy(energy)

        return None

    def PrintConfiguration(self, header=None):
        from KNARRio.output_print import PrintConfiguration
        PrintConfiguration(header, self.ndim, self.ndof, self.coords,
                           self.constraints, self.symbols, self.cell, self.pbc)

        return None

    def Cleanse(self):
        self.nim = None
        self.ndim = None
        self.coords = None
        self.forces = None
        self.r = None
        self.f = None
        self.energy = None
        self.energy0 = None
        self.mass = []

    # ==================================================================
    # "IS" methods
    # ==================================================================

    def IsTwoDee(self):
        return self.twodee

    def IsChain(self):
        return False

    def IsConstrained(self):
        return np.sum(self.constraints) != 0

    # ==================================================================
    # "Update and add" methods
    # ==================================================================

    def TakeStep(self, timestep):
        dr = self.GetR() + self.GetV() * timestep + 0.5 * self.GetA() * timestep ** 2
        self.SetR(dr)
        return

    def UpdateCoords(self):
        self.coords[self.GetMoveableAtoms()] = self.GetR().copy()
        return

    def UpdateR(self):
        listi = self.GetMoveableAtoms()
        self.r = self.coords[listi].copy()
        return

    def UpdateF(self):
        listi = self.GetMoveableAtoms()
        self.f = self.forces[listi].copy()
        return

    def UpdateV(self, timestep):
        self.SetOldV(self.GetV())
        self.v = self.v + 0.5 * (self.GetA() + self.GetOldA()) * timestep
        return

    def UpdateA(self):
        self.SetOldA(self.GetA())
        self.a = np.divide(self.GetF(), self.GetM())
        return

    def AddFC(self, x=1):
        self.forcecalls += x
        return

    def ZeroV(self):
        self.v = np.zeros(shape=(self.GetNDof(), 1))
        return

    def ZeroA(self):
        self.a = np.zeros(shape=(self.GetNDof(), 1))
        return

    def ZeroF(self):
        self.f = np.zeros(shape=(self.GetNDof(), 1))

    # ==================================================================
    # "SET" methods
    # ==================================================================

    def SetMass(self):
        mass = self.ComputeMass()
        self.mass = mass
        return

    def SetTwoDee(self, x):
        try:
            self.twodee = bool(x)
        except:
            raise TypeError("Expting boolean")
        return

    def SetOutputFile(self, string):
        self.output = string
        return

    def SetNDim(self, x):
        try:
            self.ndim = int(x)
        except:
            raise TypeError("Expecting type int")
        return

    def SetNDof(self, x):
        try:
            self.ndim = int(x)
        except:
            raise TypeError("Expecting type int")
        return

    def SetCoords(self, x):

        if len(x) != self.ndim:
            raise RuntimeError("Dimension mismatch")

        if type(x) is not np.ndarray:
            raise TypeError("numpy array expected")

        self.coords = x.copy()
        return None

    def SetR(self, x):

        if len(x) != self.ndof:
            raise RuntimeError("Dimension mismatch")
        self.r = x.copy()
        return None

    def SetOldEnergy(self):
        if self.GetEnergy() is not None:
            if type(self.energy) is float:
                self.energy0 = self.energy
            elif type(self.energy) is np.ndarray:
                self.energy0 = self.energy.copy()
        return

    def SetEnergy(self, energy):
        self.SetOldEnergy()
        try:
            self.energy = float(energy)
        except:
            raise TypeError("SetEnergy requires float")

        return None

    def SetForces(self, x):
        if len(x) != self.ndim:
            raise RuntimeError("Dimension mismatch in SetForces")

        if type(x) is not np.ndarray:
            raise TypeError("SetForces: numpy array needed")

        self.forces = x.copy()
        return None

    def SetF(self, x):
        self.f = x.copy()
        return None

    def SetV(self, x):
        if self.GetNDof() != len(x):
            raise RuntimeError("Dimension mismatch in velocity")
        self.SetOldV(self.GetV().copy())
        self.v = x.copy()
        return

    def SetOldV(self, x):
        if self.GetNDof() != len(x):
            raise RuntimeError("Dimension mismatch in velocity")
        self.v0 = x.copy()
        return

    def SetA(self, x):
        if self.GetNDof() != len(x):
            raise RuntimeError("Dimension mismatch in acceleration")
        self.a = x.copy()
        return

    def SetOldA(self, x):
        if self.GetNDof() != len(x):
            raise RuntimeError("Dimension mismatch in velocity")
        self.a0 = x.copy()
        return

    def SetHessian(self, H):

        if type(H) is not np.ndarray:
            raise TypeError("SetHessian: numpy array needed")

        a, b, c = np.shape(H)

        if c != self.nim:
            raise RuntimeError("Wrong number of Hessian matrices")

        if a != b:
            raise TypeError("Wrong dimensions of H matrix")

        self.h = H.copy()
        return

    def SetGlobalDof(self, x):

        if (x == 0 or x == 5 or x == 6):
            self.globaldof = x
        else:
            raise ValueError("Incorrect number of global degrees of freedom")
        return

    def SetSymbols(self, symbols):
        print("symbols:", symbols)
        print("self.ndim:", self.ndim)
        if len(symbols) != self.ndim:
            raise RuntimeError("Dimension mismatch in SetSymbols")

        self.symbols = symbols

        return None

    def SetCell(self, x):
        if len(x) != 3:
            raise TypeError("Dimension mismatch in SetCell, list of size 3 required.")
        self.cell = x
        return

    def SetConstraints(self, x):
        print("x:", x)
        print("len x", len(x))
        print("self.ndim:", self.ndim)
        if len(x) != self.ndim:
            raise RuntimeError("Dimension mismatch")

        if type(x) is not np.ndarray:
            raise TypeError("Numpy array needed")
        self.constraints = x

        return

    def SetMoveableAtoms(self):
        if self.constraints is None:
            raise ValueError("Constraints have not been set")

        self.moveable = np.where(self.constraints == 0)[0]
        return

    # ==================================================================
    # "GET" methods
    # ==================================================================


    def GetM(self):
        return self.mass[self.GetMoveableAtoms()]

    def GetMass(self):
        return self.mass

    def GetOutputFile(self):
        return self.output

    def GetNDim(self):
        return self.ndim

    def GetNim(self):
        return self.nim

    def GetNDimIm(self):
        return self.ndim

    def GetCoords(self):
        return self.coords

    def GetR(self):
        if self.r is None:
            raise RuntimeError("Active coordinates have not been set")
        return self.r

    def GetEnergy(self):
        return self.energy

    def GetOldEnergy(self):
        return self.energy0

    def GetCenter(self):
        from KNARRatom.utilities import GetCentroid
        return GetCentroid(self.GetNDim(), self.GetCoords())

    def GetForces(self):
        return self.forces

    def GetF(self):
        if self.f is None:
            raise RuntimeError("Active coordinates have not been set")
        return self.f

    def GetV(self):
        return self.v

    def GetOldV(self):
        return self.v0

    def GetA(self):
        return self.a

    def GetOldA(self):
        return self.a0

    def GetGlobalDof(self):
        return self.globaldof

    def GetHessian(self):
        return self.h

    def GetFC(self):
        return self.forcecalls

    def GetSymbols(self):
        return self.symbols

    def GetPBC(self):
        return self.pbc

    def GetCell(self):
        return self.cell

    def GetNDof(self):
        return self.ndof

    def GetConstraints(self):
        return self.constraints

    def GetNim(self):
        return self.nim

    def GetMoveableAtoms(self):
        return self.moveable

    def GetMoveableAtomsInList(self):
        return list(self.moveable)

    def GetInfo(self):
        listi = [self.name, self.ndim, self.ndim, self.ndof, self.nim, self.coords, self.forces, self.energy,
                 self.symbols, self.pbc, self.cell, self.constraints, self.moveable, self.r, self.f, self.v,
                 self.forcecalls]

        return listi
