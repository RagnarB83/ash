import numpy as np

from KNARRatom.atom import Atom
from KNARRjobs.utilities import PathLinearInterpol
from KNARRio.output_print import PrintConfigurationPath
from KNARRio.io import WritePath


# Author: Vilhjalmur Asgeirsson, 2019

class Path(Atom):

    # Child-class to Atom
    def __init__(self, name="unknown_path", nim=6, config1=None,
                 config2=None, insertion=None,
                 cell=None, pbc=None):

        self.name = name
        self.nim = nim
        self.config1 = config1
        self.config2 = config2
        self.insertion = insertion
        self.output = None

        self.coords = None
        self.moveable = None
        self.ndim = None
        self.ndof = None
        self.ndofIm = None
        self.ndimIm = None
        self.symbols = None
        self.constraints = None
        self.dkappa = None

        self.energy = None
        self.forces = None
        self.forcecalls = 0

        self.pbc = pbc
        self.cell = cell

        self.twodee = False
        self.setup = False
        self.ischain = True

    # ==================================================================
    # Function methods
    # ==================================================================
    def LinearInterpolate(self):

        if self.ndim is None or self.ndimIm is None or self.nim is None:
            raise TypeError("Please initialize the path before performing interpolation")

        if type(self.GetConfig1()) is not np.ndarray:
            raise TypeError("Expecting numpy array")

        if type(self.GetConfig2()) is not np.ndarray:
            raise TypeError("Expecting numpy array")

        if len(self.GetConfig2()) != len(self.GetConfig1()):
            raise ValueError("Dimension mismatch")

        rp = PathLinearInterpol(self.GetNDimIm(), self.nim,
                                self.GetConfig1(), self.GetConfig2(),
                                self.GetPBC(), self.GetCell())

        self.SetCoords(rp)
        self.MIC()
        self.setup = True
        return

    def MinRMSD(self):

        from KNARRatom.utilities import MinimizeRotation
        if self.IsConstrained():
            return
        ndim = self.GetNDimIm()
        nim = self.GetNim()
        newpath = np.zeros(shape=(ndim * nim, 1))
        for i in range(1, nim):
            target_coords = self.GetCoords()[(i - 1) * ndim:i * ndim].copy()
            prod_coords = self.GetCoords()[i * ndim:(i + 1) * ndim].copy()
            prod_coords, target_coords = MinimizeRotation(ndim, target_coords,
                                                          prod_coords)
            if i == 1:
                newpath[(i - 1) * ndim:i * ndim] = target_coords

            newpath[i * ndim:(i + 1) * ndim] = prod_coords

        self.SetCoords(newpath)
        return

    def PrintPath(self, header=None):
        PrintConfigurationPath(header, self.GetNDim(), self.GetNDimIm(), self.GetNim(), self.GetNDof(),
                               self.GetCoords(), self.GetConstraints(), self.GetSymbols(),
                               self.GetCell(), self.GetPBC())
        return None

    def WritePath(self, fname):
        WritePath(fname, self.GetNDimIm(), self.GetNim(), self.GetCoords(),
                  self.GetSymbols(), self.GetEnergy())
        return None

    # ==================================================================
    # "Update and add" methods
    # ==================================================================

    # ==================================================================
    # "SET" methods
    # ==================================================================

    def SetNim(self,x):
        assert x > 0
        self.nim = x

    def SetNDimIm(self, x):
        try:
            self.ndimIm = int(x)
        except:
            raise TypeError("Expecting type int")
        return

    def SetNDofIm(self, x):
        try:
            self.ndofIm = int(x)
        except:
            raise TypeError("Expecting type int")
        return
    def SetNDof(self, x):
        self.ndof = x

    def SetNDim(self, x):
        try:
            self.ndim = int(x)
        except:
            raise TypeError("Expecting type int")
        return


    def Setdkappa(self, x):
        assert type(x) is np.ndarray
        self.dkappa = x.copy()
        return

    def SetEnergy(self, energy, x=None):
        if x is None:
            self.SetOldEnergy()
            assert type(energy) == np.ndarray
            if self.energy is not None:
                assert len(energy) == len(self.GetEnergy())
            self.energy = energy
            return
        else:
            assert x >= 0
            assert x < self.GetNim()

            self.SetOldEnergy(x)
            self.energy[x] = energy
            return

    def SetOldEnergy(self, x=None):
        if x is None:
            if self.energy is not None:
                self.energy0 = self.energy.copy()
        else:
            if self.energy is not None:
                self.energy0[x] = self.energy[x]
        return

    def SetConfig1(self, x):
        if len(x) != self.GetNDimIm():
            raise ValueError("Dimension mismatch")

        if type(x) is not np.ndarray:
            raise TypeError("numpy array expected")

        self.config1 = x
        return

    def SetConfig2(self, x):
        if len(x) != self.GetNDimIm():
            raise ValueError("Dimension mismatch")

        if type(x) is not np.ndarray:
            raise TypeError("numpy array expected")

        self.config2 = x
        return

    def SetInsertionConfig(self, x):
        if len(x) != self.GetNDimIm():
            raise ValueError("Dimension mismatch")

        if type(x) is not np.ndarray:
            raise TypeError("numpy array expected")

        self.insertion = x
        return

    # ==================================================================
    # "GET" methods
    # ==================================================================

    def GetNDimIm(self):
        return self.ndimIm

    def GetNDofIm(self):
        return self.ndofIm

    def GetNDim(self):
        return self.ndim

    def GetInsertionConfig(self):
        return self.insertion

    def Getdkappa(self):
        return self.dkappa

    def GetConfig1(self):
        return self.config1

    def GetConfig2(self):
        return self.config2

    def GetEnergy(self, x=None):
        if x is None:
            return self.energy
        else:
            assert x > 0
            assert x < self.GetNim()
            return self.energy[x]
