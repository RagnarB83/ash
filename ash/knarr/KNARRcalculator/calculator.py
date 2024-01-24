import os
import numpy as np

import KNARRsettings

from KNARRcalculator.utilities import GetProgramPath, ReadORCATemplateFile, CheckORCATemplateFile, ReadEONTemplateFile
from KNARRcalculator.xtb import XTB, XTBHess
from KNARRcalculator.eon import EON
from KNARRcalculator.orca import ORCA, ORCAHess
from KNARRcalculator.idpp import IDPP
#from KNARRcalculator.asepot import ASEPot
#from KNARRcalculator.cuh2 import EAM
from KNARRcalculator.mb import MBG, MB
from KNARRcalculator.peaks import Peaks
from KNARRcalculator.lepsho import LEPSHO, LEPSHOGauss
from KNARRcalculator.debug import Debug
from KNARRcalculator.bobdebug import BobDebug
from KNARRcalculator.henkelman  import Henkelman, Henkelman10D, Henkelman20D, Henkelman100D
from KNARRcalculator.lennardjones import LennardJones
from KNARRcalculator.henkelman_gauss import HenkelmanGaussBoosted


# Author: Vilhjalmur Asgeirsson, 2019.

class Calculator(object):

    def __init__(self, name=None, path=None, fd_step=0.0, template_file=None, charge=0, multiplicity=1, ncore=1):
        self.name = name  # User-known name for the calculator
        self.charge = charge
        self.multiplicity = multiplicity

        self.ncore = ncore  # Number of cores to perform the calculation on

        self.gradient = True  # does the calculator support analytical gradient computations?
        self.hessian = False  # does the calculator support analytical hessian computations?
        self.fd_step = fd_step  # finite difference step for numerical hessian and gradient

        self.template_file = template_file  # if template files go with the calculator
        self.template = None  # list containing the template file

        self.program = None  # Actual 'calculator' function
        self.programhess = None  # same as program but for Hessian

        self.twodee = False
        self.path_to_calculator = None  # Path to executables
        self.path = path
        self.pbc = False  # Does the calculator support PBC?
        self.setup = False

    # =================================================
    # Initializing the calculator - based on "name"
    # =================================================
    def Setup(self):

        if self.path is None:
            self.path = os.getcwd()

        # Setup the calculator based off "name"
        if self.name.upper() == "ORCA":
            self.path_to_calculator = GetProgramPath(self.name)
            self.program = ORCA
            self.programhess = ORCAHess
            self.gradient = True
            self.hessian = True
            self.pbc = False
            if self.template_file is None:
                raise RuntimeError("ORCA requires a template file")

            self.template = ReadORCATemplateFile(self.template_file)
            CheckORCATemplateFile(self.template)

        elif self.name.upper() == "XTB":
            self.path_to_calculator = GetProgramPath(self.name)
            self.program = XTB
            self.programhess = XTBHess
            self.gradient = True
            self.hessian = True
            self.pbc = False
            self.template_file = None
            self.template = None

        elif self.name.upper() == "ASEPOT":

            try:
                import ase
            except:
                raise ImportError("Unable to import ASE")

            self.path_to_calculator = os.getcwd()
            self.program = ASEPot
            self.programhess = None
            self.gradient = True
            self.hessian = False
            self.pbc = True
            self.template_file = None
            self.template = None

        elif self.name.upper() == "IDPP":
            self.path_to_calculator = os.getcwd()
            self.program = IDPP
            self.programhess = None
            self.gradient = True
            self.hessian = False
            self.pbc = True
            self.template_file = None
            self.template = None

        #elif self.name.upper() == "EAM":
            #self.path_to_calculator = os.getcwd()
            #self.program = EAM
            #self.programhess = None
            #self.gradient = True
            #self.hessian = False
            #self.pbc = True
            #self.template_file = None
            #self.template = None

        elif self.name.upper() == "EONCLIENT":
            print('**Warning: point-job of eonclient needs to print out forces.dat and energy.dat file')
            self.path_to_calculator = GetProgramPath(self.name)
            self.program = EON
            self.programhess = None
            self.gradient = True
            self.hessian = False
            self.pbc = True
            if self.template_file is None:
                raise RuntimeError("EON requires a template file")
            self.template = ReadEONTemplateFile(self.template_file)
            
        elif self.name.upper() == "MBG" or self.name.upper() == "MB":
            self.path_to_calculator = os.getcwd()
            if self.name.upper() == "MBG":
                self.program = MBG
            else:
                self.program = MB

            self.programhess = None
            self.gradient = True
            self.hessian = False
            self.pbc = False
            self.template_file = None
            self.template = None

        elif self.name.upper() == "PEAKS":
            self.path_to_calculator = os.getcwd()
            self.program = Peaks
            self.programhess = None
            self.gradient = True
            self.hessian = False
            self.pbc = False
            self.template_file = None
            self.template = None

        elif self.name.upper() == "LENNARDJONES":
            self.path_to_calculator = os.getcwd()
            self.program = LennardJones
            self.programhess = None
            self.gradient = True
            self.hessian = False
            self.pbc = False
            self.template_file = None
            self.template = None

        elif self.name.upper() == "DEBUG":
            self.path_to_calculator = os.getcwd()
            self.program = Debug
            self.programhess = None
            self.gradient = True
            self.hessian = False
            self.pbc = False
            self.template_file = None
            self.template = None

        elif self.name.upper() == "BOBDEBUG":
            self.path_to_calculator = os.getcwd()
            self.program = BobDebug
            self.programhess = None
            self.gradient = True
            self.hessian = False
            self.pbc = False
            self.template_file = None
            self.template = None

        elif self.name.upper() == "HENKELMAN":
            self.path_to_calculator = os.getcwd()
            self.program = Henkelman
            self.programhess = None
            self.gradient = True
            self.hessian = False
            self.pbc = False
            self.template_file = None
            self.template = None

        elif self.name.upper() == "HENKELMANGAUSS":
            self.path_to_calculator = os.getcwd()
            self.program = HenkelmanGaussBoosted
            self.programhess = None
            self.gradient = True
            self.hessian = False
            self.pbc = False
            self.template_file = None
            self.template = None

        elif self.name.upper() == "LEPSHO" or self.name.upper() == "LEPSHOGAUSS":
            self.path_to_calculator = os.getcwd()
            if self.name.upper() == "LEPSHO":
                self.program = LEPSHO
            else:
                self.program = LEPSHOGauss

            self.programhess = None
            self.gradient = True
            self.hessian = False
            self.pbc = False
            self.template_file = None
            self.template = None

        else:
            raise NotImplementedError("Calculator/potential not found")

        self.setup = True
        return None

    # =================================================

    def GetTemplate(self):
        return self.template

    def GetQCPath(self):

        if not self.setup:
            raise RuntimeError("Calculator %s has not been setup yet" % self.name)

        return self.path_to_calculator

    def GetCharge(self):

        if not self.setup:
            raise RuntimeError("Calculator %s has not been setup yet" % self.name)

        return self.charge

    def GetPBC(self):

        if not self.setup:
            raise RuntimeError("Calculator %s has not been setup yet" % self.name)

        return self.pbc

    def SetPBC(self, x=False):
        try:
            self.pbc = bool(x)
        except:
            raise TypeError("Expecting boolean argument")

    def SetCharge(self, x=1):
        try:
            self.charge = int(x)
        except:
            raise TypeError("Charge is an integer number")
        return

    def SetMultiplicity(self, x=1):
        try:
            self.multiplicity = int(x)
        except:
            raise TypeError("Multiplicity is an integer number")
        return

    def GetMultiplicity(self):

        if not self.setup:
            raise RuntimeError("Calculator %s has not been setup yet" % self.name)

        return self.multiplicity

    def numhess(self, atoms):

        if atoms.GetNim() > 1:
            raise NotImplementedError()

        if atoms.r is None:
            atoms.UpdateR()

        dr = self.fd_step
        x0 = atoms.GetR().copy()
        H = np.zeros(shape=(atoms.GetNDof(), atoms.GetNDof(), atoms.GetNim()))
        for i in range(atoms.GetNDof()):
            atoms.r[i] -= dr
            atoms.UpdateCoords()
            self.Compute(atoms)
            atoms.UpdateF()
            F1 = atoms.GetF().copy()
            atoms.SetR(x0)

            atoms.r[i] += dr
            atoms.UpdateCoords()
            self.Compute(atoms)
            atoms.UpdateF()
            F2 = atoms.GetF().copy()
            atoms.SetR(x0)

            tmp = -(F2 - F1) / (2.0 * dr)  # its negative because I use the forces!
            H[i, :, 0] = tmp.T

        atoms.SetR(x0)
        atoms.UpdateCoords()
        atoms.SetHessian(H)
        return H

    def ComputeCosSq(self, atoms):
        self.Compute(atoms)
        atoms.UpdateF()
        if atoms.IsTwoDee():
            grad = -atoms.GetF()[0:2]  # for 2d
        else:
            grad = -atoms.GetF()
        grad = grad / np.linalg.norm(grad)
        self.Compute(atoms, computeHessian=True)
        hessian = atoms.GetHessian()

        if atoms.IsTwoDee():
            hessian = hessian[0:2, 0:2, 0]  # for 2d
        eigenval, eigenvec = np.linalg.eigh(hessian)
        a = np.argmin(eigenval)
        omega = eigenvec[:, a]
        grad = grad / np.linalg.norm(grad)
        omega = omega / np.linalg.norm(omega)
        ndim = atoms.GetNDof()
        if atoms.IsTwoDee():
            ndim = 2
        omega = np.reshape(omega, (ndim, 1))
        return (np.dot(grad.T, omega))** 2

    def ComputeCosSqGrad(self, atoms):

        if atoms.GetNim() > 1:
            raise NotImplementedError()

        if atoms.r is None:
            atoms.UpdateR()

        dr = self.fd_step
        x0 = atoms.GetR().copy()
        grad = np.zeros(shape=(atoms.GetNDof(), 1))
        for i in range(atoms.GetNDof()):
            atoms.r[i] -= dr
            atoms.UpdateCoords()
            objf_l=self.ComputeCosSq(atoms)
            atoms.SetR(x0)

            atoms.r[i] += dr
            atoms.UpdateCoords()
            objf_r = self.ComputeCosSq(atoms)
            atoms.SetR(x0)

            grad[i] = (objf_r - objf_l) / (2.0*dr)
        return grad

    # ==================================================================
    # Electronic structure computation routine
    # ==================================================================
    def Compute(self, atoms, list_to_compute=None, computeHessian=False):
        if not self.setup:
            raise RuntimeError("Calculator %s has not been setup yet" % self.name)

        if not self.gradient:
            raise NotImplementedError("Calculator has to have gradient available")

        #  ENERGY AND  GRADIENT COMPUTATION
        if not computeHessian:
            # NUMERICAL GRADIENT COMPUTATION
            if not self.gradient:
                raise NotImplementedError()
            else:
                self.program(self, atoms, list_to_compute)
        else:
            # NUMERICAL HESSIAN COMPUTATION
            if not self.hessian:
                self.numhess(atoms)
            # ANALYTICAL HESSIAN COMPUTATION
            else:
                self.programhess(self, atoms)

        return None

    # =================================================

    def GetNCore(self):

        if not self.setup:
            raise RuntimeError("Calculator %s has not been setup yet" % self.name)

        return self.ncore

    def SetNCore(self, x=1):
        try:
            self.ncore = int(x)
        except:
            raise TypeError("NCore is an integer number")
        return

    def Delete(self):
        if not self.setup:
            raise RuntimeError("Calculator %s has not been setup yet - nothing to delete" % self.name)

        self.setup = False
        self.program = None
        return

    def PrintInfo(self):
        return None

    def GetInfo(self):
        listi = [self.name, self.charge, self.multiplicity, self.ncore, self.gradient, self.hessian, self.fd_step,
                 self.template_file, self.template, self.program, self.path_to_calculator, self.path, self.pbc,
                 self.setup]

        return listi
