#New interface to Knarr so that we don't have to change too much
#Assumes that Knarr directory exists inside Yggdrasill (for now at least)

from yggdrasill import *
import numpy as np
import sys
import os
#This makes Knarr part of python path
#Recommended way?
yggpath = os.path.dirname(yggdrasill.__file__)
sys.path.insert(0,yggpath+'/knarr')

from KNARRatom.utilities import InitializeAtomObject, InitializePathObject
from KNARRjobs.path import DoPathInterpolation
from KNARRio.io import ReadTraj
from KNARRjobs.neb import DoNEB
import KNARRatom.atom

#Define manual dicts here
#These will be reasonable defaults that can be overridden by special keywords in Yggdrasill NEB object
path_parameters = {"METHOD": "DOUBLE", "INTERPOLATION": "IDPP", "NIMAGES": 6,
              "INSERT_CONFIG": None, "IDPP_MAX_ITER": 2000,
              "IDPP_SPRINGCONST": 10.0, "IDPP_TIME_STEP": 0.01,
              "IDPP_MAX_MOVE": 0.1, "IDPP_MAX_F": 0.01, "IDPP_RMS_F": 0.005}

neb_settings = {"PATH": "neb.xyz",
              "CLIMBING": True,
              "TANGENT": "IMPROVED",
              "SPRINGTYPE": "DISTANCE",
              "PERP_SPRINGTYPE": "",
              "ENERGY_WEIGHTED": True, "SPRINGCONST": 1.0, "SPRINGCONST2": 10.0,
              "MIN_RMSD": True, "REMOVE_EXTERNAL_FORCE": True,
              "FREE_END": False, "FREE_END_TYPE": 'PERP', "FREE_END_ENERGY": 0.0,
              "FREE_END_ENERGY2": 0.0, "FREE_END_KAPPA": 0.0,
              "CONV_TYPE": "ALL", "TOL_SCALE": 10, "TOL_MAX_FCI": 0.026, "TOL_RMS_FCI": 0.013,
              "TOL_MAX_F": 0.026, "TOL_RMS_F": 0.013, "TOL_TURN_ON_CI": 1.0,
              "ZOOM": False,
              "TOL_TURN_ON_ZOOM": 0.5,
              "REPARAM": 0, "TOL_REPARAM": 0.0,
              "INTERPOLATION_TYPE": "LINEAR",
              "AUTO_ZOOM": True, "ZOOM_OFFSET": 1, "ZOOM_ALPHA": 0.5,
              "RESTART_OPT_ON_CI": True,
              "LBFGS_REPARAM_ON_RESTART": False
              }

optimizer = {"OPTIM_METHOD": "LBFGS", "MAX_ITER": 1000, "TOL_MAX_FORCE": 0.01,
             "TOL_RMS_FORCE": 0.005, "TIME_STEP": 0.01, "MAX_MOVE": 0.1, "RESTART_ON_SCALING": True,
             "LBFGS_MEMORY": 20,
             "LBFGS_DAMP": 1.0,
             "FD_STEP": 0.001,
             "LINESEARCH": None}

#Path generator
def Knarr_pathgenerator(nebsettings,path_parameters,react,prod):
    sett = nebsettings
    type_of_method_string = path_parameters["METHOD"].upper()
    if type_of_method_string == "SINGLE":
        prod_is_needed = False
    elif type_of_method_string == "DOUBLE":
        prod_is_needed = True
    else:
        raise TypeError("Either choose single or double ended path generation")

    nim = path_parameters["NIMAGES"]
    path = InitializePathObject(nim, react)

    if prod_is_needed:
        # Check product
        if react.GetNDim() != prod.GetNDim():
            raise RuntimeError("Reactant / product do not match")
        if react.GetSymbols() != prod.GetSymbols():
            raise RuntimeError("Reactant / product do not match")
        path.SetConfig2(prod.GetCoords())

        # check insertion
        insertion = path_parameters["INSERT_CONFIG"]
        if insertion is not None:
            insertion = InitializeAtomObject(name="insertion", input_config=path_parameters["INSERT_CONFIG"],
                                             pbc=main_control["PBC"])
            if insertion.GetSymbols() != react.GetSymbols():
                raise ValueError("Insertion does not match reactant / product")
            path.SetInsertionConfig(insertion.GetCoords())
    else:
        prod = None
    DoPathInterpolation(path, path_parameters)

#Convert coordinates list to Knarr-type array
def coords_to_Knarr(coords):
    coords_xyz=[]
    for i in coords:
        coords_xyz.append([i[0]]);coords_xyz.append([i[1]]);coords_xyz.append([i[2]])
    coords_xyz_np=np.array(coords_xyz)
    return coords_xyz_np

#Wrapper around Yggdrasill object
class KnarrCalculator:
    def __init__(self,theory,fragment1,fragment2,runmode='serial'):
        self.forcecalls=0
        self.theory=theory
        #Yggdrasill fragments for reactant and product
        #Used for element list and keep track of full system if QM/MM
        self.fragment1=fragment1
        self.fragment2=fragment2
        self.runmode=runmode
    def Compute(self,path, list_to_compute=[]):
        #
        counter=0
        F = np.zeros(shape=(path.GetNDimIm() * path.GetNim(), 1))
        E = np.zeros(shape=(path.GetNim(), 1))
        numatoms=int(path.ndofIm/3)
        if self.runmode=='serial':
            for image_number in list_to_compute:
                image_coords_1d = path.GetCoords()[image_number * path.ndimIm : (image_number + 1) * path.ndimIm]
                image_coords=np.reshape(image_coords_1d, (numatoms, 3))
                # Request Engrad calc
                En_image, Grad_image = self.theory.run(current_coords=image_coords, elems=self.fragment1.elems, Grad=True)
                counter += 1
                #Energies array for all images
                E[image_number]=En_image
                #Forces array for all images
                #Todo: Check units
                F[image_number* path.ndimIm : (image_number + 1) * path.ndimIm] = -1 * np.reshape(Grad_image,(int(path.ndofIm),1))
                print(F[image_number])
        elif self.runmode=='parallel':
            print("not yet done")
            exit()
        #Note: F and E is list/arrays of Forces and and Energy for all images in list provided
        path.SetForces(F)
        path.SetEnergy(E)
        #Forcecalls
        path.AddFC(counter)
    def AddFC(self, x=1):
        self.forcecalls += x
        return



#Yggdrasill NEB function. Calls Knarr
def NEB(reactant=None, product=None, theory=None, images=None, interpolation=None):
    if reactant==None or product==None or theory==None:
        print("You need to provide reactant and product fragment and a theory to NEB")
        exit()
    numatoms = reactant.numatoms

    #Override some default settings
    if images is not None:
        path_parameters["NIMAGES"]=images
    if interpolation is not None:
        path_parameters["INTERPOLATION"]=interpolation

    #Create Knarr calculator from Yggdrasill theory
    calculator = KnarrCalculator(theory, fragment1=reactant, fragment2=product)

    #Zero-valued constraints list. We probably won't use constraints for now
    constr = np.zeros(shape=(numatoms * 3, 1))

    # Symbols list for Knarr
    Knarr_symbols = [y for y in reactant.elems for i in range(3)]

    # Create KNARR Atom objects. Used in path generation
    react = KNARRatom.atom.Atom(coords=coords_to_Knarr(reactant.coords), symbols=Knarr_symbols, ndim=numatoms * 3,
                                ndof=numatoms * 3, constraints=constr, pbc=False)
    prod = KNARRatom.atom.Atom(coords=coords_to_Knarr(product.coords), symbols=Knarr_symbols, ndim=numatoms * 3,
                               ndof=numatoms * 3, constraints=constr, pbc=False)

    # Generate path via Knarr_pathgenerator
    Knarr_pathgenerator(neb_settings, path_parameters, react, prod)

    #Reading initial path from XYZ file. Hardcoded as knarr_path.xyz
    rp, ndim, nim, symb = ReadTraj("knarr_path.xyz")
    path = InitializePathObject(nim, react)
    path.SetCoords(rp)

    #Now starting NEB from path object, using neb_settings and optimizer settings
    DoNEB(path, calculator, neb_settings, optimizer)